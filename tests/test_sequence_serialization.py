"""统一设计文档 §23 Level 0 / §20：序列化与约束。

模型往返（含 a 依赖参数）、LangevinConfig 新字段默认关闭（向后兼容）、
SequenceConstraints 的构造校验与 logits 掩码。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import aa_index_of, randomize

from socg.dynamics import LangevinConfig
from socg.model.socg import ModelConfig, SOCGModel, load_model, save_model
from socg.sequence import SequenceConstraints, UniformSequenceProposal
from socg.topology import CGTopology


def test_save_load_roundtrip_with_a_dependent_params(tmp_path):
    """突变参数（w_pair_type 非零）下 save/load 逐位一致，ΔU 接口可用。"""
    topo = CGTopology.from_sequence("ACDEFGHIKLMN")
    rng = np.random.default_rng(13)
    coords = rng.normal(0, 0.3, size=(64, topo.n, 3)) + 1.0
    from socg.model.priors import Priors
    priors = Priors.fit(topo, coords, 300.0)
    model = SOCGModel(topo, priors, ModelConfig()).double()
    randomize(model, 0.3, seed=19)
    path = tmp_path / "model.pt"
    save_model(path, model, meta={"system": "mixed"})
    model2 = load_model(path)                      # v1 -> v1：strict、无告警
    R = torch.as_tensor(coords[:2])
    s = torch.as_tensor(rng.integers(0, 3, size=(2, topo.n)), dtype=torch.long)
    a = aa_index_of(model)
    site = torch.tensor([2, 5])
    aa_new = torch.tensor([7, 1])
    with torch.no_grad():
        assert torch.equal(model.energy(R, s, a), model2.energy(R, s, a))
        assert torch.equal(model.mutation_delta_u_local(R, s, a, site, aa_new),
                           model2.mutation_delta_u_local(R, s, a, site, aa_new))
    assert model2._meta["system"] == "mixed"


def test_langevin_config_defaults_keep_sequence_moves_off():
    """新字段默认关闭：既有构造不触发序列移动，也不需要提议算子。"""
    cfg = LangevinConfig(temperature=300.0)
    assert cfg.seq_interval == 0 and cfg.seq_update == "full"
    cfg_local = LangevinConfig(temperature=300.0, seq_interval=10, seq_update="local")
    assert cfg_local.seq_interval == 10 and cfg_local.seq_update == "local"
    with pytest.raises(ValueError, match="seq_update"):
        LangevinConfig(temperature=300.0, seq_interval=1, seq_update="bogus")


def test_proposal_without_seq_interval_rejected(toy_topology, toy_priors):
    from socg.dynamics import CGSimulator
    model = SOCGModel(toy_topology, toy_priors, ModelConfig())
    R0 = torch.zeros(2, toy_topology.n, 3)
    s0 = torch.zeros(2, toy_topology.n, dtype=torch.long)
    cfg = LangevinConfig(temperature=300.0, seq_interval=0)
    with pytest.raises(ValueError, match="seq_interval"):
        CGSimulator(model, cfg, R0, s0, aa_index_of(model),
                    sequence_proposal=UniformSequenceProposal())


def test_constraints_from_topology_variants():
    topo = CGTopology.from_sequence("GPAACDEFGH")
    interior = SequenceConstraints.from_topology(topo, mutable="interior")
    assert bool(interior.mutable_mask[0]) is False
    assert bool(interior.mutable_mask[5]) is True
    assert int(interior.mutable_mask.sum()) == topo.n - 2

    restricted = SequenceConstraints.from_topology(
        topo, mutable="all", allowed_letters={3: "KRH", 4: "AGPC"},
        forbidden_letters={4: "C"})
    logits = torch.zeros(1, topo.n, 20)
    masked = restricted.mask_logits(logits)
    alph = "ACDEFGHIKLMNPQRSTVWY"
    # 位点 1 不在 allowed 字典里 -> 整行禁
    assert bool(torch.isinf(masked[0, 1, :]).all())
    # 位点 3 allowed = {K,R,H}
    assert float(masked[0, 3, alph.index("K")]) == 0.0
    assert float(masked[0, 3, alph.index("A")]) == float("-inf")
    assert float(masked[0, 3, alph.index("R")]) == 0.0
    # 位点 4 allowed ∩ 禁 C：C 禁、A 允许
    assert float(masked[0, 4, alph.index("C")]) == float("-inf")
    assert float(masked[0, 4, alph.index("A")]) == 0.0

    with pytest.raises(ValueError, match="mutable"):
        SequenceConstraints.from_topology(topo, mutable="whatever")
    with pytest.raises(ValueError, match="out of range"):
        SequenceConstraints.from_topology(topo, mutable="all",
                                          allowed_letters={99: "A"})


def test_constraints_tied_positions_validation():
    topo = CGTopology.from_sequence("GPAACDEFGH")
    with pytest.raises(ValueError, match="immutable"):
        SequenceConstraints.from_topology(
            topo, mutable="interior", tied_positions=((0, 3),))   # 0 首残基不可变
    ok = SequenceConstraints.from_topology(topo, mutable="all",
                                           tied_positions=((2, 3, 4),))
    assert ok.tied_positions == ((2, 3, 4),)
    with pytest.raises(ValueError, match="invalid tied"):
        SequenceConstraints.from_topology(topo, mutable="all",
                                          tied_positions=((1,),))


def test_constraints_validate_sequence():
    topo = CGTopology.from_sequence("AAAAAA")
    cons = SequenceConstraints.from_topology(topo, mutable="all",
                                             allowed_letters={2: "AG"})
    a_bad = torch.zeros(1, 6, dtype=torch.long)          # 全 A
    cons.validate_sequence(a_bad)                        # A 在 allowed 内 -> 通过
    a_bad[0, 2] = 5                                      # G 也可以
    cons.validate_sequence(a_bad)
    a_bad[0, 2] = 8                                      # I 不在 {A,G} 内
    with pytest.raises(ValueError, match="violates allowed_aa"):
        cons.validate_sequence(a_bad)
