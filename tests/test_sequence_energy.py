"""统一设计文档 §10/§29：能量分项与突变能量学。

验证 mutation_delta_u 的 local 路径（§29.2）与批量 20-AA 路径（§29.3）
都数学恒等于全能量差分（浮点路径不同，用 1e-9 容差）；energy_terms 求和
与 energy() 一致；替换为当前氨基酸的 ΔU 恒为 0。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import aa_index_of, randomize

from socg.model.socg import ModelConfig, SOCGModel


@pytest.fixture
def mixed_model():
    """20 种氨基酸混合序列的 float64 模型（覆盖全部 res_class 与类型对）。"""
    from conftest import ideal_helix

    from socg.model.priors import Priors
    from socg.topology import CGTopology
    topo = CGTopology.from_sequence("ACDEFGHIKLMNPQRSTVWY")
    rng = np.random.default_rng(11)
    coords = ideal_helix(topo.n).astype(np.float64)[None].repeat(64, axis=0)
    coords = coords + rng.normal(0.0, 0.02, size=coords.shape)
    priors = Priors.fit(topo, coords, 300.0)
    model = SOCGModel(topo, priors, ModelConfig()).double()
    return randomize(model, 0.3, seed=17)


@pytest.fixture
def mixed_inputs(mixed_model):
    rng = np.random.default_rng(23)
    n = mixed_model.topo.n
    R = torch.as_tensor(rng.normal(0.3, 0.08, size=(4, n, 3)) + 1.0)
    s = torch.as_tensor(rng.integers(0, 3, size=(4, n)), dtype=torch.long)
    return R, s, aa_index_of(mixed_model)


def test_energy_terms_sum_matches_energy(mixed_model, mixed_inputs):
    R, s, a = mixed_inputs
    terms = mixed_model.energy_terms(R, s, a)
    expected_keys = {"bond", "angle", "dihedral", "eps", "w_nn",
                     "pair_state", "pair_type", "prior"}
    assert set(terms) == expected_keys
    total = sum(terms.values())
    assert torch.allclose(total, mixed_model.energy(R, s, a), rtol=1e-10, atol=1e-10)


def test_energy_terms_no_pair_type_when_disabled(toy_topology, toy_priors,
                                                 toy_coords):
    model = SOCGModel(toy_topology, toy_priors, ModelConfig(pair_types=False))
    R = torch.as_tensor(toy_coords[:2].astype(np.float64))
    s = torch.zeros(2, toy_topology.n, dtype=torch.long)
    terms = model.energy_terms(R, s, aa_index_of(model))
    assert float(terms["pair_type"].abs().sum()) == 0.0


def _sample_sites(model, B, rng, n):
    return torch.as_tensor(rng.integers(1, n - 1, size=B), dtype=torch.long)


def test_mutation_delta_u_local_matches_full(mixed_model, mixed_inputs):
    R, s, a = mixed_inputs
    B, n = R.shape[0], R.shape[1]
    rng = np.random.default_rng(31)
    model = mixed_model
    site = _sample_sites(model, B, rng, n)
    aa_new = torch.as_tensor(rng.integers(0, 20, size=B), dtype=torch.long)
    d_local = model.mutation_delta_u_local(R, s, a, site, aa_new)
    d_full = model.mutation_delta_u_full(R, s, a, site, aa_new)
    assert torch.allclose(d_local, d_full, rtol=1e-9, atol=1e-9)


def test_mutation_delta_u_all_aa_matches_full(mixed_model, mixed_inputs):
    R, s, a = mixed_inputs
    B, n = R.shape[0], R.shape[1]
    rng = np.random.default_rng(37)
    model = mixed_model
    site = _sample_sites(model, B, rng, n)
    d_all = model.mutation_delta_u_all_aa(R, s, a, site)
    assert d_all.shape == (B, 20)
    # 替换为当前氨基酸：ΔU 精确为 0
    aa_site = a[torch.arange(B), site]
    assert torch.equal(d_all[torch.arange(B), aa_site],
                       torch.zeros(B, dtype=d_all.dtype))
    # 与全能量差分一致（每个 replica 抽 3 种替换核对）
    for b in range(B):
        for k in rng.choice(20, size=3, replace=False):
            d_full = model.mutation_delta_u_full(
                R[b:b + 1], s[b:b + 1], a[b:b + 1],
                site[b:b + 1], torch.tensor([int(k)]))
            assert torch.allclose(d_all[b, int(k)], d_full[0],
                                  rtol=1e-9, atol=1e-9), (b, int(k))


def test_mutation_delta_u_k1(toy_topology, toy_priors):
    """K=1 基线：突变 ΔU 仍精确（eps 项在 K=1 时为常数，角度/类型对仍贡献）。"""
    from conftest import ideal_helix
    model = SOCGModel(toy_topology, toy_priors, ModelConfig(n_states=1)).double()
    randomize(model, 0.3, seed=41)
    R = torch.as_tensor(ideal_helix(6).astype(np.float64))[None]
    s = torch.zeros(1, 6, dtype=torch.long)
    a = aa_index_of(model)
    site = torch.tensor([2])
    d_local = model.mutation_delta_u_local(R, s, a, site, torch.tensor([5]))
    d_full = model.mutation_delta_u_full(R, s, a, site, torch.tensor([5]))
    assert torch.allclose(d_local, d_full, rtol=1e-9, atol=1e-9)


def test_mutation_delta_u_terminal_site_pair_only(mixed_model, mixed_inputs):
    """首尾残基没有角度顶点项：ΔU 只能来自 eps 与类型对（仍须与全差分一致）。"""
    R, s, a = mixed_inputs
    model = mixed_model
    site = torch.tensor([0, model.topo.n - 1])
    aa_new = torch.tensor([3, 9])
    d_local = model.mutation_delta_u_local(R, s, a, site, aa_new)
    d_full = model.mutation_delta_u_full(R, s, a, site, aa_new)
    assert torch.allclose(d_local, d_full, rtol=1e-9, atol=1e-9)
