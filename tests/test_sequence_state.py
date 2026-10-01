"""统一设计文档 §8/§9/§15：运行时序列状态与提议算子接口。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from socg.constants import AA_ALPHABET
from socg.sequence import SequenceProposal, SequenceState, UniformSequenceProposal
from socg.topology import (
    RES_CLASS_GLY,
    RES_CLASS_OTHER,
    RES_CLASS_PRO,
    RES_CLASS_TABLE,
    CGTopology,
)


def test_res_class_table_covers_all_20():
    assert RES_CLASS_TABLE.shape == (20,)
    assert int(RES_CLASS_TABLE[AA_ALPHABET.index("G")]) == RES_CLASS_GLY
    assert int(RES_CLASS_TABLE[AA_ALPHABET.index("P")]) == RES_CLASS_PRO
    for ch in "ACDEFGHIKLMNPQRSTVWY":
        if ch not in "GP":
            assert int(RES_CLASS_TABLE[AA_ALPHABET.index(ch)]) == RES_CLASS_OTHER


def test_res_class_from_a_matches_topology():
    topo = CGTopology.from_sequence("GPAACDEFGHIKLMNPQRSTVWYP")
    a = torch.as_tensor(topo.aa_index, dtype=torch.long)
    table = torch.as_tensor(RES_CLASS_TABLE, dtype=torch.long)
    assert torch.equal(table[a], torch.as_tensor(topo.res_class, dtype=torch.long))


def test_sequence_state_from_topology_and_batch():
    topo = CGTopology.from_sequence("ACDEFG")
    st = SequenceState.from_topology(topo)
    assert st.mutable_mask is None
    batch = st.to_batch(4)
    assert batch.aa_index.shape == (4, 6)
    assert torch.equal(batch.aa_index[2], st.aa_index)
    # 已是目标 batch 时原样返回同一视图
    again = batch.to_batch(4)
    assert again.aa_index is batch.aa_index
    with pytest.raises(ValueError):
        batch.to_batch(3)


def test_sequence_state_clone_is_independent():
    st = SequenceState.from_topology(CGTopology.from_sequence("ACDEFG")).to_batch(2)
    st2 = st.clone()
    st2.aa_index[0, 0] = (st2.aa_index[0, 0] + 1) % 20
    assert not torch.equal(st2.aa_index, st.aa_index)


def test_mutable_mask_validation():
    topo = CGTopology.from_sequence("ACDEFG")
    with pytest.raises(ValueError):
        SequenceState.from_topology(topo, mutable=torch.ones(5, dtype=torch.bool))
    st = SequenceState.from_topology(topo, mutable=torch.tensor(
        [False, True, True, True, True, False]))
    b = st.to_batch(3)
    assert b.mutable_mask.shape == (3, 6)
    assert not bool(b.mutable_mask[:, 0].any())


def test_proposal_base_is_abstract():
    prop = SequenceProposal()
    R = torch.zeros(1, 4, 3)
    a = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(NotImplementedError):
        prop.logits(R, a)
    with pytest.raises(NotImplementedError):
        prop.sample(R, a, torch.zeros(1, dtype=torch.long))
    with pytest.raises(NotImplementedError):
        prop.log_q(R, a, a, torch.zeros(1, dtype=torch.long))


def test_uniform_proposal_never_repeats_current_aa():
    topo = CGTopology.from_sequence("ACDEFGHIKLMN")
    st = SequenceState.from_topology(topo).to_batch(512)
    prop = UniformSequenceProposal()
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(1)
    sites = torch.as_tensor(rng.integers(0, topo.n, size=512), dtype=torch.long)
    new = prop.sample(torch.zeros(512, topo.n, 3), st.aa_index, sites, generator=gen)
    old = st.aa_index[torch.arange(512), sites]
    assert not bool((new == old).any())             # 19 种替代，绝不重复当前值
    counts = torch.bincount(new, minlength=20)
    assert int((counts > 0).sum()) == 20            # 20 种氨基酸全部出现
    # log_q：均匀提议，log(1/19)
    logq = prop.log_q(torch.zeros(512, topo.n, 3), st.aa_index,
                      st.aa_index.clone(), sites)
    assert torch.allclose(logq, torch.full_like(logq, -np.log(19)))


def test_uniform_proposal_logits_mask_immutable():
    topo = CGTopology.from_sequence("AAAAAA")
    st = SequenceState.from_topology(
        topo, mutable=torch.tensor([False, True, True, True, True, False])).to_batch(1)
    prop = UniformSequenceProposal()
    logits = prop.logits(torch.zeros(1, 6, 3), st.aa_index, st.mutable_mask)
    assert logits.shape == (1, 6, 20)
    assert bool(torch.isinf(logits[:, [0, 5], :]).all())
    assert bool(torch.isfinite(logits[:, 1:5, :]).all())
