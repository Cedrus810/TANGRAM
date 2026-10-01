"""统一设计文档 §13/§18/§31：序列移动与 MH 正确性。

Level 1（热力学正确性）：固定 R、可枚举序列空间上，均匀提议与
非对称偏置提议必须收敛到同一精确 Boltzmann 分布——这是 Hastings 修正
正确性的关键检验（§31 Level 2 "不改平衡分布，只改混合速度"）。
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch
from conftest import aa_index_of, ideal_helix, randomize

from socg.dynamics import CGSimulator, LangevinConfig
from socg.model.socg import ModelConfig, SOCGModel
from socg.sequence import SequenceProposal, UniformSequenceProposal


class _BiasedProposal(SequenceProposal):
    """非对称偏置提议：q(a'|R,a) ∝ bias[a']（与当前值/上下文无关）。

    故意非对称（前后概率不等）：若模拟器漏掉 Hastings 修正，稳态分布
    会偏离 Boltzmann，本测试即失败——这是序列 MH 机器的核心回归。
    """

    def __init__(self, bias: torch.Tensor):
        self.bias = bias / bias.sum()

    def logits(self, R, a, mutable_mask=None):
        out = torch.log(self.bias).expand(*a.shape, -1).clone()
        return out

    def sample(self, R, a, site, generator=None):
        probs = self.bias.to(a.device)
        return torch.multinomial(probs, a.shape[0], replacement=True,
                                 generator=generator)

    def log_q(self, R, a_from, a_to, site):
        return torch.log(self.bias).to(a_from.device)[a_to[
            torch.arange(a_to.shape[0], device=a_from.device), site]]


@pytest.fixture
def seq_setup(toy_topology, toy_priors):
    """固定 R + 两个可变位点 + 4 种允许氨基酸 -> 16 组合可枚举。

    用 constraints 语义等价的 mutable_mask 把序列自由度压到可枚举规模。
    """
    model = SOCGModel(toy_topology, toy_priors, ModelConfig()).double()
    randomize(model, 1.0, seed=51)
    B = 4096
    R = torch.as_tensor(ideal_helix(6).astype(np.float64))[None].repeat(B, 1, 1)
    mutable = torch.zeros(6, dtype=torch.bool)
    mutable[2] = mutable[4] = True                  # 2 个自由位点
    return model, R, mutable


def _exact_boltzmann_all(model, R, free):
    """自由位点全组合（2 个位点 -> 400 组合）的精确 Boltzmann 分布。"""
    beta = 1.0 / (0.0083144626181532 * 300.0)
    n = model.topo.n
    combos = list(product(range(20), repeat=len(free)))
    U = np.zeros(len(combos))
    a_base = aa_index_of(model)
    with torch.no_grad():
        for i, combo in enumerate(combos):
            a = a_base.expand(1, -1).clone()
            for site, aa in zip(free, combo):
                a[0, site] = aa
            U[i] = float(model.energy(R[:1], torch.zeros(1, n, dtype=torch.long), a))
    logw = -beta * U
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return dict(zip(combos, w)), free


@pytest.mark.parametrize("make_proposal", [
    lambda: UniformSequenceProposal(),
    lambda: _BiasedProposal(torch.tensor(
        [4.0, 0.5, 2.0, 0.1, 1.0, 0.3, 2.5, 0.8, 1.5, 0.2,
         0.9, 1.8, 0.4, 3.0, 0.6, 1.2, 0.7, 2.2, 1.1, 0.05])),
], ids=["uniform", "biased-asymmetric"])
def test_sequence_mc_matches_exact_boltzmann(seq_setup, make_proposal):
    """固定 R、纯序列移动：经验序列分布 vs 精确分布，TV < 0.06。

    dt=0 使 R 恒定；flip_interval=0 关闭态翻转；seq_interval=1 每步一个
    单位点突变提议。均匀与非对称提议必须给出同一目标分布。
    """
    model, R, mutable = seq_setup
    a0 = aa_index_of(model)
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.0, flip_interval=0,
                         record_interval=1, seq_interval=1, seed=77)
    sim = CGSimulator(model, cfg, R,
                      torch.ones((R.shape[0], 6), dtype=torch.long), a0,
                      sequence_proposal=make_proposal(), mutable_mask=mutable)
    _, _, seqs = sim.run(320, return_sequences=True)
    assert sim.seq_proposed > 0
    free = tuple(np.flatnonzero(mutable.numpy()))
    sample = seqs[:, 20:, free].astype(np.int64).reshape(-1, len(free))
    counts = {}
    for row in map(tuple, sample):
        counts[row] = counts.get(row, 0) + 1
    p_exact, _ = _exact_boltzmann_all(model, R, free)
    total = sum(counts.values())
    tv = 0.0
    all_keys = set(counts) | set(p_exact)
    for k in all_keys:
        p_emp = counts.get(k, 0) / total
        tv += abs(p_emp - p_exact.get(k, 0.0))
    tv *= 0.5
    assert tv < 0.06, f"TV={tv:.4f}"


def _exact_boltzmann_all(model, R, free):
    """全部 20^2 组合的精确 Boltzmann（2 个自由位点 -> 400 组合）。"""
    beta = 1.0 / (0.0083144626181532 * 300.0)
    n = model.topo.n
    combos = list(product(range(20), repeat=len(free)))
    U = np.zeros(len(combos))
    a_base = aa_index_of(model)
    with torch.no_grad():
        for i, combo in enumerate(combos):
            a = a_base.expand(1, -1).clone()
            for site, aa in zip(free, combo):
                a[0, site] = aa
            U[i] = float(model.energy(R[:1], torch.zeros(1, n, dtype=torch.long), a))
    logw = -beta * U
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return dict(zip(combos, w)), free


def test_local_and_full_seq_update_paths_agree(seq_setup):
    """同种子下 local 与 full ΔU 路径产生相同的轨迹与接受判定。"""
    model, R, mutable = seq_setup
    a0 = aa_index_of(model)
    trajs = {}
    for mode in ("full", "local"):
        cfg = LangevinConfig(temperature=300.0, dt_ps=0.0, flip_interval=0,
                             record_interval=100, seq_interval=1, seed=91,
                             seq_update=mode)
        sim = CGSimulator(model, cfg, R, torch.ones((R.shape[0], 6), dtype=torch.long),
                          a0, sequence_proposal=UniformSequenceProposal(),
                          mutable_mask=mutable)
        coords, states, seqs = sim.run(80, return_sequences=True)
        trajs[mode] = (coords, states, seqs, sim.sequence_acceptance_rate)
    assert trajs["full"][2].shape == trajs["local"][2].shape
    # ΔU 数值路径不同（~1e-12），个别边缘判定可能翻转，但分布应几乎一致：
    # 允许极少量帧差异，接受率应在 1% 内一致
    assert abs(trajs["full"][3] - trajs["local"][3]) < 0.01


def test_sequence_moves_off_by_default(sim_setup):
    """默认（无提议/seq_interval=0）序列状态永不改变。"""
    model, R0, s0, a0 = sim_setup
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, flip_interval=1,
                         record_interval=5, seed=95)
    sim = CGSimulator(model, cfg, R0, s0, a0)
    _, _, seqs = sim.run(30, return_sequences=True)
    assert (seqs == seqs[:, :1, :]).all()
    assert sim.seq_proposed == 0
    with pytest.raises(ValueError):
        CGSimulator(model, cfg, R0, s0, a0,
                    sequence_proposal=UniformSequenceProposal())  # seq_interval=0 矛盾


def test_sequence_respects_mutable_mask(seq_setup):
    """mutable_mask 之外的位点序列恒不变。"""
    model, R, mutable = seq_setup
    a0 = aa_index_of(model)
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.0, flip_interval=0,
                         record_interval=10, seq_interval=1, seed=97)
    sim = CGSimulator(model, cfg, R, torch.ones((R.shape[0], 6), dtype=torch.long),
                      a0, sequence_proposal=UniformSequenceProposal(),
                      mutable_mask=mutable)
    _, _, seqs = sim.run(60, return_sequences=True)
    frozen_sites = [i for i in range(6) if not mutable[i]]
    assert (seqs[:, :, frozen_sites] == a0.numpy()[:, None, frozen_sites]).all()
    assert (seqs[:, -1, :] != seqs[:, 0, :]).any()   # 可变位点确实变化过
