"""T9: 混合 Langevin + Metropolis 模拟器。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from socg.dynamics import CGSimulator, LangevinConfig
from socg.model.socg import ModelConfig, SOCGModel


@pytest.fixture
def sim_setup(toy_topology, toy_priors):
    from conftest import ideal_helix, randomize

    model = SOCGModel(toy_topology, toy_priors, ModelConfig()).double()
    randomize(model, 1.0, seed=21)
    rng = np.random.default_rng(0)
    R0 = np.tile(ideal_helix(6).astype(np.float64)[None], (8, 1, 1))
    R0 = R0 + rng.normal(0, 0.01, size=R0.shape)
    s0 = rng.integers(0, 3, size=(8, 6))
    a0 = torch.as_tensor(toy_topology.aa_index, dtype=torch.long)
    return model, torch.as_tensor(R0), torch.as_tensor(s0, dtype=torch.long), a0


def test_exact_state_sampling_fixed_r(toy_topology, toy_priors):
    """固定 R、只做翻转：经验分布 vs 穷举 Boltzmann 分布，TV < 0.03。

    用 dt=0 使动力学成为恒等变换（c1=exp(0)=1，无噪声、无位移）。
    """
    from itertools import product

    from conftest import ideal_helix, randomize

    model = SOCGModel(toy_topology, toy_priors, ModelConfig()).double()
    randomize(model, 1.0, seed=31)
    R = torch.as_tensor(ideal_helix(6).astype(np.float64))[None]
    R = R.repeat(2048, 1, 1)
    s0 = torch.ones((2048, 6), dtype=torch.long)   # 冻结位=1，可动位起点相同
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.0, flip_interval=1,
                         record_interval=1, seed=99)
    a = torch.as_tensor(toy_topology.aa_index, dtype=torch.long)
    sim = CGSimulator(model, cfg, R, s0, a)
    coords, states = sim.run(121)
    # 第 t 帧之前已做 t-1 个 sweep（首个 sweep 在第 1 步之后）：
    # 取 t = 21..120 -> 每个 replica 预热 >= 20 个 sweep 后采集 100 个 sweep，所有 replica 合并
    assert states.shape[1] == 122
    sample = states[:, 21:121, 1:5].reshape(-1, 4).astype(np.int64)   # 4 个可动位点
    assert sample.shape == (2048 * 100, 4)
    counts = np.zeros((3,) * 4)
    np.add.at(counts, tuple(sample.T), 1.0)
    p_emp = counts / counts.sum()
    # 穷举 Boltzmann
    beta = 1.0 / (0.0083144626181532 * 300.0)
    p_bolt = np.zeros((3,) * 4)
    with torch.no_grad():
        for combo in product(range(3), repeat=4):
            s = torch.ones((1, 6), dtype=torch.long)
            s[0, 1:5] = torch.tensor(combo)
            p_bolt[combo] = np.exp(-beta * float(model.energy(R[:1], s, a)))
    p_bolt /= p_bolt.sum()
    tv = 0.5 * np.abs(p_emp - p_bolt).sum()
    assert tv < 0.03


def test_nve_energy_conservation(sim_setup):
    model, R0, s0, a0 = sim_setup
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=0.0,
                         flip_interval=0, record_interval=1, seed=1)
    n = 6
    kT = 0.0083144626181532 * 300.0
    sim = CGSimulator(model, cfg, R0, s0, a0)
    traj = []
    for _ in range(2000):
        sim._step()
        traj.append(float(sim.potential_energy().mean() + sim.kinetic_energy().mean()))
    traj = np.asarray(traj)
    assert traj.std() / (n * kT) < 0.01
    # 无单调漂移
    half = len(traj) // 2
    assert abs(traj[half:].mean() - traj[:half].mean()) / (n * kT) < 0.01


def test_equipartition(sim_setup):
    model, R0, s0, a0 = sim_setup
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=5.0,
                         flip_interval=0, record_interval=1000, seed=2)
    sim = CGSimulator(model, cfg, R0, s0, a0)
    sim.run(1000)                                  # 预热
    ke = []
    for _ in range(3000):
        sim._step()
        ke.append(float(sim.kinetic_energy().mean()))
    n_dof = 3 * 6
    kT = 0.0083144626181532 * 300.0
    mean_ke_per_dof = np.mean(ke) / n_dof
    assert mean_ke_per_dof == pytest.approx(kT / 2, rel=0.03)


def test_frozen_sites_never_flip(sim_setup):
    model, R0, s0, a0 = sim_setup
    s0 = s0.clone()
    s0[:, 0] = 2                                    # 冻结位的初值（会被归位）
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=1.0,
                         flip_interval=1, record_interval=5, seed=3)
    sim = CGSimulator(model, cfg, R0, s0, a0)
    coords, states = sim.run(100)
    assert (states[:, :, 0] == 1).all()             # 冻结位恒为 FROZEN_STATE
    assert (states[:, :, 5] == 1).all()
    assert (states[:, :, 1:5] != 1).any()           # 可动位确实翻转过
    assert 0.01 < sim.acceptance_rate < 0.99


def test_k1_never_flips(toy_topology, toy_priors):
    from conftest import ideal_helix

    model = SOCGModel(toy_topology, toy_priors, ModelConfig(n_states=1)).double()
    R0 = torch.as_tensor(ideal_helix(6).astype(np.float64))[None].repeat(4, 1, 1)
    s0 = torch.randint(0, 3, (4, 6))
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.0, flip_interval=1,
                         record_interval=1, seed=4)
    sim = CGSimulator(model, cfg, R0, s0, torch.as_tensor(toy_topology.aa_index,
                                                          dtype=torch.long))
    coords, states = sim.run(50)
    assert sim.proposed == 0
    assert sim.acceptance_rate == 0.0
    assert (states == 0).all()                      # prepare_states 全置 0


def test_explosion_raises_runtimeerror(sim_setup):
    # 触发真正的数值爆炸：键先验刚度 1e10（ω·dt ≈ 27 ≫ 2，BAOAB 指数发散直至溢出）。
    # 注：计划原文“把某个参数设为 1e6”不可靠——等权 RBF 叠加近似平台，力并不大。
    model, R0, s0, a0 = sim_setup
    with torch.no_grad():
        model.priors.k_bond.fill_(1e10)
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=0.0,
                         flip_interval=0, record_interval=10, seed=5)
    sim = CGSimulator(model, cfg, R0, s0, a0)
    with pytest.raises(RuntimeError) as excinfo:
        sim.run(500)
    assert "step" in str(excinfo.value) and "replica" in str(excinfo.value)


def test_record_shapes(sim_setup):
    model, R0, s0, a0 = sim_setup
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=1.0,
                         flip_interval=0, record_interval=10, seed=6)
    sim = CGSimulator(model, cfg, R0, s0, a0)
    coords, states = sim.run(45)
    assert coords.shape == (8, 5, 6, 3)             # T = 45//10 + 1
    assert coords.dtype == np.float32
    assert states.dtype == np.int8


def test_freeze_mode_isolates_exploded_replica(sim_setup):
    # 审阅 P1-6：一个 replica 爆炸 -> 冻结并记录，其余继续；默认 raise 模式不变
    model, R0, s0, a0 = sim_setup
    R0 = R0.clone()
    R0[3, 2] = float("nan")                        # 只让 replica 3 出现非有限值
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=1.0,
                         flip_interval=5, record_interval=10, seed=7)
    with pytest.raises(RuntimeError):
        CGSimulator(model, cfg, R0, s0, a0).run(20)
    sim = CGSimulator(model, cfg, R0, s0, a0, on_nonfinite="freeze")
    coords, states = sim.run(50)
    mask = sim.exploded_mask
    assert mask.tolist() == [False, False, False, True, False, False, False, False]
    assert sim.exploded_step[3] == 1 and (sim.exploded_step[~mask] == -1).all()
    assert np.isfinite(coords[~mask]).all()
    assert np.ptp(coords[0], axis=0).max() > 0     # 其余 replica 仍在运动


def test_run_record_initial_false_no_duplicate(sim_setup):
    model, R0, s0, a0 = sim_setup
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, flip_interval=0,
                         record_interval=10, seed=8)
    sim = CGSimulator(model, cfg, R0, s0, a0)
    c1, _ = sim.run(20)
    c2, _ = sim.run(20, record_initial=False)
    assert c1.shape[1] == 3 and c2.shape[1] == 2
    assert not np.array_equal(c1[:, -1], c2[:, 0])
