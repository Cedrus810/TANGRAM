"""T10: 联合拟合（力匹配 + 态伪似然）。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from socg.data import CGDataset, CGTrajectory
from socg.dynamics import CGSimulator, LangevinConfig
from socg.fit import FitConfig, fit, force_matching_loss, pseudo_likelihood_nll
from socg.model.socg import ModelConfig, SOCGModel


def _toy_dataset(toy_topology, toy_coords, model_for_forces, seed=0):
    """coords = toy_coords；forces = 指定模型在 (coords, 真实态) 下的力。"""
    rng = np.random.default_rng(seed)
    n = toy_topology.n
    states = rng.integers(0, 3, size=(toy_coords.shape[0], n))
    states[:, toy_topology.frozen_mask] = 1
    R = torch.as_tensor(toy_coords.astype(np.float64))
    s = torch.as_tensor(states, dtype=torch.long)
    with torch.no_grad():
        F = model_for_forces.forces(R, s).numpy().astype(np.float32)
    trj = CGTrajectory(coords=toy_coords, forces=F, states=states, dt_ps=1.0)
    return CGDataset(topology=toy_topology, temperature=300.0, trajectories=[trj])


def test_pl_uniform_states_converges_to_ln3(toy_topology, toy_coords, toy_model):
    """态标签与 R 完全独立、均匀随机 -> 拟合后（留出帧上）PL ≈ ln 3（±0.02）。"""
    rng = np.random.default_rng(0)
    n = toy_topology.n
    states = rng.integers(0, 3, size=(toy_coords.shape[0], n))
    states[:, toy_topology.frozen_mask] = 1
    R = torch.as_tensor(toy_coords.astype(np.float64))
    s = torch.as_tensor(states, dtype=torch.long)
    with torch.no_grad():
        F = toy_model.forces(R, s).numpy().astype(np.float32)  # 先验力，与 s 无关
    half = toy_coords.shape[0] // 2

    def ds(sl):
        trj = CGTrajectory(coords=toy_coords[sl], forces=F[sl], states=states[sl], dt_ps=1.0)
        return CGDataset(topology=toy_topology, temperature=300.0, trajectories=[trj])

    train, held = ds(slice(0, half)), ds(slice(half, None))
    model = SOCGModel(toy_topology, toy_model.priors, ModelConfig()).double()
    cfg = FitConfig(device="cpu", max_steps=800, batch_frames=512, lr=5e-3,
                    stride=2, eval_every=800, seed=0)
    fit(model, train, held, cfg)
    coords, _, states_h = held.frames(stride=1)
    Rv = torch.as_tensor(coords.astype(np.float64))
    sv = torch.as_tensor(states_h, dtype=torch.long)
    with torch.no_grad():
        pl = float(pseudo_likelihood_nll(model, Rv, sv, 1.0 / (0.0083144626181532 * 300.0)))
    assert pl == pytest.approx(np.log(3), abs=0.02)


def test_k1_pl_is_zero_and_no_grad_to_eps_wnn(toy_topology, toy_coords, toy_model):
    model = SOCGModel(toy_topology, toy_model.priors, ModelConfig(n_states=1)).double()
    rng = np.random.default_rng(1)
    R = torch.as_tensor(toy_coords[:16].astype(np.float64))
    s = torch.as_tensor(rng.integers(0, 3, size=(16, 6)), dtype=torch.long)
    pl = pseudo_likelihood_nll(model, R, s, 1.0 / 2.4943)
    assert float(pl) == 0.0
    assert not pl.requires_grad                     # PL 不连接任何参数
    # 总损失（FM + PL）对 eps、w_nn 的梯度为空或恒 0：K=1 时它们只是常数能量偏移
    F_ref = torch.zeros_like(R)
    loss = force_matching_loss(model, R, s, F_ref, torch.tensor(1.0, dtype=R.dtype)) + pl
    grads = torch.autograd.grad(loss, [model.eps, model.w_nn], allow_unused=True)
    for g in grads:
        assert g is None or float(g.abs().max()) == 0.0


def test_frozen_sites_excluded_from_pl(toy_model_double):
    """改变冻结位点的标签，损失不变。"""
    model = toy_model_double
    from conftest import randomize

    randomize(model, 0.3, seed=3)
    rng = np.random.default_rng(2)
    R = torch.as_tensor(rng.normal(0, 0.4, size=(4, 6, 3)))
    beta = 1.0 / 2.4943
    s1 = torch.as_tensor(rng.integers(0, 3, size=(4, 6)), dtype=torch.long)
    s2 = s1.clone()
    s2[:, 0] = (s1[:, 0] + 1) % 3      # 冻结位点 0 换标签
    s2[:, 5] = (s1[:, 5] + 2) % 3      # 冻结位点 5 换标签
    pl1 = pseudo_likelihood_nll(model, R, s1, beta)
    pl2 = pseudo_likelihood_nll(model, R, s2, beta)
    assert float(pl1) == pytest.approx(float(pl2), rel=1e-12)


def test_pl_gradient_flows(toy_model_double):
    """K=3 的 PL 对参数有非零梯度（可训练）。"""
    model = toy_model_double
    from conftest import randomize

    randomize(model, 0.3, seed=4)
    rng = np.random.default_rng(5)
    R = torch.as_tensor(rng.normal(0, 0.4, size=(8, 6, 3)))
    s = torch.as_tensor(rng.integers(0, 3, size=(8, 6)), dtype=torch.long)
    pl = pseudo_likelihood_nll(model, R, s, 1.0 / 2.4943)
    g = torch.autograd.grad(pl, model.w_pair_state)[0]
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


@pytest.mark.slow
def test_teacher_student_self_consistency(toy_topology, toy_priors):
    """随机教师 -> 模拟数据（力加噪）-> 从零拟合学生。

    留出帧上：学生力 vs 教师无噪声力的相对 RMSE < 0.15；
    条件态分布 P(s_i | R, s_-i) 平均 TV < 0.05。CPU 上 < 90 s。
    """
    from conftest import ideal_helix, randomize

    torch.manual_seed(0)
    teacher = SOCGModel(toy_topology, toy_priors, ModelConfig()).double()
    randomize(teacher, 0.3, seed=42)

    # 教师生成数据：B=32, 800 步, 每 1 步记录 -> 25.6k 帧
    rng = np.random.default_rng(0)
    R0 = np.tile(ideal_helix(6).astype(np.float64)[None], (32, 1, 1))
    R0 += rng.normal(0, 0.02, size=R0.shape)
    s0 = rng.integers(0, 3, size=(32, 6))
    cfg = LangevinConfig(temperature=300.0, dt_ps=0.002, friction_per_ps=5.0,
                         flip_interval=10, record_interval=1, seed=11)
    sim = CGSimulator(teacher, cfg, torch.as_tensor(R0), torch.as_tensor(s0, dtype=torch.long))
    coords, states = sim.run(800)
    coords = coords.reshape(-1, 6, 3).astype(np.float64)
    states = states.reshape(-1, 6).astype(np.int64)

    # 教师无噪声力 + 0.5×力标准差的高斯噪声
    R_all = torch.as_tensor(coords)
    s_all = torch.as_tensor(states, dtype=torch.long)
    with torch.no_grad():
        F_true = teacher.forces(R_all, s_all).numpy()
    noise_std = 0.5 * F_true.std()
    rng = np.random.default_rng(1)
    F_noisy = (F_true + rng.normal(0, noise_std, F_true.shape)).astype(np.float32)

    # 按帧切分 train/holdout（模拟器帧间强相关，仅作拟合自洽检查）
    n = coords.shape[0]
    hold = np.arange(0, n, 5)                       # 20% 留出
    mask = np.ones(n, dtype=bool); mask[hold] = False
    trj_tr = CGTrajectory(coords=coords[mask].astype(np.float32), forces=F_noisy[mask],
                          states=states[mask], dt_ps=0.002)
    ds_tr = CGDataset(topology=toy_topology, temperature=300.0, trajectories=[trj_tr])

    student = SOCGModel(toy_topology, toy_priors, ModelConfig()).double()
    fcfg = FitConfig(device="cpu", max_steps=400, batch_frames=256, lr=5e-3,
                     pl_weight=1.0, l2=1e-4, stride=1, eval_every=400, seed=0)
    fit(student, ds_tr, ds_tr, fcfg)

    beta = 1.0 / (0.0083144626181532 * 300.0)
    R_h = torch.as_tensor(coords[hold])
    s_h = torch.as_tensor(states[hold], dtype=torch.long)
    with torch.no_grad():
        F_s = student.forces(R_h, s_h).numpy()
    rel_rmse = np.sqrt(np.mean((F_s - F_true[hold]) ** 2)) / np.sqrt(np.mean(F_true[hold] ** 2))
    assert rel_rmse < 0.15

    # 条件态分布 TV
    movable = np.where(~toy_topology.frozen_mask)[0]
    with torch.no_grad():
        U_t = torch.stack([teacher.energy(R_h, _with_state(s_h, i, k))
                           for i in movable for k in range(3)])
        U_s = torch.stack([student.energy(R_h, _with_state(s_h, i, k))
                           for i in movable for k in range(3)])
    K = 3
    U_t = U_t.reshape(len(movable), K, -1)          # (site, k, B)
    U_s = U_s.reshape(len(movable), K, -1)
    p_t = torch.softmax(-beta * U_t, dim=1)
    p_s = torch.softmax(-beta * U_s, dim=1)
    tv = 0.5 * (p_s - p_t).abs().sum(dim=1).mean(dim=-1)   # (site,)
    assert float(tv.mean()) < 0.05


def _with_state(s, site, k):
    out = s.clone()
    out[:, site] = k
    return out


def test_force_matching_loss_value(toy_model_double):
    model = toy_model_double
    R = torch.as_tensor(np.random.default_rng(9).normal(0, 0.4, size=(4, 6, 3)))
    s = torch.zeros((4, 6), dtype=torch.long)
    F_ref = torch.zeros_like(R)
    f_var = torch.tensor(1.0)
    loss = force_matching_loss(model, R, s, F_ref, f_var)
    expected = float((model.forces(R, s) ** 2).mean())
    assert float(loss) == pytest.approx(expected, rel=1e-6)
