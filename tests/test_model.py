"""T8: SOCG v0 势函数。"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from conftest import aa_index_of

from socg.model.socg import ModelConfig, SOCGModel, load_model, save_model
from socg.states import FROZEN_STATE


def _random_coords(rng, n, B=3):
    return rng.normal(0, 0.4, size=(B, n, 3))


def test_forces_match_finite_difference(toy_model_double, toy_coords):
    # Review Focus 5 的力正确性：float64、随机参数、h=1e-6
    model = toy_model_double
    from conftest import randomize

    randomize(model, 0.1, seed=11)
    rng = np.random.default_rng(0)
    R = torch.as_tensor(toy_coords[:2].astype(np.float64))
    s = torch.as_tensor(rng.integers(0, 3, size=(2, 6)), dtype=torch.long)
    a = aa_index_of(model)
    F = model.forces(R, s, a).detach().numpy()

    U = lambda r: model.energy(torch.as_tensor(r), s, a).detach().numpy()
    h = 1e-6
    for b in range(2):
        for i in range(6):
            for c in range(3):
                Rp = R.numpy().copy(); Rp[b, i, c] += h
                Rm = R.numpy().copy(); Rm[b, i, c] -= h
                fd = -(U(Rp)[b] - U(Rm)[b]) / (2 * h)
                denom = max(abs(fd), abs(F[b, i, c]), 1e-6)
                assert abs(F[b, i, c] - fd) / denom < 1e-5


def test_rotation_translation_invariance(toy_model_double):
    from conftest import randomize

    model = toy_model_double
    randomize(model, 0.2, seed=5)
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.4, size=(2, 6, 3))
    # 随机旋转
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    s = torch.as_tensor(rng.integers(0, 3, size=(2, 6)), dtype=torch.long)
    a = aa_index_of(model)
    e1 = model.energy(torch.as_tensor(R), s, a)
    e2 = model.energy(torch.as_tensor(R @ Q.T + 10.0), s, a)
    assert torch.max(torch.abs(e1 - e2)) < 1e-8


def test_energy_depends_on_state_k3(toy_model_double):
    from conftest import randomize

    model = toy_model_double
    randomize(model, 0.3, seed=7)
    rng = np.random.default_rng(2)
    R = torch.as_tensor(rng.normal(0, 0.4, size=(1, 6, 3)))
    # 位点 1 可动；位点 0 冻结（会被强制归位为 FROZEN_STATE）
    a = aa_index_of(model)
    e = [model.energy(R, torch.as_tensor([[1, k, 2, 0, 1, 1]]), a).item()
         for k in range(3)]
    assert len(set(e)) == 3


def test_k1_insensitive_to_states(toy_model_double):
    # Review Focus 4：K=1 基线对任意 s 输入给出完全相同的能量
    model = SOCGModel(toy_model_double.topo, toy_model_double.priors,
                      ModelConfig(n_states=1)).double()
    from conftest import randomize

    randomize(model, 0.3, seed=9)
    rng = np.random.default_rng(3)
    R = torch.as_tensor(rng.normal(0, 0.4, size=(4, 6, 3)))
    s = torch.as_tensor(rng.integers(0, 3, size=(4, 6)), dtype=torch.long)
    a = aa_index_of(model)
    e_ref = model.energy(R, s, a)
    for k in range(3):
        s2 = torch.full_like(s, k)
        assert torch.allclose(model.energy(R, s2, a), e_ref)


def test_prepare_states_freezes(toy_model):
    s = torch.zeros((2, 6), dtype=torch.long)
    out = toy_model.prepare_states(s)
    assert out[0, 0].item() == FROZEN_STATE
    assert out[0, 5].item() == FROZEN_STATE
    assert out[0, 2].item() == 0
    s1 = toy_model.prepare_states(s)  # K=3: 冻结位强制 1
    assert (s1[:, [0, 5]] == FROZEN_STATE).all()


def test_k1_prepare_all_zero(toy_model):
    model = SOCGModel(toy_model.topo, toy_model.priors, ModelConfig(n_states=1))
    s = torch.as_tensor([[2, 1, 0, 2, 1, 0]], dtype=torch.long)
    assert (model.prepare_states(s) == 0).all()


def test_prior_guards_rbf_range(toy_model, toy_coords):
    # Review Focus 5：参数为 0 时先验兜住 RBF 覆盖区之外（两项分别单独检验）
    model = toy_model.double()                       # 所有可学习参数为 0
    base = toy_coords[0].astype(np.float64)
    s0 = torch.zeros((1, 6), dtype=torch.long)
    kT300 = 0.0083144626181532 * 300.0

    def U(R, m=model):
        return m.energy(torch.as_tensor(R[None]), s0, aa_index_of(m)).item()

    # 1) 非局部 pair (0,3) 压到 0.2 nm：只看排斥项的贡献（同一构型下把 σ 置 0 作对照）
    R1 = base.copy()
    R1[3] = R1[0] + np.array([0.2, 0.0, 0.0])
    no_rep = copy.deepcopy(model)
    no_rep.priors.sigma.zero_()
    assert U(R1) - U(R1, no_rep) > 10 * kT300
    # 2) 只拉长键 2-3 到 0.6 nm：尾部 3..5 沿 2->3 方向刚性平移，角与二面角不变
    R2 = base.copy()
    u = (R2[3] - R2[2]) / np.linalg.norm(R2[3] - R2[2])
    R2[3:] += (0.6 - np.linalg.norm(base[3] - base[2])) * u
    assert np.linalg.norm(R2[3] - R2[2]) == pytest.approx(0.6)
    assert U(R2) - U(base) > 10 * kT300


def test_save_load_roundtrip(toy_model_double, tmp_path):
    from conftest import randomize

    randomize(toy_model_double, 0.3, seed=13)
    path = tmp_path / "model.pt"
    save_model(path, toy_model_double, meta={"system": "ala6"})
    model2 = load_model(path)
    rng = np.random.default_rng(4)
    R = torch.as_tensor(rng.normal(0, 0.4, size=(2, 6, 3)))
    s = torch.as_tensor(rng.integers(0, 3, size=(2, 6)), dtype=torch.long)
    a = aa_index_of(toy_model_double)
    assert torch.allclose(toy_model_double.energy(R, s, a), model2.energy(R, s, a),
                          rtol=0, atol=0)
    assert model2._meta["system"] == "ala6"
