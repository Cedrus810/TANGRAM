"""固定序列回归门（统一设计文档 §30 Phase 1 Gate / §31 Level 0）。

tests/data/ 下的基准由 scripts/dump_regression_reference.py 在**重构前的 v0
代码**上生成（20 种氨基酸混合序列，float64，3 个配置）。本测试断言：

1. v1 代码在固定序列 a=topo.aa_index 下逐位复现 v0 的 U 与 F
   （U_new(R, s, a_fixed) == U_old(R, s)）；
2. v0 checkpoint（含已废弃的 res_class/aa_index 序列缓冲）能被 v1 的
   load_model 加载（旧键被过滤并告警）；
3. a 以 (N,) 与 (B, N) 两种形状传入给出相同结果。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from socg.model.socg import load_model

DATA = "tests/data"
CONFIGS = ("k3", "k1", "nopt")


@pytest.fixture(scope="module")
def reference():
    return np.load(f"{DATA}/regression_v0_reference.npz")


def _load_v0_model(name, reference):
    ref = reference
    with pytest.warns(UserWarning, match="legacy sequence buffers"):
        model = load_model(f"{DATA}/regression_v0_{name}.pt")
    R = torch.as_tensor(ref["R"])
    s = torch.as_tensor(ref["s"], dtype=torch.long)
    a = torch.as_tensor(ref["a"], dtype=torch.long)
    return model, R, s, a


@pytest.mark.parametrize("name", CONFIGS)
def test_fixed_sequence_reproduces_v0_energy_and_forces(reference, name):
    model, R, s, a = _load_v0_model(name, reference)
    with torch.no_grad():
        U = model.energy(R, s, a)
        F = model.forces(R, s, a)
    assert torch.equal(U, torch.as_tensor(reference[f"U_{name}"]))
    assert torch.equal(F, torch.as_tensor(reference[f"F_{name}"]))


@pytest.mark.parametrize("name", CONFIGS)
def test_a_broadcast_1d_equals_2d(reference, name):
    model, R, s, a = _load_v0_model(name, reference)
    u1 = model.energy(R, s, a)
    u2 = model.energy(R, s, a.unsqueeze(0).expand(R.shape[0], -1))
    assert torch.equal(u1, u2)


def test_wrong_a_shape_raises(reference):
    model, R, s, a = _load_v0_model("k3", reference)
    with pytest.raises(ValueError, match="a must be"):
        model.energy(R, s, a[:-1])
    with pytest.raises(ValueError, match="a must be"):
        model.energy(R, s, a.unsqueeze(0).expand(R.shape[0] + 1, -1))


def _zero_pair_type_rows(model, *aa_values):
    """把 w_pair_type（对称化前）给定氨基酸的行列整体置零。"""
    with torch.no_grad():
        for v in aa_values:
            model.w_pair_type.data[v, :] = 0.0
            model.w_pair_type.data[:, v] = 0.0


def test_mutation_locality_same_res_class(reference):
    """突变局域性（设计文档 §31）：A→V 同为 OTHER 类，res_class 不变，
    a_i 只进入 w_pair_type 项 -> 置零新旧氨基酸的类型对行列后 ΔU 精确为 0。"""
    model, R, s, a = _load_v0_model("k3", reference)
    site = 6                                    # 序列 "ACDEFGHIKLMNPQRSTVWY" 的 H
    aa_old, aa_new = 6, 0                       # H -> A（字母表索引 H=6），均为 OTHER
    a2 = a.clone()
    a2[site] = aa_new
    with torch.no_grad():
        dU = model.energy(R, s, a2) - model.energy(R, s, a)
    assert not torch.equal(dU, torch.zeros_like(dU))    # 类型对项确实参与
    _zero_pair_type_rows(model, aa_old, aa_new)
    with torch.no_grad():
        dU0 = model.energy(R, s, a2) - model.energy(R, s, a)
    assert torch.equal(dU0, torch.zeros_like(dU0))


def test_mutation_locality_res_class_change(reference):
    """H(F, OTHER)→G(GLY) 改变 res_class：类型对、eps、w_angle 三处全部
    置零新旧取值后 ΔU 精确为 0（bond/dihedral/w_nn/pair_state/先验均与 a 无关）。"""
    model, R, s, a = _load_v0_model("k3", reference)
    site = 4                                    # F (OTHER) -> G (GLY)
    aa_old, aa_new = 4, 5
    cls_old, cls_new = 2, 0
    a2 = a.clone()
    a2[site] = aa_new
    with torch.no_grad():
        dU = model.energy(R, s, a2) - model.energy(R, s, a)
    assert not torch.equal(dU, torch.zeros_like(dU))
    _zero_pair_type_rows(model, aa_old, aa_new)
    with torch.no_grad():
        model.eps.data[cls_old, :] = 0.0
        model.eps.data[cls_new, :] = 0.0
        model.w_angle.data[cls_old, :, :] = 0.0
        model.w_angle.data[cls_new, :, :] = 0.0
        dU0 = model.energy(R, s, a2) - model.energy(R, s, a)
    assert torch.equal(dU0, torch.zeros_like(dU0))
