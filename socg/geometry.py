"""numpy 几何工具：IUPAC 二面角、键角、Kabsch RMSD。单位：度 / nm。"""
from __future__ import annotations

import numpy as np


def dihedral_deg(p0, p1, p2, p3):
    """IUPAC 二面角（度）。输入 (..., 3)，支持广播。

    验证约定：p1=(0,0,0), p2=(0,0,1), p0=(1,0,0), p3=p2+(cos a, sin a, 0)
    => dihedral_deg == a。
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v, axis=-1) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def angle_deg(a, b, c):
    """键角（度），顶点在 b。输入 (..., 3)。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    x = np.sum(ba * bc, axis=-1)
    y = np.linalg.norm(np.cross(ba, bc, axis=-1), axis=-1)
    return np.degrees(np.arctan2(y, x))


def _rotation_matrices(T: int, seed: int | None = None) -> np.ndarray:
    """测试辅助：T 个随机旋转矩阵（正交、行列式 +1）。"""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(T, 3, 3))
    Q, R = np.linalg.qr(M)
    Q = np.swapaxes(Q, -1, -2)  # 行向量正交 -> 列向量正交
    sign = np.sign(np.linalg.det(Q))
    sign = np.where(sign == 0, 1.0, sign)
    Q = Q * sign[:, None, None]
    return Q


def kabsch_rmsd(X, ref):
    """最优叠合后的 RMSD（nm）。

    X: (T, N, 3)；ref: (N, 3)。返回 (T,)。
    """
    X = np.asarray(X, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != ref.shape:
        raise ValueError(f"X shape {X.shape} incompatible with ref {ref.shape}")
    xc = X - X.mean(axis=1, keepdims=True)
    rc = ref - ref.mean(axis=0, keepdims=True)
    # 协方差 C = xc^T rc，批量 SVD；行向量约定下 R = U·D·V^T
    C = np.einsum("tni,nj->tij", xc, rc)
    U, _S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.matmul(U, Vt)))
    d = np.where(d == 0, 1.0, d)
    D = np.zeros_like(U)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.matmul(np.matmul(U, D), Vt)               # (T,3,3)
    aligned = np.einsum("tni,tij->tnj", xc, R)
    diff = aligned - rc[None, :, :]
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1), axis=-1))
