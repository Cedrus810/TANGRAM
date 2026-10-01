"""残基内部离散态：Ramachandran 核心集 + transition-based assignment (TBA)。

态词表（锁定）：0 = A(αR)、1 = B(β/PPII)、2 = L(αL)；冻结残基恒为 1，
不参与翻转和伪似然。

TBA 语义：每个残基保持“最近一次进入的核心”对应的态；第 0 帧不在任何
核心内时取 nearest_state。输入必须是一条连续轨迹，不得跨轨迹拼接。
"""
from __future__ import annotations

import numpy as np

from .constants import wrap_deg

STATE_NAMES: tuple[str, ...] = ("A", "B", "L")
N_STATES = 3
FROZEN_STATE = 1

# (phi_deg, psi_deg, state)
CORE_CENTERS_DEG: tuple[tuple[float, float, int], ...] = (
    (-63.0, -43.0, 0),   # A (alphaR)
    (-120.0, 130.0, 1),  # B (beta / PPII)
    (-70.0, 145.0, 1),   # B (beta / PPII)
    (60.0, 45.0, 2),     # L (alphaL)
)


def _centers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cphi = np.asarray([c[0] for c in CORE_CENTERS_DEG], dtype=np.float64)
    cpsi = np.asarray([c[1] for c in CORE_CENTERS_DEG], dtype=np.float64)
    cstate = np.asarray([c[2] for c in CORE_CENTERS_DEG], dtype=np.int64)
    return cphi, cpsi, cstate


def _periodic_distance(phi: np.ndarray, psi: np.ndarray,
                       cphi: np.ndarray, cpsi: np.ndarray) -> np.ndarray:
    """到各核心中心的 (Δφ, Δψ) 欧氏距离，Δ 先折回 [-180, 180)。

    phi/psi 形状 (T,N)；返回 (T,N,n_centers)。
    """
    dphi = wrap_deg(phi[..., None] - cphi)     # (T,N,C)
    dpsi = wrap_deg(psi[..., None] - cpsi)
    return np.sqrt(dphi ** 2 + dpsi ** 2)


def core_state(phi, psi, radius: float = 40.0):
    """逐帧逐残基判断是否落在某个核心内。

    落在核心内返回该核心的态（0/1/2，多个核心命中取最近者），
    否则 -1；NaN 返回 -1。输入形状 (T,N)，单位度。
    """
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.shape != psi.shape:
        raise ValueError(f"phi shape {phi.shape} != psi shape {psi.shape}")
    cphi, cpsi, cstate = _centers()
    dist = _periodic_distance(phi, psi, cphi, cpsi)
    nan_mask = np.isnan(phi) | np.isnan(psi)
    dist = np.where(nan_mask[..., None], np.inf, dist)
    inside = dist <= radius
    dist_near = np.where(inside, dist, np.inf)
    nearest = np.argmin(dist_near, axis=-1)          # 命中核心中的最近者
    has_core = inside.any(axis=-1) & ~nan_mask
    out = np.where(has_core, cstate[nearest], -1)
    return out.astype(np.int64)


def nearest_state(phi, psi):
    """最近核心中心对应的态（与是否在核心内无关）；NaN -> -1。"""
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.shape != psi.shape:
        raise ValueError(f"phi shape {phi.shape} != psi shape {psi.shape}")
    cphi, cpsi, cstate = _centers()
    dist = _periodic_distance(phi, psi, cphi, cpsi)
    nan_mask = np.isnan(phi) | np.isnan(psi)
    dist = np.where(nan_mask[..., None], np.inf, dist)
    return cstate[np.argmin(dist, axis=-1)].astype(np.int64)


def assign_states(phi, psi, frozen, radius: float = 40.0):
    """对一条连续轨迹做 TBA 态指派。

    Parameters
    ----------
    phi, psi : (T, N) 度；冻结列可以全为 NaN。
    frozen : (N,) bool。

    Returns
    -------
    (T, N) int64。frozen 列恒为 FROZEN_STATE。
    """
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.ndim != 2 or psi.ndim != 2:
        raise ValueError("phi/psi must be 2-D (T, N)")
    if phi.shape != psi.shape:
        raise ValueError(f"phi shape {phi.shape} != psi shape {psi.shape}")
    frozen = np.asarray(frozen, dtype=bool)
    if frozen.shape != (phi.shape[1],):
        raise ValueError(f"frozen shape {frozen.shape} != ({phi.shape[1]},)")

    T, N = phi.shape
    out = np.empty((T, N), dtype=np.int64)

    core = core_state(phi, psi, radius=radius)       # (T,N)
    near = nearest_state(phi, psi)                   # (T,N)

    for j in range(N):
        if frozen[j]:
            out[:, j] = FROZEN_STATE
            continue
        col = core[:, j]
        valid = col >= 0
        if valid.all():
            out[:, j] = col
            continue
        # 第 0 帧不在核心内：视为“进入” nearest_state(第 0 帧) 对应的态，
        # 之后与普通 TBA 一样前向保持，直到第一次真正进入核心
        col = col.copy()
        if not valid[0]:
            col[0] = near[0, j]
            valid = valid.copy()
            valid[0] = True
        idx = np.maximum.accumulate(np.where(valid, np.arange(T), 0))
        out[:, j] = col[idx].astype(np.int64)

    return out
