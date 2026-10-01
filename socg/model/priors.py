"""Boltzmann 反演先验（D4）：键/角谐振子 + 软球排斥，兜住 RBF 覆盖区之外。

U_prior = Σ_bonds ½k_b(r-r0)² + Σ_angles ½k_θ(θ-θ0)² + Σ_pairs kT·(σ_bucket/r)^12
其中 k = kT/var；σ_bucket = 该 bucket 距离的 0.1 分位数 × 0.9。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..constants import KB
from ..topology import N_SEP_BUCKETS, CGTopology

_DEFAULT_SIGMA_NM = 0.4  # 空 bucket 的兜底值


class Priors(nn.Module):
    """拓扑固定的先验能量；值由 fit() 从 AA（或任意）坐标估计。"""

    def __init__(self, topo: CGTopology):
        super().__init__()
        bonds, angles, pairs = topo.bonds, topo.angles, topo.pairs
        self.register_buffer("bond_index", torch.as_tensor(bonds, dtype=torch.long))
        self.register_buffer("angle_index", torch.as_tensor(angles, dtype=torch.long))
        self.register_buffer("pair_index", torch.as_tensor(pairs, dtype=torch.long))
        self.register_buffer("pair_bucket", torch.as_tensor(topo.pair_bucket, dtype=torch.long))
        self.register_buffer("r0", torch.zeros(bonds.shape[0], dtype=torch.float64))
        self.register_buffer("k_bond", torch.zeros(bonds.shape[0], dtype=torch.float64))
        self.register_buffer("th0", torch.zeros(angles.shape[0], dtype=torch.float64))
        self.register_buffer("k_angle", torch.zeros(angles.shape[0], dtype=torch.float64))
        self.register_buffer("sigma", torch.full((N_SEP_BUCKETS,), _DEFAULT_SIGMA_NM,
                                                 dtype=torch.float64))
        self.register_buffer("temperature", torch.tensor(300.0, dtype=torch.float64))

    # ------------------------------------------------------------------ #
    # Boltzmann 反演
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, topo: CGTopology, coords: np.ndarray, temperature: float) -> "Priors":
        """coords: (T,N,3) numpy，单位 nm。"""
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 3:
            raise ValueError(f"coords must be (T,N,3), got {coords.shape}")
        inst = cls(topo)
        kT = KB * float(temperature)

        bi = topo.bonds
        ai = topo.angles
        pi = topo.pairs
        bucket = topo.pair_bucket

        # 键长：均值 + 方差（分块累积）
        r_sum = np.zeros(bi.shape[0])
        r_sq = np.zeros(bi.shape[0])
        for lo in range(0, coords.shape[0], _CHUNK):
            c = coords[lo:lo + _CHUNK]
            d = np.linalg.norm(c[:, bi[:, 0]] - c[:, bi[:, 1]], axis=-1)
            r_sum += d.sum(axis=0)
            r_sq += (d ** 2).sum(axis=0)
        T = coords.shape[0]
        r0 = r_sum / T
        r_var = np.maximum(r_sq / T - r0 ** 2, 1e-8)
        inst.r0.copy_(torch.as_tensor(r0))
        inst.k_bond.copy_(torch.as_tensor(kT / r_var))

        # 键角（弧度）
        th_sum = np.zeros(ai.shape[0])
        th_sq = np.zeros(ai.shape[0])
        for lo in range(0, T, _CHUNK):
            c = coords[lo:lo + _CHUNK]
            th = _angle_rad(c[:, ai[:, 0]], c[:, ai[:, 1]], c[:, ai[:, 2]])
            th_sum += th.sum(axis=0)
            th_sq += (th ** 2).sum(axis=0)
        th0 = th_sum / T
        th_var = np.maximum(th_sq / T - th0 ** 2, 1e-8)
        inst.th0.copy_(torch.as_tensor(th0))
        inst.k_angle.copy_(torch.as_tensor(kT / th_var))

        # pair 距离的 0.1 分位数（抽稀以控制内存）
        stride = max(1, T // 200_000)
        sub = coords[::stride]
        d = np.linalg.norm(sub[:, pi[:, 0]] - sub[:, pi[:, 1]], axis=-1)  # (T', npr)
        sigma = np.full(N_SEP_BUCKETS, _DEFAULT_SIGMA_NM)
        for b in range(N_SEP_BUCKETS):
            vals = d[:, bucket == b]
            if vals.size:
                sigma[b] = float(np.quantile(vals, 0.1) * 0.9)
        inst.sigma.copy_(torch.as_tensor(sigma))
        inst.temperature.fill_(float(temperature))
        return inst

    # ------------------------------------------------------------------ #
    # 能量
    # ------------------------------------------------------------------ #
    def forward(self, R: torch.Tensor) -> torch.Tensor:
        R = R.to(self.r0.dtype)
        bi = self.bond_index
        d = torch.norm(R[:, bi[:, 0]] - R[:, bi[:, 1]], dim=-1)
        u_bond = 0.5 * self.k_bond * (d - self.r0) ** 2

        ai = self.angle_index
        ba = R[:, ai[:, 0]] - R[:, ai[:, 1]]
        bc = R[:, ai[:, 2]] - R[:, ai[:, 1]]
        th = _torch_angle_rad(ba, bc)
        u_angle = 0.5 * self.k_angle * (th - self.th0) ** 2

        pi = self.pair_index
        dp = torch.norm(R[:, pi[:, 0]] - R[:, pi[:, 1]], dim=-1)
        kT = KB * self.temperature
        u_rep = kT * (self.sigma[self.pair_bucket].unsqueeze(0) / dp.clamp_min(1e-6)) ** 12

        return u_bond.sum(-1) + u_angle.sum(-1) + u_rep.sum(-1)


_CHUNK = 200_000


def _angle_rad(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """顶点在 b 的键角（弧度）。输入 (..., 3)。"""
    ba = a - b
    bc = c - b
    x = np.sum(ba * bc, axis=-1)
    y = np.linalg.norm(np.cross(ba, bc, axis=-1), axis=-1)
    return np.arctan2(y, x)


def _torch_angle_rad(ba: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    x = (ba * bc).sum(-1)
    y = torch.linalg.vector_norm(torch.cross(ba, bc, dim=-1), dim=-1)
    return torch.atan2(y, x)
