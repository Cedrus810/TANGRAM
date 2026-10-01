"""CG 几何 -> 伪全骨架重建（统一设计文档 §19 Option 2）。

ProteinMPNN 需要 N-CA-C-O 全骨架输入，而 TANGRAM 的 CG 状态只有 Cα 轨迹
与残基内部态 s。本模块用 NeRF（内坐标链式重建）构造伪骨架：

- 键长/键角取理想值（Engh-Huber，nm），ω = 180°（反式）；
- φ(i)/ψ(i) 取该残基当前内部态的 Ramachandran 锚点（states.CORE_CENTERS_DEG
  中该态的第一个核心中心）——链的内坐标由状态完全决定；
- 重建链的 CA 轨迹与 CG 的 R 不逐点重合（状态锚定的理想链会漂移），以全局
  Kabsch 刚体叠合对齐到 R 后输出，供提议模型条件化使用。近似误差由
  "内部态是否忠实于 AA 骨架"决定，属 Option 2 的已知限制（§19）。

单位：输入输出均为 nm（ProteinMPNN 适配层负责 nm -> Å 换算）。
"""
from __future__ import annotations

import numpy as np
import torch

from ..states import CORE_CENTERS_DEG

# 理想几何（nm / rad；Engh-Huber 常规值）
B_N_CA = 0.1458
B_CA_C = 0.1525
B_C_N = 0.1329
TH_N_CA_C = np.deg2rad(111.2)
TH_CA_C_N = np.deg2rad(116.2)
TH_C_N_CA = np.deg2rad(121.7)
B_C_O = 0.1231
TH_CA_C_O = np.deg2rad(120.5)
OMEGA_TRANS = np.pi

# 每个内部态的 φ/ψ 锚点：取该态第一个核心中心
_STATE_PHI_PSI: dict[int, tuple[float, float]] = {}
for _phi, _psi, _st in CORE_CENTERS_DEG:
    _STATE_PHI_PSI.setdefault(int(_st), (np.deg2rad(_phi), np.deg2rad(_psi)))


def _place(a: np.ndarray, b: np.ndarray, c: np.ndarray,
           bond: float, angle: float, dihedral: float) -> np.ndarray:
    """NeRF：给定三原子 a,b,c，按 (键长 c-d, 键角 b-c-d, 二面角 a-b-c-d) 放置 d。"""
    bc = c - b
    bc = bc / max(np.linalg.norm(bc), 1e-12)
    n = np.cross(b - a, bc)
    n = n / max(np.linalg.norm(n), 1e-12)
    m = np.cross(n, bc)                       # 正交基 (n, bc, m)
    d2 = np.array([-bond * np.cos(angle),
                   bond * np.cos(dihedral) * np.sin(angle),
                   bond * np.sin(dihedral) * np.sin(angle)])
    return c + d2[0] * bc + d2[1] * n + d2[2] * m


def _kabsch_align(src: np.ndarray, dst: np.ndarray,
                  apply_to: np.ndarray) -> np.ndarray:
    """把 src 刚体叠合到 dst（最小二乘），作用于 apply_to 的最后一维 (...,3)。"""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rot = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    out = (apply_to - sc).reshape(-1, 3) @ rot.T + dc
    return out.reshape(apply_to.shape)


def pseudo_backbone_numpy(R: np.ndarray, s: np.ndarray) -> np.ndarray:
    """R (N,3) CA 轨迹（nm）、s (N,) 内部态 -> (N, 4, 3) [N, CA, C, O]（nm）。

    链式 NeRF：N(i) 由 ψ(i-1) 定位，CA(i) 由 ω 定位，C(i) 由 φ(i) 定位，
    O(i) 由 ψ(i)+180° 定位；首残基用其自身态锚点起步。重建后对 CA 轨迹做
    一次 Kabsch 刚体叠合（不改变内坐标）。
    """
    R = np.asarray(R, dtype=np.float64)
    s = np.asarray(s, dtype=np.int64)
    n = R.shape[0]
    if n < 3:
        raise ValueError("need at least 3 residues")
    phi = np.asarray([_STATE_PHI_PSI[int(st)][0] for st in s])
    psi = np.asarray([_STATE_PHI_PSI[int(st)][1] for st in s])

    N = np.zeros((n, 3))
    CA = np.zeros((n, 3))
    C = np.zeros((n, 3))
    # 首残基：N 沿 x，CA 沿链方向，C 用理想角 + 任意二面角（φ(0) 无前驱，不适用）
    N[0] = np.array([0.0, 0.0, 0.0])
    CA[0] = N[0] + B_N_CA * np.array([1.0, 0.0, 0.0])
    C[0] = _place(N[0], np.array([0.0, 0.0, 0.0]), CA[0], B_CA_C, TH_N_CA_C, 0.0)
    for i in range(1, n):
        # N(i)：二面角 ψ(i-1) = N(i-1)-CA(i-1)-C(i-1)-N(i)
        N[i] = _place(N[i - 1], CA[i - 1], C[i - 1], B_C_N, TH_CA_C_N, psi[i - 1])
        # CA(i)：二面角 ω = CA(i-1)-C(i-1)-N(i)-CA(i)（反式肽键）
        CA[i] = _place(CA[i - 1], C[i - 1], N[i], B_N_CA, TH_C_N_CA, OMEGA_TRANS)
        # C(i)：二面角 φ(i) = C(i-1)-N(i)-CA(i)-C(i)
        C[i] = _place(C[i - 1], N[i], CA[i], B_CA_C, TH_N_CA_C, phi[i])
    O = np.zeros((n, 3))
    for i in range(n):
        # O(i)：二面角 N(i)-CA(i)-C(i)-O(i) = ψ(i) + 180°
        O[i] = _place(N[i], CA[i], C[i], B_C_O, TH_CA_C_O, psi[i] + np.pi)

    bb = np.stack([N, CA, C, O], axis=1)
    return _kabsch_align(bb[:, 1, :], R, bb)


def pseudo_backbone(R: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """批量版本：R (B,N,3)、s (B,N) -> (B,N,4,3) [N, CA, C, O]，单位 nm。"""
    out = np.stack([pseudo_backbone_numpy(R[b].detach().cpu().numpy(),
                                          s[b].detach().cpu().numpy())
                    for b in range(R.shape[0])])
    return torch.as_tensor(out, dtype=R.dtype, device=R.device)
