"""torch 基函数：高斯 RBF 与 Fourier。

约定：forward 输入 x 形状 (...,)，输出 (..., n_features)。
"""
from __future__ import annotations

import torch
from torch import nn


class RBF(nn.Module):
    """高斯基函数：中心均匀分布，宽度 = (hi-lo)/(n-1)。"""

    def __init__(self, lo: float, hi: float, n: int):
        super().__init__()
        if n < 2:
            raise ValueError(f"RBF needs n >= 2, got {n}")
        if not hi > lo:
            raise ValueError(f"RBF needs hi > lo, got [{lo}, {hi}]")
        centers = torch.linspace(float(lo), float(hi), n)
        width = (float(hi) - float(lo)) / (n - 1)
        self.register_buffer("centers", centers)
        self.register_buffer("width", torch.tensor(width))

    @property
    def n_features(self) -> int:
        return int(self.centers.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-0.5 * z * z)


class Fourier(nn.Module):
    """Fourier 基：forward 输出 (..., 2*order)，依次 cos(kx), sin(kx), k=1..order。"""

    def __init__(self, order: int):
        super().__init__()
        if order < 1:
            raise ValueError(f"Fourier needs order >= 1, got {order}")
        self.order = int(order)
        self.register_buffer(
            "orders", torch.arange(1, self.order + 1, dtype=torch.get_default_dtype()))

    @property
    def n_features(self) -> int:
        return 2 * self.order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1) * self.orders            # (..., order)
        cs = torch.stack([torch.cos(z), torch.sin(z)], dim=-1)  # (..., order, 2)
        return cs.reshape(*z.shape[:-1], 2 * self.order)
