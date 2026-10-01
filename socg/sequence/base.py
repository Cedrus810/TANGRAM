"""序列状态容器与提议算子抽象接口（统一设计文档 §8/§15/§17）。

序列身份 a 从拓扑元数据提升为运行时状态变量：X = (R, s, a)。本模块提供
SequenceState（a 的运行时容器）与 SequenceProposal（序列空间提议算子抽象
接口 q(a' | R, a)）。能量模型与模拟器只依赖该接口，不依赖具体提议模型。

Metropolis-Hastings 接受概率（设计文档 §6）：
    A = min[1, exp(-β ΔU) * q(a|R,a') / q(a'|R,a)]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..topology import CGTopology


@dataclass
class SequenceState:
    """运行时序列状态。

    aa_index: (N,) 或 (B, N) long 张量，氨基酸索引；
    mutable_mask: 与 aa_index 同形（或可广播）的 bool 张量；None 表示全部可变。
    """

    aa_index: torch.Tensor
    mutable_mask: torch.Tensor | None = None

    @classmethod
    def from_topology(cls, topo: CGTopology,
                      mutable: torch.Tensor | None = None) -> SequenceState:
        """从参考序列构造。mutable: bool 数组/Tensor，缺省全可变。"""
        aa = torch.as_tensor(topo.aa_index, dtype=torch.long)
        mask = None
        if mutable is not None:
            mask = torch.as_tensor(mutable, dtype=torch.bool)
            if mask.shape != aa.shape:
                raise ValueError(
                    f"mutable shape {tuple(mask.shape)} != aa shape {tuple(aa.shape)}")
        return cls(aa_index=aa, mutable_mask=mask)

    def to_batch(self, batch: int) -> SequenceState:
        """(N,) -> (B, N)；已是 (B, N) 且 batch 匹配则原样返回（不复制）。"""
        aa = self.aa_index
        if aa.dim() == 1:
            aa = aa.unsqueeze(0).expand(batch, -1)
        elif aa.shape[0] == 1 and batch > 1:
            aa = aa.expand(batch, -1)
        elif aa.shape[0] != batch:
            raise ValueError(f"aa batch {int(aa.shape[0])} != expected {batch}")
        mask = self.mutable_mask
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch, -1)
            elif mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1)
        return SequenceState(aa_index=aa, mutable_mask=mask)

    def clone(self) -> SequenceState:
        return SequenceState(
            aa_index=self.aa_index.clone(),
            mutable_mask=None if self.mutable_mask is None else self.mutable_mask.clone())

    def to(self, device: torch.device | str) -> SequenceState:
        return SequenceState(
            aa_index=self.aa_index.to(device),
            mutable_mask=None if self.mutable_mask is None else self.mutable_mask.to(device))


class SequenceProposal:
    """序列提议算子接口：q(a' | R, a)。

    实现必须满足：sample 给出的提议可由 log_q 赋予确切的前向概率（Hastings
    修正需要反向概率，见设计文档 §6/§18）。位点级提议的单位是单个可动残基。
    """

    def logits(self, R: torch.Tensor, a: torch.Tensor,
               mutable_mask: torch.Tensor | None = None) -> torch.Tensor:
        """返回位置 i 上各氨基酸的（未归一化）logit，(B, N, N_AA)。"""
        raise NotImplementedError

    def sample(self, R: torch.Tensor, a: torch.Tensor, site: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """在位点 site（(B,) long，每 replica 一个）提议新氨基酸，返回 (B,)。"""
        raise NotImplementedError

    def log_q(self, R: torch.Tensor, a_from: torch.Tensor, a_to: torch.Tensor,
              site: torch.Tensor) -> torch.Tensor:
        """log q(a_to | R, a_from) 在位点 site，(B,)。前后调用共享同一 R。"""
        raise NotImplementedError
