"""单位点均匀序列提议（Phase 2 基线提议，设计文档 §13 Stage A / §15）。

不依赖任何学习模型；前后提议概率恒等（MH 比值为 1），专门用于验证
MH 机器本身的正确性，并作为与 ProteinMPNN 提议对比的基线。
"""
from __future__ import annotations

import torch

from .base import SequenceProposal


class UniformSequenceProposal(SequenceProposal):
    """单位点均匀提议：a_i' 从除当前值外的 N_AA-1 种氨基酸均匀抽取。"""

    def logits(self, R: torch.Tensor, a: torch.Tensor,
               mutable_mask: torch.Tensor | None = None) -> torch.Tensor:
        """所有可变位点上的均匀分布 logit（不可变位点 -inf），(B, N, N_AA)。"""
        n_aa = self._alphabet_size(a)
        out = torch.zeros(*a.shape, n_aa, device=a.device, dtype=R.dtype)
        if mutable_mask is not None:
            out = out.masked_fill(~mutable_mask.unsqueeze(-1), float("-inf"))
        return out

    def sample(self, R: torch.Tensor, a: torch.Tensor, site: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
        old = a[torch.arange(a.shape[0], device=a.device), site]
        n_aa = self._alphabet_size(a)
        # 拒绝采样：重抽与当前相同的索引（N_AA=20 时期望 20/19 ≈ 1.05 次）
        cand = torch.randint(0, n_aa, old.shape, generator=generator, device=a.device)
        redo = cand == old
        while bool(redo.any()):
            cand[redo] = torch.randint(0, n_aa, (int(redo.sum()),),
                                       generator=generator, device=a.device)
            redo = cand == old
        return cand

    def log_q(self, R: torch.Tensor, a_from: torch.Tensor, a_to: torch.Tensor,
              site: torch.Tensor) -> torch.Tensor:
        n_aa = self._alphabet_size(a_from)
        return torch.full(site.shape, -float(torch.log(torch.tensor(float(n_aa - 1)))),
                          device=a_from.device)

    @staticmethod
    def _alphabet_size(a: torch.Tensor) -> int:
        return max(int(a.max().item()) + 1 if a.numel() else 0, 20)
