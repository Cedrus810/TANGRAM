"""复合提议算子（统一设计文档 §28：能量引导 logits 模式）。

EnergyGuidedProposal 用 TANGRAM 的近似局部突变能量修正基础提议的 logits：

    ell_i'(a) = ell_i(a) - beta_guide * ΔU_i(a)

并以 softmax(ell_i') 为提议分布。采样模式下它仍是**提议分布**——若要精确
平衡分布，MH 修正（log_q 的前后比值）依然必须；设计模式下可直接作为
物理引导的生成器使用。
"""
from __future__ import annotations

import torch

from .base import SequenceProposal


class EnergyGuidedProposal(SequenceProposal):
    """在基础提议的 logits 上叠加 -beta_guide * ΔU 的引导提议。

    base: 被包装的提议（如 ProteinMPNNProposal / UniformSequenceProposal）；
    delta_u_fn: 可调用对象 (R, a, site) -> (B, N_AA)，给出位点 site 上 20 种
    替换的近似突变能（通常传 SOCGModel.mutation_delta_u_all_aa）；
    beta_guide: 引导强度（1/kT 单位），0 退化为 base。
    """

    def __init__(self, base: SequenceProposal, delta_u_fn, beta_guide: float = 1.0):
        self.base = base
        self.delta_u_fn = delta_u_fn
        self.beta_guide = float(beta_guide)
        # 最近一次 logits（供 log_q 复用，避免重复评估 ΔU）
        self._logit_cache: tuple[int, torch.Tensor] | None = None

    def logits(self, R: torch.Tensor, a: torch.Tensor,
               mutable_mask: torch.Tensor | None = None) -> torch.Tensor:
        base_logits = self.base.logits(R, a, mutable_mask)
        return base_logits  # 逐位点引导走 guided_logits（依赖具体 site）

    def guided_logits(self, R: torch.Tensor, a: torch.Tensor, site: torch.Tensor,
                      mutable_mask: torch.Tensor | None = None) -> torch.Tensor:
        """位点 site 的引导 logits：(B, N_AA) = base_logits - beta_guide * ΔU_i。"""
        base_logits = self.base.logits(R, a, mutable_mask)
        rows = base_logits[torch.arange(base_logits.shape[0], device=R.device), site]
        dU = self.delta_u_fn(R, a, site)                      # (B, N_AA)
        guided = rows - self.beta_guide * dU.to(rows.dtype)
        if mutable_mask is not None:
            m = mutable_mask[torch.arange(a.shape[0], device=R.device), site]
            guided = guided.masked_fill(~m[:, None], float("-inf"))
        return guided

    def sample(self, R: torch.Tensor, a: torch.Tensor, site: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
        guided = self.guided_logits(R, a, site)
        probs = torch.softmax(guided, dim=-1)
        new = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        self._logit_cache = (id(a), guided)                   # 供 log_q 复用
        return new

    def log_q(self, R: torch.Tensor, a_from: torch.Tensor, a_to: torch.Tensor,
              site: torch.Tensor) -> torch.Tensor:
        """log q(a_to | R, a_from)；优先复用 sample 缓存的引导 logits。"""
        cached = self._logit_cache
        if cached is not None and cached[0] == id(a_from):
            guided = cached[1]
        else:
            guided = self.guided_logits(R, a_from, site)
        rows = guided[torch.arange(guided.shape[0], device=R.device), site]
        log_z = torch.logsumexp(guided, dim=-1)
        return rows - log_z
