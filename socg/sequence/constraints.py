"""序列设计约束（统一设计文档 §20）。

约束以 logits 掩码的形式作用在提议分布上：不可变位点、禁止氨基酸、
受限氨基酸集合直接屏蔽；tied_positions（绑定位点，同组位点始终保持相同
氨基酸）在采样层处理；fixed_motifs（固定基序）是 mutable_mask 的便捷构造。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..constants import AA_ALPHABET
from ..topology import CGTopology

N_AA = len(AA_ALPHABET)


@dataclass
class SequenceConstraints:
    """序列约束集合。所有数组均为残基维（(N, ...)），按需广播到 batch。

    mutable_mask: (N,) bool，True = 该位点可突变；
    allowed_aa: (N, N_AA) bool，True = 该位点允许该氨基酸（含当前值）；
    forbidden_aa: (N, N_AA) bool，True = 禁止；与 allowed_aa 取交；
    tied_positions: list[tuple[int, ...]]，同组位点共享同一氨基酸（采样层）；
    """

    mutable_mask: torch.Tensor
    allowed_aa: torch.Tensor | None = None
    forbidden_aa: torch.Tensor | None = None
    tied_positions: tuple[tuple[int, ...], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.mutable_mask.dim() != 1:
            raise ValueError("mutable_mask must be (N,)")
        n = self.mutable_mask.shape[0]
        for name in ("allowed_aa", "forbidden_aa"):
            val = getattr(self, name)
            if val is not None:
                val = torch.as_tensor(val, dtype=torch.bool)
                if val.shape != (n, N_AA):
                    raise ValueError(f"{name} must be (N,{N_AA}), got {tuple(val.shape)}")
                setattr(self, name, val)
        for group in self.tied_positions:
            if len(group) < 2 or not all(0 <= i < n for i in group):
                raise ValueError(f"invalid tied position group {group!r}")
            if not all(bool(self.mutable_mask[i]) for i in group):
                raise ValueError(f"tied group {group!r} contains immutable sites")

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #
    @classmethod
    def from_topology(cls, topo: CGTopology, *,
                      mutable: str = "interior",
                      allowed_letters: dict[int, str] | None = None,
                      forbidden_letters: dict[int, str] | None = None,
                      tied_positions: tuple[tuple[int, ...], ...] = ()) -> SequenceConstraints:
        """mutable: "interior"（首尾不可变）| "all" | "none"。

        allowed_letters / forbidden_letters: {位点: 字母串}，如 {3: "KR", 7: "AGP"}。
        """
        n = topo.n
        if mutable == "interior":
            mask = torch.as_tensor(~topo.frozen_mask, dtype=torch.bool)
        elif mutable == "all":
            mask = torch.ones(n, dtype=torch.bool)
        elif mutable == "none":
            mask = torch.zeros(n, dtype=torch.bool)
        else:
            raise ValueError(f"mutable must be 'interior'|'all'|'none', got {mutable!r}")

        def letters_to_mask(spec: dict[int, str] | None) -> torch.Tensor | None:
            if not spec:
                return None
            m = torch.zeros(n, N_AA, dtype=torch.bool)
            for site, letters in spec.items():
                if not 0 <= site < n:
                    raise ValueError(f"site {site} out of range [0,{n})")
                for ch in letters:
                    if ch not in AA_ALPHABET:
                        raise ValueError(f"unknown amino-acid letter {ch!r}")
                    m[site, AA_ALPHABET.index(ch)] = True
            return m

        return cls(mutable_mask=mask,
                   allowed_aa=letters_to_mask(allowed_letters),
                   forbidden_aa=letters_to_mask(forbidden_letters),
                   tied_positions=tied_positions)

    def fixed_motif(self, start: int, motif: str) -> SequenceConstraints:
        """便捷构造：把 [start, start+len) 设为固定基序（从可变集中移出）。"""
        mask = self.mutable_mask.clone()
        mask[start:start + len(motif)] = False
        return SequenceConstraints(mutable_mask=mask, allowed_aa=self.allowed_aa,
                                   forbidden_aa=self.forbidden_aa,
                                   tied_positions=self.tied_positions)

    # ------------------------------------------------------------------ #
    # 应用
    # ------------------------------------------------------------------ #
    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """把约束作用到 (B, N, N_AA) logits 上：不可变/不允许/禁止 -> -inf。"""
        out = logits.clone()
        out = out.masked_fill(~self.mutable_mask[None, :, None], float("-inf"))
        if self.allowed_aa is not None:
            out = out.masked_fill(~self.allowed_aa[None, :, :], float("-inf"))
        if self.forbidden_aa is not None:
            out = out.masked_fill(self.forbidden_aa[None, :, :], float("-inf"))
        return out

    def validate_sequence(self, a: torch.Tensor) -> None:
        """校验序列满足 allowed/forbidden 约束（不满足抛 ValueError）。"""
        aa = a[0] if a.dim() == 2 else a
        idx = torch.arange(aa.shape[0], device=aa.device)
        if self.allowed_aa is not None:
            ok = self.allowed_aa[idx, aa]
            if not bool(ok.all()):
                site = int((~ok).nonzero()[0])
                raise ValueError(f"site {site} aa {int(aa[site])} violates allowed_aa")
        if self.forbidden_aa is not None:
            bad = self.forbidden_aa[idx, aa]
            if bool(bad.any()):
                site = int(bad.nonzero()[0])
                raise ValueError(f"site {site} aa {int(aa[site])} is forbidden")
