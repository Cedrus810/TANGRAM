"""Cα 粗粒化拓扑：序列 -> 键/角/二面角/非局部 pair 图。

锁定的接口约定（Phase 1 文档 T1）：
- RES_CLASS_GLY=0, RES_CLASS_PRO=1, RES_CLASS_OTHER=2, N_RES_CLASSES=3
- N_SEP_BUCKETS=3；pair bucket：0 -> 序号差 3，1 -> 差 4，2 -> 差 >=5
- capped=False（两端两性离子）时首、尾残基 frozen=True
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .constants import AA_INDEX

RES_CLASS_GLY = 0
RES_CLASS_PRO = 1
RES_CLASS_OTHER = 2
N_RES_CLASSES = 3
N_SEP_BUCKETS = 3


def _res_class(letter: str) -> int:
    if letter == "G":
        return RES_CLASS_GLY
    if letter == "P":
        return RES_CLASS_PRO
    return RES_CLASS_OTHER


@dataclass(frozen=True)
class CGTopology:
    """一条序列的 Cα 拓扑。所有数组均为派生属性（property）。"""

    sequence: str
    frozen: tuple[bool, ...]

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #
    @classmethod
    def from_sequence(cls, sequence: str, capped: bool = False) -> "CGTopology":
        sequence = str(sequence)
        if len(sequence) < 4:
            raise ValueError(f"sequence too short (<4): {sequence!r}")
        for ch in sequence:
            if ch not in AA_INDEX:
                raise ValueError(f"unknown amino-acid letter {ch!r} in {sequence!r}")
        if capped:
            frozen = tuple([False] * len(sequence))
        else:
            frozen = tuple([True] + [False] * (len(sequence) - 2) + [True])
        return cls(sequence=sequence, frozen=frozen)

    def __post_init__(self) -> None:
        if len(self.frozen) != len(self.sequence):
            raise ValueError(
                f"frozen length {len(self.frozen)} != sequence length {len(self.sequence)}"
            )
        for ch in self.sequence:
            if ch not in AA_INDEX:
                raise ValueError(f"unknown amino-acid letter {ch!r}")

    # ------------------------------------------------------------------ #
    # 派生量
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.sequence)

    @property
    def frozen_mask(self) -> np.ndarray:
        return np.asarray(self.frozen, dtype=bool)

    @property
    def movable_indices(self) -> np.ndarray:
        return np.where(~self.frozen_mask)[0].astype(np.int64)

    @property
    def aa_index(self) -> np.ndarray:
        return np.asarray([AA_INDEX[ch] for ch in self.sequence], dtype=np.int64)

    @property
    def res_class(self) -> np.ndarray:
        return np.asarray([_res_class(ch) for ch in self.sequence], dtype=np.int64)

    @property
    def bonds(self) -> np.ndarray:
        return np.stack([np.arange(self.n - 1), np.arange(1, self.n)], axis=1).astype(np.int64)

    @property
    def angles(self) -> np.ndarray:
        i = np.arange(1, self.n - 1)
        return np.stack([i - 1, i, i + 1], axis=1).astype(np.int64)

    @property
    def dihedrals(self) -> np.ndarray:
        i = np.arange(self.n - 3)
        return np.stack([i, i + 1, i + 2, i + 3], axis=1).astype(np.int64)

    @property
    def pairs(self) -> np.ndarray:
        """所有 j - i >= 3 的 (i, j)，按 (i, j) 字典序。"""
        idx_i, idx_j = np.triu_indices(self.n, k=3)
        return np.stack([idx_i, idx_j], axis=1).astype(np.int64)

    @property
    def pair_bucket(self) -> np.ndarray:
        pairs = self.pairs
        sep = pairs[:, 1] - pairs[:, 0]
        bucket = np.where(sep == 3, 0, np.where(sep == 4, 1, 2))
        return bucket.astype(np.int64)

    @property
    def pair_types_index(self) -> np.ndarray:
        """bucket==2 的 pair 行号（用于 w_pair_type 项）。"""
        return np.where(self.pair_bucket == 2)[0].astype(np.int64)

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def to_json(self) -> str:
        return json.dumps({"sequence": self.sequence, "frozen": list(self.frozen)})

    @classmethod
    def from_json(cls, s: str) -> "CGTopology":
        d = json.loads(s)
        return cls(sequence=d["sequence"], frozen=tuple(bool(b) for b in d["frozen"]))
