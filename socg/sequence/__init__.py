"""序列层：运行时序列状态、提议算子、约束（统一设计文档 §15）。

对外导出：
- SequenceState / SequenceProposal（base）
- UniformSequenceProposal（random，Phase 2 基线提议）
- SequenceConstraints（constraints，§20）
- EnergyGuidedProposal（proposals，§28 能量引导 logits）
- ProteinMPNNBackend / ProteinMPNNProposal / cg_geometry_provider
  （proteinmpnn，Phase 3 适配层，可选依赖）
- pseudo_backbone / pseudo_backbone_numpy（backbone，§19 伪骨架重建）
"""
from .backbone import pseudo_backbone, pseudo_backbone_numpy
from .base import SequenceProposal, SequenceState
from .constraints import SequenceConstraints
from .proposals import EnergyGuidedProposal
from .random import UniformSequenceProposal

__all__ = [
    "EnergyGuidedProposal",
    "SequenceConstraints",
    "SequenceProposal",
    "SequenceState",
    "UniformSequenceProposal",
    "pseudo_backbone",
    "pseudo_backbone_numpy",
]


def __getattr__(name):
    """ProteinMPNN 适配层懒导出（import 即触发可选依赖解析的类延迟到属性访问）。"""
    if name in ("ProteinMPNNBackend", "ProteinMPNNProposal", "cg_geometry_provider"):
        from . import proteinmpnn as _pmpnn
        return getattr(_pmpnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
