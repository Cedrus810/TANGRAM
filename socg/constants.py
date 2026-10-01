"""全局常数与单位约定（全项目统一）。

单位：长度 nm、时间 ps、能量 kJ/mol、力 kJ/mol/nm、质量 amu、温度 K。
数据文件中角度用度，torch 模型内部用弧度。
"""
from __future__ import annotations

import math

# Boltzmann 常数，kJ/mol/K
KB: float = 0.0083144626181532

# 氨基酸字母表固定（索引 0–19），全项目唯一顺序
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# 三字母 -> 单字母
AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# 一字母 -> 三字母
AA1_TO_3: dict[str, str] = {v: k for k, v in AA3_TO_1.items()}


def kT(temperature: float) -> float:
    """热能 kT，单位 kJ/mol。"""
    return KB * float(temperature)


def beta(temperature: float) -> float:
    """1/kT，单位 mol/kJ。"""
    return 1.0 / kT(temperature)


def wrap_deg(delta: float) -> float:
    """把角度差周期折回到 [-180, 180)。"""
    return (delta + 180.0) % 360.0 - 180.0


__all__ = ["KB", "AA_ALPHABET", "AA_INDEX", "AA3_TO_1", "AA1_TO_3",
           "kT", "beta", "wrap_deg", "math"]
