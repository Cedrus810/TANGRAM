"""序列 -> 带氢 PDB（PeptideBuilder 重原子 + OpenMM 补氢）。"""
from __future__ import annotations

from pathlib import Path

from ..constants import AA1_TO_3


def build_peptide_pdb(sequence: str, phi_deg: float, psi_deg: float, out_pdb) -> Path:
    """构建一条所有内部残基 φ/ψ 相同的肽，补 OXT 与氢，写入 out_pdb。

    PeptideBuilder 只能在添加第 i+1 个残基时设定第 i 个残基的 ψ 与第 i+1
    个残基的 φ，因此内部残基（2..n-1）会同时拿到指定的 (φ, ψ)。
    """
    import PeptideBuilder  # 延迟导入，避免无关模块拖依赖
    from openmm import app

    sequence = str(sequence)
    n = len(sequence)
    for ch in sequence:
        if ch not in AA1_TO_3:
            raise ValueError(f"unknown amino-acid letter {ch!r}")
    phi_deg = float(phi_deg)
    psi_deg = float(psi_deg)
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    phis = [phi_deg] * n
    psis = [psi_deg] * n

    structure = None
    # 优先使用批量 API；老版本回退到逐残基 API
    try:
        structure = PeptideBuilder.make_structure(sequence, phis, psis)
    except (AttributeError, TypeError):
        peptide = PeptideBuilder.initialize_res(AA1_TO_3[sequence[0]])
        for i in range(1, n):
            peptide = PeptideBuilder.add_residue(peptide, sequence[i],
                                                 phis[i], psis[i - 1])
        structure = peptide

    # 末端 OXT：先试库函数，失败则按 N-CA-C 平面镜像 O 手工放置
    try:
        _add_oxt_library(structure)
    except Exception:  # noqa: BLE001 — 手工兜底
        _add_oxt_manual(structure)

    heavy_pdb = out_pdb.with_suffix(".heavy.pdb")
    _save_pdb(structure, heavy_pdb)

    # 补氢（amber14 模板，pH 7）
    pdb = app.PDBFile(str(heavy_pdb))
    ff = app.ForceField("amber14-all.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=7.0)
    with open(out_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    heavy_pdb.unlink()
    return out_pdb


def _iter_residues(structure):
    """PeptideBuilder 返回对象可能是 Structure 或 Polypeptide/链。"""
    model = structure
    if hasattr(structure, "get_models"):
        model = structure.get_models()[0] if list(structure.get_models()) else structure
    if hasattr(model, "get_residues"):
        return list(model.get_residues())
    return [structure]


def _add_oxt_library(structure) -> None:
    import PeptideBuilder

    residues = _iter_residues(structure)
    last = residues[-1]
    if "OXT" in last:
        return
    adder = getattr(PeptideBuilder, "add_terminal_OXT", None)
    if adder is None:
        raise AttributeError("PeptideBuilder.add_terminal_OXT unavailable")
    try:
        adder(structure)
    except TypeError:
        adder(last)


def _add_oxt_manual(structure) -> None:
    """OXT = O 关于 (N, CA, C) 平面的镜像（羧酸根两个氧关于该平面对称）。"""
    import numpy as np

    residues = _iter_residues(structure)
    last = residues[-1]
    if "OXT" in last:
        return
    N = last["N"].coord.astype(float)
    CA = last["CA"].coord.astype(float)
    C = last["C"].coord.astype(float)
    O = last["O"].coord.astype(float)
    normal = np.cross(CA - N, C - N)
    normal = normal / np.linalg.norm(normal)
    v = O - C
    oxt = O - 2.0 * np.dot(v, normal) * normal
    from Bio.PDB import Atom

    atom = Atom.Atom(
        name="OXT", coord=oxt, bfactor=0.0, occupancy=1.0, altloc=" ",
        fullname=" OXT", serial_number=max(a.serial_number for a in last.get_atoms()) + 1,
        element="O",
    )
    last.add(atom)


def _save_pdb(structure, path: Path) -> None:
    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))
