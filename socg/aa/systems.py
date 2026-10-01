"""AA 参考体系规格（锁定）。

D3：amber14 + GBn2 隐式溶剂，HMR 4 fs，LangevinMiddle 1/ps。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemSpec:
    name: str
    sequence: str
    temperature: float          # K
    n_replicas: int
    ns_per_replica: float       # ns
    # 各 replica 循环使用的初始 (phi, psi)，单位度
    starts: tuple[tuple[float, float], ...] = field(default_factory=tuple)


SYSTEMS: dict[str, SystemSpec] = {
    "ala10": SystemSpec(
        name="ala10",
        sequence="AAAAAAAAAA",
        temperature=300.0,
        n_replicas=4,
        ns_per_replica=500.0,
        starts=((-60.0, -40.0), (-120.0, 130.0)),
    ),
    "cln025": SystemSpec(
        name="cln025",
        sequence="YYDPETGTWY",
        temperature=340.0,
        n_replicas=8,
        ns_per_replica=1000.0,
        starts=((-120.0, 130.0), (-70.0, 145.0)),
    ),
}


def load_reference_ca(pdb_path, expected_n: int | None = None):
    """参考结构（如 5AWL）的 Cα 坐标 (n,3)，nm：只取第 1 个 model 的第 1 条链。

    多链 / 结晶水 / 配体一律忽略；altloc 由 OpenMM 取第一个。expected_n 不符时抛 ValueError
    （不要自己编造参考结构）。
    """
    import numpy as np
    from openmm import app, unit

    from ..constants import AA3_TO_1

    pdb = app.PDBFile(str(pdb_path))
    pos = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    chain = next(iter(pdb.topology.chains()))
    ca = [a.index for r in chain.residues() if r.name in AA3_TO_1
          for a in r.atoms() if a.name == "CA"]
    out = pos[ca]
    if expected_n is not None and out.shape[0] != expected_n:
        raise ValueError(f"{pdb_path}: first chain has {out.shape[0]} CA atoms, "
                         f"expected {expected_n}")
    return out
