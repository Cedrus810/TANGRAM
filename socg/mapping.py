"""AA chunk -> CG 轨迹映射：骨架二面角、replica 读取（含完整性检查）、数据集构建。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .data import CGDataset, CGTrajectory
from .geometry import dihedral_deg
from .states import assign_states
from .topology import CGTopology

_CHUNK_RE = re.compile(r"chunk(\d+)\.npz$")


def backbone_dihedrals(backbone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """骨架 (T,n,3,3)（原子序 N、CA、C）-> (phi, psi)，单位度。

    phi[:,0] 与 psi[:,-1] 无定义，置 NaN。
    """
    backbone = np.asarray(backbone, dtype=np.float64)
    if backbone.ndim != 4 or backbone.shape[2:] != (3, 3):
        raise ValueError(f"backbone must be (T,n,3,3), got {backbone.shape}")
    T, n = backbone.shape[:2]
    phi = np.full((T, n), np.nan)
    psi = np.full((T, n), np.nan)
    if n >= 2:
        # phi_i = dihedral(C(i-1), N(i), CA(i), C(i))
        phi[:, 1:] = dihedral_deg(backbone[:, :-1, 2, :],
                                  backbone[:, 1:, 0, :],
                                  backbone[:, 1:, 1, :],
                                  backbone[:, 1:, 2, :])
    if n >= 2:
        # psi_i = dihedral(N(i), CA(i), C(i), N(i+1))
        psi[:, :-1] = dihedral_deg(backbone[:, :-1, 0, :],
                                   backbone[:, :-1, 1, :],
                                   backbone[:, :-1, 2, :],
                                   backbone[:, 1:, 0, :])
    return phi, psi


def load_replica(rep_dir):
    """读取一个 replica 的全部 chunk。

    chunk 按编号排序且必须连续（0..K-1，无缺失/重复）；拼接后的
    time_ps 必须严格等差（步长 = meta.dt_save_ps）。违反即抛 ValueError。

    Returns
    -------
    (backbone (T,n,3,3) f32, forces (T,n,3) f32, time_ps (T,) f64)
    """
    rep_dir = Path(rep_dir)
    meta_path = rep_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"missing meta.json in {rep_dir}")
    meta = json.loads(meta_path.read_text())
    dt_save_ps = float(meta["dt_save_ps"])

    chunks = {}
    for f in rep_dir.glob("chunk*.npz"):
        m = _CHUNK_RE.search(f.name)
        if m:
            chunks[int(m.group(1))] = f
    if not chunks:
        raise ValueError(f"no chunks found in {rep_dir}")
    numbers = sorted(chunks)
    if numbers != list(range(len(numbers))):
        raise ValueError(f"chunk numbering has gaps/duplicates in {rep_dir}: {numbers}")

    backbones, forces, times = [], [], []
    for k in numbers:
        with np.load(chunks[k], allow_pickle=False) as z:
            backbones.append(np.asarray(z["backbone"], dtype=np.float32))
            forces.append(np.asarray(z["forces"], dtype=np.float32))
            times.append(np.asarray(z["time_ps"], dtype=np.float64))
    backbone = np.concatenate(backbones, axis=0)
    force = np.concatenate(forces, axis=0)
    time_ps = np.concatenate(times, axis=0)

    if backbone.shape[0] != time_ps.shape[0]:
        raise ValueError(f"frame count mismatch in {rep_dir}")
    if time_ps.shape[0] < 2:
        raise ValueError(f"replica {rep_dir} has fewer than 2 frames")
    if not np.all(np.diff(time_ps) > 0):
        raise ValueError(f"time_ps not strictly increasing in {rep_dir}")
    if not np.allclose(np.diff(time_ps), dt_save_ps, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"time_ps not uniformly spaced (dt={dt_save_ps}) in {rep_dir}; "
            "a chunk may be missing or duplicated")
    return backbone, force, time_ps


def replica_to_cg(rep_dir, topo: CGTopology, radius: float = 40.0) -> CGTrajectory:
    """一个 AA replica 目录 -> CGTrajectory（Cα 坐标 + 映射力 + TBA 态）。"""
    backbone, force, time_ps = load_replica(rep_dir)
    coords = np.ascontiguousarray(backbone[:, :, 1, :])          # CA
    phi, psi = backbone_dihedrals(backbone)
    states = assign_states(phi, psi, topo.frozen_mask, radius=radius)
    dt_ps = float(time_ps[1] - time_ps[0])
    return CGTrajectory(coords=coords, forces=force, states=states, dt_ps=dt_ps)


def build_dataset(system_dir, topo: CGTopology, temperature: float,
                  radius: float = 40.0) -> CGDataset:
    """data/aa/<system>/ 下每个 rep*/ 目录 -> 一条 CG 轨迹。"""
    system_dir = Path(system_dir)
    rep_dirs = sorted(d for d in system_dir.iterdir()
                      if d.is_dir() and d.name.startswith("rep"))
    if not rep_dirs:
        raise ValueError(f"no rep*/ directories under {system_dir}")
    trajs = [replica_to_cg(d, topo, radius=radius) for d in rep_dirs]
    return CGDataset(topology=topo, temperature=temperature, trajectories=trajs)
