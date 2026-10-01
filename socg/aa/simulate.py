"""OpenMM 生产跑：amber14 + GBn2、HMR、力映射、chunk 与 checkpoint 续跑。

帧约定：生产阶段每 save_every_steps 步存一帧（默认 250 步 = 1 ps）。
chunk{NNNN}.npz 内含 backbone (n_res,3,3)（N、CA、C 依次）、映射力
(n_res,3)、time_ps；每个 chunk 写完后按“chunk → checkpoint → meta”顺序原子提交。
time_ps[k] = (全局帧号 k+1) * dt_save_ps，保证跨 chunk 连续且严格等差。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# 锁定常数
FF_FILES = ("amber14-all.xml", "implicit/gbn2.xml")
TIMESTEP_PS = 0.004
FRICTION_PER_PS = 1.0
HYDROGEN_MASS_AMU = 4.0

DCD_INTERVAL_PS = 10.0
CHK_NAME = "state.chk"


def make_simulation(pdb_path, temperature: float, seed: int, platform_name: str = "CUDA"):
    """构建 OpenMM Simulation：NoCutoff、HBonds 约束、HMR、LangevinMiddle。"""
    from openmm import app, unit

    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(*FF_FILES)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=HYDROGEN_MASS_AMU * unit.amu,
    )
    from openmm import LangevinMiddleIntegrator

    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        TIMESTEP_PS * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(int(seed))
    platform = _platform(platform_name)
    sim = app.Simulation(pdb.topology, system, integrator, platform, _platform_props(platform))
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, int(seed))
    return sim


def _platform(name: str):
    from openmm import Platform

    if name == "CUDA" and Platform.getNumPlatforms() > 0:
        try:
            return Platform.getPlatformByName("CUDA")
        except Exception:  # noqa: BLE001
            pass
    return Platform.getPlatformByName(name)


def _platform_props(platform):
    name = platform.getName()
    if name == "CUDA":
        return {"Precision": "mixed", "DeterministicForces": "true"}
    return {}


# --------------------------------------------------------------------- #
# Cα 力映射
# --------------------------------------------------------------------- #
def cg_atom_indices(topology) -> dict:
    """每个残基的 N/CA/C 原子号与力映射组。

    force_groups (n_res, G)：CA + 与 CA 成键的 H；不足 G 的位置填 -1。
    """
    import openmm as mm

    residues = list(topology.residues())
    n = len(residues)
    n_idx, ca_idx, c_idx = [], [], []
    groups: list[list[int]] = []
    bonds: list[tuple[int, int]] = [(b[0].index, b[1].index) for b in topology.bonds()]
    neighbors: dict[int, list[int]] = {}
    for a, b in bonds:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    is_h = {}
    for atom in topology.atoms():
        is_h[atom.index] = atom.element is not None and atom.element == mm.element.hydrogen

    for res in residues:
        atoms = {a.name: a.index for a in res.atoms()}
        n_idx.append(atoms["N"])
        ca_idx.append(atoms["CA"])
        c_idx.append(atoms["C"])
        ca = atoms["CA"]
        grp = [ca] + sorted(h for h in neighbors.get(ca, []) if is_h.get(h, False))
        groups.append(grp)

    G = max(len(g) for g in groups)
    force_groups = np.full((n, G), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        force_groups[i, : len(g)] = g
    return {
        "N": np.asarray(n_idx, dtype=np.int64),
        "CA": np.asarray(ca_idx, dtype=np.int64),
        "C": np.asarray(c_idx, dtype=np.int64),
        "force_groups": force_groups,
    }


def mapped_forces(forces: np.ndarray, force_groups: np.ndarray) -> np.ndarray:
    """(n_atoms,3) -> (n_res,3)；-1 视为零贡献。"""
    forces = np.asarray(forces)
    valid = force_groups >= 0
    idx = np.where(valid, force_groups, 0)
    contrib = forces[idx] * valid[:, :, None]
    return contrib.sum(axis=1).astype(np.float32)


# --------------------------------------------------------------------- #
# 生产跑
# --------------------------------------------------------------------- #
def _sequence_from_topology(topology) -> str:
    from ..constants import AA3_TO_1

    three = [res.name for res in topology.residues()]
    return "".join(AA3_TO_1.get(r, "X") for r in three)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _recover_checkpoint(out_dir: Path, meta: dict) -> bytes:
    """找到与 meta 中 chk_sha256 一致的 checkpoint（state.chk 或未完成改名的 state.chk.next）。"""
    chk = out_dir / CHK_NAME
    nxt = out_dir / (CHK_NAME + ".next")
    want = meta.get("chk_sha256")
    for cand in (chk, nxt):
        if cand.exists():
            data = cand.read_bytes()
            if want is None or _sha256(data) == want:
                if cand is nxt:                       # 崩在 meta 提交之后、改名之前
                    os.replace(nxt, chk)
                elif nxt.exists():                    # 崩在 meta 提交之前：next 作废
                    nxt.unlink()
                return data
    raise ValueError(
        f"{out_dir}: no checkpoint matches meta.json (chunks_done={meta.get('chunks_done')}); "
        "refusing to resume")


def _set_aside_orphans(out_dir: Path, first_uncommitted: int) -> list[str]:
    """编号 >= first_uncommitted 的 chunk 是未提交的残留：改名留档，绝不覆盖。"""
    moved = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in sorted(out_dir.glob("chunk*.npz")):
        k = int(f.stem[len("chunk"):])
        if k >= first_uncommitted:
            dst = f.with_name(f"{f.name}.orphan-{stamp}")
            os.replace(f, dst)
            moved.append(dst.name)
    return moved


def run_replica(pdb_path, out_dir, temperature: float, ns: float, seed: int,
                platform_name: str = "CUDA", save_every_steps: int = 250,
                frames_per_chunk: int = 100_000, equil_ps: float = 100.0) -> None:
    """跑一个 replica（可断点续跑）。输出写到 out_dir。

    提交顺序（每个 chunk）：chunk 原子写入 → state.chk.next → meta.json（含
    checkpoint 的 sha256）原子提交 → state.chk.next 改名为 state.chk。任何一步
    崩溃，续跑都会回到最后一次提交的状态；未提交的 chunk 被改名留档而不是覆盖。
    checkpoint 用 OpenMM createCheckpoint，恢复积分器随机数状态。
    """
    from openmm import app, unit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    dt_save_ps = save_every_steps * TIMESTEP_PS
    total_frames = int(round(ns * 1000.0 / dt_save_ps))
    n_chunks = max(1, math.ceil(total_frames / frames_per_chunk))
    dcd_every_frames = max(1, int(round(DCD_INTERVAL_PS / dt_save_ps)))

    sim = make_simulation(pdb_path, temperature, seed, platform_name)
    topology = sim.topology
    idx = cg_atom_indices(topology)
    n_res = len(list(topology.residues()))

    resume = meta_path.exists()
    if resume:
        meta = json.loads(meta_path.read_text())
        # 续跑参数必须与首跑一致（否则帧号/时间轴会错位）
        for key, val in (("temperature", float(temperature)), ("seed", int(seed)),
                         ("save_every_steps", int(save_every_steps)),
                         ("frames_per_chunk", int(frames_per_chunk))):
            if key in meta and meta[key] != val:
                raise ValueError(f"{out_dir}: resume with {key}={val} but meta has {meta[key]}")
        start_chunk = int(meta["chunks_done"])
        frames_done = int(meta.get("frames_done", start_chunk * frames_per_chunk))
        if frames_done != start_chunk * frames_per_chunk:
            raise ValueError(
                f"{out_dir}: last committed chunk is partial ({frames_done} frames); "
                "cannot extend without a time gap")
        sim.context.loadCheckpoint(_recover_checkpoint(out_dir, meta))
        if "current_step" in meta:
            sim.currentStep = int(meta["current_step"])
        orphans = _set_aside_orphans(out_dir, start_chunk)
        if orphans:
            meta.setdefault("orphans", []).extend(orphans)
    else:
        meta = {"chunks_done": 0, "frames_done": 0, "ns_day": 0.0}
        # 首个 chunk 提交前崩溃留下的残留：留档后从头开始
        orphans = _set_aside_orphans(out_dir, 0)
        if orphans:
            meta["orphans"] = orphans
        for stale in (out_dir / CHK_NAME, out_dir / (CHK_NAME + ".next"), out_dir / "viz.dcd"):
            if stale.exists():
                stale.unlink()
        start_chunk = 0
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, int(seed))
        equil_steps = int(round(equil_ps * 1000.0 / TIMESTEP_PS))
        if equil_steps > 0:
            sim.step(equil_steps)

    meta.update({
        "sequence": _sequence_from_topology(topology),
        "temperature": float(temperature),
        "seed": int(seed),
        "dt_save_ps": dt_save_ps,
        "platform": sim.context.getPlatform().getName(),
        "save_every_steps": int(save_every_steps),
        "frames_per_chunk": int(frames_per_chunk),
        "equil_ps": float(equil_ps),
    })

    # 可视化 DCD（每 10 ps 一帧）：只在 chunk 提交后写入，续跑时以 r+b 追加
    dcd_path = out_dir / "viz.dcd"
    dcd_append = dcd_path.exists() and dcd_path.stat().st_size > 0
    dcd_fh = open(dcd_path, "r+b" if dcd_append else "wb")
    dcd_file = app.DCDFile(dcd_fh, topology, dt=DCD_INTERVAL_PS * unit.picoseconds,
                           firstStep=0, interval=1, append=dcd_append)

    wall_start = time.time()
    frames_this_session = 0
    try:
        for chunk in range(start_chunk, n_chunks):
            lo = chunk * frames_per_chunk
            hi = min((chunk + 1) * frames_per_chunk, total_frames)
            if hi <= lo:
                break
            n_frames = hi - lo
            backbone = np.zeros((n_frames, n_res, 3, 3), dtype=np.float32)
            forces_out = np.zeros((n_frames, n_res, 3), dtype=np.float32)
            time_ps = (np.arange(lo, hi, dtype=np.float64) + 1.0) * dt_save_ps
            dcd_frames = []

            for f in range(n_frames):
                sim.step(save_every_steps)
                state = sim.context.getState(getPositions=True, getForces=True)
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                frc = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer)
                backbone[f, :, 0, :] = pos[idx["N"]]
                backbone[f, :, 1, :] = pos[idx["CA"]]
                backbone[f, :, 2, :] = pos[idx["C"]]
                forces_out[f] = mapped_forces(frc, idx["force_groups"])
                if (lo + f + 1) % dcd_every_frames == 0:
                    dcd_frames.append(np.array(pos))

            # ---- 提交 ----
            _atomic_savez(out_dir / f"chunk{chunk:04d}.npz",
                          backbone=backbone, forces=forces_out, time_ps=time_ps)
            chk_bytes = sim.context.createCheckpoint()
            _atomic_write_bytes(out_dir / (CHK_NAME + ".next"), chk_bytes)
            frames_this_session += n_frames
            elapsed_days = (time.time() - wall_start) / 86400.0
            meta["chunks_done"] = chunk + 1
            meta["frames_done"] = int(hi)
            meta["current_step"] = int(sim.currentStep)
            meta["chk_sha256"] = _sha256(chk_bytes)
            session_ns = frames_this_session * dt_save_ps / 1000.0
            meta["ns_day"] = float(session_ns / elapsed_days) if elapsed_days > 0 else 0.0
            _atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode())
            os.replace(out_dir / (CHK_NAME + ".next"), out_dir / CHK_NAME)

            for p in dcd_frames:
                dcd_file.writeModel(p * unit.nanometer)
            dcd_fh.flush()
    finally:
        dcd_fh.close()
