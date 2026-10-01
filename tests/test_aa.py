"""T3: AA 体系构建、力映射与（慢速）生产跑冒烟。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from socg.aa.build import build_peptide_pdb
from socg.aa.simulate import (FF_FILES, HYDROGEN_MASS_AMU, TIMESTEP_PS,
                              cg_atom_indices, mapped_forces)
from socg.geometry import dihedral_deg
from socg.states import assign_states


@pytest.fixture(scope="module")
def ala5_pdb(tmp_path_factory):
    pytest.importorskip("PeptideBuilder")          # TODO(env)：openmm_dev 暂未安装
    out = tmp_path_factory.mktemp("aa") / "ala5.pdb"
    return build_peptide_pdb("AAAAA", -60.0, -40.0, out)


@pytest.fixture(scope="module")
def gag_pdb(tmp_path_factory):
    pytest.importorskip("PeptideBuilder")
    out = tmp_path_factory.mktemp("aa2") / "gag.pdb"
    return build_peptide_pdb("GAG", -60.0, -40.0, out)


def test_phi_psi_roundtrip(ala5_pdb):
    # Review Focus 2：符号约定错误会把螺旋判成 αL 且不报错
    import openmm.unit as unit
    from openmm import app

    pdb = app.PDBFile(str(ala5_pdb))
    pos = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    res = list(pdb.topology.residues())
    idx = [{a.name: a.index for a in r.atoms()} for r in res]
    for i in range(1, len(res) - 1):
        phi = dihedral_deg(pos[idx[i - 1]["C"]], pos[idx[i]["N"]],
                           pos[idx[i]["CA"]], pos[idx[i]["C"]])
        psi = dihedral_deg(pos[idx[i]["N"]], pos[idx[i]["CA"]],
                           pos[idx[i]["C"]], pos[idx[i + 1]["N"]])
        assert phi == pytest.approx(-60.0, abs=1.0)
        assert psi == pytest.approx(-40.0, abs=1.0)
    # 内部残基全部指派为 A 态
    T = 1
    n = len(res)
    backbone = np.zeros((T, n, 3, 3))
    for j, r in enumerate(res):
        backbone[0, j, 0] = pos[idx[j]["N"]]
        backbone[0, j, 1] = pos[idx[j]["CA"]]
        backbone[0, j, 2] = pos[idx[j]["C"]]
    from socg.mapping import backbone_dihedrals
    from socg.topology import CGTopology

    phi, psi = backbone_dihedrals(backbone)
    topo = CGTopology.from_sequence("AAAAA")
    states = assign_states(phi, psi, topo.frozen_mask)
    assert (states[0, 1:-1] == 0).all()


def test_cg_atom_indices_group_sizes(ala5_pdb, gag_pdb):
    from openmm import app

    topo = app.PDBFile(str(ala5_pdb)).topology
    idx = cg_atom_indices(topo)
    assert idx["force_groups"].shape[1] == 2        # Ala: CA + HA
    topo_g = app.PDBFile(str(gag_pdb)).topology
    idx_g = cg_atom_indices(topo_g)
    fg = idx_g["force_groups"]
    assert fg.shape[1] == 3                         # Gly: CA + HA2 + HA3
    assert (fg[0] >= 0).sum() == 3 and (fg[2] >= 0).sum() == 3   # G-A-G 的两个 Gly
    assert (fg[1] >= 0).sum() == 2                  # 中间的 Ala：CA + HA，余位 -1


def test_force_mapping_finite_difference(ala5_pdb):
    # 双精度 Reference 平台，把残基 3 的 CA(+H) 沿 x 平移 ±1e-5 nm
    import openmm as mm
    from openmm import app, unit

    pdb = app.PDBFile(str(ala5_pdb))
    ff = app.ForceField(*FF_FILES)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                             constraints=None)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integrator,
                         mm.Platform.getPlatformByName("Reference"))
    idx = cg_atom_indices(pdb.topology)
    groups = idx["force_groups"]
    res_i = 2
    atoms = [int(g) for g in groups[res_i] if g >= 0]
    x0 = np.asarray(pdb.positions.value_in_unit(unit.nanometer)).copy()

    def energy(x):
        sim.context.setPositions(x * unit.nanometer)
        return sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    def mapped_fx(x):
        sim.context.setPositions(x * unit.nanometer)
        f = sim.context.getState(getForces=True).getForces(asNumpy=True)
        f = np.asarray(f.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        return float(mapped_forces(f, groups)[res_i, 0])

    h = 1e-5
    e_plus = energy(_shift(x0, atoms, +h))
    e_minus = energy(_shift(x0, atoms, -h))
    fd = (e_plus - e_minus) / (2 * h)
    mf = mapped_fx(x0)
    assert -mf == pytest.approx(fd, rel=1e-3)


def _shift(x, atoms, h):
    out = x.copy()
    out[atoms, 0] += h
    return out


@pytest.mark.slow
def test_run_replica_smoke(tmp_path, ala5_pdb):
    from socg.aa.simulate import run_replica

    out_dir = tmp_path / "rep00"
    run_replica(ala5_pdb, out_dir, 300.0, ns=0.002, seed=17, platform_name="CPU",
                save_every_steps=25, frames_per_chunk=10, equil_ps=0.0)
    files = sorted(out_dir.glob("chunk*.npz"))
    assert len(files) == 2                              # 20 帧 / 10 帧 = 2 chunks
    with np.load(files[0]) as z:
        assert z["backbone"].shape == (10, 5, 3, 3)
        assert z["forces"].shape == (10, 5, 3)
        t = z["time_ps"]
    assert np.allclose(np.diff(t), 25 * TIMESTEP_PS)
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["chunks_done"] == 2
    assert meta["sequence"] == "AAAAA"
    assert meta["temperature"] == 300.0
    assert meta["seed"] == 17


@pytest.mark.slow
def test_run_replica_resume(tmp_path, ala5_pdb):
    # Review Focus 3：中断续跑不覆盖已完成 chunk，时间连续
    from socg.aa.simulate import run_replica

    out_dir = tmp_path / "rep01"
    run_replica(ala5_pdb, out_dir, 300.0, ns=0.001, seed=17, platform_name="CPU",
                save_every_steps=25, frames_per_chunk=10, equil_ps=0.0)
    with np.load(out_dir / "chunk0000.npz") as z:
        first_chunk = {k: z[k].copy() for k in z.files}
        t0_end = float(z["time_ps"][-1])
    # 续跑更长的 ns：只写新的 chunk
    run_replica(ala5_pdb, out_dir, 300.0, ns=0.002, seed=17, platform_name="CPU",
                save_every_steps=25, frames_per_chunk=10, equil_ps=0.0)
    with np.load(out_dir / "chunk0000.npz") as z:
        for k in first_chunk:
            assert np.array_equal(first_chunk[k], z[k])   # 永不覆盖
    with np.load(out_dir / "chunk0001.npz") as z:
        t1_start = float(z["time_ps"][0])
    assert t1_start == pytest.approx(t0_end + 25 * TIMESTEP_PS)


@pytest.mark.slow
def test_run_replica_crash_recovery(tmp_path, ala5_pdb):
    # 审阅 P1-5：模拟两种崩溃现场，续跑必须回到最后提交状态、不覆盖任何 chunk
    import shutil

    from socg.aa.simulate import CHK_NAME, run_replica
    from socg.mapping import load_replica

    kw = dict(temperature=300.0, seed=17, platform_name="CPU",
              save_every_steps=25, frames_per_chunk=10, equil_ps=0.0)
    out_dir = tmp_path / "rep02"
    run_replica(ala5_pdb, out_dir, ns=0.001, **kw)          # 提交了 chunk0
    chunk0 = (out_dir / "chunk0000.npz").read_bytes()
    # 现场 A：chunk1 已写入但 meta 未提交（孤儿 chunk）
    shutil.copy(out_dir / "chunk0000.npz", out_dir / "chunk0001.npz")
    # 现场 B：meta 已提交、checkpoint 改名未完成（只剩 state.chk.next）
    os.replace(out_dir / CHK_NAME, out_dir / (CHK_NAME + ".next"))

    run_replica(ala5_pdb, out_dir, ns=0.002, **kw)
    assert (out_dir / "chunk0000.npz").read_bytes() == chunk0     # 永不覆盖
    assert list(out_dir.glob("chunk0001.npz.orphan-*"))            # 孤儿被留档
    assert (out_dir / CHK_NAME).exists() and not (out_dir / (CHK_NAME + ".next")).exists()
    backbone, forces, t = load_replica(out_dir)                    # 两道闸门都通过
    assert backbone.shape[0] == 20
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["chunks_done"] == 2 and meta["orphans"]


def test_constants():
    assert TIMESTEP_PS == 0.004
    assert HYDROGEN_MASS_AMU == 4.0
