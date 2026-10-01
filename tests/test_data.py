"""T4: CGTrajectory / CGDataset 与 AA->CG 映射。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from socg.data import CGDataset, CGTrajectory
from socg.mapping import backbone_dihedrals, load_replica
from socg.topology import CGTopology


def _make_chunk(path: Path, frames: int, n_res: int, t_start: float, dt: float,
                seed: int = 0):
    rng = np.random.default_rng(seed)
    backbone = rng.normal(0, 0.1, size=(frames, n_res, 3, 3)).astype(np.float32)
    forces = rng.normal(0, 1.0, size=(frames, n_res, 3)).astype(np.float32)
    time_ps = (np.arange(frames) + t_start / dt + 1.0) * dt
    np.savez(path, backbone=backbone, forces=forces, time_ps=time_ps.astype(np.float64))
    return backbone, forces, time_ps


@pytest.fixture
def rep_dirs(tmp_path):
    topo = CGTopology.from_sequence("AAAAAAAAAA")
    dirs = []
    for r, offset in ((0, 0.0), (1, 40.0)):
        d = tmp_path / f"rep{r:02d}"
        d.mkdir()
        dt = 1.0
        meta = {"sequence": topo.sequence, "temperature": 300.0, "seed": r,
                "dt_save_ps": dt, "platform": "CPU", "chunks_done": 3, "ns_day": 0.0}
        (d / "meta.json").write_text(json.dumps(meta))
        for c in range(3):
            _make_chunk(d / f"chunk{c:04d}.npz", frames=5, n_res=10,
                        t_start=(c * 5) * dt, dt=dt, seed=100 + c + 10 * r)
        dirs.append(d)
    return dirs, topo


def test_dataset_two_trajectories(rep_dirs, tmp_path):
    (dirs, topo) = rep_dirs
    from socg.mapping import build_dataset

    ds = build_dataset(tmp_path, topo, 300.0)
    assert len(ds.trajectories) == 2
    assert all(t.n_frames == 15 for t in ds.trajectories)      # 3 chunk × 5 帧

    out = tmp_path / "cg" / "test.npz"
    out.parent.mkdir(exist_ok=True)
    ds.save(out)
    ds2 = CGDataset.load(out)
    assert ds2.topology == ds.topology
    assert ds2.temperature == ds.temperature
    for a, b in zip(ds.trajectories, ds2.trajectories):
        assert (a.coords == b.coords).all()
        assert (a.states == b.states).all()
        assert (a.forces == b.forces).all()
        assert a.dt_ps == b.dt_ps


def test_missing_chunk_raises(rep_dirs):
    # 删除中间的 chunk1（共 3 个）
    (dirs, topo) = rep_dirs
    (dirs[0] / "chunk0001.npz").unlink()
    with pytest.raises(ValueError):
        load_replica(dirs[0])


def test_duplicate_chunk_time_regression_raises(rep_dirs):
    (dirs, topo) = rep_dirs
    # 复制 chunk1 -> chunk3：编号连续，但时间回退，只能被“严格等差”这道闸抓住
    data = (dirs[0] / "chunk0001.npz").read_bytes()
    (dirs[0] / "chunk0003.npz").write_bytes(data)
    with pytest.raises(ValueError):
        load_replica(dirs[0])


def test_split_val(rep_dirs):
    (dirs, topo) = rep_dirs
    from socg.mapping import build_dataset

    ds = build_dataset(tmp_path_dir(rep_dirs), topo, 300.0)
    train, val = ds.split([1])
    assert len(val.trajectories) == 1
    assert len(train.trajectories) == 1
    assert (val.trajectories[0].coords == ds.trajectories[1].coords).all()


def tmp_path_dir(rep_dirs):
    return rep_dirs[0][0].parent


def test_frames_stride_and_concat(rep_dirs):
    (dirs, topo) = rep_dirs
    from socg.mapping import build_dataset

    ds = build_dataset(dirs[0].parent, topo, 300.0)
    coords, forces, states = ds.frames(stride=2)
    assert coords.shape[0] == sum(t.n_frames // 2 + (t.n_frames % 2) for t in ds.trajectories)
    assert forces is not None
    assert coords.shape[1:] == (10, 3)
    assert states.shape == coords.shape[:2]


def test_backbone_dihedrals_consistent_with_build():
    pytest.importorskip("PeptideBuilder")
    from socg.aa.build import build_peptide_pdb

    import openmm.app as app
    import openmm.unit as unit

    pdb_path = Path(__file__).parent / "_tmp_build_test.pdb"
    build_peptide_pdb("AAAAA", -60.0, -40.0, pdb_path)
    try:
        pdb = app.PDBFile(str(pdb_path))
        pos = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
        res = list(pdb.topology.residues())
        idx = [{a.name: a.index for a in r.atoms()} for r in res]
        backbone = np.zeros((1, len(res), 3, 3))
        for j, r in enumerate(res):
            backbone[0, j, 0] = pos[idx[j]["N"]]
            backbone[0, j, 1] = pos[idx[j]["CA"]]
            backbone[0, j, 2] = pos[idx[j]["C"]]
        phi, psi = backbone_dihedrals(backbone)
        assert np.isnan(phi[0, 0]) and np.isnan(psi[0, -1])
        assert phi[0, 2] == pytest.approx(-60.0, abs=1.0)
        assert psi[0, 2] == pytest.approx(-40.0, abs=1.0)
    finally:
        pdb_path.unlink(missing_ok=True)


def test_trajectory_positional_order():
    # 规格锁定的字段顺序：coords, forces, states, dt_ps
    trj = CGTrajectory(np.zeros((5, 4, 3)), None, np.zeros((5, 4)), 1.0)
    assert trj.forces is None and trj.dt_ps == 1.0


def test_trajectory_validation():
    with pytest.raises(ValueError):
        CGTrajectory(coords=np.zeros((5, 4, 3)), forces=None, states=np.zeros((5, 5)),
                     dt_ps=1.0)
    with pytest.raises(ValueError):
        CGTrajectory(coords=np.zeros((5, 4, 3)), forces=None, states=np.zeros((5, 4)),
                     dt_ps=0.0)
