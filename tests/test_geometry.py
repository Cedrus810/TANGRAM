"""T3: 几何工具（二面角符号约定、Kabsch RMSD）。"""
import numpy as np
import pytest

from socg.geometry import angle_deg, dihedral_deg, kabsch_rmsd


@pytest.mark.parametrize("a", [60.0, -60.0, 179.0])
def test_dihedral_convention(a):
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 1.0])
    p0 = np.array([1.0, 0.0, 0.0])
    p3 = p2 + np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), 0.0])
    assert dihedral_deg(p0, p1, p2, p3) == pytest.approx(a, abs=1e-6)


def test_angle_deg():
    b = np.zeros(3)
    a = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert angle_deg(a, b, c) == pytest.approx(90.0, abs=1e-9)
    assert angle_deg(a, b, -a) == pytest.approx(180.0, abs=1e-9)


def _random_rotation(rng):
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_kabsch_rmsd_rigid_motion():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(8, 3))
    T = 5
    X = np.zeros((T, 8, 3))
    for t in range(T):
        R = _random_rotation(rng)
        shift = rng.normal(size=3) * 10
        X[t] = ref @ R.T + shift
    rmsd = kabsch_rmsd(X, ref)
    assert (np.abs(rmsd) < 1e-6).all()
