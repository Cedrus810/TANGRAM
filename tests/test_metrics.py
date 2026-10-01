"""T6: 分布度量。"""
from __future__ import annotations

import numpy as np
import pytest

from socg.analysis.metrics import (LN2, bimodal_threshold, cg_local_features,
                                   hist_js, js_divergence, state_populations)


def test_js_identical_zero():
    p = np.asarray([0.2, 0.3, 0.5])
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_disjoint_ln2():
    p = np.asarray([1.0, 0.0, 0.0])
    q = np.asarray([0.0, 0.0, 1.0])
    assert js_divergence(p, q) == pytest.approx(LN2, rel=1e-9)


def test_hist_js_1d():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20_000)
    assert hist_js(a, a, bins=40) == pytest.approx(0.0, abs=1e-12)
    b = rng.normal(50, 1, 20_000)                   # 支撑集不相交 -> ln2
    assert hist_js(a, b, bins=40) == pytest.approx(LN2, rel=1e-6)
    c = rng.normal(0.5, 1, 20_000)                  # 共同分箱：部分重叠 -> 介于 0 与 ln2
    assert 0.0 < hist_js(a, c, bins=40) < 0.2


def test_hist_js_2d():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (5000, 2))
    assert hist_js(a, a, bins=30) == pytest.approx(0.0, abs=1e-12)
    b = rng.normal(3, 1, (5000, 2))
    assert hist_js(a, b, bins=30) > 0.5


def test_state_populations():
    states = np.asarray([[0, 1, 2], [0, 1, 1], [2, 2, 2]])
    pops = state_populations(states, 3)
    assert pops.shape == (3, 3)
    assert pops[0, 0] == pytest.approx(2 / 3)
    # 行 = 时间、列 = 残基：残基 1 的序列为 [1,1,2]，残基 2 为 [2,1,2]
    assert pops[1, 1] == pytest.approx(2 / 3)
    assert pops[2, 0] == pytest.approx(0.0)
    assert np.allclose(pops.sum(axis=1), 1.0)


def test_cg_local_features_straight_chain():
    # 直线链：theta = 180°，tau 无定义但应为常数
    coords = np.linspace(0, 1, 6).reshape(1, 6, 1) * np.asarray([1.0, 0, 0])
    theta, tau = cg_local_features(coords)
    assert theta.shape == (1, 4)
    assert (np.abs(theta - 180.0) < 1e-6).all()
    assert tau.shape == (1, 3)                      # (T, N-3)


def test_cg_local_features_shapes():
    rng = np.random.default_rng(2)
    coords = rng.normal(0, 0.3, size=(7, 8, 3))
    theta, tau = cg_local_features(coords)
    assert theta.shape == (7, 6)
    assert tau.shape == (7, 5)


def test_bimodal_threshold():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0.1, 0.02, 5000), rng.normal(0.4, 0.05, 5000)])
    thr = bimodal_threshold(x)
    assert 0.15 < thr < 0.3


def test_bimodal_threshold_unimodal_fallback():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 5000)
    thr = bimodal_threshold(x)
    assert abs(thr) < 0.5
