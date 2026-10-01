"""T6: 动力学统计库。"""
from __future__ import annotations

import numpy as np
import pytest

from socg.analysis.kinetics import (integrated_acf_time, lagged_pairs,
                                    lifetimes, msm_its, tica, vamp2_score)


def _two_state_chain(n: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        s[t] = s[t - 1]
        if rng.random() < p:
            s[t] = 1 - s[t - 1]
    return s


def test_lagged_pairs_no_cross_trajectory():
    # Review Focus 1：10 帧两条、lag=3 -> 14 对，不跨轨迹
    t1 = np.arange(10, dtype=float).reshape(-1, 1)
    t2 = (np.arange(10, dtype=float) + 100).reshape(-1, 1)
    X0, X1 = lagged_pairs([t1, t2], 3)
    assert X0.shape[0] == 14
    for a, b in zip(X0[:, 0], X1[:, 0]):
        assert b - a == 3
    # 每对同属一条轨迹（轨迹 1 的值 < 100，轨迹 2 的值 >= 100）
    assert ((X0[:, 0] < 100) == (X1[:, 0] < 100)).all()


def test_lagged_pairs_skips_short():
    X0, X1 = lagged_pairs([np.zeros((3, 2)), np.ones((5, 2))], 3)
    assert X0.shape[0] == 2                 # 只有 5 帧那条给出 2 对


def test_msm_its_two_state():
    # p=0.01、2×1e5 步：lag=1 和 5 的 ITS 都 ≈ -1/ln(1-2p) ≈ 49.5（10% 以内）
    target = -1.0 / np.log(1 - 0.02)
    dtraj = _two_state_chain(200_000, 0.01, seed=1)
    its1 = msm_its([dtraj], 1, 1, 1.0)[0]
    its5 = msm_its([dtraj], 5, 1, 1.0)[0]
    assert its1 == pytest.approx(target, rel=0.10)
    assert its5 == pytest.approx(target, rel=0.10)


def test_tica_finds_slow_signal():
    rng = np.random.default_rng(2)
    T = 40_000
    slow = _two_state_chain(T, 0.05, seed=3) * 2.0 - 1.0
    x = np.stack([rng.normal(0, 1, T), slow + 0.3 * rng.normal(0, 1, T)], axis=1)
    res = tica([x], 5, 2)
    tic1 = res.transform(x)[:, 0]
    corr = np.corrcoef(tic1, slow)[0, 1]
    assert abs(corr) > 0.9


def test_vamp2_noise_vs_signal():
    rng = np.random.default_rng(4)
    T = 40_000
    slow = _two_state_chain(T, 0.02, seed=5) * 2.0 - 1.0
    sig = (slow + 0.3 * rng.normal(0, 1, T)).reshape(-1, 1)
    noise = rng.normal(0, 1, (T, 1))
    train = [sig[: T // 2], noise[: T // 2]]
    test = [sig[T // 2:], noise[T // 2:]]
    score_sig = vamp2_score(train[:1], test[:1], 5, 5)
    score_noise = vamp2_score(train[1:], test[1:], 5, 5)
    assert score_noise == pytest.approx(1.0, abs=0.05)
    assert score_sig > score_noise + 0.5


def test_lifetimes():
    states = np.asarray([[0, 0, 1, 1, 1, 0, 0, 0, 1]], dtype=np.int64).T  # (9,1)
    lt = lifetimes(states, 1.0)
    assert np.array_equal(lt[(0, 1)], [3.0])
    assert np.array_equal(lt[(0, 0)], [3.0])


def test_lifetimes_drop_truncated_ends():
    # runs: 1×1, 0×2, 1×3, 0×3, 1×1 —— 首尾两段被轨迹边界截断，剔除
    states = np.asarray([[1, 0, 0, 1, 1, 1, 0, 0, 0, 1]], dtype=np.int64).T
    lt = lifetimes(states, 2.0)
    assert np.array_equal(lt[(0, 0)], [4.0, 6.0])   # 内部 0×2、0×3，dt=2
    assert np.array_equal(lt[(0, 1)], [6.0])        # 内部 1×3，dt=2


def test_integrated_acf_time_white_noise():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 100_000)
    tau = integrated_acf_time([x], 1.0, 50)
    assert -1.5 < tau < 1.5


def test_integrated_acf_time_known_ar1():
    # AR(1)：x_t = φ x_{t-1} + ε，τ_int ≈ dt·(0.5 + φ/(1-φ))
    phi = 0.9
    rng = np.random.default_rng(7)
    n = 400_000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))
    tau = integrated_acf_time([x], 1.0, 200)
    expected = 0.5 + phi / (1 - phi)
    assert tau == pytest.approx(expected, rel=0.2)


def test_subsample_to_dt():
    from socg.analysis.kinetics import subsample_to_dt

    x = np.arange(100)
    sub, dt = subsample_to_dt(x, 0.1, 1.0)
    assert dt == pytest.approx(1.0)
    assert list(sub[:3]) == [0, 10, 20]
    sub, dt = subsample_to_dt(x, 2.0, 1.0)          # 已比目标粗：不再下采样
    assert dt == 2.0 and sub.shape[0] == 100


def test_pooled_mean_lifetime_resolution_invariant():
    # 同一条两态链，以 0.1 与 1.0 的帧间隔记录，下采样到同一 Δt 后平均驻留一致
    s_fine = np.repeat(_two_state_chain(20_000, 0.02, seed=8), 10)[:, None]   # dt=0.1
    s_coarse = s_fine[::10]                                                  # dt=1.0
    from socg.analysis.kinetics import pooled_mean_lifetime

    m_f, n_f, dt_f = pooled_mean_lifetime([s_fine], [0], 0.1, 1.0)
    m_c, n_c, dt_c = pooled_mean_lifetime([s_coarse], [0], 1.0, 1.0)
    assert dt_f == pytest.approx(dt_c)
    assert m_f == pytest.approx(m_c, rel=1e-9) and n_f == n_c


def test_vamp2_no_dimension_bias():
    # 审阅 P1-3：高维纯噪声特征的 CV 得分不应随维度上升（仍约为 1）
    rng = np.random.default_rng(10)
    T = 4000
    tr = [rng.normal(0, 1, (T, 30))]
    te = [rng.normal(0, 1, (T, 30))]
    assert vamp2_score(tr, te, 5, 5) < 1.03


def test_vamp2_collinear_onehot_ok():
    # one-hot 全列（与常数列共线）不应数值爆炸
    s = _two_state_chain(20_000, 0.02, seed=12)
    oh = np.eye(2)[s]
    score = vamp2_score([oh[:10_000]], [oh[10_000:]], 5, 5)
    assert 1.0 < score < 2.0 + 1e-9


def test_subsample_to_dt_rejects_nan():
    from socg.analysis.kinetics import subsample_to_dt

    with pytest.raises(ValueError):
        subsample_to_dt(np.arange(10), float("nan"), 1.0)
