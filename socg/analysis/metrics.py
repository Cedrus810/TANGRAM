"""分布度量：JS 散度（自然对数，上界 ln2）、直方图 JS、态布居、CG 局部特征、双峰阈值。"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

LN2 = float(np.log(2.0))


def js_divergence(p, q) -> float:
    """Jensen–Shannon 散度（自然对数，上界 ln2）。输入直方图（自动归一化）。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"histogram shape mismatch: {p.shape} vs {q.shape}")
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / p.size)
    q = q / q.sum() if q.sum() > 0 else np.full_like(q, 1.0 / q.size)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def hist_js(a, b, bins: int = 50, range=None) -> float:
    """样本 -> 直方图（共同 bins/range）-> JS。支持一维 (n,) 与二维 (n,2) 样本。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1 and b.ndim == 1:
        if range is None:
            range = (float(min(a.min(), b.min())), float(max(a.max(), b.max())))
        ha, _ = np.histogram(a, bins=bins, range=range)
        hb, _ = np.histogram(b, bins=bins, range=range)
    elif a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1] == 2:
        if range is None:
            lo = np.minimum(a.min(axis=0), b.min(axis=0))
            hi = np.maximum(a.max(axis=0), b.max(axis=0))
            rng = [(float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))]
        else:
            rng = range
        ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=rng)
        hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=rng)
    else:
        raise ValueError(f"unsupported sample shapes {a.shape}, {b.shape}")
    return js_divergence(ha, hb)


def state_populations(states, n_states: int) -> np.ndarray:
    """(T,N) int -> (N, n_states) 每个残基的态布居。"""
    states = np.asarray(states, dtype=np.int64)
    if states.ndim != 2:
        raise ValueError("states must be (T, N)")
    return np.stack([(states == k).mean(axis=0) for k in range(n_states)], axis=1)


def cg_local_features(coords):
    """Cα 局部几何特征。

    coords: (T,N,3)。返回 (theta_deg (T,N-2), tau_deg (T,N-3))：
    theta_i 为 (i-1,i,i+1) 的键角；tau_i 为 (i..i+3) 的赝二面角。
    """
    from ..geometry import angle_deg, dihedral_deg

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3:
        raise ValueError("coords must be (T, N, 3)")
    theta = angle_deg(coords[:, :-2], coords[:, 1:-1], coords[:, 2:])
    tau = dihedral_deg(coords[:, :-3], coords[:, 1:-2], coords[:, 2:-1], coords[:, 3:])
    return theta, tau


def bimodal_threshold(x, bins: int = 100) -> float:
    """两个最高峰之间的最低点（用于 RMSD 折叠阈值）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 10:
        raise ValueError("not enough samples for bimodal_threshold")
    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(counts.astype(np.float64), sigma=max(1.0, bins / 50.0))
    peaks = [i for i in range(1, bins - 1)
             if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    if len(peaks) < 2:
        return float(np.median(x))
    peaks = sorted(sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2])
    lo, hi = peaks
    valley = lo + int(np.argmin(smooth[lo:hi + 1]))
    return float(centers[valley])
