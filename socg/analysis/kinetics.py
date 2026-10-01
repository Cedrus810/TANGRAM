"""动力学与统计分析库（D6：纯 numpy/scipy 实现，不依赖 deeptime）。

关键约定：所有时间相关计算都在每条轨迹内部进行，绝不跨轨迹拼接。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2, vq


# --------------------------------------------------------------------- #
# lag 配对与 TICA / VAMP
# --------------------------------------------------------------------- #
def lagged_pairs(trajs, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """在每条轨迹内部做 lag 配对，返回 (X0, X1)。

    长度 <= lag 的轨迹跳过；任何配对都不会跨越轨迹边界。
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    X0, X1 = [], []
    for x in trajs:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] <= lag:
            continue
        X0.append(x[:-lag])
        X1.append(x[lag:])
    if not X0:
        a = np.asarray(trajs[0]) if len(trajs) else np.zeros((0, 0))
        return np.zeros((0, a.shape[-1])), np.zeros((0, a.shape[-1]))
    return np.concatenate(X0, axis=0), np.concatenate(X1, axis=0)


@dataclass
class TICAResult:
    mean: np.ndarray          # (d,)
    proj: np.ndarray          # (d, dim)
    eigenvalues: np.ndarray   # (dim,)

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) @ self.proj


def _inv_sqrt(C: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """对称半正定矩阵的伪逆平方根：丢弃 w <= rtol·max(w) 的零空间方向。

    不做截断放大——例如 one-hot 特征与常数列共线时，零空间方向直接置零。
    """
    w, V = np.linalg.eigh(C)
    wmax = float(w.max()) if w.size else 0.0
    keep = w > max(wmax * rtol, 1e-300)
    Vk = V[:, keep]
    return (Vk * (1.0 / np.sqrt(w[keep]))) @ Vk.T


def tica(trajs, lag: int, dim: int) -> TICAResult:
    """时序独立成分分析（协方差白化 + 对称 Koopman 特征分解）。

    trajs: list[(T,d)]；返回前 dim 个慢方向（按特征值降序）。
    """
    X0, X1 = lagged_pairs(trajs, lag)
    if X0.shape[0] < 2:
        raise ValueError("not enough frames for tica")
    mean = 0.5 * (X0.mean(axis=0) + X1.mean(axis=0))   # 与对称化的 C0 一致
    X0c = X0 - mean
    X1c = X1 - mean
    n = X0.shape[0]
    C0 = (X0c.T @ X0c + X1c.T @ X1c) / (2.0 * n)
    Ct = (X0c.T @ X1c + X1c.T @ X0c) / (2.0 * n)
    S = _inv_sqrt(C0)
    M = S @ Ct @ S
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1][:dim]
    proj = S @ V[:, order]
    return TICAResult(mean=mean, proj=proj, eigenvalues=w[order])


def vamp2_score(train_trajs, test_trajs, lag: int, k: int) -> float:
    """交叉验证 VAMP-2 分数（包含常数奇异值 1；纯噪声约为 1）。

    特征内部增广一列常数 1（不扣均值），因此常数方向的奇异值 1 总被计入。
    标准 CV 做法（Wu & Noé 2020）：在训练集上求 Koopman 矩阵的前 k 个奇异
    向量 U、V；在测试集上用测试协方差计算
        ‖(UᵀC00ᵗᵉ U)^{-1/2} UᵀC01ᵗᵉ V (VᵀC11ᵗᵉ V)^{-1/2}‖_F²。
    子空间只由训练集决定，因此不会因特征维度增加而产生乐观偏差。
    """
    X0tr, X1tr = lagged_pairs(train_trajs, lag)
    X0te, X1te = lagged_pairs(test_trajs, lag)
    if X0tr.shape[0] < 2 or X0te.shape[0] < 2:
        raise ValueError("not enough frames for vamp2_score")

    def augment(X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    X0tr, X1tr, X0te, X1te = map(augment, (X0tr, X1tr, X0te, X1te))

    def covs(X0, X1):
        n = X0.shape[0]
        return X0.T @ X0 / n, X1.T @ X1 / n, X0.T @ X1 / n

    C00, C11, C01 = covs(X0tr, X1tr)
    A = _inv_sqrt(C00)
    B = _inv_sqrt(C11)
    Uk, _s, Vkt = np.linalg.svd(A @ C01 @ B)
    U = A @ Uk[:, :k]
    V = B @ Vkt[:k].T

    T00, T11, T01 = covs(X0te, X1te)
    M = _inv_sqrt(U.T @ T00 @ U) @ (U.T @ T01 @ V) @ _inv_sqrt(V.T @ T11 @ V)
    return float(np.sum(M ** 2))


# --------------------------------------------------------------------- #
# 离散化与 MSM / ITS
# --------------------------------------------------------------------- #
def discretize(trajs, n_clusters: int, seed: int, max_fit_frames: int = 200_000):
    """k-means 离散化（拟合帧数上限 max_fit_frames），返回每条轨迹的 dtraj。"""
    flat = np.concatenate([np.asarray(x, dtype=np.float64) for x in trajs], axis=0)
    if flat.shape[0] > max_fit_frames:
        rng = np.random.default_rng(seed)
        idx = rng.choice(flat.shape[0], size=max_fit_frames, replace=False)
        flat_fit = flat[idx]
    else:
        flat_fit = flat
    centroids, _ = kmeans2(flat_fit, n_clusters, iter=50, minit="++",
                           missing="warn", check_finite=False, seed=seed)
    return [vq(np.asarray(x, dtype=np.float64), centroids)[0].astype(np.int64)
            for x in trajs]


def _largest_connected(C_sym: np.ndarray) -> np.ndarray:
    n = C_sym.shape[0]
    visited = np.zeros(n, dtype=bool)
    best: list[int] = []
    for start in range(n):
        if visited[start] or C_sym[start].sum() == 0:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(C_sym[u] > 0)[0]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(int(v))
        if len(comp) > len(best):
            best = comp
    return np.asarray(sorted(best), dtype=np.int64)


def msm_its(dtrajs, lag: int, n_its: int, dt: float) -> np.ndarray:
    """对称化计数的 MSM 的前 n_its 个非平凡隐含时间尺度（跳过 λ=1）。

    只保留最大连通集；返回长度恰为 n_its 的数组，不足处用 inf 填充。
    """
    n_states = max(int(d.max()) for d in dtrajs) + 1
    C = np.zeros((n_states, n_states), dtype=np.float64)
    for d in dtrajs:
        d = np.asarray(d)
        if d.shape[0] <= lag:
            continue
        np.add.at(C, (d[:-lag], d[lag:]), 1.0)
    C_sym = C + C.T
    ix = _largest_connected(C_sym)
    m = len(ix)
    its = np.full(n_its, np.inf)
    if m < 2:
        return its
    sub = C_sym[np.ix_(ix, ix)]
    rowsum = sub.sum(axis=1, keepdims=True)
    # T = sub / rowsum 与对称矩阵 D^{-1/2} sub D^{-1/2} 相似 -> 特征值为实数
    d_sqrt = np.sqrt(rowsum.ravel())
    S = sub / np.outer(d_sqrt, d_sqrt)
    S = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(S)
    w = np.sort(w)[::-1]
    w = np.clip(w[1:1 + n_its], 0.0, 1.0)
    for i, lam in enumerate(w):
        if lam <= 0.0:
            its[i] = 0.0
        elif lam >= 1.0:
            its[i] = np.inf
        else:
            its[i] = -lag * dt / np.log(lam)
    return its


# --------------------------------------------------------------------- #
# 驻留时间与 ACF
# --------------------------------------------------------------------- #
def lifetimes(states, dt: float) -> dict:
    """逐残基态驻留时间（剔除首尾两段被轨迹边界截断的驻留）。

    states: (T,N) int。返回 {(res, state): 1D array}，时间为 run 长度 × dt。
    """
    states = np.asarray(states, dtype=np.int64)
    if states.ndim != 2:
        raise ValueError("states must be (T, N)")
    T, N = states.shape
    out: dict[tuple[int, int], list[float]] = {}
    for j in range(N):
        col = states[:, j]
        change = np.flatnonzero(col[1:] != col[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [T]])
        runs = ends - starts
        vals = col[starts]
        # 首尾两段被截断，剔除
        interior = runs[1:-1] if len(runs) >= 3 else np.zeros(0, dtype=np.int64)
        interior_vals = vals[1:-1] if len(runs) >= 3 else vals[:0]
        for v, r in zip(interior_vals, interior):
            out.setdefault((j, int(v)), []).append(float(r) * dt)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _normalized_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.ones(max_lag + 1)
    x = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    acf = acf / np.arange(n, n - max_lag - 1, -1)
    if acf[0] <= 0:
        return np.zeros(max_lag + 1)
    return acf / acf[0]


def integrated_acf_time(x_trajs, dt: float, max_lag: int) -> float:
    """积分自相关时间 τ = dt·(0.5 + Σ_{l=1..max_lag} C(l))，多条轨迹平均 C。"""
    acfs = [_normalized_acf(np.asarray(x), max_lag) for x in x_trajs
            if np.asarray(x).shape[0] > max_lag + 2]
    if not acfs:
        return 0.0
    C = np.mean(acfs, axis=0)
    return float(dt * (0.5 + C[1:].sum()))


def subsample_to_dt(x: np.ndarray, frame_dt: float, target_dt: float):
    """沿第 0 轴下采样，使帧间隔尽量接近 target_dt（不会比 frame_dt 更细）。

    返回 (x_sub, actual_dt)，actual_dt = stride·frame_dt，stride = max(1, round(target/frame))。
    用于在同一物理时间分辨率下比较 AA 与 CG 的驻留时间（驻留对采样间隔敏感）。
    """
    if not (np.isfinite(frame_dt) and frame_dt > 0 and np.isfinite(target_dt) and target_dt > 0):
        raise ValueError(f"frame_dt/target_dt must be finite and positive, got {frame_dt}, {target_dt}")
    stride = max(1, int(round(float(target_dt) / float(frame_dt))))
    return np.asarray(x)[::stride], stride * float(frame_dt)


def pooled_mean_lifetime(states_list, residues, frame_dt: float, target_dt: float):
    """多条轨迹、指定残基的平均驻留时间（先下采样到 target_dt，再合并所有驻留段取均值）。

    返回 (mean_lifetime, n_segments, actual_dt)；没有完整驻留段时 mean 为 nan。
    """
    vals = []
    actual = float(frame_dt)
    residues = np.asarray(residues, dtype=np.int64)
    for s in states_list:
        s_sub, actual = subsample_to_dt(np.asarray(s)[:, residues], frame_dt, target_dt)
        for v in lifetimes(s_sub, actual).values():
            vals.extend(v.tolist())
    mean = float(np.mean(vals)) if vals else float("nan")
    return mean, len(vals), actual
