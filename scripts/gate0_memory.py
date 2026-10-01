#!/usr/bin/env python3
"""Gate 0 — H11 记忆检验（只用 AA 数据，不阻断施工）。

特征：X_R  = 所有 |i-j| >= 2 的 Cα 间距
      X_Rs = X_R 后拼接非冻结残基的态 one-hot

判据（每个都写 value/threshold/pass）：
  G0.1: lag=100 时 ΔVAMP2 = VAMP2(X_Rs) - VAMP2(X_R) > 2·sqrt(σ_R² + σ_Rs²)
  G0.2: X_Rs 的收敛 lag <= X_R 的收敛 lag 的一半
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAMP_LAGS = [10, 50, 100, 200, 500]
ITS_LAGS = [1, 2, 5, 10, 20, 50, 100, 200]


def ca_distance_features(coords: np.ndarray) -> np.ndarray:
    """所有 |i-j| >= 2 的 Cα 间距，(T, n_pairs)。"""
    T, N, _ = coords.shape
    pairs = [(i, j) for i in range(N) for j in range(i + 2, N)]
    d = np.stack([np.linalg.norm(coords[:, i] - coords[:, j], axis=-1)
                  for i, j in pairs], axis=1)
    return d


def features(dataset, with_states: bool):
    """每条轨迹的特征列表。"""
    topo = dataset.topology
    movable = np.where(~topo.frozen_mask)[0]
    feats = []
    for trj in dataset.trajectories:
        X_R = ca_distance_features(trj.coords.astype(np.float64))
        if not with_states:
            feats.append(X_R)
            continue
        # 每残基只取 K-1 列 one-hot：K 列之和恒为 1，会与 VAMP 的常数列共线
        onehot = np.eye(3)[trj.states[:, movable]][..., :-1]   # (T, nmov, 2)
        feats.append(np.hstack([X_R, onehot.reshape(onehot.shape[0], -1)]))
    return feats


def convergence_lag(its_values: dict[int, float], lags: list[int]) -> int | None:
    """收敛 lag：最小的 L，使得从 L 起相邻 lag 之间 t1 的相对变化都 < 10%。

    至少要有一对相邻 lag 满足条件；否则视为在所给 lag 网格内未收敛，返回 None。
    """
    known = [l for l in lags if l in its_values and np.isfinite(its_values[l])]

    def small(a, b):
        return abs(its_values[b] - its_values[a]) / max(abs(its_values[a]), 1e-12) < 0.10

    for i in range(len(known) - 1):
        if all(small(known[j], known[j + 1]) for j in range(i, len(known) - 1)):
            return known[i]
    return None


def its_pipeline(feats, n_clusters: int = 100, dim: int = 4, tica_lag: int = 100,
                 seed: int = 0):
    """TICA(dim) -> k-means -> ITS(lag 网格，前 3 个)。"""
    from socg.analysis.kinetics import discretize, msm_its, tica

    tic = tica(feats, tica_lag, dim)
    proj = [tic.transform(x) for x in feats]
    dtrajs = discretize(proj, n_clusters, seed)
    out = {}
    for lag in ITS_LAGS:
        its = msm_its(dtrajs, lag, 3, 1.0)
        out[lag] = [float(v) for v in its]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from socg.analysis.kinetics import vamp2_score
    from socg.data import CGDataset

    system = args.system
    dataset = CGDataset.load(ROOT / "data" / "cg" / f"{system}.npz")
    feats_R = features(dataset, with_states=False)
    feats_Rs = features(dataset, with_states=True)
    lags = [l for l in VAMP_LAGS
            if l < min(x.shape[0] for x in feats_R + feats_Rs)]

    vamp_R, vamp_Rs = [], []
    for lag in lags:
        scores_R, scores_Rs = [], []
        n_tr = len(feats_R)
        for hold in range(n_tr):
            train_R = [x for i, x in enumerate(feats_R) if i != hold]
            train_Rs = [x for i, x in enumerate(feats_Rs) if i != hold]
            scores_R.append(vamp2_score(train_R, [feats_R[hold]], lag, args.k))
            scores_Rs.append(vamp2_score(train_Rs, [feats_Rs[hold]], lag, args.k))
        vamp_R.append({"lag": lag, "mean": float(np.mean(scores_R)),
                       "std": float(np.std(scores_R)), "scores": scores_R})
        vamp_Rs.append({"lag": lag, "mean": float(np.mean(scores_Rs)),
                        "std": float(np.std(scores_Rs)), "scores": scores_Rs})

    def at(lag, arr):
        for e in arr:
            if e["lag"] == lag:
                return e
        raise KeyError(lag)

    eR, eRs = at(100, vamp_R), at(100, vamp_Rs)
    delta = eRs["mean"] - eR["mean"]
    err = 2.0 * np.sqrt(eR["std"] ** 2 + eRs["std"] ** 2)
    g01 = {"value": float(delta), "threshold": float(err), "pass": bool(delta > err)}

    its_R = its_pipeline(feats_R, seed=args.seed)
    its_Rs = its_pipeline(feats_Rs, seed=args.seed)
    conv_R = convergence_lag({k: v[0] for k, v in its_R.items() if v}, ITS_LAGS)
    conv_Rs = convergence_lag({k: v[0] for k, v in its_Rs.items() if v}, ITS_LAGS)
    # 未收敛记为 "not_converged"：X_R 未收敛而 X_Rs 收敛 -> PASS；X_Rs 未收敛 -> FAIL
    thr = (conv_R / 2.0) if conv_R is not None else "not_converged(X_R)"
    if conv_Rs is None:
        g02_pass = False
    elif conv_R is None:
        g02_pass = True
    else:
        g02_pass = conv_Rs <= thr
    g02 = {
        "value": conv_Rs if conv_Rs is not None else "not_converged",
        "threshold": thr,
        "pass": bool(g02_pass),
    }

    report = {
        "system": system,
        "vamp2_X_R": vamp_R,
        "vamp2_X_Rs": vamp_Rs,
        "its_X_R": its_R,
        "its_X_Rs": its_Rs,
        "convergence_lag_X_R": conv_R,        # null = 在 lag 网格内未收敛
        "convergence_lag_X_Rs": conv_Rs,
        "criteria": {"G0.1": g01, "G0.2": g02},
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # ITS-lag 图（PNG 不进 git）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, its, name in ((axes[0], its_R, "X_R"), (axes[1], its_Rs, "X_Rs")):
            for i in range(3):
                lags_known = sorted(int(l) for l in its if len(its[l]) > i
                                    and np.isfinite(its[l][i]))
                ax.plot(lags_known, [its[l][i] for l in lags_known],
                        marker="o", label=f"t{i+1}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("lag (frames)"); ax.set_title(name); ax.legend()
        axes[0].set_ylabel("ITS (frames)")
        fig.tight_layout()
        fig.savefig(ROOT / "reports" / f"gate0_its_{system}.png", dpi=150)
    except Exception as exc:  # noqa: BLE001 — 画图失败不阻断报告
        report["plot_error"] = str(exc)

    out = ROOT / "reports" / f"gate0_{system}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[gate0] {system}: G0.1 = {g01}")
    print(f"[gate0] {system}: G0.2 = {g02}")
    print(f"[gate0] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
