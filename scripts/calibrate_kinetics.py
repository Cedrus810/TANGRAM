#!/usr/bin/env python3
"""动力学标定（T11）：只用两个全局自由参数 α_R 与 flip_interval，只用局部观测量。

- α_R（时间缩放）：让 CG 与 AA 的 d(i,i+2) 积分 ACF 时间（非末端残基平均）相等；
- flip_interval ∈ {1,3,10,30,100}：缩放后中心残基平均驻留时间最接近 AA 者。
  基线（K=1）只标定 α_R。

两个参数联合确定：每个 flip 候选各自重新定 α_R（翻转频率会改变局部 ACF），
最终写出的 α_R 与选中的 flip 对应。所有 replica 都作为独立轨迹参与统计。
驻留时间在同一物理时间分辨率（AA 帧间隔）下比较，该分辨率写入报告，
evaluate.py 的 K2 使用同一分辨率。

结果写入 reports/calib_<system>.json；此后 evaluate.py 的所有动力学判据
都是预测，不允许再调参。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLIP_GRID = [1, 3, 10, 30, 100]
AA_MAX_LAG_PS = 200.0
CG_DT_PS = 0.002
ACCEPT_RANGE = (0.01, 0.90)


def dij2_series(coords: np.ndarray) -> list[np.ndarray]:
    """一条轨迹的 d(i,i+2) 序列列表（每个非末端 i 一条），各 (T,)。"""
    T, N, _ = coords.shape
    return [np.linalg.norm(coords[:, i] - coords[:, i + 2], axis=-1)
            for i in range(1, N - 2)]


def mean_acf_time(series_by_res: list[list[np.ndarray]], dt: float, max_lag: int) -> float:
    """每个残基在所有轨迹上求积分 ACF 时间，再对残基取平均。"""
    from socg.analysis.kinetics import integrated_acf_time

    taus = []
    for trajs in series_by_res:
        usable = [x for x in trajs if x.shape[0] > max_lag + 2]
        if usable:
            taus.append(integrated_acf_time(usable, dt, max_lag))
    return float(np.mean(taus)) if taus else float("nan")


def series_by_residue(coords_list: list[np.ndarray]) -> list[list[np.ndarray]]:
    per_traj = [dij2_series(c) for c in coords_list]
    return [list(x) for x in zip(*per_traj)]


def run_cg(model, pool_coords, pool_states, temperature, batch, steps, record_every,
           flip_interval, seed, device):
    """跑一批 CG（freeze 模式），返回 (d(i,i+2) 按残基的序列, 各 replica 态轨迹, 接受率, 爆炸数)。"""
    import torch

    from socg.dynamics import CGSimulator, LangevinConfig

    rng = np.random.default_rng(seed)
    pick = rng.choice(pool_coords.shape[0], size=batch, replace=True)
    cfg = LangevinConfig(temperature=temperature, dt_ps=CG_DT_PS, friction_per_ps=1.0,
                         flip_interval=flip_interval, record_interval=record_every, seed=seed)
    sim = CGSimulator(model, cfg,
                      torch.as_tensor(pool_coords[pick].astype(np.float32), device=device),
                      torch.as_tensor(pool_states[pick].astype(np.int64), device=device),
                      torch.as_tensor(model.topo.aa_index, dtype=torch.long,
                                      device=device),
                      on_nonfinite="freeze")
    coords, states = sim.run(steps)
    ok = ~sim.exploded_mask
    coords_list = [coords[b].astype(np.float64) for b in np.flatnonzero(ok)]
    states_list = [states[b].astype(np.int64) for b in np.flatnonzero(ok)]
    return (series_by_residue(coords_list), states_list, sim.acceptance_rate,
            int((~ok).sum()))


def fit_alpha(aa_tau: float, cg_series, cg_frame_dt: float, n_frames: int,
              n_iter: int = 4) -> tuple[float, float, int]:
    """α_R = τ_AA / τ_CG(raw)。CG 的 ACF 截断 lag 对应同一物理时长 AA_MAX_LAG_PS，
    因为它依赖 α，迭代求自洽解。返回 (alpha, tau_cg_raw, max_lag_frames)。"""
    alpha, tau_raw, max_lag = 1.0, float("nan"), 1
    for _ in range(n_iter):
        max_lag = int(AA_MAX_LAG_PS / alpha / cg_frame_dt)
        max_lag = max(1, min(max_lag, n_frames // 2))
        tau_raw = mean_acf_time(cg_series, cg_frame_dt, max_lag)
        if not (np.isfinite(tau_raw) and tau_raw > 0):
            return float("nan"), tau_raw, max_lag
        alpha = aa_tau / tau_raw
    return float(alpha), float(tau_raw), max_lag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--model-k3", required=True)
    parser.add_argument("--model-k1", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cg-steps", type=int, default=200_000)
    parser.add_argument("--cg-batch", type=int, default=256)
    parser.add_argument("--cg-record-every", type=int, default=20)
    args = parser.parse_args()

    import torch

    from socg.analysis.kinetics import pooled_mean_lifetime
    from socg.data import CGDataset
    from socg.model.socg import load_model

    system = args.system
    dataset = CGDataset.load(ROOT / "data" / "cg" / f"{system}.npz")
    topo = dataset.topology
    aa_dt = dataset.trajectories[0].dt_ps
    life_dt = aa_dt                                   # 驻留时间比较的物理分辨率

    movable = np.where(~topo.frozen_mask)[0]
    central = movable[(movable >= topo.n // 3) & (movable < 2 * topo.n // 3)]

    # ---- AA 参考量 ----
    aa_series = series_by_residue([t.coords.astype(np.float64) for t in dataset.trajectories])
    aa_tau = mean_acf_time(aa_series, aa_dt, int(AA_MAX_LAG_PS / aa_dt))
    aa_life, aa_nseg, _ = pooled_mean_lifetime([t.states for t in dataset.trajectories],
                                               central, aa_dt, life_dt)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    pool_coords, _, pool_states = dataset.frames(stride=1)
    cg_frame_dt = CG_DT_PS * args.cg_record_every
    n_frames = args.cg_steps // args.cg_record_every + 1

    report: dict = {
        "system": system, "aa_tau_acf_ps": aa_tau,
        "aa_central_lifetime_ps": aa_life, "aa_central_lifetime_segments": aa_nseg,
        "central_residues": central.tolist(), "lifetime_dt_ps": life_dt,
        "cg": {"steps": args.cg_steps, "batch": args.cg_batch, "dt_ps": CG_DT_PS,
               "record_every": args.cg_record_every},
        "models": {},
    }

    for tag, model_path in (("k3", args.model_k3), ("k1", args.model_k1)):
        model = load_model(model_path, map_location=device)
        flip_grid = FLIP_GRID if model.config.n_states > 1 else [0]
        scan = []
        for flip in flip_grid:
            series, states_list, acc, n_bad = run_cg(
                model, pool_coords, pool_states, dataset.temperature, args.cg_batch,
                args.cg_steps, args.cg_record_every, flip, args.seed + flip, device)
            alpha, tau_raw, max_lag = fit_alpha(aa_tau, series, cg_frame_dt, n_frames)
            entry = {"flip_interval": flip, "alpha_R": alpha, "cg_tau_acf_raw_ps": tau_raw,
                     "acf_max_lag_frames": max_lag, "acceptance_rate": float(acc),
                     "exploded_replicas": n_bad}
            if flip > 0 and not (np.isfinite(alpha) and alpha > 0):
                # ACF 积分非正（轨迹太短/噪声）→ α_R 无定义：记为失败候选，不参与选择
                entry.update({"scaled_lifetime_ps": float("nan"), "lifetime_segments": 0,
                              "lifetime_actual_dt_ps": float("nan"),
                              "log_ratio_error": float("inf"),
                              "note": "alpha_R undefined (non-positive integrated ACF)"})
            elif flip > 0:
                life, nseg, actual_dt = pooled_mean_lifetime(
                    states_list, central, cg_frame_dt * alpha, life_dt)
                err = (abs(np.log(life / aa_life))
                       if np.isfinite(life) and life > 0 and aa_life > 0 else float("inf"))
                entry.update({"scaled_lifetime_ps": life, "lifetime_segments": nseg,
                              "lifetime_actual_dt_ps": actual_dt, "log_ratio_error": err})
            scan.append(entry)
            print(f"[calibrate] {system} {tag} flip={flip}: alpha_R={alpha:.4g} "
                  f"acc={acc:.3f} exploded={n_bad}"
                  + (f" life={entry['scaled_lifetime_ps']:.4g}ps (AA {aa_life:.4g})"
                     if flip > 0 else ""))

        if flip_grid == [0]:
            best = scan[0]
        else:
            best = min(scan, key=lambda e: e["log_ratio_error"])
        acc = best["acceptance_rate"]
        report["models"][tag] = {
            "alpha_R": best["alpha_R"],
            "flip_interval": best["flip_interval"] if flip_grid != [0] else None,
            "scan": scan,
            "checks": {
                "acceptance_rate": ({"value": acc, "threshold": list(ACCEPT_RANGE),
                                     "pass": bool(ACCEPT_RANGE[0] <= acc <= ACCEPT_RANGE[1])}
                                    if flip_grid != [0] else None),
                "no_explosion": {"value": best["exploded_replicas"], "threshold": 0,
                                 "pass": best["exploded_replicas"] == 0},
            },
        }

    out = ROOT / "reports" / f"calib_{system}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[calibrate] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
