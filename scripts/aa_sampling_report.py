#!/usr/bin/env python3
"""AA 采样验收报告：scripts/aa_sampling_report.py --system ala10|cln025

输出 reports/aa_sampling_<system>.json：
- 每条轨迹时长、每个残基的态布居
- ala10：平均螺旋度（非冻结残基中 A 态比例）与 replica 间标准差
- cln025：相对 data/ref/5AWL.pdb（model 1 的 Cα）的 RMSD 直方图、
  bimodal 折叠阈值、折叠分数、折叠/解折叠跃迁次数（双阈值 ±0.02 nm）
- 采样验收判据（T5 验收表）

依赖 T6 的 bimodal_threshold / lifetimes；需要先运行 build_dataset。
5AWL.pdb 下载：https://files.rcsb.org/download/5AWL.pdb -> data/ref/5AWL.pdb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF_PDB = ROOT / "data" / "ref" / "5AWL.pdb"


def folded_transitions(rmsd: np.ndarray, thr: float, margin: float = 0.02) -> int:
    """双阈值跃迁计数：进入 RMSD < thr-margin 或 > thr+margin 才算一次跃迁。"""
    last_side = 0  # 最近一次所在的一侧：+1 折叠侧、-1 解折叠侧、0 尚未离开中间带
    transitions = 0
    for r in rmsd:
        if r < thr - margin:
            side = 1
        elif r > thr + margin:
            side = -1
        else:
            continue      # 中间带：不改变所在侧
        if side != last_side and last_side != 0:
            transitions += 1
        last_side = side
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    args = parser.parse_args()

    from socg.aa.systems import SYSTEMS
    from socg.analysis.metrics import bimodal_threshold, state_populations

    spec = SYSTEMS[args.system]
    dataset_path = ROOT / "data" / "cg" / f"{spec.name}.npz"
    if not dataset_path.exists():
        raise SystemExit(f"missing {dataset_path}; run scripts/build_dataset.py first")

    from socg.data import CGDataset

    dataset = CGDataset.load(dataset_path)
    topo = dataset.topology
    report: dict = {
        "system": spec.name,
        "temperature_K": spec.temperature,
        "n_trajectories": len(dataset.trajectories),
        "trajectories": [],
    }

    durations = []
    pops_per_traj = []
    for trj in dataset.trajectories:
        durations.append(trj.duration_ps / 1000.0)
        pops = state_populations(trj.states, 3)
        pops_per_traj.append(pops)
        report["trajectories"].append({
            "frames": trj.n_frames,
            "duration_ns": trj.duration_ps / 1000.0,
            "dt_ps": trj.dt_ps,
        })

    pops_arr = np.stack(pops_per_traj)                       # (n_traj, N, 3)
    mean_pops = pops_arr.mean(axis=0)
    report["mean_state_populations"] = mean_pops.tolist()

    if spec.name == "ala10":
        movable = ~topo.frozen_mask
        helicity = float(mean_pops[movable, 0].mean())
        per_traj_helicity = np.array([p[~topo.frozen_mask, 0].mean() for p in pops_arr])
        report["helicity"] = {
            "mean": helicity,
            "std_across_replicas": float(per_traj_helicity.std(ddof=0)),
            "per_replica": per_traj_helicity.tolist(),
        }
        report["acceptance"] = {
            "min_duration_ns": {"value": float(min(durations)), "threshold": 450.0,
                                "pass": bool(min(durations) >= 450.0)},
            "helicity_std": {"value": report["helicity"]["std_across_replicas"],
                             "threshold": 0.1,
                             "pass": bool(report["helicity"]["std_across_replicas"] < 0.1)},
        }
    else:
        if not REF_PDB.exists():
            raise SystemExit(
                f"missing {REF_PDB}; download https://files.rcsb.org/download/5AWL.pdb "
                "(不要自己编造参考结构)")
        from socg.aa.systems import load_reference_ca

        ref_ca = load_reference_ca(REF_PDB, expected_n=topo.n)
        from socg.geometry import kabsch_rmsd

        all_rmsd = []
        for trj in dataset.trajectories:
            all_rmsd.append(kabsch_rmsd(trj.coords.astype(np.float64), ref_ca))
        rmsd_pool = np.concatenate(all_rmsd)
        thr = bimodal_threshold(rmsd_pool)
        folded_frac = float((rmsd_pool < thr).mean())
        trans = sum(folded_transitions(r, thr) for r in all_rmsd)
        report["rmsd"] = {
            "histogram": np.histogram(rmsd_pool, bins=50)[0].tolist(),
            "bin_edges_nm": np.histogram(rmsd_pool, bins=50)[1].tolist(),
            "folded_threshold_nm": thr,
            "folded_fraction": folded_frac,
            "fold_unfold_transitions": int(trans),
        }
        report["acceptance"] = {
            "transitions": {"value": int(trans), "threshold": 10,
                            "pass": bool(trans >= 10)},
            "folded_fraction": {"value": folded_frac, "threshold": [0.2, 0.9],
                                "pass": bool(0.2 <= folded_frac <= 0.9)},
        }

    out = ROOT / "reports" / f"aa_sampling_{spec.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[aa_sampling_report] wrote {out}")
    for key, val in report.get("acceptance", {}).items():
        print(f"  {key}: {val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
