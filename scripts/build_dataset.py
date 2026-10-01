#!/usr/bin/env python3
"""AA 数据 -> CG 数据集：scripts/build_dataset.py --system ala10

写出 data/cg/<system>.npz，并打印每条轨迹的帧数与每个残基的态布居。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--radius", type=float, default=40.0)
    args = parser.parse_args()

    from socg.aa.systems import SYSTEMS
    from socg.mapping import build_dataset
    from socg.states import STATE_NAMES
    from socg.topology import CGTopology

    spec = SYSTEMS[args.system]
    topo = CGTopology.from_sequence(spec.sequence)
    aa_dir = ROOT / "data" / "aa" / spec.name
    dataset = build_dataset(aa_dir, topo, spec.temperature, radius=args.radius)

    out_path = ROOT / "data" / "cg" / f"{spec.name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(out_path)

    print(f"[build_dataset] {spec.name}: {len(dataset.trajectories)} trajectories "
          f"-> {out_path}")
    for i, trj in enumerate(dataset.trajectories):
        pops = np.stack([(trj.states == k).mean(axis=0) for k in range(3)], axis=1)
        line = " ".join(f"{p:.3f}" for p in pops.mean(axis=0))
        print(f"  traj {i}: frames={trj.n_frames} ({trj.duration_ps:.0f} ps)  "
              f"mean state pop A/B/L = {line}")
        print(f"            per-residue pops ({', '.join(STATE_NAMES)}):")
        for j in range(topo.n):
            frozen = " (frozen)" if topo.frozen[j] else ""
            print(f"              res {j} {spec.sequence[j]}: "
                  + " ".join(f"{p:.3f}" for p in pops[j]) + frozen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
