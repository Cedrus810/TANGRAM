#!/usr/bin/env python3
"""CG 模拟器性能基准：scripts/bench_cg.py --n-res 10 --replicas 1024 --steps 2000

报告 steps/s 与 replica·ns/day。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-res", type=int, default=10)
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--flip-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    from socg.dynamics import CGSimulator, LangevinConfig
    from socg.model.priors import Priors
    from socg.model.socg import ModelConfig, SOCGModel
    from socg.topology import CGTopology
    sys.path.insert(0, str(ROOT / "tests"))
    from conftest import ideal_helix, randomize  # noqa: E402

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    topo = CGTopology.from_sequence("A" * args.n_res, capped=True)
    coords = ideal_helix(args.n_res).astype(np.float64)
    rng = np.random.default_rng(args.seed)
    batch = np.tile(coords[None], (args.replicas, 1, 1))
    batch = batch + rng.normal(0.0, 0.02, size=batch.shape)
    priors = Priors.fit(topo, batch[:64], 300.0)
    model = SOCGModel(topo, priors, ModelConfig()).to(device=device)
    randomize(model, 0.3, seed=args.seed)

    cfg = LangevinConfig(temperature=300.0, dt_ps=args.dt, friction_per_ps=1.0,
                         flip_interval=args.flip_interval, record_interval=max(args.steps, 1),
                         seed=args.seed)
    sim = CGSimulator(model, cfg,
                      torch.as_tensor(batch, dtype=torch.float32, device=device),
                      torch.zeros((args.replicas, args.n_res), dtype=torch.long),
                      torch.as_tensor(topo.aa_index, dtype=torch.long, device=device))
    # 预热
    sim.run(min(50, args.steps))
    t0 = time.time()
    sim.run(args.steps)
    wall = time.time() - t0
    steps_per_s = args.steps / wall
    rep_ns_day = steps_per_s * args.dt * args.replicas / 1000.0 * 86400.0
    print(f"n_res={args.n_res} replicas={args.replicas} steps={args.steps} "
          f"device={device} model_dtype={model.w_bond.dtype}")
    print(f"steps/s        = {steps_per_s:,.0f}")
    print(f"replica·ns/day = {rep_ns_day:,.1f}")
    print(f"flip acceptance = {sim.acceptance_rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
