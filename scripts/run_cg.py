#!/usr/bin/env python3
"""CG 生产模拟：scripts/run_cg.py --model models/ala10_k3.pt --replicas 1024 --steps 500000 \
       --friction 1.0 --flip-interval 10 --out data/cgsim/ala10_k3.npz

初始 (R,s) 从 AA 数据集中随机抽帧；输出 npz 包含 coords、states、
record_dt_ps、acceptance_rate 与完整 config JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--flip-interval", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--record-interval", type=int, default=500)
    parser.add_argument("--dataset", default=None,
                        help="AA CG 数据集（默认 data/cg/<system>.npz，system 取自模型元数据）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-chunk-steps", type=int, default=0,
                        help=">0 时每推进这么多步写一个 part 文件（须为 record-interval 的倍数）")
    parser.add_argument("--seq-interval", type=int, default=0,
                        help=">0 时每推进这么多步做一次均匀提议的单位点序列突变（Phase 2 基线）")
    parser.add_argument("--seq-update", default="full", choices=["full", "local"],
                        help="序列移动 ΔU 路径：full=两次全能量，local=局部重算（§29.2）")
    args = parser.parse_args()

    import torch

    from socg.data import CGDataset
    from socg.dynamics import CGSimulator, LangevinConfig
    from socg.model.socg import load_model

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = load_model(args.model, map_location=device)
    system = getattr(model, "_meta", {}).get("system", "system")
    dataset_path = Path(args.dataset) if args.dataset else ROOT / "data" / "cg" / f"{system}.npz"
    dataset = CGDataset.load(dataset_path)

    coords_pool, _, states_pool = dataset.frames(stride=1)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(coords_pool.shape[0], size=args.replicas, replace=True)
    R0 = coords_pool[pick].astype(np.float32)
    s0 = states_pool[pick].astype(np.int64)

    cfg = LangevinConfig(temperature=dataset.temperature, dt_ps=args.dt,
                         friction_per_ps=args.friction, flip_interval=args.flip_interval,
                         record_interval=args.record_interval, seed=args.seed,
                         seq_interval=args.seq_interval, seq_update=args.seq_update)
    proposal = None
    if args.seq_interval > 0:
        from socg.sequence import UniformSequenceProposal
        proposal = UniformSequenceProposal()
    sim = CGSimulator(model.to(device=device), cfg,
                      torch.as_tensor(R0, device=device),
                      torch.as_tensor(s0, device=device),
                      torch.as_tensor(model.topo.aa_index, dtype=torch.long,
                                      device=device),
                      on_nonfinite="freeze", sequence_proposal=proposal)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_chunk_steps > 0:
        if args.save_chunk_steps % args.record_interval:
            raise SystemExit("--save-chunk-steps 必须是 --record-interval 的倍数")
        done, part = 0, 0
        while done < args.steps:
            n = min(args.save_chunk_steps, args.steps - done)
            coords, states = sim.run(n, record_initial=(part == 0))
            np.savez(out.with_suffix(f".part{part:03d}.npz"), coords=coords, states=states)
            done += n
            part += 1
    else:
        if args.seq_interval > 0:
            coords, states, sequences = sim.run(args.steps, return_sequences=True)
        else:
            coords, states = sim.run(args.steps)

    exploded = sim.exploded_mask
    payload = {
        "record_dt_ps": float(cfg.dt_ps * cfg.record_interval),
        "temperature_K": dataset.temperature,
        "alpha_R_note": "physical time = record_dt_ps * alpha_R (see reports/calib_*.json)",
        "acceptance_rate": sim.acceptance_rate,
        # 爆炸统计（T11：不许静默丢弃）；evaluate 会排除这些 replica 并报告比例
        "exploded_mask": exploded,
        "exploded_step": sim.exploded_step,
        "config": json.dumps(asdict(cfg)),
        "model": str(args.model),
        "dataset": str(dataset_path),
    }
    if args.save_chunk_steps > 0:
        payload["n_parts"] = part
        payload["steps_per_part"] = args.save_chunk_steps
    else:
        payload["coords"] = coords
        payload["states"] = states
        if args.seq_interval > 0:
            payload["sequences"] = sequences
            payload["sequence_acceptance_rate"] = sim.sequence_acceptance_rate
    np.savez(out, **payload)
    n_bad = int(exploded.sum())
    seq_note = (f" seq_accept={sim.sequence_acceptance_rate:.3f}"
                if args.seq_interval > 0 else "")
    print(f"[run_cg] wrote {out}: replicas={args.replicas} steps={args.steps} "
          f"acceptance={sim.acceptance_rate:.3f}{seq_note} "
          f"exploded={n_bad}/{args.replicas}")
    if n_bad:
        steps_bad = sim.exploded_step[exploded]
        print(f"[run_cg] WARNING: exploded replicas {np.flatnonzero(exploded).tolist()[:20]} "
              f"at steps {steps_bad.tolist()[:20]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
