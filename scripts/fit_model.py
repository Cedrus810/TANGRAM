#!/usr/bin/env python3
"""联合拟合：scripts/fit_model.py --system ala10 --n-states 3 --val-replica 3 --out models/ala10_k3.pt

Priors 只用训练集拟合；训练历史写入 reports/fit_<system>_k<K>.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--n-states", type=int, choices=(1, 3), default=3)
    parser.add_argument("--val-replica", type=int, required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    import torch

    from socg.data import CGDataset
    from socg.fit import FitConfig, fit
    from socg.model.priors import Priors
    from socg.model.socg import ModelConfig, SOCGModel, save_model
    from socg.topology import CGTopology

    system = args.system
    dataset = CGDataset.load(ROOT / "data" / "cg" / f"{system}.npz")
    topo = CGTopology.from_json(dataset.topology.to_json())
    train, val = dataset.split([args.val_replica])
    print(f"[fit_model] {system}: {len(train.trajectories)} train / "
          f"{len(val.trajectories)} val trajectories")

    coords_tr, _, states_tr = train.frames(stride=args.stride)
    priors = Priors.fit(topo, coords_tr, train.temperature)
    config = ModelConfig(n_states=args.n_states)
    model = SOCGModel(topo, priors, config)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[fit_model] CUDA unavailable, falling back to CPU")
        device = "cpu"
    fit_config = FitConfig(device=device, stride=args.stride,
                           **({"max_steps": args.max_steps} if args.max_steps else {}))
    history = fit(model, train, val, fit_config)

    out = Path(args.out) if args.out else ROOT / "models" / f"{system}_k{args.n_states}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(out, model, meta={
        "system": system, "val_replica": args.val_replica,
        "temperature": train.temperature, "stride": args.stride,
    })
    report_path = ROOT / "reports" / f"fit_{system}_k{args.n_states}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "system": system, "n_states": args.n_states,
        "val_replica": args.val_replica,
        "config": {k: getattr(fit_config, k) for k in fit_config.__dataclass_fields__},
        "history": history,
    }, indent=2))
    print(f"[fit_model] saved {out} and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
