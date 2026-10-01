#!/usr/bin/env python3
"""跑一个 AA replica：scripts/run_aa.py --system ala10 --replica 0 [--platform CUDA] [--ns X]

seed = 1000*k + 17；初始 (φ,ψ) 取 starts[k % len(starts)]；
输出 data/aa/<system>/rep<kk>/。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(("ala10", "cln025")))
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--ns", type=float, default=None,
                        help="覆盖 ns_per_replica（续跑时写更大的值即可延长）")
    args = parser.parse_args()

    from socg.aa.build import build_peptide_pdb
    from socg.aa.simulate import run_replica
    from socg.aa.systems import SYSTEMS

    spec = SYSTEMS[args.system]
    k = args.replica
    if not 0 <= k < spec.n_replicas:
        raise SystemExit(f"replica 必须在 [0, {spec.n_replicas}) 内")
    seed = 1000 * k + 17
    phi, psi = spec.starts[k % len(spec.starts)]
    ns = spec.ns_per_replica if args.ns is None else args.ns

    out_dir = ROOT / "data" / "aa" / spec.name / f"rep{k:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / f"{spec.name}.pdb"
    if not pdb_path.exists():
        print(f"[run_aa] building {spec.sequence} at phi={phi}, psi={psi}")
        build_peptide_pdb(spec.sequence, phi, psi, pdb_path)

    print(f"[run_aa] system={spec.name} replica={k} seed={seed} T={spec.temperature}K "
          f"ns={ns} platform={args.platform} -> {out_dir}")
    run_replica(pdb_path, out_dir, spec.temperature, ns, seed,
                platform_name=args.platform)
    print("[run_aa] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
