#!/usr/bin/env python
"""生成固定序列回归基准（统一设计文档 §30 Phase 0/§31 Level 0）。

用**当前**代码构建三个配置的模型（K=3 带类型对 / K=1 / K=3 无类型对），
在共享输入 (R, s, a) 上保存 checkpoint 与能量/力参考值到 tests/data/。
重构后 tests/test_sequence_regression.py 逐位复现这些数字，即
"U_new(R, s, a_fixed) == U_old(R, s)" 回归门 + v0 checkpoint 兼容门。

用法：python scripts/dump_regression_reference.py [--out tests/data]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import ideal_helix, randomize

from socg.model.priors import Priors
from socg.model.socg import ModelConfig, SOCGModel, save_model
from socg.topology import CGTopology

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"          # 覆盖全部 20 种氨基酸与 3 个 res_class
N_RES = len(SEQUENCE)
BATCH = 3
NOISE_SEED = 20261001
STATE_SEED = 20261002
PARAM_SEED = 7
TEMPERATURE = 300.0

CONFIGS = {
    "k3": ModelConfig(n_states=3, pair_types=True),
    "k1": ModelConfig(n_states=1, pair_types=True),
    "nopt": ModelConfig(n_states=3, pair_types=False),
}


def main() -> int:
    out_dir = ROOT / "tests" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = CGTopology.from_sequence(SEQUENCE)
    rng = np.random.default_rng(NOISE_SEED)
    R = ideal_helix(N_RES).astype(np.float64)[None].repeat(BATCH, axis=0)
    R = R + rng.normal(0.0, 0.02, size=R.shape)

    rng_s = np.random.default_rng(STATE_SEED)
    s = rng_s.integers(0, 3, size=(BATCH, N_RES))
    s[:, topo.frozen_mask] = 1                  # 冻结位归位（energy 内部亦会强制）

    a = torch.as_tensor(topo.aa_index, dtype=torch.long)
    R_t = torch.as_tensor(R)                    # float64
    s_t = torch.as_tensor(s, dtype=torch.long)

    # 先验只依赖 R（fit 确定性），三个配置共用同一 coords
    coords_for_priors = R + rng.normal(0.0, 0.01, size=R.shape)
    priors = Priors.fit(topo, coords_for_priors, TEMPERATURE)

    ref: dict[str, np.ndarray] = {
        "R": R, "s": s.astype(np.int64), "a": topo.aa_index,
        "sequence": np.array(SEQUENCE), "temperature": np.array(TEMPERATURE),
    }
    for name, config in CONFIGS.items():
        model = SOCGModel(topo, priors, config).double()
        randomize(model, 0.3, seed=PARAM_SEED)
        # 签名自适应：v0 为 energy(R,s)，v1（序列运行时化）起为 energy(R,s,a)
        try:
            with torch.no_grad():
                U = model.energy(R_t, s_t, a)
                F = model.forces(R_t, s_t, a)
        except TypeError:
            with torch.no_grad():
                U = model.energy(R_t, s_t)
                F = model.forces(R_t, s_t)
        ref[f"U_{name}"] = U.numpy()
        ref[f"F_{name}"] = F.numpy()
        ckpt = out_dir / f"regression_v0_{name}.pt"
        save_model(ckpt, model, meta={"purpose": "phase1-sequence-regression"})
        print(f"[dump] {name}: U={float(U.mean()):+.6f} kJ/mol  -> {ckpt.relative_to(ROOT)}")

    npz = out_dir / "regression_v0_reference.npz"
    np.savez(npz, **ref)
    print(f"[dump] inputs + reference U/F -> {npz.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
