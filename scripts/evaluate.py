#!/usr/bin/env python3
"""Gate 1 评估：scripts/evaluate.py --system <name> --socg <npz> --k1 <npz>

判据表（每项写 value/threshold/pass 到 reports/gate1_<system>.json）：
  E1  每残基态布居 max|P_SOCG − P_AA|                  ≤ 0.05
  E2a  Cα 角 θ 直方图 JS（全残基合并）                  SOCG ≤ 0.02
  E2b  赝二面角 τ 直方图 JS                             SOCG ≤ 0.03
  E2c  按态条件化 P(θ|s_i) 的 JS（检验态边界力匹配偏差） 每态 ≤ 0.05
  E3   AA TICA(lag100) 投影 (TIC1,TIC2) FES 的 JS       SOCG ≤ 0.10 且 ≤ 基线
  E4a  ala10 平均螺旋度 |Δ|                             ≤ 0.05
  E4b  cln025 折叠分数 |Δ|（AA bimodal 阈值）           ≤ 0.10
  K1   |log(t1/t2)_CG − log(t1/t2)_AA|                  SOCG < 基线
  K2   非中心残基平均驻留时间（缩放后）比值              ∈ [0.5, 2]（仅 SOCG）
  K3   cln025 折叠/解折叠 MFPT 比值（缩放后）           仅报告
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF_PDB = ROOT / "data" / "ref" / "5AWL.pdb"
TICA_LAG = 100
TICA_DIM = 4
N_CLUSTERS = 100
FES_BINS = 50


def crit(value, threshold, ok) -> dict:
    return {"value": value, "threshold": threshold, "pass": bool(ok)}


def ca_distance_features(coords: np.ndarray) -> np.ndarray:
    T, N, _ = coords.shape
    pairs = [(i, j) for i in range(N) for j in range(i + 2, N)]
    return np.stack([np.linalg.norm(coords[:, i] - coords[:, j], axis=-1)
                     for i, j in pairs], axis=1)


def load_cg_npz(npz_path: Path):
    """读取 run_cg 输出（支持分片 part 文件），排除爆炸的 replica。

    返回 (coords (B',T,N,3), states (B',T,N), record_dt_ps, explosion_info)。
    """
    z = np.load(npz_path, allow_pickle=False)
    rec_dt = float(z["record_dt_ps"])
    if "n_parts" in z.files:
        parts = [np.load(npz_path.with_suffix(f".part{p:03d}.npz"), allow_pickle=False)
                 for p in range(int(z["n_parts"]))]
        coords = np.concatenate([p["coords"] for p in parts], axis=1)
        states = np.concatenate([p["states"] for p in parts], axis=1)
    else:
        coords, states = z["coords"], z["states"]
    B = coords.shape[0]
    exploded = z["exploded_mask"] if "exploded_mask" in z.files else np.zeros(B, bool)
    info = {"n_replicas": int(B), "n_exploded": int(exploded.sum()),
            "fraction": float(exploded.mean()) if B else 0.0,
            "steps": (z["exploded_step"][exploded].tolist()
                      if "exploded_step" in z.files else [])}
    keep = ~exploded
    return coords[keep], states[keep], rec_dt, info


def cg_feats_states(npz_path: Path):
    coords, states, rec_dt, info = load_cg_npz(npz_path)
    feats = [ca_distance_features(coords[b].astype(np.float64))
             for b in range(coords.shape[0])]
    sts = [states[b].astype(np.int64) for b in range(states.shape[0])]
    cds = [coords[b].astype(np.float64) for b in range(coords.shape[0])]
    return feats, sts, cds, rec_dt, info


def pooled(arrs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrs, axis=0)


def rmsd_to_ref(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
    from socg.geometry import kabsch_rmsd

    return kabsch_rmsd(coords, ref)


def mfpt_fold(rmsd: np.ndarray, thr: float, dt: float, margin: float = 0.02):
    """双阈值轨道 -> (mean U->F MFPT, mean F->U MFPT)，单位 = dt。"""
    side = np.zeros(rmsd.shape[0], dtype=int)
    side[rmsd < thr - margin] = 1
    side[rmsd > thr + margin] = -1
    filled = np.zeros_like(side)
    last = 0
    for i, s in enumerate(side):
        if s != 0:
            last = s
        filled[i] = last
    times_u2f, times_f2u = [], []
    t_in = 0
    cur = filled[0]
    for i in range(1, filled.shape[0]):
        if filled[i] != cur:
            wait = (i - t_in) * dt
            if cur == -1 and filled[i] == 1:
                times_u2f.append(wait)
            elif cur == 1 and filled[i] == -1:
                times_f2u.append(wait)
            t_in = i
            cur = filled[i]
    f = float(np.mean(times_u2f)) if times_u2f else float("nan")
    u = float(np.mean(times_f2u)) if times_f2u else float("nan")
    return f, u


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--socg", required=True, help="K=3 CG 生产 npz")
    parser.add_argument("--k1", required=True, help="K=1 基线生产 npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from socg.analysis.kinetics import (discretize, msm_its, pooled_mean_lifetime,
                                        tica)
    from socg.analysis.metrics import (bimodal_threshold, cg_local_features,
                                       hist_js, state_populations)
    from socg.data import CGDataset
    from socg.states import N_STATES

    system = args.system
    aa = CGDataset.load(ROOT / "data" / "cg" / f"{system}.npz")
    topo = aa.topology
    movable = np.where(~topo.frozen_mask)[0]
    central = movable[(movable >= topo.n // 3) & (movable < 2 * topo.n // 3)]
    central_set = set(central.tolist())
    noncentral = np.array([m for m in movable if m not in central_set], dtype=int)

    aa_feats = [ca_distance_features(t.coords.astype(np.float64)) for t in aa.trajectories]
    aa_states = [t.states for t in aa.trajectories]
    aa_coords = [t.coords.astype(np.float64) for t in aa.trajectories]
    aa_dt = aa.trajectories[0].dt_ps

    socg_feats, socg_states, socg_coords, socg_rec_dt, socg_expl = cg_feats_states(ROOT / args.socg)
    k1_feats, k1_states, k1_coords, k1_rec_dt, k1_expl = cg_feats_states(ROOT / args.k1)
    calib = json.loads((ROOT / "reports" / f"calib_{system}.json").read_text())
    life_dt = float(calib.get("lifetime_dt_ps", aa_dt))
    alpha_socg = float(calib["models"]["k3"]["alpha_R"])
    alpha_k1 = float(calib["models"]["k1"]["alpha_R"])
    socg_phys_dt = socg_rec_dt * alpha_socg
    k1_phys_dt = k1_rec_dt * alpha_k1

    criteria: dict[str, dict] = {}
    details: dict = {}

    # ---------------- E1 态布居 ----------------
    # 先沿时间合并所有轨迹的态，再求每残基布居 -> (N, K)
    aa_pops = state_populations(pooled(aa_states), N_STATES)
    socg_pops = state_populations(pooled(socg_states), N_STATES)
    e1 = float(np.abs(socg_pops - aa_pops)[movable].max())
    criteria["E1"] = crit(e1, 0.05, e1 <= 0.05)
    details["state_populations"] = {"aa": aa_pops.tolist(), "socg": socg_pops.tolist()}

    # ---------------- E2 局部分布 ----------------
    aa_loc = [cg_local_features(c) for c in aa_coords]
    sc_loc = [cg_local_features(c) for c in socg_coords]
    # 所有残基合并成一维样本（θ: (T,N-2)、τ: (T,N-3) -> 展平）
    aa_theta = pooled([x[0] for x in aa_loc]).ravel()
    aa_tau = pooled([x[1] for x in aa_loc]).ravel()
    sc_theta = pooled([x[0] for x in sc_loc]).ravel()
    sc_tau = pooled([x[1] for x in sc_loc]).ravel()
    e2a = hist_js(sc_theta, aa_theta, bins=90,
                  range=(float(min(aa_theta.min(), sc_theta.min())),
                         float(max(aa_theta.max(), sc_theta.max()))))
    e2b = hist_js(sc_tau, aa_tau, bins=90,
                  range=(float(min(aa_tau.min(), sc_tau.min())),
                         float(max(aa_tau.max(), sc_tau.max()))))
    criteria["E2a"] = crit(e2a, 0.02, e2a <= 0.02)
    criteria["E2b"] = crit(e2b, 0.03, e2b <= 0.03)

    # E2c：按态条件化的 P(theta | s_i)（非冻结残基，逐态合并所有残基）
    def cond_theta(coords_list, states_list):
        per_state = [[] for _ in range(N_STATES)]
        for c, s in zip(coords_list, states_list):
            theta, _ = cg_local_features(c)         # (T, N-2)，顶点 1..N-2
            sv = s[:, 1:-1]                          # 与 theta 顶点对齐
            for k in range(N_STATES):
                per_state[k].append(theta[sv == k])
        return [pooled(x) if x else np.zeros(0) for x in per_state]

    aa_cond = cond_theta(aa_coords, aa_states)
    sc_cond = cond_theta(socg_coords, socg_states)
    e2c = {}
    lo = float(aa_theta.min())
    hi = float(aa_theta.max())
    for k in range(N_STATES):
        if aa_cond[k].size and sc_cond[k].size:
            e2c[k] = hist_js(sc_cond[k], aa_cond[k], bins=90, range=(lo, hi))
        else:
            e2c[k] = float("nan")
    criteria["E2c"] = crit(max(e2c.values()), 0.05, max(e2c.values()) <= 0.05)
    details["E2c_per_state"] = e2c

    # ---------------- E3 FES ----------------
    tic = tica(aa_feats, TICA_LAG, TICA_DIM)
    aa_proj = [tic.transform(x) for x in aa_feats]
    sc_proj = [tic.transform(x) for x in socg_feats]
    k1_proj = [tic.transform(x) for x in k1_feats]
    aa2 = pooled([p[:, :2] for p in aa_proj])
    lo1, hi1 = np.quantile(aa2[:, 0], [0.001, 0.999])
    lo2, hi2 = np.quantile(aa2[:, 1], [0.001, 0.999])
    rng2d = [(float(lo1), float(hi1)), (float(lo2), float(hi2))]

    def fes_js(proj_list):
        return hist_js(pooled([p[:, :2] for p in proj_list]), aa2,
                       bins=FES_BINS, range=rng2d)

    e3_socg = fes_js(sc_proj)
    e3_k1 = fes_js(k1_proj)
    criteria["E3"] = crit(e3_socg, min(0.10, e3_k1),
                          e3_socg <= 0.10 and e3_socg <= e3_k1)
    details["E3"] = {"socg_js": e3_socg, "k1_js": e3_k1}

    # ---------------- E4 ----------------
    if system == "ala10":
        aa_h = float(aa_pops[movable, 0].mean())
        sc_h = float(socg_pops[movable, 0].mean())
        e4 = abs(sc_h - aa_h)
        criteria["E4a"] = crit(e4, 0.05, e4 <= 0.05)
        details["helicity"] = {"aa": aa_h, "socg": sc_h}
    else:
        from socg.aa.systems import load_reference_ca

        ref_ca = load_reference_ca(REF_PDB, expected_n=topo.n)
        aa_rmsd = pooled([rmsd_to_ref(c, ref_ca) for c in aa_coords])
        sc_rmsd = pooled([rmsd_to_ref(c, ref_ca) for c in socg_coords])
        k1_rmsd = pooled([rmsd_to_ref(c, ref_ca) for c in k1_coords])
        thr = bimodal_threshold(aa_rmsd)
        aa_f = float((aa_rmsd < thr).mean())
        sc_f = float((sc_rmsd < thr).mean())
        k1_f = float((k1_rmsd < thr).mean())
        e4 = abs(sc_f - aa_f)
        criteria["E4b"] = crit(e4, 0.10, e4 <= 0.10)
        details["folding"] = {"threshold_nm": thr, "aa": aa_f, "socg": sc_f, "k1": k1_f}

    # ---------------- K1 ITS 比值 ----------------
    dtraj_aa = discretize(aa_proj, N_CLUSTERS, seed=0)
    dtraj_sc = discretize(sc_proj, N_CLUSTERS, seed=0)
    dtraj_k1 = discretize(k1_proj, N_CLUSTERS, seed=0)
    its_lags = [20, 50, 100, 200]
    its_aa = {l: msm_its(dtraj_aa, l, 3, aa_dt) for l in its_lags}
    its_sc = {l: msm_its(dtraj_sc, l, 3, socg_phys_dt) for l in its_lags}
    its_k1 = {l: msm_its(dtraj_k1, l, 3, k1_phys_dt) for l in its_lags}

    def log_ratio(its):
        t1, t2 = its[0], its[1]
        if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
            return float("nan")
        return float(np.log(t1 / t2))

    lr_aa = np.mean([log_ratio(its_aa[l]) for l in its_lags])
    lr_sc = np.mean([log_ratio(its_sc[l]) for l in its_lags])
    lr_k1 = np.mean([log_ratio(its_k1[l]) for l in its_lags])
    err_sc = abs(lr_sc - lr_aa)
    err_k1 = abs(lr_k1 - lr_aa)
    criteria["K1"] = crit(err_sc, err_k1, err_sc < err_k1)
    details["K1"] = {"log_ratio_aa": lr_aa, "log_ratio_socg": lr_sc,
                     "log_ratio_k1": lr_k1,
                     "its": {"aa": {str(k): v.tolist() for k, v in its_aa.items()},
                             "socg": {str(k): v.tolist() for k, v in its_sc.items()},
                             "k1": {str(k): v.tolist() for k, v in its_k1.items()}}}

    # ---------------- K2 非中心残基驻留 ----------------
    # 与标定相同的物理时间分辨率（calib 的 lifetime_dt_ps）下比较驻留时间
    aa_life, aa_nseg, aa_ldt = pooled_mean_lifetime(aa_states, noncentral, aa_dt, life_dt)
    sc_life, sc_nseg, sc_ldt = pooled_mean_lifetime(socg_states, noncentral, socg_phys_dt, life_dt)
    k1_life, _, _ = pooled_mean_lifetime(k1_states, noncentral, k1_phys_dt, life_dt)
    k2 = sc_life / aa_life if aa_life > 0 else float("nan")
    criteria["K2"] = crit(k2, [0.5, 2], bool(np.isfinite(k2) and 0.5 <= k2 <= 2))
    details["K2"] = {"aa_ps": aa_life, "socg_ps": sc_life, "k1_ps": k1_life,
                     "segments": {"aa": aa_nseg, "socg": sc_nseg},
                     "lifetime_dt_ps": {"target": life_dt, "aa": aa_ldt, "socg": sc_ldt}}

    # ---------------- K3 MFPT（仅 cln025，报告项） ----------------
    if system == "cln025":
        # ref_ca 与 thr 已在 E4b 分支计算
        aa_mf = [mfpt_fold(rmsd_to_ref(c, ref_ca), thr, aa_dt) for c in aa_coords]
        sc_mf = [mfpt_fold(rmsd_to_ref(c, ref_ca), thr, socg_phys_dt) for c in socg_coords]

        def safe_mean(xs):
            xs = [x for x in xs if np.isfinite(x)]
            return float(np.mean(xs)) if xs else float("nan")

        aa_mfpt_u2f = safe_mean([x[0] for x in aa_mf])
        aa_mfpt_f2u = safe_mean([x[1] for x in aa_mf])
        sc_mfpt_u2f = safe_mean([x[0] for x in sc_mf])
        sc_mfpt_f2u = safe_mean([x[1] for x in sc_mf])
        details["K3"] = {
            "aa_mfpt_unfold_to_fold_ps": aa_mfpt_u2f,
            "aa_mfpt_fold_to_unfold_ps": aa_mfpt_f2u,
            "socg_mfpt_unfold_to_fold_ps": sc_mfpt_u2f,
            "socg_mfpt_fold_to_unfold_ps": sc_mfpt_f2u,
            "ratio_u2f": sc_mfpt_u2f / aa_mfpt_u2f if aa_mfpt_u2f else float("nan"),
            "ratio_f2u": sc_mfpt_f2u / aa_mfpt_f2u if aa_mfpt_f2u else float("nan"),
        }

    # ---------------- 图 ----------------
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 态布居
    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.35
    x = np.arange(topo.n)
    for k, name in enumerate(("A", "B", "L")):
        ax.bar(x + (k - 1) * width, aa_pops[:, k], width, label=f"AA {name}",
               alpha=0.7)
        ax.bar(x + (k - 1) * width, socg_pops[:, k], width, label=f"SOCG {name}",
               fill=False, edgecolor="C0")
    ax.set_xlabel("residue"); ax.set_ylabel("population"); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(fig_dir / f"{system}_pops.png", dpi=150)
    # theta / tau
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, a, b, name in ((axes[0], aa_theta, sc_theta, "theta"),
                           (axes[1], aa_tau, sc_tau, "tau")):
        lo_, hi_ = min(a.min(), b.min()), max(a.max(), b.max())
        ax.hist(a, bins=90, range=(lo_, hi_), alpha=0.6, label="AA", density=True)
        ax.hist(b, bins=90, range=(lo_, hi_), alpha=0.6, label="SOCG", density=True)
        ax.set_yscale("log"); ax.set_xlabel(f"{name} (deg)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / f"{system}_local.png", dpi=150)
    # FES 三联图
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    allp = np.vstack([aa2, pooled([p[:, :2] for p in sc_proj]),
                      pooled([p[:, :2] for p in k1_proj])])
    rr = [(float(allp[:, 0].min()), float(allp[:, 0].max())),
          (float(allp[:, 1].min()), float(allp[:, 1].max()))]
    for ax, pl, name in zip(axes, (aa2, pooled([p[:, :2] for p in sc_proj]),
                                   pooled([p[:, :2] for p in k1_proj])),
                            ("AA", "SOCG", "K=1")):
        h, xe, ye = np.histogram2d(pl[:, 0], pl[:, 1], bins=60, range=rr)
        p = (h / h.sum()).T
        fes = -np.log(np.where(p > 0, p, np.nan))
        ax.imshow(fes, origin="lower", aspect="auto",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis")
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(fig_dir / f"{system}_fes.png", dpi=150)
    # ITS-lag
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, its, dt in (("AA", its_aa, aa_dt), ("SOCG", its_sc, socg_phys_dt),
                          ("K=1", its_k1, k1_phys_dt)):
        lags_sorted = sorted(int(l) for l in its)
        t1 = [its[l][0] for l in lags_sorted]
        ax.plot([l * dt for l in lags_sorted], t1, marker="o", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("lag (ps)"); ax.set_ylabel("t1 (ps)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / f"{system}_its.png", dpi=150)

    # ---------------- 写报告 ----------------
    out = Path(args.out) if args.out else ROOT / "reports" / f"gate1_{system}.json"
    out.write_text(json.dumps({
        "system": system,
        "criteria": criteria,
        "details": details,
        "alpha_R": {"socg": alpha_socg, "k1": alpha_k1},
        "explosions": {"socg": socg_expl, "k1": k1_expl},
        "physical_dt_ps": {"socg": socg_phys_dt, "k1": k1_phys_dt},
    }, indent=2, default=float))
    print(f"[evaluate] {system}")
    for k, v in criteria.items():
        print(f"  {k}: value={v['value']} threshold={v['threshold']} pass={v['pass']}")
    print(f"[evaluate] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
