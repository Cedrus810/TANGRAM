"""联合拟合：力匹配 (FM) + 态伪似然 (PL) + L2（目标函数在 T10 中锁定）。

L = FM + λ_PL·PL + λ_L2·Σ‖θ‖²
FM = mean‖F_model(R,s) − F_ref‖² / var(F_ref)
PL = −mean over (帧, 可动位点 i) of log softmax_k(−β U(R, s with s_i := k))[s_i]
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .constants import KB
from .data import CGDataset
from .model.socg import SOCGModel


@dataclass(frozen=True)
class FitConfig:
    batch_frames: int = 256
    max_steps: int = 20000
    lr: float = 5e-3
    pl_weight: float = 1.0
    l2: float = 1e-4
    stride: int = 5
    eval_every: int = 500
    device: str = "cuda"
    seed: int = 0


def force_matching_loss(model: SOCGModel, R: torch.Tensor, s: torch.Tensor,
                        a: torch.Tensor, F_ref: torch.Tensor,
                        f_var: torch.Tensor) -> torch.Tensor:
    F_model = model.forces(R, s, a, create_graph=True)
    return ((F_model - F_ref) ** 2).mean() / f_var


def pseudo_likelihood_nll(model: SOCGModel, R: torch.Tensor, s: torch.Tensor,
                          a: torch.Tensor,
                          beta_: float | torch.Tensor) -> torch.Tensor:
    """态伪似然 NLL（带参数梯度，可反传）。

    K=1 时恒为 0（其梯度中不含 eps、w_nn）；冻结位点不参与。
    PL = −mean over (帧, 可动位点 i) of log softmax_k(−β U(R, s_i:=k, a))[s_i]
    """
    if model.config.n_states == 1:
        return R.sum() * 0.0
    movable = torch.as_tensor(np.where(~model.topo.frozen_mask)[0],
                              dtype=torch.long, device=R.device)
    K = model.config.n_states
    rows = torch.arange(R.shape[0], device=R.device)
    with torch.no_grad():
        s_base = model.prepare_states(s)
    total = R.sum() * 0.0
    for site in movable:
        U = torch.empty((R.shape[0], K), device=R.device, dtype=R.dtype)
        for k in range(K):
            s_k = s_base.clone()
            s_k[:, site] = k
            U[:, k] = model.energy(R, s_k, a)
        logp = torch.log_softmax(-beta_ * U, dim=-1)
        total = total - logp[rows, s_base[:, site]].sum()
    return total / (R.shape[0] * movable.numel())


def _eval_metrics(model, coords, states, a, forces, f_var, beta_, device,
                  chunk: int = 4096) -> dict:
    model.eval()
    n = coords.shape[0]
    fm_sum, pl_sum, r2_res, r2_tot = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for lo in range(0, n, chunk):
            R = torch.as_tensor(coords[lo:lo + chunk], device=device)
            s = torch.as_tensor(states[lo:lo + chunk], device=device, dtype=torch.long)
            F = torch.as_tensor(forces[lo:lo + chunk], device=device)
            F_model = model.forces(R, s, a)
            fm_sum += float(((F_model - F) ** 2).sum()) / float(f_var) / (F.shape[0] * F.shape[1] * 3)
            pl_sum += float(pseudo_likelihood_nll(model, R, s, a, beta_)) * R.shape[0]
            r2_res += float(((F_model - F) ** 2).sum())
            r2_tot += float(((F - F.mean(dim=0, keepdim=True)) ** 2).sum())
    model.train()
    force_r2 = 1.0 - r2_res / r2_tot if r2_tot > 0 else 0.0
    return {"fm": fm_sum, "pl": pl_sum / max(n, 1), "force_r2": force_r2}


def fit(model: SOCGModel, train: CGDataset, val: CGDataset,
        config: FitConfig) -> list[dict]:
    """训练循环；返回每 eval_every 步的 train/val 指标历史。"""
    device = config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = model.to(device=device)
    model.train()

    coords_tr, forces_tr, states_tr = train.frames(stride=config.stride)
    coords_va, forces_va, states_va = val.frames(stride=config.stride)
    if forces_tr is None:
        raise ValueError("training data has no forces")
    f_var_np = forces_tr.astype(np.float64).var(axis=0).mean()
    f_var = torch.tensor(max(f_var_np, 1e-12), device=device)
    beta_ = 1.0 / (KB * train.temperature)
    # 运行时序列状态（统一设计文档 §8）：拟合期的 a 来自数据集拓扑的参考序列
    a = torch.as_tensor(train.topology.aa_index, dtype=torch.long, device=device)

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    n = coords_tr.shape[0]
    history: list[dict] = []
    start = time.time()
    for step in range(1, config.max_steps + 1):
        idx = rng.choice(n, size=min(config.batch_frames, n), replace=False)
        R = torch.as_tensor(coords_tr[idx], device=device)
        s = torch.as_tensor(states_tr[idx], device=device, dtype=torch.long)
        F = torch.as_tensor(forces_tr[idx], device=device)
        s = model.prepare_states(s)

        fm = force_matching_loss(model, R, s, a, F, f_var)
        pl = pseudo_likelihood_nll(model, R, s, a, beta_)
        l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        loss = fm + config.pl_weight * pl + config.l2 * l2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            tr_m = _eval_metrics(model, coords_tr, states_tr, a, forces_tr, f_var, beta_, device)
            va_m = _eval_metrics(model, coords_va, states_va, a, forces_va, f_var, beta_, device)
            history.append({
                "step": step,
                "wall_s": time.time() - start,
                "loss": float(loss.detach()),
                "train_fm": tr_m["fm"], "val_fm": va_m["fm"],
                "train_pl": tr_m["pl"], "val_pl": va_m["pl"],
                "train_force_R2": tr_m["force_r2"], "val_force_R2": va_m["force_r2"],
            })
            print(f"[fit] step {step}: train FM={tr_m['fm']:.4f} PL={tr_m['pl']:.4f} "
                  f"R2={tr_m['force_r2']:.3f} | val FM={va_m['fm']:.4f} "
                  f"PL={va_m['pl']:.4f} R2={va_m['force_r2']:.3f}")
    return history
