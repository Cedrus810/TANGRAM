"""混合 Langevin + Metropolis 态/序列翻转 CG 模拟器（D1；统一设计文档 §7/§17/§18）。

状态 X = (R, s, a)：R 用 BAOAB 积分；每 flip_interval 步做一次 flip_sweep
（M 次单点提议，M = 可动位点数，各 replica 独立）：位点在可动位点中均匀抽取，
新态在另外 K-1 个态中均匀抽取，接受概率 min(1, exp(-βΔU))。K=1 或
flip_interval=0 时从不翻转。

序列移动（统一设计文档 §6/§13/§18）：每 seq_interval 步做一次
sequence_flip_sweep——由 SequenceProposal 在可突变位点上提议 a_i'，按
Metropolis–Hastings 接受：

    A = min[1, exp(-βΔU) · q(a|R,a') / q(a'|R,a)]

seq_update="full" 用两次全能量差分求 ΔU；"local" 用模型的局部 ΔU（§29.2，
只重算依赖 a_i 的项）。无提议算子或 seq_interval=0 时 K_a 为恒等核。

每步检查非有限值：默认 on_nonfinite="raise" 抛 RuntimeError（含步数与
replica 编号）；on_nonfinite="freeze" 时把出事的 replica 冻结在最后一个
有限构型（不再积分、不再翻转/突变），其余继续，并记录 exploded_mask /
exploded_step，供生产模拟统计爆炸比例（T11：不许静默丢弃）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import KB
from .model.socg import SOCGModel
from .sequence.base import SequenceProposal


@dataclass(frozen=True)
class LangevinConfig:
    temperature: float
    dt_ps: float = 0.002
    friction_per_ps: float = 1.0
    mass_amu: float = 110.0
    flip_interval: int = 10     # 0 表示关闭翻转
    record_interval: int = 500
    seed: int = 0
    seq_interval: int = 0       # 0 表示关闭序列移动
    seq_update: str = "full"    # "full" | "local"（§29.2 局部 ΔU）


class CGSimulator:
    """批量 (B 个 replica) 混合动力学。状态 (R, s, a)。"""

    def __init__(self, model: SOCGModel, config: LangevinConfig,
                 R0: torch.Tensor, s0: torch.Tensor, a0: torch.Tensor,
                 v0: torch.Tensor | None = None,
                 on_nonfinite: str = "raise",
                 sequence_proposal: SequenceProposal | None = None,
                 mutable_mask: torch.Tensor | None = None):
        if on_nonfinite not in ("raise", "freeze"):
            raise ValueError(f"on_nonfinite must be 'raise' or 'freeze', got {on_nonfinite!r}")
        if config.seq_update not in ("full", "local"):
            raise ValueError(
                f"seq_update must be 'full' or 'local', got {config.seq_update!r}")
        if sequence_proposal is not None and config.seq_interval <= 0:
            raise ValueError("sequence_proposal given but seq_interval <= 0")
        self.model = model
        self.config = config
        self.on_nonfinite = on_nonfinite
        self.seq_proposal = sequence_proposal
        device = model.w_bond.device
        dtype = model.w_bond.dtype
        self.R = R0.detach().to(device=device, dtype=dtype).clone()
        if self.R.ndim != 3:
            raise ValueError(f"R0 must be (B,N,3), got {tuple(self.R.shape)}")
        self.B, self.N = int(self.R.shape[0]), int(self.R.shape[1])
        self.s = model.prepare_states(
            torch.as_tensor(s0, dtype=torch.long, device=device)).clone()
        if self.s.shape != (self.B, self.N):
            raise ValueError(f"s0 shape {tuple(self.s.shape)} != {(self.B, self.N)}")
        self.a = self._normalize_a0(a0, device)
        if v0 is None:
            self.v = torch.zeros_like(self.R)
        else:
            self.v = v0.detach().to(device=device, dtype=dtype).clone()

        self.beta = 1.0 / (KB * float(config.temperature))
        self.mass = float(config.mass_amu)
        self.c1 = float(np.exp(-config.friction_per_ps * config.dt_ps))
        self.sqrt_1mc1sq = float(np.sqrt(max(0.0, 1.0 - self.c1 ** 2)))
        self.sigma_v = float(np.sqrt(KB * config.temperature / self.mass))
        self.movable = torch.as_tensor(
            np.where(~model.topo.frozen_mask)[0], dtype=torch.long, device=device)
        # 序列可突变位点：外部 mutable_mask 优先，否则用拓扑内部残基（首尾除外）
        if mutable_mask is not None:
            m = torch.as_tensor(mutable_mask, dtype=torch.bool, device=device)
            if m.dim() == 2 and m.shape[0] == 1:
                m = m[0]
            if tuple(m.shape) != (self.N,):
                raise ValueError(f"mutable_mask must be (N,) = ({self.N},), got {tuple(m.shape)}")
            self.seq_mutable = m.nonzero().squeeze(-1)
        else:
            self.seq_mutable = self.movable
        self.k = int(model.config.n_states)

        self.alive = torch.ones(self.B, dtype=torch.bool, device=device)
        self._frozen_R = self.R.clone()
        self.exploded_step = np.full(self.B, -1, dtype=np.int64)

        self._rng = torch.Generator(device=str(device))
        self._rng.manual_seed(int(config.seed))
        self._forces = self._compute_forces()
        self._U = self._potential()
        self._step_count = 0
        self.accepted = 0
        self.proposed = 0
        self.seq_accepted = 0
        self.seq_proposed = 0
        self._records_R: list[torch.Tensor] = []
        self._records_s: list[torch.Tensor] = []
        self._records_a: list[torch.Tensor] = []

    def _normalize_a0(self, a0: torch.Tensor, device) -> torch.Tensor:
        """a0: (N,) 或 (B, N) long -> (B, N)（clone，序列移动按 replica 突变）。"""
        if not isinstance(a0, torch.Tensor):
            a0 = torch.as_tensor(a0)
        a0 = a0.to(device=device, dtype=torch.long)
        if a0.dim() == 1:
            a0 = a0.unsqueeze(0).expand(self.B, -1)
        if a0.dim() != 2 or tuple(a0.shape) != (self.B, self.N):
            raise ValueError(
                f"a0 must be (N,) or (B,N) = ({self.B},{self.N}), got {tuple(a0.shape)}")
        return a0.clone()

    # ------------------------------------------------------------------ #
    # 基础量
    # ------------------------------------------------------------------ #
    def _compute_forces(self) -> torch.Tensor:
        with torch.no_grad():
            f = self.model.forces(self.R, self.s, self.a)
        return f

    def potential_energy(self) -> torch.Tensor:
        with torch.no_grad():
            return self.model.energy(self.R, self.s, self.a)

    def kinetic_energy(self) -> torch.Tensor:
        return 0.5 * self.mass * self.v.pow(2).sum(dim=(1, 2))

    @property
    def exploded_mask(self) -> np.ndarray:
        return (~self.alive).cpu().numpy()

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted) / float(self.proposed) if self.proposed else 0.0

    @property
    def sequence_acceptance_rate(self) -> float:
        return (float(self.seq_accepted) / float(self.seq_proposed)
                if self.seq_proposed else 0.0)

    # ------------------------------------------------------------------ #
    # 积分
    # ------------------------------------------------------------------ #
    def _check_finite(self, tensor: torch.Tensor, what: str) -> None:
        flat_bad = ~torch.isfinite(tensor.reshape(self.B, -1)).all(dim=-1)
        if bool(flat_bad.any()):
            rep = int(flat_bad.nonzero()[0])
            raise RuntimeError(
                f"{what} contains non-finite values at step {self._step_count}, "
                f"replica {rep}")

    def _freeze_nonfinite(self, R_prev: torch.Tensor) -> None:
        """freeze 模式：新出现非有限值的 replica 回退到 R_prev 并冻结。"""
        ok = (torch.isfinite(self.R).reshape(self.B, -1).all(dim=-1)
              & torch.isfinite(self._forces).reshape(self.B, -1).all(dim=-1))
        newly = self.alive & ~ok
        if bool(newly.any()):
            self._frozen_R[newly] = R_prev[newly]
            self.alive &= ~newly
            self.exploded_step[newly.cpu().numpy()] = self._step_count
        dead = ~self.alive
        if bool(dead.any()):
            self.R[dead] = self._frozen_R[dead]
            self.v[dead] = 0.0
            self._forces[dead] = 0.0

    def _step(self) -> None:
        dt = self.config.dt_ps
        freeze = self.on_nonfinite == "freeze"
        if freeze:
            R_prev = self.R.clone()
        half = 0.5 * dt / self.mass
        # B
        self.v += half * self._forces
        # A
        self.R += 0.5 * dt * self.v
        # O
        if self.sqrt_1mc1sq > 0.0:
            noise = torch.randn(self.v.shape, generator=self._rng,
                                device=self.v.device, dtype=self.v.dtype)
            self.v = self.c1 * self.v + self.sqrt_1mc1sq * self.sigma_v * noise
        # A
        self.R += 0.5 * dt * self.v
        # F + B
        self._forces = self._compute_forces()
        self.v += half * self._forces
        self._step_count += 1
        if freeze:
            self._freeze_nonfinite(R_prev)
        else:
            self._check_finite(self.R, "positions")
            self._check_finite(self._forces, "forces")

    # ------------------------------------------------------------------ #
    # Metropolis 态翻转
    # ------------------------------------------------------------------ #
    def flip_sweep(self) -> None:
        """M 次单点提议（M = 可动位点数），各 replica 独立。"""
        if self.k <= 1 or self.config.flip_interval <= 0:
            return
        self._U = self._potential()      # R 自上次翻转后已移动，先刷新基准能量
        M = int(self.movable.numel())
        rows = torch.arange(self.B, device=self.R.device)
        for _ in range(M):
            site_col = torch.randint(0, M, (self.B,), generator=self._rng,
                                     device=self.R.device)
            sites = self.movable[site_col]
            shift = torch.randint(0, self.k - 1, (self.B,), generator=self._rng,
                                  device=self.R.device) + 1
            new_state = (self.s[rows, sites] + shift) % self.k
            s_new = self.s.clone()
            s_new[rows, sites] = new_state
            with torch.no_grad():
                U_new = self.model.energy(self.R, s_new, self.a)
            dU = U_new - self._U
            u = torch.rand(self.B, generator=self._rng, device=self.R.device)
            accept = torch.log(u) < -self.beta * dU
            accept &= torch.isfinite(dU) & self.alive
            self.proposed += self.B
            self.accepted += int(accept.sum())
            if bool(accept.any()):
                idx = accept.nonzero().squeeze(-1)
                self.s[idx, sites[idx]] = new_state[idx]
                self._U = torch.where(accept, U_new, self._U)
        # 翻转结束后重算力（与能量）
        self._forces = self._compute_forces()
        if not bool(self.alive.all()):
            self._forces[~self.alive] = 0.0
        self._U = self._potential()

    # ------------------------------------------------------------------ #
    # Metropolis–Hastings 序列移动（统一设计文档 §6/§18）
    # ------------------------------------------------------------------ #
    def sequence_flip_sweep(self) -> None:
        """每个 replica 一次单位点突变提议 + MH 接受。

        提议由 self.seq_proposal 给出；log_q(R, a_from, a_to, site) 分别在
        (a→a') 与 (a'→a) 上求值得到 Hastings 修正。ΔU 用 full（两次全能量）
        或 local（只重算 a 依赖项，§29.2）路径。
        """
        if self.seq_proposal is None or self.config.seq_interval <= 0:
            return
        self._U = self._potential()      # 刷新基准能量（R 已移动）
        M = int(self.seq_mutable.numel())
        if M == 0:
            return
        rows = torch.arange(self.B, device=self.R.device)
        site_col = torch.randint(0, M, (self.B,), generator=self._rng,
                                 device=self.R.device)
        sites = self.seq_mutable[site_col]
        with torch.no_grad():
            new_aa = self.seq_proposal.sample(self.R, self.a, sites,
                                              generator=self._rng)
            if self.config.seq_update == "local":
                dU = self.model.mutation_delta_u_local(self.R, self.s, self.a,
                                                       sites, new_aa)
            else:
                a_new = self.a.clone()
                a_new[rows, sites] = new_aa
                dU = (self.model.energy(self.R, self.s, a_new)
                      - self.model.energy(self.R, self.s, self.a))
            # Hastings：log q(a'→a) - log q(a→a')（同一 R；同一被突变位点）
            a_new = self.a.clone()
            a_new[rows, sites] = new_aa
            log_q_forward = self.seq_proposal.log_q(self.R, self.a, a_new, sites)
            log_q_reverse = self.seq_proposal.log_q(self.R, a_new, self.a, sites)
            log_accept = -self.beta * dU + log_q_reverse - log_q_forward
            u = torch.rand(self.B, generator=self._rng, device=self.R.device)
            accept = torch.log(u) < log_accept
            accept &= torch.isfinite(dU) & self.alive
            self.seq_proposed += self.B
            self.seq_accepted += int(accept.sum())
            if bool(accept.any()):
                idx = accept.nonzero().squeeze(-1)
                self.a[idx, sites[idx]] = new_aa[idx]
        # 序列变化改变能量 -> 重算力与能量
        self._forces = self._compute_forces()
        if not bool(self.alive.all()):
            self._forces[~self.alive] = 0.0
        self._U = self._potential()

    def _potential(self) -> torch.Tensor:
        with torch.no_grad():
            return self.model.energy(self.R, self.s, self.a)

    # ------------------------------------------------------------------ #
    # 记录与主循环
    # ------------------------------------------------------------------ #
    def _record_frame(self) -> None:
        self._records_R.append(self.R.detach().to(torch.float32).cpu().clone())
        self._records_s.append(self.s.detach().to(torch.int8).cpu().clone())
        self._records_a.append(self.a.detach().to(torch.int8).cpu().clone())

    def run(self, n_steps: int, record_initial: bool = True,
            return_sequences: bool = False):
        """推进 n_steps，返回 (coords (B,T,N,3) f32, states (B,T,N) int8)；
        return_sequences=True 时追加序列轨迹 (B,T,N) int8 作第三返回值。

        record_initial=True 时先记录当前帧；之后在全局步数为 r 的倍数时追加。
        从 0 步起调用时 T = n_steps//r + 1。分段调用时，后续段传
        record_initial=False，避免段边界重复记录。
        """
        r = max(1, int(self.config.record_interval))
        if record_initial:
            self._record_frame()
        for _ in range(n_steps):
            if (self.config.flip_interval > 0 and self.k > 1
                    and self._step_count % self.config.flip_interval == 0
                    and self._step_count > 0):
                self.flip_sweep()
            if (self.seq_proposal is not None
                    and self._step_count % self.config.seq_interval == 0
                    and self._step_count > 0):
                self.sequence_flip_sweep()
            self._step()
            if self._step_count % r == 0:
                self._record_frame()
        if not self._records_R:
            coords = np.zeros((self.B, 0, self.N, 3), np.float32)
            states = np.zeros((self.B, 0, self.N), np.int8)
            if return_sequences:
                return coords, states, np.zeros((self.B, 0, self.N), np.int8)
            return coords, states
        coords = torch.stack(self._records_R, dim=1).numpy()
        states = torch.stack(self._records_s, dim=1).numpy()
        seqs = torch.stack(self._records_a, dim=1).numpy()
        self._records_R, self._records_s, self._records_a = [], [], []
        if return_sequences:
            return coords, states, seqs
        return coords, states
