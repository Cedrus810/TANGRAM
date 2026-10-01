"""SOCG v0：operator-valued CG 势函数（能量函数在 Phase 1 文档 T8 中锁定）。

统一设计文档 §10 起，序列 a 是运行时状态变量：

U(R,s,a) = U_prior(R)
         + Σ_bonds     RBF_bond(r_{i,i+1}) · w_bond
         + Σ_angles    RBF_angle(θ_i) · w_angle[c(a_i), s_i]        # 顶点 i
         + Σ_dihedrals Fourier(τ) · w_dihed[s_j, s_k]              # 中心键 j-k
         + Σ_i eps[c(a_i), s_i]
         + Σ_i w_nn[s_i, s_{i+1}]
         + Σ_pairs     RBF_pair(r_ij) · sym(w_pair_state)[bucket_ij, s_i, s_j]
         + Σ_{pairs, bucket=2} RBF_pair(r_ij) · sym(w_pair_type)[a_i, a_j]

其中 c(a) = RES_CLASS_TABLE[a] 为运行时查表（§9），模型不再持有序列缓冲；
topo 里的 sequence 只是构造时的参考序列（序列化用），运行时 a 可以不同。

所有可学习参数初始化为 0；模型对参数线性。角度在模型内部用弧度。
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..states import FROZEN_STATE
from ..topology import N_RES_CLASSES, N_SEP_BUCKETS, RES_CLASS_TABLE, CGTopology
from .basis import RBF, Fourier
from .priors import Priors

N_AA = len(RES_CLASS_TABLE)


@dataclass(frozen=True)
class ModelConfig:
    n_states: int = 3
    n_rbf_bond: int = 12
    bond_range: tuple[float, float] = (0.33, 0.43)
    n_rbf_angle: int = 24
    angle_range: tuple[float, float] = (1.2, 2.8)   # 弧度
    fourier_order: int = 4
    n_rbf_pair: int = 24
    pair_range: tuple[float, float] = (0.35, 1.5)
    pair_types: bool = True


def _torch_dihedral(p0, p1, p2, p3):
    """IUPAC 二面角（弧度），与 numpy 版约定一致。输入 (..., 3)。"""
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


class SOCGModel(nn.Module):
    """带内部离散态的 Cα CG 模型。energy(R, s, a) 与 forces(R, s, a)。

    a 为运行时序列状态：(N,) 或 (B, N) long 张量；(N,) 按 batch 广播。
    模型不持有序列缓冲，res_class 由 RES_CLASS_TABLE[a] 运行时派生。
    """

    def __init__(self, topo: CGTopology, priors: Priors, config: ModelConfig):
        super().__init__()
        self.topo = topo
        self.priors = priors
        self.config = config
        K = config.n_states

        self.bond_basis = RBF(*config.bond_range, config.n_rbf_bond)
        self.angle_basis = RBF(*config.angle_range, config.n_rbf_angle)
        self.pair_basis = RBF(*config.pair_range, config.n_rbf_pair)
        self.dihedral_basis = Fourier(config.fourier_order)

        self.w_bond = nn.Parameter(torch.zeros(config.n_rbf_bond))
        self.w_angle = nn.Parameter(torch.zeros(N_RES_CLASSES, K, config.n_rbf_angle))
        self.w_dihed = nn.Parameter(torch.zeros(K, K, 2 * config.fourier_order))
        self.eps = nn.Parameter(torch.zeros(N_RES_CLASSES, K))
        self.w_nn = nn.Parameter(torch.zeros(K, K))
        self.w_pair_state = nn.Parameter(
            torch.zeros(N_SEP_BUCKETS, K, K, config.n_rbf_pair))
        if config.pair_types:
            self.w_pair_type = nn.Parameter(torch.zeros(N_AA, N_AA, config.n_rbf_pair))
        else:
            self.register_parameter("w_pair_type", None)

        self.register_buffer("bond_index", torch.as_tensor(topo.bonds, dtype=torch.long))
        self.register_buffer("angle_index", torch.as_tensor(topo.angles, dtype=torch.long))
        self.register_buffer("dihedral_index",
                             torch.as_tensor(topo.dihedrals, dtype=torch.long))
        self.register_buffer("pair_index", torch.as_tensor(topo.pairs, dtype=torch.long))
        self.register_buffer("pair_bucket", torch.as_tensor(topo.pair_bucket, dtype=torch.long))
        self.register_buffer("pair_type_rows",
                             torch.as_tensor(topo.pair_types_index, dtype=torch.long))
        self.register_buffer("res_class_of_aa",
                             torch.as_tensor(RES_CLASS_TABLE, dtype=torch.long))
        self.register_buffer("frozen_idx",
                             torch.as_tensor(np.where(topo.frozen_mask)[0], dtype=torch.long))

        # ---- 突变能量学索引（非持久化：仅拓扑派生，不进 state_dict）---- #
        # 顶点 i 的角度行号（首尾残基无角度，-1）
        vertex_angle = np.full(topo.n, -1, dtype=np.int64)
        for row, (_i0, i1, _i2) in enumerate(topo.angles):
            vertex_angle[i1] = row
        self.register_buffer("vertex_angle_index",
                             torch.as_tensor(vertex_angle, dtype=torch.long),
                             persistent=False)
        # 每个残基在 bucket==2 类型对中的伙伴与对应 pair 行号（补齐 -1 到等长）
        pt_rows = topo.pair_types_index
        pr = topo.pairs[pt_rows]
        partners: list[list[int]] = [[] for _ in range(topo.n)]
        prow: list[list[int]] = [[] for _ in range(topo.n)]
        for row, (i, j) in enumerate(pr):
            partners[i].append(int(j))
            prow[i].append(int(pt_rows[row]))
            partners[j].append(int(i))
            prow[j].append(int(pt_rows[row]))
        pmax = max((len(p) for p in partners), default=0) or 1
        pt_partners = np.full((topo.n, pmax), -1, dtype=np.int64)
        pt_pair_row = np.zeros((topo.n, pmax), dtype=np.int64)
        for i in range(topo.n):
            pt_partners[i, :len(partners[i])] = partners[i]
            pt_pair_row[i, :len(prow[i])] = prow[i]
        self.register_buffer("pt_partners",
                             torch.as_tensor(pt_partners, dtype=torch.long),
                             persistent=False)
        self.register_buffer("pt_pair_row",
                             torch.as_tensor(pt_pair_row, dtype=torch.long),
                             persistent=False)

    # ------------------------------------------------------------------ #
    # 状态处理
    # ------------------------------------------------------------------ #
    def prepare_states(self, s: torch.Tensor) -> torch.Tensor:
        """K=1 时全置 0；K>1 时把冻结位点置为 FROZEN_STATE。"""
        if not isinstance(s, torch.Tensor):
            s = torch.as_tensor(s, dtype=torch.long)
        s = s.to(device=self.w_bond.device, dtype=torch.long)
        if self.config.n_states == 1:
            return torch.zeros_like(s)
        s = s.clone()
        if self.frozen_idx.numel():
            s[..., self.frozen_idx] = FROZEN_STATE
        return s

    def _normalize_a(self, R: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """把 a 归一成与 R 同 batch 的 (B, N) long 张量。"""
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a)
        a = a.to(device=R.device, dtype=torch.long)
        if a.dim() == 1:
            a = a.unsqueeze(0).expand(R.shape[0], -1)
        if a.dim() != 2 or tuple(a.shape) != tuple(R.shape[:2]):
            raise ValueError(
                f"a must be (N,) or (B,N) matching R {tuple(R.shape[:2])}, "
                f"got {tuple(a.shape)}")
        return a

    # ------------------------------------------------------------------ #
    # 能量
    # ------------------------------------------------------------------ #
    def energy(self, R: torch.Tensor, s: torch.Tensor,
               a: torch.Tensor) -> torch.Tensor:
        """R (B,N,3)、s (B,N)、a (N,) 或 (B,N) -> (B,)。冻结位点在内部被强制归位。"""
        dtype = self.w_bond.dtype
        R = R.to(dtype)
        s = self.prepare_states(s)
        a = self._normalize_a(R, a)
        K = self.config.n_states
        cls_all = self.res_class_of_aa[a]                                # (B,N)

        # bonds
        bi = self.bond_index
        d = torch.norm(R[:, bi[:, 0]] - R[:, bi[:, 1]], dim=-1)          # (B,nb)
        U = (self.bond_basis(d) * self.w_bond).sum(-1).sum(-1)

        # angles（顶点 i：c(a_i), s_i）
        ai = self.angle_index                                            # (na,3)
        ba = R[:, ai[:, 0]] - R[:, ai[:, 1]]
        bc = R[:, ai[:, 2]] - R[:, ai[:, 1]]
        x = (ba * bc).sum(-1)
        y = torch.linalg.vector_norm(torch.cross(ba, bc, dim=-1), dim=-1)
        theta = torch.atan2(y, x)
        cls_v = cls_all[:, ai[:, 1]]                                     # (B,na)
        s_v = s[:, ai[:, 1]]                                             # (B,na)
        w_a = self.w_angle.reshape(N_RES_CLASSES * K, -1)[cls_v * K + s_v]
        U = U + (self.angle_basis(theta) * w_a).sum(-1).sum(-1)

        # dihedrals（中心键 j-k：s_j, s_k）
        di = self.dihedral_index                                         # (nd,4)
        tau = _torch_dihedral(R[:, di[:, 0]], R[:, di[:, 1]],
                              R[:, di[:, 2]], R[:, di[:, 3]])
        s_j, s_k = s[:, di[:, 1]], s[:, di[:, 2]]
        w_d = self.w_dihed.reshape(K * K, -1)[s_j * K + s_k]
        U = U + (self.dihedral_basis(tau) * w_d).sum(-1).sum(-1)

        # eps 与相邻态耦合
        U = U + self.eps.reshape(-1)[cls_all * K + s].sum(-1)
        U = U + self.w_nn.reshape(K * K)[s[:, :-1] * K + s[:, 1:]].sum(-1)

        # 非局部 pair（态依赖，对称化）
        pi = self.pair_index
        dp = torch.norm(R[:, pi[:, 0]] - R[:, pi[:, 1]], dim=-1)         # (B,np)
        phi_p = self.pair_basis(dp)
        bkt = self.pair_bucket                                           # (np,)
        w_ps = 0.5 * (self.w_pair_state + self.w_pair_state.transpose(1, 2))
        idx = (bkt[None, :] * K * K + s[:, pi[:, 0]] * K + s[:, pi[:, 1]])
        U = U + (phi_p * w_ps.reshape(-1, phi_p.shape[-1])[idx]).sum(-1).sum(-1)

        # 氨基酸类型 pair（仅 bucket==2，对称化）
        if self.w_pair_type is not None and self.pair_type_rows.numel():
            rows = self.pair_type_rows
            pr = pi[rows]                                                # (npt,2)
            dp2 = dp[:, rows]
            idx2 = a[:, pr[:, 0]] * N_AA + a[:, pr[:, 1]]                # (B,npt)
            w_pt = 0.5 * (self.w_pair_type + self.w_pair_type.transpose(0, 1))
            U = U + (self.pair_basis(dp2)
                     * w_pt.reshape(N_AA * N_AA, -1)[idx2]).sum(-1).sum(-1)

        return U + self.priors(R)

    # ------------------------------------------------------------------ #
    # 能量分项（统一设计文档 §10：U = U_geom + U_conf + U_chem + ...）
    # ------------------------------------------------------------------ #
    def energy_terms(self, R: torch.Tensor, s: torch.Tensor,
                     a: torch.Tensor) -> dict[str, torch.Tensor]:
        """把能量分解为各项（每项 (B,)）；sum(terms) 与 energy() 数学恒等
        （求和路径不同，浮点意义下允许 ~1e-12 相对差）。"""
        dtype = self.w_bond.dtype
        R = R.to(dtype)
        s = self.prepare_states(s)
        a = self._normalize_a(R, a)
        K = self.config.n_states
        cls_all = self.res_class_of_aa[a]

        bi = self.bond_index
        d = torch.norm(R[:, bi[:, 0]] - R[:, bi[:, 1]], dim=-1)
        terms = {"bond": (self.bond_basis(d) * self.w_bond).sum(-1).sum(-1)}

        ai = self.angle_index
        ba = R[:, ai[:, 0]] - R[:, ai[:, 1]]
        bc = R[:, ai[:, 2]] - R[:, ai[:, 1]]
        theta = torch.atan2(torch.linalg.vector_norm(torch.cross(ba, bc, dim=-1), dim=-1),
                            (ba * bc).sum(-1))
        w_a = self.w_angle.reshape(N_RES_CLASSES * K, -1)[cls_all[:, ai[:, 1]] * K
                                                         + s[:, ai[:, 1]]]
        terms["angle"] = (self.angle_basis(theta) * w_a).sum(-1).sum(-1)

        di = self.dihedral_index
        tau = _torch_dihedral(R[:, di[:, 0]], R[:, di[:, 1]],
                              R[:, di[:, 2]], R[:, di[:, 3]])
        w_d = self.w_dihed.reshape(K * K, -1)[s[:, di[:, 1]] * K + s[:, di[:, 2]]]
        terms["dihedral"] = (self.dihedral_basis(tau) * w_d).sum(-1).sum(-1)

        terms["eps"] = self.eps.reshape(-1)[cls_all * K + s].sum(-1)
        terms["w_nn"] = self.w_nn.reshape(K * K)[s[:, :-1] * K + s[:, 1:]].sum(-1)

        pi = self.pair_index
        dp = torch.norm(R[:, pi[:, 0]] - R[:, pi[:, 1]], dim=-1)
        w_ps = 0.5 * (self.w_pair_state + self.w_pair_state.transpose(1, 2))
        idx = (self.pair_bucket[None, :] * K * K + s[:, pi[:, 0]] * K + s[:, pi[:, 1]])
        terms["pair_state"] = (self.pair_basis(dp)
                               * w_ps.reshape(-1, dp.shape[-1])[idx]).sum(-1).sum(-1)

        if self.w_pair_type is not None and self.pair_type_rows.numel():
            rows = self.pair_type_rows
            idx2 = a[:, pi[rows][:, 0]] * N_AA + a[:, pi[rows][:, 1]]
            w_pt = 0.5 * (self.w_pair_type + self.w_pair_type.transpose(0, 1))
            terms["pair_type"] = (self.pair_basis(dp[:, rows])
                                  * w_pt.reshape(N_AA * N_AA, -1)[idx2]).sum(-1).sum(-1)
        else:
            terms["pair_type"] = torch.zeros_like(terms["bond"])

        terms["prior"] = self.priors(R).to(terms["bond"].dtype)
        return terms

    # ------------------------------------------------------------------ #
    # 突变能量学（统一设计文档 §29.2/§29.3）
    # ------------------------------------------------------------------ #
    def _site_theta(self, R: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        """顶点为 site 的键角（每 replica 一项；首尾无角度处为 0），(B,)。"""
        B = R.shape[0]
        rows = self.vertex_angle_index[site]
        valid = rows >= 0
        tri = self.angle_index[rows.clamp_min(0)]                    # (B,3)
        ba = R[torch.arange(B, device=R.device), tri[:, 0]] - R[
            torch.arange(B, device=R.device), tri[:, 1]]
        bc = R[torch.arange(B, device=R.device), tri[:, 2]] - R[
            torch.arange(B, device=R.device), tri[:, 1]]
        theta = torch.atan2(torch.linalg.vector_norm(torch.cross(ba, bc, dim=-1), dim=-1),
                            (ba * bc).sum(-1))
        return torch.where(valid, theta, torch.zeros_like(theta))

    def _site_pair_basis(self, R: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        """site 类型对伙伴距离的 RBF 基：(B, Pmax, n_rbf)（无伙伴处为 0）。"""
        B = R.shape[0]
        prow = self.pt_pair_row[site]                                # (B,Pmax)
        pi = self.pair_index
        dp = torch.norm(R[:, pi[:, 0]] - R[:, pi[:, 1]], dim=-1)     # (B,np)
        basis = self.pair_basis(dp[torch.arange(B, device=R.device)[:, None], prow])
        return basis * (self.pt_partners[site] >= 0)[:, :, None].to(basis.dtype)

    def mutation_delta_u_all_aa(self, R: torch.Tensor, s: torch.Tensor,
                                a: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        """位点 site 上 20 种替换的精确 ΔU（只重算 a 依赖项），(B, N_AA)。

        替换为当前氨基酸的列恒为 0（同 gather 逐位相消）。§29.3 批量打分。
        """
        B = R.shape[0]
        site = site.to(device=R.device, dtype=torch.long)
        a = self._normalize_a(R, a)
        s = self.prepare_states(s)
        K = self.config.n_states
        cls_c = self.res_class_of_aa                                 # (20,)
        aa_site = a[torch.arange(B, device=R.device), site]          # (B,)
        s_site = s[torch.arange(B, device=R.device), site]           # (B,)
        cls_old = self.res_class_of_aa[aa_site]

        # 角度（顶点 = site）
        theta = self._site_theta(R, site)                            # (B,)
        basis_t = self.angle_basis(theta)                            # (B,nrbf)
        w_new = self.w_angle.reshape(N_RES_CLASSES * K, -1)[
            cls_c[None, :] * K + s_site[:, None]]                    # (B,20,nrbf)
        w_old = self.w_angle.reshape(N_RES_CLASSES * K, -1)[
            cls_old * K + s_site]                                    # (B,nrbf)
        d_angle = (basis_t[:, None, :] * (w_new - w_old[:, None, :])).sum(-1)

        # eps（onsite）
        eps_flat = self.eps.reshape(-1)
        d_eps = eps_flat[cls_c[None, :] * K + s_site[:, None]] \
            - eps_flat[cls_old * K + s_site][:, None]

        # 类型对（bucket==2，涉及 site 的伙伴）
        basis_p = self._site_pair_basis(R, site)                     # (B,Pmax,nrbf)
        partner = self.pt_partners[site].clamp_min(0)                # (B,Pmax)
        aa_p = a[torch.arange(B, device=R.device)[:, None], partner]  # (B,Pmax)
        if self.w_pair_type is not None:
            w_pt = 0.5 * (self.w_pair_type + self.w_pair_type.transpose(0, 1))
            w_flat = w_pt.reshape(N_AA * N_AA, -1)
            idx_new = cls_c[None, :, None] * N_AA + aa_p[:, None, :]  # (B,20,Pmax)
            idx_old = (aa_site * N_AA + aa_p)[:, None, :]             # (B,1,Pmax)
            d_pair = (basis_p[:, None, :, :]
                      * (w_flat[idx_new] - w_flat[idx_old])).sum(-1).sum(-1)
        else:
            d_pair = torch.zeros_like(d_angle)

        return d_angle + d_eps + d_pair

    def mutation_delta_u_local(self, R: torch.Tensor, s: torch.Tensor,
                               a: torch.Tensor, site: torch.Tensor,
                               aa_new: torch.Tensor) -> torch.Tensor:
        """单位点精确 ΔU 的局部路径（§29.2）：每 replica 一个新氨基酸，(B,)。

        与全能量差分 mutation_delta_u_full 数学恒等（浮点路径不同）。
        """
        B = R.shape[0]
        site = site.to(device=R.device, dtype=torch.long)
        aa_new = aa_new.to(device=R.device, dtype=torch.long)
        K = self.config.n_states
        a = self._normalize_a(R, a)
        s = self.prepare_states(s)
        ar = torch.arange(B, device=R.device)
        aa_old = a[ar, site]
        cls_new = self.res_class_of_aa[aa_new]
        cls_old = self.res_class_of_aa[aa_old]
        s_site = s[ar, site]

        theta = self._site_theta(R, site)
        w_flat = self.w_angle.reshape(N_RES_CLASSES * K, -1)
        d_angle = ((self.angle_basis(theta)
                    * (w_flat[cls_new * K + s_site] - w_flat[cls_old * K + s_site]))
                   .sum(-1))
        eps_flat = self.eps.reshape(-1)
        d_eps = eps_flat[cls_new * K + s_site] - eps_flat[cls_old * K + s_site]

        basis_p = self._site_pair_basis(R, site)
        partner = self.pt_partners[site].clamp_min(0)
        aa_p = a[ar[:, None], partner]
        if self.w_pair_type is not None:
            w_pt = 0.5 * (self.w_pair_type + self.w_pair_type.transpose(0, 1))
            w_flat2 = w_pt.reshape(N_AA * N_AA, -1)
            d_pair = (basis_p * (w_flat2[aa_new[:, None] * N_AA + aa_p]
                                 - w_flat2[aa_old[:, None] * N_AA + aa_p])
                      ).sum(-1).sum(-1)
        else:
            d_pair = torch.zeros_like(d_angle)
        return d_angle + d_eps + d_pair

    def mutation_delta_u_full(self, R: torch.Tensor, s: torch.Tensor,
                              a: torch.Tensor, site: torch.Tensor,
                              aa_new: torch.Tensor) -> torch.Tensor:
        """全能量差分参考实现（慢路径，供测试/校验）：U(R,s,a') - U(R,s,a)。"""
        B = R.shape[0]
        a = self._normalize_a(R, a)
        site = site.to(device=R.device, dtype=torch.long)
        aa_new = aa_new.to(device=R.device, dtype=torch.long)
        ar = torch.arange(B, device=R.device)
        a_new = a.clone()
        a_new[ar, site] = aa_new
        return self.energy(R, s, a_new) - self.energy(R, s, a)

    # ------------------------------------------------------------------ #
    # 力
    # ------------------------------------------------------------------ #
    def forces(self, R: torch.Tensor, s: torch.Tensor, a: torch.Tensor,
               create_graph: bool = False) -> torch.Tensor:
        """等于 -∂U/∂R，(B,N,3)。

        内部强制 enable_grad：调用方处于 torch.no_grad() 中也能求力
        （此时结果不带参数图）。
        """
        dtype = self.w_bond.dtype
        with torch.enable_grad():
            R = R.to(dtype).detach().clone().requires_grad_(True)
            U = self.energy(R, s, a)
            grad = torch.autograd.grad(U.sum(), R, create_graph=create_graph)[0]
        if not create_graph:
            grad = grad.detach()
        return -grad

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def save(self, path, meta: dict | None = None) -> None:
        save_model(path, self, meta=meta)


def save_model(path, model: SOCGModel, meta: dict | None = None) -> None:
    torch.save({
        "config": asdict(model.config),
        "topology_json": model.topo.to_json(),
        "priors_state": model.priors.state_dict(),
        "model_state": model.state_dict(),
        "meta": meta if meta is not None else {},
    }, str(path))


# v0 -> v1 兼容：这些键是新增的常量缓冲，旧 checkpoint 里没有，模型默认值即正确值
_COMPAT_MISSING_OK = frozenset({"res_class_of_aa"})


def load_model(path, map_location=None) -> SOCGModel:
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    topo = CGTopology.from_json(ckpt["topology_json"])
    config = ModelConfig(**ckpt["config"])
    priors = Priors(topo)
    priors.load_state_dict(ckpt["priors_state"])
    model = SOCGModel(topo, priors, config)
    # load_state_dict 会按现有张量的 dtype 强转；先对齐成保存时的 dtype，保证往返逐位一致
    state = dict(ckpt["model_state"])
    # v0 checkpoint 含 res_class/aa_index 序列缓冲（统一设计文档 §8 已移除）：
    # 丢弃多余键，序列改为运行时传入
    own = set(model.state_dict().keys())
    dropped = sorted(k for k in state if k not in own)
    if dropped:
        warnings.warn(
            f"checkpoint {path} contains legacy sequence buffers not in this "
            f"model version, dropping: {dropped}; sequence is now a runtime "
            f"argument (energy(R, s, a))", stacklevel=2)
        state = {k: v for k, v in state.items() if k in own}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if name in state and state[name].dtype != t.dtype:
            t.data = t.data.to(state[name].dtype)
    result = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if k not in _COMPAT_MISSING_OK]
    if bad_missing:
        raise RuntimeError(f"checkpoint {path} is missing required keys: {bad_missing}")
    model._meta = dict(ckpt.get("meta", {}))
    return model
