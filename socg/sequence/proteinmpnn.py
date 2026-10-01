"""ProteinMPNN 提议算子适配层（统一设计文档 §5.2/§15/§19，Phase 3）。

角色锁定：ProteinMPNN 只做**学习型提议算子** q(a'|R,a)，目标分布永远是
TANGRAM 的 exp(-βU)。本模块包含

- ProteinMPNNBackend：对 ProteinMPNN 官方实现的薄包装（可选依赖，懒加载；
  需要仓库路径与权重路径）；
- ProteinMPNNProposal：SequenceProposal 实现。单位点提议采用
  "被突变位点最后解码"的固定策略：q(a_i'|R, a_¬i) 由一次前向的 logits 行
  给出；反向概率 q(a_i|R, a'_¬i) 的条件（R 与其余位点、解码策略）与之完全
  相同，因此前向/反向共享同一行 logits，Hastings 修正近乎免费（§6.1）。

换算约定：TANGRAM 内部 nm；ProteinMPNN 使用 Å——backend 接口收 nm，内部 ×10。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .backbone import pseudo_backbone
from .base import SequenceProposal

_TANGRAM_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_PRESETS = {"vanilla": "vanilla_model_weights.pt",
            "soluble": "soluble_model_weights.pt"}


class ProteinMPNNBackend:
    """ProteinMPNN 官方实现的懒加载包装。

    repo_dir: ProteinMPNN 仓库根（含 protein_mpnn_utils.py）；
    checkpoint: "vanilla" | "soluble" 预设或权重 .pt 路径。
    """

    def __init__(self, repo_dir: str | Path, checkpoint: str | Path = "vanilla"):
        self.repo_dir = Path(repo_dir)
        self.checkpoint = Path(checkpoint) if str(checkpoint).endswith(".pt") else checkpoint
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        try:
            import protein_mpnn_utils  # 可选依赖，懒加载
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ProteinMPNN adapter requires the reference implementation: "
                "clone https://github.com/dauparas/ProteinMPNN and pass its "
                f"directory as repo_dir (got {self.repo_dir!r})") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.checkpoint).endswith(".pt"):
            ckpt = torch.load(str(self.checkpoint), map_location=device,
                              weights_only=False)
        else:
            if str(self.checkpoint) not in _PRESETS:
                raise ValueError(f"unknown checkpoint preset {self.checkpoint!r}")
            ckpt = torch.load(str(self.repo_dir / "vanilla_model_weights"
                                  / _PRESETS[str(self.checkpoint)]),
                              map_location=device, weights_only=False)
        model_state, letters = ckpt["model_state_dict"], ckpt["letters"]
        args = protein_mpnn_utils.ArgParser(defaults=True)   # 官方默认超参
        hidden_dim = 128 * args.num_edges
        model = protein_mpnn_utils.ProteinMPNN(
            num_letters=len(letters), node_features=hidden_dim,
            edge_features=hidden_dim, num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            augment_eps=args.backbone_noise, k_neighbors=args.hidden_dim)
        model.load_state_dict(model_state)
        model.to(device).eval()
        self._model = model
        self._letters = letters
        self._device = device
        self._aa_order = {ch: i for i, ch in enumerate(letters)}

    # ------------------------------------------------------------------ #
    def _decode_logits(self, X, mask, chain_mask, chain_idx, S, order, bias):
        """官方模型的单前向调用接缝。

        签名以 protein_mpnn_utils.ProteinMPNN.forward 为准
        （(X, S, mask, chain_mask, chain_idx, order, bias) 形参名/顺序
        随版本略有差异），接入时按所装版本核对一次即可。
        返回 (B, N, num_letters) 的 log-probs。
        """
        return self._model(X, S, mask, chain_mask, chain_idx, order, bias)

    def conditional_logits(self, backbone_nm: torch.Tensor, a: torch.Tensor,
                           site: torch.Tensor) -> torch.Tensor:
        """位点 site 的条件 log-probs：(B, 20)（TANGRAM 标准字母表）。

        backbone_nm: (B, N, 4, 3) [N, CA, C, O]，单位 nm；
        a: (B, N) 当前序列（TANGRAM 索引）；site: (B,) 最后解码位点。
        """
        self._ensure_loaded()
        device = self._device
        B, N = a.shape
        X = backbone_nm.to(device).detach() * 10.0            # nm -> Å
        mask = torch.ones(B, N, device=device)
        chain_mask = torch.zeros(B, N, device=device)         # 0 = 可设计
        chain_idx = torch.zeros(B, N, device=device, dtype=torch.long)
        # 解码顺序：随机置换「其余 N-1 个位点」，site 固定排最后
        other = torch.argsort(torch.rand(B, N, device=device), dim=-1)
        is_site = other == site[:, None]
        other[is_site] = N + 1                                # 把 site 挤出前段
        other.sort(dim=-1)
        order = torch.cat([other[:, :-1] % N, site[:, None].to(device)], dim=1)
        bias = torch.zeros(B, N, len(self._letters), device=device)
        idx_map = torch.as_tensor([self._aa_order[ch] for ch in _TANGRAM_ALPHABET],
                                  device=device)
        S = idx_map[a.to(device)].long()
        with torch.no_grad():
            logits = self._decode_logits(X, mask, chain_mask, chain_idx, S,
                                         order, bias)
        out = logits[torch.arange(B, device=device), site.to(device)]
        # ProteinMPNN 字母表 -> TANGRAM 20 种标准氨基酸（官方字母表缺的补 -inf）
        full = torch.full((B, 20), float("-inf"), device=out.device,
                          dtype=out.dtype)
        for t_idx, ch in enumerate(_TANGRAM_ALPHABET):
            if ch in self._aa_order:
                full[:, t_idx] = out[:, self._aa_order[ch]]
        return full


class ProteinMPNNProposal(SequenceProposal):
    """SequenceProposal 实现：q(a_i'|R, a_¬i) ∝ exp(conditional_logits)。

    geometry: callable(R) -> (B,N,4,3)。CG 状态只有 Cα 时传
    ``cg_geometry_provider(s)``（内部用内部态 s 重建伪骨架，§19 Option 2）。
    mutable_mask: (N,) 或 (B,N) bool，None = 全部可变。
    """

    def __init__(self, backend: ProteinMPNNBackend, geometry,
                 mutable_mask: torch.Tensor | None = None):
        self.backend = backend
        self.geometry = geometry
        self.mutable_mask = mutable_mask
        self._cache: tuple[int, torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ #
    def _site_logits(self, R: torch.Tensor, a: torch.Tensor,
                     site: torch.Tensor) -> torch.Tensor:
        backbone = self.geometry(R)
        logits = self.backend.conditional_logits(backbone, a, site)
        if self.mutable_mask is not None:
            m = self.mutable_mask
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(a.shape[0], -1)
            rows = m[torch.arange(a.shape[0], device=a.device), site]
            logits = logits.masked_fill(~rows[:, None], float("-inf"))
        return logits

    def logits(self, R: torch.Tensor, a: torch.Tensor,
               mutable_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError(
            "ProteinMPNN proposal is site-conditional; use sample()/log_q()")

    def sample(self, R: torch.Tensor, a: torch.Tensor, site: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
        logits = self._site_logits(R, a, site)
        probs = torch.softmax(logits, dim=-1)
        new = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        self._cache = (a.clone(), site.clone(), logits)  # 供两个方向的 log_q 复用
        return new

    def log_q(self, R: torch.Tensor, a_from: torch.Tensor, a_to: torch.Tensor,
              site: torch.Tensor) -> torch.Tensor:
        """log q(a_to | R, a_from)。

        位点 site 最后解码时，条件分布只依赖 (R, 其余位点)。前向（a_from =
        采样时序列）与反向（a_from = 突变后序列）的上下文完全相同——仅 site
        处的氨基酸不同——因此两者共享同一次前向的同一行 logits（§6.1）。
        缓存命中条件：site 相同且 a_from 与缓存序列仅在 site 处可能不同。
        """
        cached = self._cache
        if cached is not None:
            a_c, site_c, logits = cached
            same_site = torch.equal(site_c, site.to(site_c.device))
            if same_site:
                diff = a_from.to(a_c.device) != a_c
                diff[torch.arange(a_from.shape[0], device=a_c.device), site_c] = False
                if not bool(diff.any()):
                    logits = cached[2]
                else:
                    logits = self._site_logits(R, a_from, site)
            else:
                logits = self._site_logits(R, a_from, site)
        else:
            logits = self._site_logits(R, a_from, site)
        rows = logits[torch.arange(logits.shape[0], device=R.device), site]
        log_z = torch.logsumexp(logits, dim=-1)
        return rows - log_z


def cg_geometry_provider(s: torch.Tensor):
    """构造 geometry callable：从 (R, s) 重建伪骨架（闭包捕获内部态 s）。"""
    def geometry(R: torch.Tensor) -> torch.Tensor:
        return pseudo_backbone(R, s)
    return geometry
