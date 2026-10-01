"""统一设计文档 §19/§31：伪骨架重建与 ProteinMPNN 适配层。

伪骨架（无外部依赖）验证理想几何与态锚点；适配层 MH 通路用注入的
FakeBackend 验证（前向/反向共享 logits、约束掩码、归一化）；真实
ProteinMPNN 后端测试在缺依赖时整体跳过。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from socg.sequence import UniformSequenceProposal
from socg.sequence.backbone import pseudo_backbone_numpy
from socg.sequence.base import SequenceProposal
from socg.sequence.proteinmpnn import ProteinMPNNProposal, cg_geometry_provider


# --------------------------------------------------------------------- #
# 伪骨架重建（Option 2）
# --------------------------------------------------------------------- #
def test_pseudo_backbone_ideal_geometry():
    """键长精确等于理想值（NeRF 构造的内部一致性）。"""
    rng = np.random.default_rng(3)
    n = 12
    s = rng.integers(0, 3, size=n)
    R = rng.normal(0, 0.3, size=(n, 3)) + 1.0
    bb = pseudo_backbone_numpy(R, s)                    # (n,4,3) N,CA,C,O
    assert bb.shape == (n, 4, 3)
    assert np.allclose(np.linalg.norm(bb[:, 0] - bb[:, 1], axis=-1), 0.1458, atol=1e-9)
    assert np.allclose(np.linalg.norm(bb[:, 1] - bb[:, 2], axis=-1), 0.1525, atol=1e-9)
    assert np.allclose(np.linalg.norm(bb[:-1, 2] - bb[1:, 0], axis=-1), 0.1329, atol=1e-9)
    assert np.allclose(np.linalg.norm(bb[:, 2] - bb[:, 3], axis=-1), 0.1231, atol=1e-9)


def test_pseudo_backbone_phi_psi_match_state_anchors():
    """重建链的 φ/ψ 逐残基等于内部态的 Ramachandran 锚点（首残基除外）。"""
    from socg.geometry import backbone_dihedrals
    from socg.states import CORE_CENTERS_DEG

    anchor = {}
    for phi, psi, st in CORE_CENTERS_DEG:
        anchor.setdefault(st, (phi, psi))
    rng = np.random.default_rng(5)
    n = 10
    s = rng.integers(0, 3, size=n)
    R = rng.normal(0, 0.3, size=(n, 3)) + 1.0
    bb = pseudo_backbone_numpy(R, s)
    phi, psi = backbone_dihedrals(bb[:, 1, :])          # CA 轨迹的 φ/ψ（度）
    for i in range(1, n):
        assert phi[i] == pytest.approx(anchor[int(s[i])][0], abs=1e-4)
        assert psi[i] == pytest.approx(anchor[int(s[i])][1], abs=1e-4)


def test_pseudo_backbone_batch_shape_and_alignment():
    """批量接口形状正确；重建 CA 与输入 R 的偏差有限（刚体叠合生效）。"""
    from socg.sequence.backbone import pseudo_backbone
    rng = np.random.default_rng(7)
    B, n = 3, 10
    R = torch.as_tensor(rng.normal(0, 0.3, size=(B, n, 3)) + 1.0)
    s = torch.as_tensor(rng.integers(0, 3, size=(B, n)), dtype=torch.long)
    bb = pseudo_backbone(R, s)
    assert bb.shape == (B, n, 4, 3)
    assert bb.dtype == R.dtype
    # 每条链叠合后 CA 均值偏差有限（不要求逐点贴合，Option 2 本就是近似）
    dev = (bb[:, :, 1, :] - R).norm(dim=-1).mean(dim=-1)
    assert float(dev.max()) < 0.6


# --------------------------------------------------------------------- #
# ProteinMPNN 提议通路（FakeBackend 注入，不依赖官方实现）
# --------------------------------------------------------------------- #
class _FakeBackend:
    """记录调用并返回确定的 logits：位点 site 的分布偏向 (site+aa) mod 20。"""

    def __init__(self):
        self.calls = []

    def conditional_logits(self, backbone_nm, a, site):
        self.calls.append((a.clone(), site.clone()))
        B = a.shape[0]
        logits = torch.full((B, 20), -3.0)
        for b in range(B):
            logits[b, (int(site[b]) + 3) % 20] = 0.0
            logits[b, (int(site[b]) + 7) % 20] = -1.0
        return logits


def test_pmpnn_proposal_forward_reverse_share_logits():
    """单位点前向/反向上下文相同 -> 全程仅一次前向（设计文档 §6.1）。"""
    n = 8
    a = torch.as_tensor(np.random.default_rng(1).integers(0, 20, size=(2, n)))
    R = torch.ones(2, n, 3)
    backend = _FakeBackend()
    prop = ProteinMPNNProposal(backend, geometry=lambda R: R[:, :, None].expand(-1, -1, 4))
    site = torch.tensor([3, 5])
    new = prop.sample(R, a, site, generator=torch.Generator().manual_seed(0))
    a_new = a.clone()
    a_new[torch.arange(2), site] = new
    log_fwd = prop.log_q(R, a, a_new, site)     # 前向 q(a'\u4f4d | a)
    log_rev = prop.log_q(R, a_new, a, site)     # 反向 q(a\u4f4d | a')：上下文只差 site
    assert len(backend.calls) == 1               # 单位点提议全程一次前向

    # 与 FakeBackend 的确定性 logits 手工对照
    ref = torch.full((2, 20), -3.0)
    for b in range(2):
        ref[b, (int(site[b]) + 3) % 20] = 0.0
        ref[b, (int(site[b]) + 7) % 20] = -1.0
    log_norm = torch.logsumexp(ref, dim=-1)
    old = a[torch.arange(2), site]
    assert torch.allclose(log_fwd, ref[torch.arange(2), new] - log_norm)
    assert torch.allclose(log_rev, ref[torch.arange(2), old] - log_norm)


def test_pmpnn_proposal_masks_immutable_sites():
    n = 6
    a = torch.full((1, n), 2, dtype=torch.long)
    R = torch.ones(1, n, 3)
    backend = _FakeBackend()
    mutable = torch.zeros(n, dtype=torch.bool)
    mutable[1] = True
    prop = ProteinMPNNProposal(backend, geometry=lambda R: R[:, :, None].expand(-1, -1, 4),
                               mutable_mask=mutable)
    gen = torch.Generator().manual_seed(2)
    new = torch.tensor([prop.sample(R, a, torch.tensor([1]), generator=gen).item()
                        for _ in range(30)])
    assert not bool((new == 2).all())                     # 可变位点确实在被提议
    logits = prop._site_logits(R, a, torch.tensor([0]))   # 不可变位点 -> 全行 -inf
    assert bool(torch.isinf(logits).all())


def test_cg_geometry_provider_closes_over_states():
    R = torch.ones(2, 10, 3)
    s = torch.zeros(2, 10, dtype=torch.long)
    geometry = cg_geometry_provider(s)
    bb = geometry(R)
    assert bb.shape == (2, 10, 4, 3)
    assert torch.isfinite(bb).all()


def test_real_proteinmpnn_backend_skips_without_dependency():
    """真实后端：无 ProteinMPNN 仓库时给出带指引的 ImportError。"""
    pytest.importorskip("protein_mpnn_utils", reason="ProteinMPNN 未安装")
    from socg.sequence.proteinmpnn import ProteinMPNNBackend
    backend = ProteinMPNNBackend(repo_dir="/nonexistent")
    with pytest.raises(ImportError, match="github.com/dauparas/ProteinMPNN"):
        backend._ensure_loaded()


def test_base_and_uniform_offers_remain_abstract():
    """SequenceProposal 基类不可直接采样；Uniform 提议的 log_q 归一。"""
    prop = SequenceProposal()
    with pytest.raises(NotImplementedError):
        prop.sample(torch.ones(1, 4, 3), torch.zeros(1, 4, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long))
    uni = UniformSequenceProposal()
    a = torch.zeros(2, 5, dtype=torch.long)
    logq = uni.log_q(torch.ones(2, 5, 3), a, a.clone(), torch.zeros(2, dtype=torch.long))
    assert torch.allclose(logq, torch.full((2,), -np.log(19)))
