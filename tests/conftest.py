"""共享夹具（T8–T10）：toy 拓扑 / 理想螺旋 / toy 坐标 / 参数随机化。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from socg.model.priors import Priors
from socg.model.socg import ModelConfig, SOCGModel
from socg.topology import CGTopology


def ideal_helix(n: int, radius: float = 0.23, rise: float = 0.15,
                twist_deg: float = 100.0) -> np.ndarray:
    """理想螺旋 Cα 坐标 (n,3)，相邻间距约 0.383 nm。"""
    ang = np.deg2rad(twist_deg) * np.arange(n)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang),
                     rise * np.arange(n)], axis=1).astype(np.float32)


def randomize(model: torch.nn.Module, scale: float, seed: int = 0) -> torch.nn.Module:
    """把所有可学习参数设成 N(0, scale²)。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)
    return model


def aa_index_of(topo_or_model) -> torch.Tensor:
    """取参考序列的运行时 a（(N,) long）。模型或拓扑皆可。"""
    topo = getattr(topo_or_model, "topo", topo_or_model)
    return torch.as_tensor(topo.aa_index, dtype=torch.long)


@pytest.fixture
def toy_topology() -> CGTopology:
    # "AAAAAA"：两性离子，4 个可动位点（首尾冻结）
    return CGTopology.from_sequence("AAAAAA")


@pytest.fixture
def toy_a(toy_topology) -> torch.Tensor:
    return aa_index_of(toy_topology)


@pytest.fixture
def toy_coords(toy_topology) -> np.ndarray:
    rng = np.random.default_rng(20260928)
    base = ideal_helix(toy_topology.n)
    noise = rng.normal(0.0, 0.01, size=(2000, toy_topology.n, 3))
    return (base[None, :, :] + noise).astype(np.float32)


@pytest.fixture
def toy_priors(toy_topology, toy_coords) -> Priors:
    return Priors.fit(toy_topology, toy_coords, 300.0)


@pytest.fixture
def toy_model(toy_topology, toy_priors) -> SOCGModel:
    return SOCGModel(toy_topology, toy_priors, ModelConfig())


@pytest.fixture
def toy_model_double(toy_model) -> SOCGModel:
    return toy_model.double()
