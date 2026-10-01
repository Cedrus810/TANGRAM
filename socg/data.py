"""CG 轨迹与数据集：npz 读写、按轨迹划分、训练抽帧。

一条“轨迹”= 一个 replica 的连续时间序列；任何时间相关计算都不得跨轨迹拼接。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import CGTopology


@dataclass
class CGTrajectory:
    """coords (T,N,3) f32 nm；forces (T,N,3) f32 或 None；states (T,N) int64。"""

    coords: np.ndarray
    forces: np.ndarray | None
    states: np.ndarray
    dt_ps: float

    def __post_init__(self) -> None:
        self.coords = np.ascontiguousarray(self.coords, dtype=np.float32)
        self.states = np.ascontiguousarray(self.states, dtype=np.int64)
        if self.coords.ndim != 3:
            raise ValueError(f"coords must be (T,N,3), got {self.coords.shape}")
        T, N, _ = self.coords.shape
        if self.states.shape != (T, N):
            raise ValueError(f"states shape {self.states.shape} != {(T, N)}")
        if self.forces is not None:
            self.forces = np.ascontiguousarray(self.forces, dtype=np.float32)
            if self.forces.shape != (T, N, 3):
                raise ValueError(f"forces shape {self.forces.shape} != {(T, N, 3)}")
        if not (self.dt_ps > 0):
            raise ValueError(f"dt_ps must be positive, got {self.dt_ps}")

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_res(self) -> int:
        return int(self.coords.shape[1])

    @property
    def duration_ps(self) -> float:
        return self.n_frames * self.dt_ps


@dataclass
class CGDataset:
    topology: CGTopology
    temperature: float
    trajectories: list[CGTrajectory] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # IO（npz，禁止 pickle）
    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        arrays: dict[str, np.ndarray] = {
            "topology_json": np.asarray(self.topology.to_json()),
            "temperature": np.asarray(self.temperature, dtype=np.float64),
            "n_trajs": np.asarray(len(self.trajectories), dtype=np.int64),
        }
        for i, trj in enumerate(self.trajectories):
            arrays[f"coords_{i}"] = trj.coords
            arrays[f"states_{i}"] = trj.states
            arrays[f"dt_{i}"] = np.asarray(trj.dt_ps, dtype=np.float64)
            arrays[f"has_forces_{i}"] = np.asarray(trj.forces is not None, dtype=bool)
            if trj.forces is not None:
                arrays[f"forces_{i}"] = trj.forces
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path) -> "CGDataset":
        with np.load(str(path), allow_pickle=False) as z:
            topo = CGTopology.from_json(str(z["topology_json"].item()
                                            if hasattr(z["topology_json"], "item")
                                            else z["topology_json"]))
            temperature = float(z["temperature"])
            n = int(z["n_trajs"])
            trajs = []
            for i in range(n):
                forces = z[f"forces_{i}"] if bool(z[f"has_forces_{i}"]) else None
                trajs.append(CGTrajectory(
                    coords=z[f"coords_{i}"],
                    forces=forces,
                    states=z[f"states_{i}"],
                    dt_ps=float(z[f"dt_{i}"]),
                ))
        return cls(topology=topo, temperature=temperature, trajectories=trajs)

    # ------------------------------------------------------------------ #
    # 划分与抽帧
    # ------------------------------------------------------------------ #
    def split(self, val_indices: list[int]) -> tuple["CGDataset", "CGDataset"]:
        """按轨迹划分（不按帧）。"""
        n = len(self.trajectories)
        val_indices = list(val_indices)
        for i in val_indices:
            if not 0 <= i < n:
                raise ValueError(f"val index {i} out of range [0, {n})")
        val_set = set(val_indices)
        train = [self.trajectories[i] for i in range(n) if i not in val_set]
        val = [self.trajectories[i] for i in val_indices]
        return (CGDataset(self.topology, self.temperature, train),
                CGDataset(self.topology, self.temperature, val))

    def frames(self, stride: int = 1):
        """按轨迹各自抽帧后再拼接（仅供训练用）。

        Returns
        -------
        (coords (M,N,3), forces (M,N,3) 或 None, states (M,N))
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")
        coords, states, forces = [], [], []
        has_forces = all(t.forces is not None for t in self.trajectories) and len(self.trajectories) > 0
        for t in self.trajectories:
            coords.append(t.coords[::stride])
            states.append(t.states[::stride])
            if has_forces:
                forces.append(t.forces[::stride])
        coords = np.concatenate(coords, axis=0) if coords else np.zeros((0, 0, 3), np.float32)
        states = np.concatenate(states, axis=0) if states else np.zeros((0, 0), np.int64)
        forces_out = np.concatenate(forces, axis=0) if has_forces else None
        return coords, forces_out, states
