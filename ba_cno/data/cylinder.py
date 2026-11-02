"""CylinderFlow (MeshGraphNets) adapter.

Data: 2D incompressible flow past a cylinder (COMSOL), 1000/100/100
trajectories x 600 steps, dt = 0.01, ~1.9k mesh nodes, varying cylinder size,
position and inflow. Convert with ``scripts/cylinder/convert.py`` first.

Protocol (kept identical to Pfaff et al. 2021 so RMSE is comparable):
    inputs  : velocity_t (+ N(0, 0.02) noise on NORMAL nodes during training),
              one-hot node type; edge features [dx, dy, |d|]
    targets : v_{t+1} - v_t (on NORMAL and OUTFLOW nodes) and p_{t+1} (auxiliary)
    rollout : 599 steps; inflow / wall nodes are reset to the ground truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..graph import cells_to_edges, edge_geometry, lumped_cell_volume, reverse_edge_permutation


class NodeType:
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


@dataclass
class CylinderMesh:
    """Static per-trajectory graph and geometry, built once."""

    pos: torch.Tensor  # (N, 2)
    node_type: torch.Tensor  # (N,)
    cells: torch.Tensor  # (C, 3)
    edge_index: torch.Tensor  # (2, E)
    reverse_edge: torch.Tensor  # (E,)
    edge_features: torch.Tensor  # (E, 3)
    node_volume: torch.Tensor  # (N,) normalised to mean 1
    loss_mask: torch.Tensor  # (N,) NORMAL | OUTFLOW
    boundary_mask: torch.Tensor  # (N,) not NORMAL
    interior_edge_mask: torch.Tensor  # (E,) both ends NORMAL
    cylinder_mask: torch.Tensor  # (N,) wall nodes on the cylinder (not channel walls)
    cylinder_edges: torch.Tensor  # (2, S) surface segments of the cylinder
    diameter: float
    centre: torch.Tensor  # (2,)

    @property
    def num_nodes(self) -> int:
        return self.pos.shape[0]

    def node_type_onehot(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.node_type, NodeType.SIZE).float()


def build_mesh(pos: np.ndarray, node_type: np.ndarray, cells: np.ndarray) -> CylinderMesh:
    pos_t = torch.as_tensor(pos, dtype=torch.float32)
    nt = torch.as_tensor(node_type, dtype=torch.long).view(-1)
    cells_t = torch.as_tensor(cells, dtype=torch.long)

    edge_index = cells_to_edges(cells_t)
    reverse = reverse_edge_permutation(edge_index, pos_t.shape[0])
    volume = lumped_cell_volume(pos_t, cells_t)
    volume = volume / volume.mean()

    normal = nt == NodeType.NORMAL
    loss_mask = normal | (nt == NodeType.OUTFLOW)
    interior = normal[edge_index[0]] & normal[edge_index[1]]

    # Cylinder wall = WALL_BOUNDARY nodes away from the top/bottom channel walls.
    wall = nt == NodeType.WALL_BOUNDARY
    y = pos_t[:, 1]
    tol = 1e-3 * float(y.max() - y.min())
    cylinder = wall & (y > y.min() + tol) & (y < y.max() - tol)
    cyl_pos = pos_t[cylinder]
    centre = 0.5 * (cyl_pos.max(0).values + cyl_pos.min(0).values) if cylinder.any() else pos_t.mean(0)
    diameter = float((cyl_pos[:, 0].max() - cyl_pos[:, 0].min())) if cylinder.any() else float("nan")

    src, dst = edge_index
    on_surface = cylinder[src] & cylinder[dst] & (src < dst)
    cylinder_edges = edge_index[:, on_surface]

    return CylinderMesh(
        pos=pos_t,
        node_type=nt,
        cells=cells_t,
        edge_index=edge_index,
        reverse_edge=reverse,
        edge_features=edge_geometry(pos_t, edge_index),
        node_volume=volume,
        loss_mask=loss_mask,
        boundary_mask=~normal,
        interior_edge_mask=interior,
        cylinder_mask=cylinder,
        cylinder_edges=cylinder_edges,
        diameter=diameter,
        centre=centre,
    )


class CylinderTrajectory:
    def __init__(self, path: str | Path) -> None:
        data = np.load(path)
        self.path = Path(path)
        self.dt = float(data["dt"]) if "dt" in data else 0.01
        self.velocity = torch.as_tensor(data["velocity"], dtype=torch.float32)  # (T, N, 2)
        self.pressure = torch.as_tensor(data["pressure"], dtype=torch.float32).view(self.velocity.shape[0], -1)
        self.mesh = build_mesh(data["mesh_pos"], data["node_type"], data["cells"])

    @property
    def num_steps(self) -> int:
        return self.velocity.shape[0]

    @property
    def inflow_speed(self) -> float:
        """Mean inflow speed (reference velocity for Strouhal / Reynolds)."""

        inflow = self.mesh.node_type == NodeType.INFLOW
        return float(self.velocity[0, inflow, 0].mean())


def node_inputs(velocity: torch.Tensor, mesh: CylinderMesh) -> torch.Tensor:
    return torch.cat([velocity, mesh.node_type_onehot()], dim=-1)


class CylinderFlowDataset(Dataset):
    """Random one-step training pairs ``(trajectory, t)``."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_trajectories: int | None = None,
        noise_std: float = 0.02,
        cache: bool = True,
    ) -> None:
        self.files = sorted((Path(root) / split).glob("traj_*.npz"))
        if max_trajectories is not None:
            self.files = self.files[:max_trajectories]
        if not self.files:
            raise FileNotFoundError(f"No trajectories in {Path(root) / split}. Run scripts/cylinder/convert.py first.")
        self.noise_std = float(noise_std)
        self._cache: dict[int, CylinderTrajectory] = {}
        self.use_cache = cache
        first = self.trajectory(0)
        self.steps_per_traj = first.num_steps - 1

    def trajectory(self, idx: int) -> CylinderTrajectory:
        if idx in self._cache:
            return self._cache[idx]
        traj = CylinderTrajectory(self.files[idx])
        if self.use_cache:
            self._cache[idx] = traj
        return traj

    def __len__(self) -> int:
        return len(self.files) * self.steps_per_traj

    def __getitem__(self, index: int) -> dict:
        traj_idx, t = divmod(index, self.steps_per_traj)
        traj = self.trajectory(traj_idx)
        mesh = traj.mesh
        v_t = traj.velocity[t]
        if self.noise_std > 0:
            noise = torch.randn_like(v_t) * self.noise_std
            noise[mesh.node_type != NodeType.NORMAL] = 0.0
            v_t = v_t + noise
        target = torch.cat([traj.velocity[t + 1] - v_t, traj.pressure[t + 1].unsqueeze(-1)], dim=-1)
        return {
            "node_features": node_inputs(v_t, mesh),
            "edge_features": mesh.edge_features,
            "edge_index": mesh.edge_index,
            "reverse_edge": mesh.reverse_edge,
            "node_volume": mesh.node_volume,
            "boundary_mask": mesh.boundary_mask,
            "interior_edge_mask": mesh.interior_edge_mask,
            "loss_mask": mesh.loss_mask,
            "target": target,
        }


def collate_graphs(samples: list[dict]) -> dict:
    """Concatenate several graphs into one disconnected graph."""

    out: dict[str, torch.Tensor] = {}
    node_offset = edge_offset = 0
    edge_parts, reverse_parts = [], []
    for s in samples:
        edge_parts.append(s["edge_index"] + node_offset)
        reverse_parts.append(s["reverse_edge"] + edge_offset)
        node_offset += s["node_features"].shape[0]
        edge_offset += s["edge_index"].shape[1]
    out["edge_index"] = torch.cat(edge_parts, dim=1)
    out["reverse_edge"] = torch.cat(reverse_parts)
    for key in samples[0]:
        if key not in ("edge_index", "reverse_edge"):
            out[key] = torch.cat([s[key] for s in samples], dim=0)
    return out


def shuffled_indices(n: int, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx
