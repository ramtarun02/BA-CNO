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
import struct
import zipfile
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


def mmap_npz(path: str | Path, key: str) -> np.ndarray:
    """Memory-map one array of an uncompressed ``.npz`` (as written by ``np.savez``).

    Reads only the headers; pages are loaded on access, so a cached trajectory
    costs almost no RAM. Copy-on-write, so torch gets a writable buffer.
    """

    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{key} is compressed; cannot memory-map")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(fh)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(fh)
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="c", shape=shape, offset=offset, order="F" if fortran else "C")


def raise_open_file_limit(target: int = 65536) -> None:
    """Lift the soft RLIMIT_NOFILE towards ``target`` (capped by the hard limit).

    Python's mmap duplicates the file descriptor, so each memory-mapped
    trajectory holds two open files; 1000 of them exceed the common 1024 default.
    """

    try:
        import resource
    except ImportError:  # not POSIX
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if soft != resource.RLIM_INFINITY and soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


class CylinderTrajectory:
    """One trajectory. ``mmap=True`` leaves velocity/pressure on disk until indexed."""

    def __init__(self, path: str | Path, mmap: bool = False) -> None:
        data = np.load(path)
        self.path = Path(path)
        self.dt = float(data["dt"]) if "dt" in data else 0.01
        if mmap:
            velocity, pressure = mmap_npz(path, "velocity"), mmap_npz(path, "pressure")
        else:
            velocity, pressure = data["velocity"], data["pressure"]
        self.velocity = torch.from_numpy(velocity) if mmap else torch.as_tensor(velocity, dtype=torch.float32)  # (T, N, 2)
        self.pressure = (torch.from_numpy(pressure) if mmap else torch.as_tensor(pressure, dtype=torch.float32)).view(self.velocity.shape[0], -1)
        self.mesh = build_mesh(data["mesh_pos"], data["node_type"], data["cells"])

    @property
    def num_steps(self) -> int:
        return self.velocity.shape[0]

    @property
    def inflow_speed(self) -> float:
        """Bulk inflow speed U_mean = (1/H) * integral u dy (reference velocity for Strouhal / Reynolds).

        Integrated with the trapezoid rule, with no-slip zeros at the channel walls.
        A plain node average is ~30% low because the inflow nodes cluster near the walls.
        """

        mesh = self.mesh
        inflow = (mesh.node_type == NodeType.INFLOW).nonzero().view(-1)
        y = mesh.pos[inflow, 1].double()
        order = torch.argsort(y)
        y, u = y[order], self.velocity[0, inflow[order], 0].double()
        y_lo, y_hi = float(mesh.pos[:, 1].min()), float(mesh.pos[:, 1].max())
        zero = u.new_zeros(1)
        y = torch.cat([y.new_tensor([y_lo]), y, y.new_tensor([y_hi])])
        u = torch.cat([zero, u, zero])
        return float(torch.trapz(u, y) / (y_hi - y_lo))


def wake_probe_index(mesh: CylinderMesh, diameters_downstream: float = 2.0) -> int:
    """Mesh node nearest the wake centreline, ``diameters_downstream`` D behind the cylinder centre."""

    target = mesh.centre + torch.tensor([diameters_downstream * mesh.diameter, 0.0])
    return int(torch.argmin(torch.linalg.norm(mesh.pos - target, dim=-1)))


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
        mmap: bool = True,
    ) -> None:
        self.files = sorted((Path(root) / split).glob("traj_*.npz"))
        if max_trajectories is not None:
            self.files = self.files[:max_trajectories]
        if not self.files:
            raise FileNotFoundError(f"No trajectories in {Path(root) / split}. Run scripts/cylinder/convert.py first.")
        self.noise_std = float(noise_std)
        self._cache: dict[int, CylinderTrajectory] = {}
        self.use_cache = cache
        self.mmap = mmap
        if mmap and cache:
            raise_open_file_limit()
        first = self.trajectory(0)
        self.steps_per_traj = first.num_steps - 1

    def trajectory(self, idx: int) -> CylinderTrajectory:
        if idx in self._cache:
            return self._cache[idx]
        traj = CylinderTrajectory(self.files[idx], mmap=self.mmap)
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
