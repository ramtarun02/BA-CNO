from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .gram_dataset import GRAMDataset, SampleRecord

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional fallback
    cKDTree = None

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import coalesce
except ImportError as exc:  # pragma: no cover - training extra required
    raise ImportError(
        "GRAMGraphDataset requires torch-geometric. Install the training extras first."
    ) from exc


def build_knn_edge_index(
    pos: np.ndarray,
    k_neighbors: int,
    *,
    undirected: bool = False,
) -> torch.Tensor:
    """Build a sparse k-NN edge list for one point cloud."""

    pos = np.asarray(pos, dtype=np.float32)
    num_nodes = int(pos.shape[0])
    if num_nodes < 2:
        raise ValueError("At least two nodes are required to build a graph.")

    k_neighbors = max(1, min(int(k_neighbors), num_nodes - 1))

    if cKDTree is not None:
        tree = cKDTree(pos)
        _, neighbours = tree.query(pos, k=k_neighbors + 1)
        neighbours = np.asarray(neighbours, dtype=np.int64)[:, 1:]
    else:  # pragma: no cover - slow fallback
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        neighbours = np.argpartition(dist, kth=k_neighbors, axis=1)[:, :k_neighbors]

    source = np.repeat(np.arange(num_nodes, dtype=np.int64), k_neighbors)
    target = neighbours.reshape(-1)
    edge_index = torch.from_numpy(np.stack([source, target], axis=0))

    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes, reduce="mean")
    return edge_index.long().contiguous()


class GRAMGraphDataset(GRAMDataset):
    """Load GRaM samples as sparse PyG graphs for node-level mini-batch training."""

    def __init__(
        self,
        data_root: str | Path,
        records: Optional[list[SampleRecord]] = None,
        *,
        k_neighbors: int = 8,
        graph_cache_dir: str | Path | None = None,
        undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_root, records=records, **kwargs)
        self.k_neighbors = int(k_neighbors)
        self.undirected = bool(undirected)
        self.graph_cache_dir = Path(graph_cache_dir) if graph_cache_dir is not None else None
        if self.graph_cache_dir is not None:
            self.graph_cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> Data:
        sample = super().__getitem__(index)
        record = self.records[index]
        edge_index = self._load_or_build_edge_index(record, sample)

        graph = Data(
            x=torch.as_tensor(np.asarray(sample["input_features"], dtype=np.float32)),
            edge_index=edge_index,
            pos=torch.as_tensor(np.asarray(sample["pos"], dtype=np.float32)),
        )
        graph.y_velocity = torch.as_tensor(
            np.asarray(sample["velocity_out"], dtype=np.float32)
        ).permute(1, 0, 2).contiguous()

        pressure = sample.get("pressure")
        if pressure is not None:
            pressure = np.asarray(pressure, dtype=np.float32)
            target_pressure = pressure[5:] if pressure.shape[0] >= 10 else pressure[-5:]
            graph.y_pressure = torch.as_tensor(target_pressure).transpose(0, 1).contiguous()

        graph.airfoil_mask = torch.as_tensor(
            np.asarray(sample["airfoil_mask"], dtype=np.bool_),
            dtype=torch.bool,
        )
        graph.file_name = record.file_name
        graph.simulation_id = int(record.simulation_id)
        graph.metadata = dict(sample.get("metadata", {}))
        return graph

    def _load_or_build_edge_index(self, record: SampleRecord, sample: dict[str, Any]) -> torch.Tensor:
        cache_path = self._cache_path(record, int(sample["pos"].shape[0]))
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu").long().contiguous()

        edge_index = build_knn_edge_index(
            sample["pos"],
            self.k_neighbors,
            undirected=self.undirected,
        )
        if cache_path is not None:
            torch.save(edge_index.to(torch.int32).cpu(), cache_path)
        return edge_index.long().contiguous()

    def _cache_path(self, record: SampleRecord, num_nodes: int) -> Path | None:
        if self.graph_cache_dir is None:
            return None

        stem = Path(record.file_path).stem
        cache_name = (
            f"{stem}__nodes{num_nodes}__k{self.k_neighbors}"
            f"__undir{int(self.undirected)}.pt"
        )
        return self.graph_cache_dir / cache_name
