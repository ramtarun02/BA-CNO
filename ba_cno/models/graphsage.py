from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class PointEncoder(nn.Module):
    """Small pointwise encoder for per-node input features."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _compute_knn_single(pos: torch.Tensor, k: int, chunk_size: int | None = None) -> torch.Tensor:
    """Compute k-NN indices for one point cloud without materializing the full NxN matrix."""

    num_points = pos.shape[0]
    k = max(1, min(int(k), num_points - 1))

    if chunk_size is None:
        max_pairwise_entries = 4_000_000
        chunk_size = max(1, min(num_points, max_pairwise_entries // max(num_points, 1)))

    neighbour_chunks = []
    for start in range(0, num_points, chunk_size):
        stop = min(start + chunk_size, num_points)
        dist = torch.cdist(pos[start:stop], pos)
        row_idx = torch.arange(stop - start, device=pos.device)
        self_idx = torch.arange(start, stop, device=pos.device)
        dist[row_idx, self_idx] = float("inf")
        neighbour_chunks.append(dist.topk(k, dim=-1, largest=False).indices)

    return torch.cat(neighbour_chunks, dim=0)


def compute_knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Compute batched k-NN indices from positions shaped ``(B, N, 3)``."""

    if pos.dim() != 3:
        raise ValueError(f"Expected pos with shape (B, N, 3), got {tuple(pos.shape)}")

    _, num_points, _ = pos.shape
    if num_points < 2:
        raise ValueError("At least two points are required to build a graph.")

    return torch.stack([_compute_knn_single(batch_pos, k) for batch_pos in pos], dim=0)


def knn_to_edge_index(neighbour_idx: torch.Tensor) -> torch.Tensor:
    """Convert batched neighbour indices shaped ``(B, N, K)`` into ``edge_index``."""

    if neighbour_idx.dim() == 2:
        neighbour_idx = neighbour_idx.unsqueeze(0)
    if neighbour_idx.dim() != 3:
        raise ValueError(f"Expected neighbour_idx with shape (B, N, K), got {tuple(neighbour_idx.shape)}")

    batch_size, num_nodes, k = neighbour_idx.shape
    offsets = torch.arange(batch_size, device=neighbour_idx.device).view(batch_size, 1, 1) * num_nodes
    source = torch.arange(num_nodes, device=neighbour_idx.device).view(1, num_nodes, 1).expand(batch_size, -1, k)

    source = (source + offsets).reshape(-1)
    target = (neighbour_idx + offsets).reshape(-1)
    edge_index = torch.stack([source, target], dim=0)
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


class ResidualBlock(nn.Module):
    """Residual GraphSAGE block optimized for sparse point-cloud graphs."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = SAGEConv(hidden_dim, hidden_dim, aggr="mean", normalize=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.norm1(x + h)

        h = self.ffn(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.norm2(x + h)
        return F.relu(x)


class BaselineGNN(nn.Module):
    """Sparse GraphSAGE baseline for node-wise multi-step CFD prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        k_neighbors: int = 12,
        output_steps: int = 5,
        predict_pressure: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.k_neighbors = int(k_neighbors)
        self.output_steps = int(output_steps)
        self.predict_pressure = bool(predict_pressure)
        self.dropout = float(dropout)

        self.encoder = PointEncoder(self.in_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.hidden_dim, dropout=self.dropout) for _ in range(self.num_layers)]
        )

        output_channels = 3 + (1 if self.predict_pressure else 0)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_steps * output_channels),
        )

    def forward(
        self,
        input_features: torch.Tensor,
        pos: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_features.dim() == 3:
            if input_features.shape[0] != 1:
                raise ValueError(
                    "Dense batched inputs are only supported for batch size 1. "
                    f"Got shape {tuple(input_features.shape)}"
                )
            input_features = input_features[0]
            if pos is not None and pos.dim() == 3:
                pos = pos[0]

        if edge_index is None:
            if pos is None:
                raise ValueError("Either `edge_index` or `pos` must be provided.")

            with torch.no_grad():
                knn_pos = pos.detach()
                if knn_pos.dim() == 2:
                    knn_pos = knn_pos.unsqueeze(0)
                use_cpu_knn = knn_pos.device.type == "mps" and knn_pos.shape[1] >= 8192
                if use_cpu_knn:
                    neighbour_idx = compute_knn(knn_pos.cpu(), self.k_neighbors).to(knn_pos.device)
                else:
                    neighbour_idx = compute_knn(knn_pos, self.k_neighbors)
                edge_index = knn_to_edge_index(neighbour_idx)

            if input_features.dim() == 3:
                input_features = input_features.reshape(-1, input_features.shape[-1])
        elif edge_index.dim() != 2:
            raise ValueError(f"Expected edge_index with shape (2, E), got {tuple(edge_index.shape)}")

        x = self.encoder(input_features)
        for block in self.blocks:
            x = block(x, edge_index)

        decoded = self.decoder(x)
        num_nodes = decoded.shape[0]

        if self.predict_pressure:
            decoded = decoded.view(num_nodes, self.output_steps, 4)
            return {
                "velocity": decoded[..., :3].contiguous(),
                "pressure": decoded[..., 3].contiguous(),
            }

        velocity = decoded.view(num_nodes, self.output_steps, 3)
        return {"velocity": velocity.contiguous()}

