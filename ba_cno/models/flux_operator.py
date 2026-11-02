from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphsage import PointEncoder, compute_knn, knn_to_edge_index


def _build_node_boundary_features(
    airfoil_mask: torch.Tensor | None,
    boundary_distance: torch.Tensor | None,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Assemble lightweight geometry features used inside each flux block."""

    num_nodes = pos.shape[0]
    device = pos.device
    dtype = pos.dtype

    if airfoil_mask is None:
        airfoil_feature = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
    else:
        airfoil_feature = airfoil_mask.to(device=device, dtype=dtype).view(num_nodes, 1)

    if boundary_distance is None:
        boundary_feature = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
    else:
        boundary_feature = torch.log1p(boundary_distance.to(device=device, dtype=dtype).view(num_nodes, 1).clamp_min(0.0))

    return torch.cat([airfoil_feature, boundary_feature], dim=-1)


def _antisymmetry_penalty(
    edge_index: torch.Tensor,
    messages: torch.Tensor,
    num_nodes: int,
    airfoil_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softly encourage paired edge messages to be antisymmetric away from boundaries."""

    if edge_index.numel() == 0 or messages.numel() == 0:
        return messages.new_zeros(())

    src, dst = edge_index
    num_edges = src.numel()
    keys = src.to(torch.int64) * int(num_nodes) + dst.to(torch.int64)
    reverse_keys = dst.to(torch.int64) * int(num_nodes) + src.to(torch.int64)

    sorted_keys, perm = torch.sort(keys)
    reverse_pos = torch.searchsorted(sorted_keys, reverse_keys)
    clamped_pos = reverse_pos.clamp(max=max(num_edges - 1, 0))
    matches = (reverse_pos < num_edges) & (sorted_keys[clamped_pos] == reverse_keys)

    if airfoil_mask is not None:
        airfoil_mask = airfoil_mask.to(device=src.device, dtype=torch.bool)
        interior = ~(airfoil_mask[src] | airfoil_mask[dst])
        matches = matches & interior

    pair_idx = torch.nonzero(matches, as_tuple=False).squeeze(-1)
    if pair_idx.numel() == 0:
        return messages.new_zeros(())

    reverse_idx = perm[clamped_pos[pair_idx]]
    unique_pairs = src[pair_idx] < dst[pair_idx]
    pair_idx = pair_idx[unique_pairs]
    reverse_idx = reverse_idx[unique_pairs]

    if pair_idx.numel() == 0:
        return messages.new_zeros(())

    return (messages[pair_idx] + messages[reverse_idx]).pow(2).mean()


class FluxOperatorBlock(nn.Module):
    """Finite-volume-inspired residual block with geometry-aware local flux updates."""

    def __init__(self, hidden_dim: int, edge_hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_hidden_dim = int(edge_hidden_dim or hidden_dim)
        self.dropout = float(dropout)

        pair_dim = 2 * self.hidden_dim + 11
        self.flux_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_dim, self.hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_dim, 1),
        )
        self.self_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 5, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        node_boundary_features: torch.Tensor,
        airfoil_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        src, dst = edge_index

        rel_pos = pos[src] - pos[dst]
        rel_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        rel_dir = rel_pos / (rel_dist + 1e-6)

        pair_features = torch.cat(
            [
                x[dst],
                x[src] - x[dst],
                rel_pos,
                rel_dist,
                rel_dir,
                node_boundary_features[dst],
                node_boundary_features[src],
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.gate_mlp(pair_features))
        messages = gate * self.flux_mlp(pair_features)

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        degree = torch.bincount(dst, minlength=x.shape[0]).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        aggregated = aggregated / degree.clamp_min(1.0)

        self_input = torch.cat([x, node_boundary_features, pos], dim=-1)
        self_update = self.self_mlp(self_input)

        x = self.norm1(x + F.dropout(aggregated + self_update, p=self.dropout, training=self.training))
        h = self.ffn(x)
        x = self.norm2(x + F.dropout(h, p=self.dropout, training=self.training))

        aux_losses = {
            "flux_antisymmetry": _antisymmetry_penalty(
                edge_index=edge_index,
                messages=messages,
                num_nodes=x.shape[0],
                airfoil_mask=airfoil_mask,
            ),
            "mean_gate": gate.mean().detach(),
        }
        return F.relu(x), aux_losses


class DirectMultiHorizonHead(nn.Module):
    """Direct decoder that predicts all future velocity frames in one shot."""

    def __init__(self, hidden_dim: int, output_steps: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.output_steps = int(output_steps)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.output_steps * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        velocity = self.net(x)
        return velocity.view(x.shape[0], self.output_steps, 3)


class FVMNeuralOperator(nn.Module):
    """Geometry-conditioned sparse neural operator with local flux-style updates."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        k_neighbors: int = 8,
        output_steps: int = 5,
        dropout: float = 0.1,
        edge_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.k_neighbors = int(k_neighbors)
        self.output_steps = int(output_steps)
        self.dropout = float(dropout)

        self.encoder = PointEncoder(self.in_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FluxOperatorBlock(
                    hidden_dim=self.hidden_dim,
                    edge_hidden_dim=edge_hidden_dim or self.hidden_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.decoder = DirectMultiHorizonHead(
            hidden_dim=self.hidden_dim,
            output_steps=self.output_steps,
            dropout=self.dropout,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        pos: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        airfoil_mask: torch.Tensor | None = None,
        boundary_distance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if input_features.dim() == 3:
            if input_features.shape[0] != 1:
                raise ValueError(
                    "Dense batched inputs are only supported for batch size 1. "
                    f"Got shape {tuple(input_features.shape)}"
                )
            input_features = input_features[0]
            if pos is not None and pos.dim() == 3:
                pos = pos[0]
            if airfoil_mask is not None and airfoil_mask.dim() == 2:
                airfoil_mask = airfoil_mask[0]
            if boundary_distance is not None and boundary_distance.dim() == 2:
                boundary_distance = boundary_distance[0]

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
        elif edge_index.dim() != 2:
            raise ValueError(f"Expected edge_index with shape (2, E), got {tuple(edge_index.shape)}")

        if pos is None:
            pos = input_features[:, :3]

        if airfoil_mask is None and input_features.shape[-1] >= 4:
            airfoil_mask = input_features[:, 3] > 0.5
        if boundary_distance is None and input_features.shape[-1] >= 5:
            boundary_distance = input_features[:, 4]

        x = self.encoder(input_features)
        node_boundary_features = _build_node_boundary_features(
            airfoil_mask=airfoil_mask,
            boundary_distance=boundary_distance,
            pos=pos,
        )

        flux_penalty = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(
                x,
                pos=pos,
                edge_index=edge_index,
                node_boundary_features=node_boundary_features,
                airfoil_mask=airfoil_mask,
            )
            flux_penalty = flux_penalty + aux["flux_antisymmetry"]

        velocity = self.decoder(x)
        return {
            "velocity": velocity.contiguous(),
            "aux_losses": {
                "flux_antisymmetry": flux_penalty / max(len(self.blocks), 1),
            },
        }
