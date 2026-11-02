"""Reference MeshGraphNets (Pfaff et al., ICLR 2021) encode-process-decode model.

This is the published baseline we compare against on CylinderFlow. It is kept
deliberately close to the original: 128 latent, 2-hidden-layer MLPs with
LayerNorm, edge-state updates, sum aggregation, residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..graph import scatter_sum
from .mlp import mlp


class GraphNetBlock(nn.Module):
    def __init__(self, latent: int) -> None:
        super().__init__()
        self.edge_mlp = mlp(3 * latent, latent, latent)
        self.node_mlp = mlp(2 * latent, latent, latent)

    def forward(self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        e_new = self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))
        agg = scatter_sum(e_new, dst, h.shape[0])
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + h_new, e + e_new


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        out_dim: int,
        latent: int = 128,
        num_layers: int = 15,
    ) -> None:
        super().__init__()
        self.node_encoder = mlp(node_in, latent, latent)
        self.edge_encoder = mlp(edge_in, latent, latent)
        self.blocks = nn.ModuleList([GraphNetBlock(latent) for _ in range(num_layers)])
        self.decoder = mlp(latent, latent, out_dim, layer_norm=False)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        edge_index = batch["edge_index"]
        h = self.node_encoder(batch["node_features"])
        e = self.edge_encoder(batch["edge_features"])
        for block in self.blocks:
            h, e = block(h, e, edge_index)
        return {"output": self.decoder(h), "aux_losses": {}}
