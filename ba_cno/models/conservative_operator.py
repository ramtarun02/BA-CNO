"""BA-CNO: Boundary-Aware Conservative Neural Operator.

Same encode-process-decode skeleton (and parameter budget) as MeshGraphNets, so
any difference in results comes from the finite-volume inductive bias and not
from model size. Two knobs carry the scientific question:

``flux_mode`` (processor messages)
    ``"none"``  ordinary learned messages  (== MeshGraphNets block)
    ``"soft"``  ordinary messages + penalty ``||F_ij + F_ji||^2``   (GRaM version)
    ``"hard"``  antisymmetric by construction ``F_ij = 1/2 (phi_ij - phi_ji)``,
                scaled by a symmetric learned face weight ``w_ij = w_ji``.
                Newton's-third-law / finite-volume flux balance in latent space.

``decoder`` (how the physical update is produced)
    ``"pointwise"``        per-node MLP (standard)
    ``"flux_divergence"``  the update of the conserved channels is a discrete
                           flux divergence: ``du_i = (sum_j G_ij + b_i S_i) / V_i``
                           with ``G_ij = -G_ji`` and sources ``S_i`` only on
                           boundary nodes. This guarantees, to round-off,
                           ``sum_i V_i du_i = sum_{i in boundary} S_i``: the
                           interior cannot create or destroy momentum.

Honest scope: the guarantee is on the discrete graph induced by the mesh (or a
k-NN graph), with lumped dual volumes. It is a discrete conservation statement,
not a claim that the network solves the Navier-Stokes equations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph import scatter_sum
from .mlp import mlp

FLUX_MODES = ("none", "soft", "hard")
DECODERS = ("pointwise", "flux_divergence")


def _antisymmetry_penalty(phi: torch.Tensor, reverse: torch.Tensor, edge_mask: torch.Tensor | None) -> torch.Tensor:
    residual = (phi + phi[reverse]).pow(2).sum(-1)
    if edge_mask is not None:
        residual = residual[edge_mask]
    return residual.mean() if residual.numel() else phi.new_zeros(())


class ConservativeFluxBlock(nn.Module):
    def __init__(self, latent: int, flux_mode: str = "hard") -> None:
        super().__init__()
        if flux_mode not in FLUX_MODES:
            raise ValueError(f"flux_mode must be one of {FLUX_MODES}, got {flux_mode!r}")
        self.flux_mode = flux_mode
        self.edge_mlp = mlp(3 * latent, latent, latent)
        self.face_weight = mlp(latent, latent // 2, 1, num_hidden=1, layer_norm=False)
        self.node_mlp = mlp(2 * latent, latent, latent)

    def forward(self, h, e, edge_index, reverse, interior_edges=None):
        src, dst = edge_index
        phi = self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))  # message j(src) -> i(dst)

        aux = {}
        if self.flux_mode == "hard":
            flux = 0.5 * (phi - phi[reverse])
            omega = self.face_weight(e)
            weight = F.softplus(0.5 * (omega + omega[reverse]))  # symmetric, positive
            message = weight * flux  # antisymmetric x symmetric = antisymmetric
        else:
            message = phi
            if self.flux_mode == "soft":
                aux["flux_antisymmetry"] = _antisymmetry_penalty(phi, reverse, interior_edges)

        agg = scatter_sum(message, dst, h.shape[0])
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        e = e + phi
        return h, e, aux


class FluxDivergenceDecoder(nn.Module):
    """Decode a conservative update ``du_i = (sum_j G_ij + b_i S_i) / V_i``."""

    def __init__(self, latent: int, conserved_dim: int, extra_dim: int = 0) -> None:
        super().__init__()
        self.conserved_dim = conserved_dim
        self.extra_dim = extra_dim
        self.edge_flux = mlp(3 * latent, latent, conserved_dim, layer_norm=False)
        self.boundary_source = mlp(latent, latent, conserved_dim, layer_norm=False)
        self.extra = mlp(latent, latent, extra_dim, layer_norm=False) if extra_dim else None

    def forward(self, h, e, edge_index, reverse, node_volume, boundary_mask):
        src, dst = edge_index
        g = self.edge_flux(torch.cat([e, h[src], h[dst]], dim=-1))
        g = 0.5 * (g - g[reverse])
        net_flux = scatter_sum(g, dst, h.shape[0])
        source = self.boundary_source(h) * boundary_mask.to(h.dtype).unsqueeze(-1)
        du = (net_flux + source) / node_volume.unsqueeze(-1)
        if self.extra is not None:
            du = torch.cat([du, self.extra(h)], dim=-1)
        return du


class BACNO(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        out_dim: int,
        latent: int = 128,
        num_layers: int = 15,
        flux_mode: str = "hard",
        decoder: str = "pointwise",
        conserved_dim: int | None = None,
    ) -> None:
        super().__init__()
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
        self.decoder_type = decoder
        self.node_encoder = mlp(node_in, latent, latent)
        self.edge_encoder = mlp(edge_in, latent, latent)
        self.blocks = nn.ModuleList([ConservativeFluxBlock(latent, flux_mode) for _ in range(num_layers)])
        if decoder == "flux_divergence":
            conserved_dim = out_dim if conserved_dim is None else conserved_dim
            self.decoder = FluxDivergenceDecoder(latent, conserved_dim, out_dim - conserved_dim)
        else:
            self.decoder = mlp(latent, latent, out_dim, layer_norm=False)

    def forward(self, batch: dict) -> dict:
        edge_index = batch["edge_index"]
        reverse = batch["reverse_edge"]
        interior = batch.get("interior_edge_mask")
        h = self.node_encoder(batch["node_features"])
        e = self.edge_encoder(batch["edge_features"])

        penalties = []
        for block in self.blocks:
            h, e, aux = block(h, e, edge_index, reverse, interior)
            if "flux_antisymmetry" in aux:
                penalties.append(aux["flux_antisymmetry"])

        if self.decoder_type == "flux_divergence":
            out = self.decoder(h, e, edge_index, reverse, batch["node_volume"], batch["boundary_mask"])
        else:
            out = self.decoder(h)

        aux_losses = {}
        if penalties:
            aux_losses["flux_antisymmetry"] = torch.stack(penalties).mean()
        return {"output": out, "aux_losses": aux_losses}
