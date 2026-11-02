from __future__ import annotations

import torch.nn as nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, num_hidden: int = 2, layer_norm: bool = True) -> nn.Sequential:
    """MeshGraphNets-style MLP: ``num_hidden`` ReLU layers, linear output, optional LayerNorm."""

    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_hidden):
        layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)
