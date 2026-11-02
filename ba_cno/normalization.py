from __future__ import annotations

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Per-channel standardisation fitted once on training data.

    ``center=False`` only rescales. Use it for targets produced by the
    flux-divergence decoder: subtracting a mean would add a spurious uniform
    source and break the exact conservation property.
    """

    def __init__(self, size: int, center: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.center = bool(center)
        self.eps = eps
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, data: torch.Tensor) -> "Normalizer":
        data = data.reshape(-1, data.shape[-1]).double()
        mean = data.mean(0) if self.center else torch.zeros(data.shape[-1], dtype=data.dtype)
        std = (data - mean).pow(2).mean(0).sqrt().clamp_min(self.eps)
        self.mean.copy_(mean.float())
        self.std.copy_(std.float())
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
