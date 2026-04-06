from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def compute_knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Compute batched k-NN indices from positions shaped ``(B, N, 3)``."""

    if pos.dim() != 3:
        raise ValueError(f"Expected pos with shape (B, N, 3), got {tuple(pos.shape)}")

    batch_size, num_points, _ = pos.shape
    if num_points < 2:
        raise ValueError("At least two points are required to build a graph.")

    k = max(1, min(int(k), num_points - 1))
    dist = torch.cdist(pos, pos)
    eye = torch.eye(num_points, device=pos.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k, dim=-1, largest=False).indices


class ResidualBlock(nn.Module):
    """Simple residual message-passing block for a baseline geometric GNN."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        neighbour_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, hidden_dim = x.shape
        k = neighbour_idx.shape[-1]

        idx_features = neighbour_idx.unsqueeze(-1).expand(batch_size, -1, -1, hidden_dim)
        gathered_x = torch.gather(
            x.unsqueeze(1).expand(-1, x.shape[1], -1, -1),
            2,
            idx_features,
        )

        idx_pos = neighbour_idx.unsqueeze(-1).expand(batch_size, -1, -1, pos.shape[-1])
        gathered_pos = torch.gather(
            pos.unsqueeze(1).expand(-1, pos.shape[1], -1, -1),
            2,
            idx_pos,
        )

        center_x = x.unsqueeze(2).expand(-1, -1, k, -1)
        center_pos = pos.unsqueeze(2).expand(-1, -1, k, -1)
        relative_pos = gathered_pos - center_pos

        msg_input = torch.cat([center_x, gathered_x, relative_pos], dim=-1)
        messages = self.message_mlp(msg_input)
        aggregated = messages.mean(dim=2)

        delta = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        out = self.norm(x + delta)
        return F.relu(out)


class BaselineGNN(nn.Module):
    """Baseline k-NN graph network for 5-step flow prediction.

    Parameters
    ----------
    in_dim:
        Per-point input feature dimension.
    hidden_dim:
        Width of the latent node embedding.
    num_layers:
        Number of residual graph blocks.
    k_neighbors:
        Number of nearest neighbours used for message passing.
    output_steps:
        Number of future frames to predict.
    predict_pressure:
        If ``True``, also predict one scalar pressure value per future frame.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        k_neighbors: int = 12,
        output_steps: int = 5,
        predict_pressure: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.k_neighbors = int(k_neighbors)
        self.output_steps = int(output_steps)
        self.predict_pressure = bool(predict_pressure)

        self.encoder = PointEncoder(self.in_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(self.hidden_dim) for _ in range(self.num_layers)])

        output_channels = 3 + (1 if self.predict_pressure else 0)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_steps * output_channels),
        )

    def forward(self, input_features: torch.Tensor, pos: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)

        if input_features.shape[:2] != pos.shape[:2]:
            raise ValueError(
                "input_features and pos must agree on batch and node dimensions: "
                f"got {tuple(input_features.shape)} vs {tuple(pos.shape)}"
            )

        neighbour_idx = compute_knn(pos, self.k_neighbors)
        x = self.encoder(input_features)
        for block in self.blocks:
            x = block(x, pos, neighbour_idx)

        decoded = self.decoder(x)
        batch_size, num_points, _ = decoded.shape

        if self.predict_pressure:
            decoded = decoded.view(batch_size, num_points, self.output_steps, 4)
            velocity = decoded[..., :3].permute(0, 2, 1, 3).contiguous()
            pressure = decoded[..., 3].permute(0, 2, 1).contiguous()
            return {"velocity": velocity, "pressure": pressure}

        velocity = decoded.view(batch_size, num_points, self.output_steps, 3)
        velocity = velocity.permute(0, 2, 1, 3).contiguous()
        return {"velocity": velocity}

