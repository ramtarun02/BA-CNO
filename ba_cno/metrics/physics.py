"""Physics-faithfulness metrics that a CFD engineer would ask for.

Gradients use weighted least-squares reconstruction on the graph (inverse
distance squared weights), the same family of schemes used in unstructured
finite-volume solvers. Every metric works on the ground truth too, so each
reported number has a CFD reference value next to it.
"""

from __future__ import annotations

import math

import torch

from ..graph import scatter_sum


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """RMSE over nodes (and time / channels), as used by MeshGraphNets."""

    err = (pred - true).pow(2)
    if mask is not None:
        err = err[..., mask, :]
    return err.mean().sqrt()


def rollout_rmse_curve(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cumulative RMSE vs rollout length; ``pred``/``true`` are ``(T, N, C)``.

    ``curve[t]`` is the RMSE averaged over steps ``0..t``, which is how
    MeshGraphNets reports RMSE-1, RMSE-50 and RMSE-all.
    """

    err = (pred - true).pow(2)
    if mask is not None:
        err = err[:, mask]
    per_step = err.mean(dim=(1, 2))
    return (per_step.cumsum(0) / torch.arange(1, per_step.numel() + 1, device=per_step.device)).sqrt()


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - true) / torch.linalg.norm(true).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Differential operators on the graph
# ---------------------------------------------------------------------------

def lsq_gradient(field: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Weighted least-squares gradient. ``field`` (N, C), ``pos`` (N, D) -> (N, C, D)."""

    src, dst = edge_index  # neighbour j = src, centre i = dst
    num_nodes, dim = pos.shape
    dx = pos[src] - pos[dst]  # (E, D)
    w = 1.0 / dx.pow(2).sum(-1).clamp_min(1e-20)  # (E,)
    df = field[src] - field[dst]  # (E, C)

    a = scatter_sum(w[:, None, None] * dx[:, :, None] * dx[:, None, :], dst, num_nodes)  # (N, D, D)
    b = scatter_sum(w[:, None, None] * dx[:, :, None] * df[:, None, :], dst, num_nodes)  # (N, D, C)
    a = a + 1e-12 * torch.eye(dim, device=pos.device, dtype=pos.dtype)
    grad = torch.linalg.solve(a, b)  # (N, D, C)
    return grad.transpose(1, 2)


def divergence(velocity: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    grad = lsq_gradient(velocity, pos, edge_index)
    return torch.diagonal(grad, dim1=1, dim2=2).sum(-1)


def vorticity_2d(velocity: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    grad = lsq_gradient(velocity, pos, edge_index)
    return grad[:, 1, 0] - grad[:, 0, 1]  # dv/dx - du/dy


# ---------------------------------------------------------------------------
# Integral quantities
# ---------------------------------------------------------------------------

def kinetic_energy(velocity: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """0.5 * sum V |u|^2 ; works on (N, C) or (T, N, C)."""

    return 0.5 * (velocity.pow(2).sum(-1) * volume).sum(-1)


def total_momentum(velocity: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    return (velocity * volume.unsqueeze(-1)).sum(-2)


def enstrophy_2d(velocity: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    w = vorticity_2d(velocity, pos, edge_index)
    return 0.5 * (w.pow(2) * volume).sum()


# ---------------------------------------------------------------------------
# Unsteady behaviour
# ---------------------------------------------------------------------------

def dominant_frequency(signal: torch.Tensor, dt: float, discard: int = 0) -> float:
    """Peak frequency of a 1D signal (Hann window, parabolic peak refinement)."""

    x = signal[discard:].double()
    x = x - x.mean()
    n = x.numel()
    if n < 8:
        return float("nan")
    spec = torch.fft.rfft(x * torch.hann_window(n, dtype=x.dtype)).abs()
    spec[0] = 0.0
    k = int(torch.argmax(spec))
    if 0 < k < spec.numel() - 1:
        a, b, c = math.log(spec[k - 1] + 1e-30), math.log(spec[k] + 1e-30), math.log(spec[k + 1] + 1e-30)
        denom = a - 2 * b + c
        k = k + (0.5 * (a - c) / denom if denom != 0 else 0.0)
    return float(k / (n * dt))


def strouhal(signal: torch.Tensor, dt: float, length: float, velocity: float, discard: int = 0) -> float:
    return dominant_frequency(signal, dt, discard) * length / velocity


def temporal_psd(signal: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = signal.double() - signal.double().mean()
    n = x.numel()
    spec = torch.fft.rfft(x * torch.hann_window(n, dtype=x.dtype)).abs().pow(2) / n
    freq = torch.fft.rfftfreq(n, d=dt)
    return freq, spec


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------

def no_slip_error(velocity: torch.Tensor, wall_mask: torch.Tensor) -> torch.Tensor:
    return velocity[..., wall_mask, :].pow(2).sum(-1).mean().sqrt()


def pressure_force_2d(pressure: torch.Tensor, pos: torch.Tensor, wall_edges: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """Pressure force on a closed 2D body from its boundary segments.

    ``wall_edges`` (2, S): undirected surface segments. Normals point away from
    ``centre`` (valid for convex bodies such as the cylinder). Returns (2,) force
    per unit depth, ``-sum p n ds``. Viscous shear is not included.
    """

    a, b = wall_edges
    seg = pos[b] - pos[a]
    length = torch.linalg.norm(seg, dim=-1)
    normal = torch.stack([seg[:, 1], -seg[:, 0]], dim=-1) / length.clamp_min(1e-12).unsqueeze(-1)
    mid = 0.5 * (pos[a] + pos[b])
    flip = ((mid - centre) * normal).sum(-1) < 0
    normal[flip] = -normal[flip]
    p_face = 0.5 * (pressure[..., a] + pressure[..., b])
    return -(p_face.unsqueeze(-1) * normal * length.unsqueeze(-1)).sum(-2)
