"""Autoregressive rollout for one-step models on fixed meshes."""

from __future__ import annotations

import torch

from .data.cylinder import CylinderTrajectory


@torch.no_grad()
def rollout_cylinder(model, normalizers: dict, traj: CylinderTrajectory, num_steps: int | None = None, device="cpu") -> torch.Tensor:
    """Roll a trained model out from ``v_0``. Returns predicted velocity ``(T, N, 2)``.

    Nodes outside ``loss_mask`` (inflow, walls) are reset to the ground truth,
    exactly as in MeshGraphNets, so the boundary conditions are imposed.
    """

    model.eval()
    mesh = traj.mesh
    steps = traj.num_steps - 1 if num_steps is None else int(num_steps)
    gt = traj.velocity.to(device)
    keep = mesh.loss_mask.to(device)

    batch = {
        "edge_index": mesh.edge_index.to(device),
        "reverse_edge": mesh.reverse_edge.to(device),
        "edge_features": normalizers["edge"](mesh.edge_features.to(device)),
        "node_volume": mesh.node_volume.to(device),
        "boundary_mask": mesh.boundary_mask.to(device),
        "interior_edge_mask": mesh.interior_edge_mask.to(device),
    }
    onehot = mesh.node_type_onehot().to(device)

    v = gt[0].clone()
    out = [v]
    for t in range(steps):
        batch["node_features"] = normalizers["node"](torch.cat([v, onehot], dim=-1))  # as node_inputs, on device
        pred = model(batch)["output"]
        dv = normalizers["target"].inverse(pred)[:, :2]
        v_next = torch.where(keep.unsqueeze(-1), v + dv, gt[t + 1])
        out.append(v_next)
        v = v_next
    return torch.stack(out, dim=0)
