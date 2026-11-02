"""Core correctness tests: run with ``python -m pytest tests -q``.

These check the claims the case study rests on, on a synthetic triangle mesh:
1. hard-mode processor messages are exactly antisymmetric;
2. the flux-divergence decoder conserves sum_i V_i du_i up to boundary sources;
3. the least-squares gradient is exact for linear fields;
4. the TFRecord / Example parser round-trips a hand-built record;
5. one training step runs for every model in the ablation ladder.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import torch

from ba_cno.data.cylinder import build_mesh, collate_graphs
from ba_cno.data.tfrecord import decode_trajectory, iter_tfrecord, parse_example
from ba_cno.graph import scatter_sum
from ba_cno.metrics import physics as P
from ba_cno.models import build_model
from ba_cno.models.conservative_operator import ConservativeFluxBlock


def grid_mesh(n: int = 8):
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    pos = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
    cells = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            cells += [[a, b, d], [a, d, c]]
    node_type = np.zeros(n * n, dtype=np.int32)
    edge = (pos[:, 0] == 0) | (pos[:, 0] == 1) | (pos[:, 1] == 0) | (pos[:, 1] == 1)
    node_type[edge] = 6
    node_type[pos[:, 0] == 0] = 4
    return pos, node_type, np.asarray(cells, dtype=np.int32)


def make_batch(mesh, node_in: int = 11):
    torch.manual_seed(0)
    n = mesh.num_nodes
    return {
        "node_features": torch.randn(n, node_in),
        "edge_features": mesh.edge_features,
        "edge_index": mesh.edge_index,
        "reverse_edge": mesh.reverse_edge,
        "node_volume": mesh.node_volume,
        "boundary_mask": mesh.boundary_mask,
        "interior_edge_mask": mesh.interior_edge_mask,
        "loss_mask": mesh.loss_mask,
        "target": torch.randn(n, 3),
    }


def test_hard_messages_are_antisymmetric():
    mesh = build_mesh(*grid_mesh())
    block = ConservativeFluxBlock(16, "hard")
    h = torch.randn(mesh.num_nodes, 16)
    e = torch.randn(mesh.edge_index.shape[1], 16)
    src, dst = mesh.edge_index
    phi = block.edge_mlp(torch.cat([e, h[src], h[dst]], -1))
    flux = 0.5 * (phi - phi[mesh.reverse_edge])
    omega = block.face_weight(e)
    w = torch.nn.functional.softplus(0.5 * (omega + omega[mesh.reverse_edge]))
    msg = w * flux
    assert torch.allclose(msg, -msg[mesh.reverse_edge], atol=1e-6)
    assert torch.allclose(scatter_sum(msg, dst, mesh.num_nodes).sum(0), torch.zeros(16), atol=1e-4)


def test_flux_decoder_conserves_momentum():
    mesh = build_mesh(*grid_mesh())
    model = build_model("ba_cno", 11, 3, 3, latent=32, num_layers=3, flux_mode="hard",
                        decoder="flux_divergence", conserved_dim=2)
    batch = make_batch(mesh)
    out = model(batch)["output"]
    du = out[:, :2]
    total = (du * mesh.node_volume.unsqueeze(-1)).sum(0)
    # only boundary sources may change total momentum: zero them and check
    batch_no_src = dict(batch, boundary_mask=torch.zeros_like(mesh.boundary_mask))
    du0 = model(batch_no_src)["output"][:, :2]
    total0 = (du0 * mesh.node_volume.unsqueeze(-1)).sum(0)
    assert torch.allclose(total0, torch.zeros(2), atol=1e-4), total0
    assert not torch.allclose(total, torch.zeros(2), atol=1e-4)  # sources are active


def test_lsq_gradient_exact_for_linear_field():
    mesh = build_mesh(*grid_mesh())
    f = (2.0 * mesh.pos[:, :1] - 3.0 * mesh.pos[:, 1:2])
    g = P.lsq_gradient(f, mesh.pos, mesh.edge_index)
    assert torch.allclose(g[:, 0, 0], torch.full_like(g[:, 0, 0], 2.0), atol=1e-3)
    assert torch.allclose(g[:, 0, 1], torch.full_like(g[:, 0, 1], -3.0), atol=1e-3)
    u = torch.cat([mesh.pos[:, 1:2], -mesh.pos[:, :1]], -1)  # solid-body rotation: div 0, vort -2
    assert P.divergence(u, mesh.pos, mesh.edge_index).abs().max() < 1e-3
    assert torch.allclose(P.vorticity_2d(u, mesh.pos, mesh.edge_index), torch.full((mesh.num_nodes,), -2.0), atol=1e-3)


def test_strouhal_recovers_frequency():
    dt, f0 = 0.01, 3.7
    t = torch.arange(600) * dt
    assert abs(P.dominant_frequency(torch.sin(2 * torch.pi * f0 * t), dt) - f0) < 0.05


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def test_tfrecord_roundtrip(tmp_path):
    vel = np.random.rand(4, 5, 2).astype(np.float32)
    cells = np.arange(6, dtype=np.int32).reshape(1, 2, 3)
    meta = {"trajectory_length": 4, "dt": 0.01, "features": {
        "velocity": {"type": "dynamic", "shape": [4, -1, 2], "dtype": "float32"},
        "cells": {"type": "static", "shape": [1, -1, 3], "dtype": "int32"}}}

    def feature(raw: bytes) -> bytes:
        return _ld(1, _ld(1, raw))  # Feature{bytes_list{value}}

    entries = b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, feature(v.tobytes())))
                       for k, v in {"velocity": vel, "cells": cells}.items())
    example = _ld(1, entries)
    record = struct.pack("<Q", len(example)) + b"\0" * 4 + example + b"\0" * 4
    path = tmp_path / "x.tfrecord"
    path.write_bytes(record * 2)
    payloads = list(iter_tfrecord(path))
    assert len(payloads) == 2
    traj = decode_trajectory(parse_example(payloads[0]), meta)
    assert np.array_equal(traj["velocity"], vel)
    assert np.array_equal(traj["cells"], cells[0])


def test_every_model_trains_one_step():
    mesh = build_mesh(*grid_mesh())
    configs = [
        ("mgn", {}),
        ("ba_cno", {"flux_mode": "soft"}),
        ("ba_cno", {"flux_mode": "hard"}),
        ("ba_cno", {"flux_mode": "hard", "decoder": "flux_divergence", "conserved_dim": 2}),
        ("ba_cno", {"flux_mode": "none", "decoder": "flux_divergence", "conserved_dim": 2}),
    ]
    batch = collate_graphs([make_batch(mesh), make_batch(mesh)])
    for name, kw in configs:
        model = build_model(name, 11, 3, 3, latent=32, num_layers=2, **kw)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        out = model(batch)
        loss = (out["output"] - batch["target"]).pow(2).mean() + sum(out["aux_losses"].values(), torch.zeros(()))
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
