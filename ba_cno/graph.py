"""Graph utilities shared by every benchmark adapter and model.

Everything here is plain PyTorch (no torch_geometric) so that it runs the same
on a MacBook (MPS/CPU) and on Colab/Kaggle GPUs.

Conventions
-----------
``edge_index`` has shape ``(2, E)`` with rows ``(src, dst)``. A message on edge
``e`` travels from ``src[e]`` to ``dst[e]`` and is accumulated at ``dst[e]``.
All graphs used with the conservative operator must be *bidirectional*: for
every edge ``(i, j)`` the edge ``(j, i)`` also exists. ``reverse_edge_permutation``
returns, for each edge, the index of its twin.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Scatter helpers
# ---------------------------------------------------------------------------

def scatter_sum(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = values.new_zeros((num_nodes,) + tuple(values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def scatter_mean(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    total = scatter_sum(values, index, num_nodes)
    count = torch.bincount(index, minlength=num_nodes).clamp_min(1).to(values.dtype)
    return total / count.view(-1, *([1] * (values.dim() - 1)))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def cells_to_edges(cells: torch.Tensor) -> torch.Tensor:
    """Unique bidirectional edges from triangular (or polygonal) cells ``(C, k)``."""

    k = cells.shape[1]
    pairs = [torch.stack([cells[:, a], cells[:, (a + 1) % k]], dim=0) for a in range(k)]
    edges = torch.cat(pairs, dim=1)
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    edges = torch.unique(edges, dim=1)
    return edges[:, edges[0] != edges[1]]


def knn_graph(pos: torch.Tensor, k: int, chunk_size: int | None = None) -> torch.Tensor:
    """Symmetrised k-NN graph for a single point cloud ``(N, d)``.

    Avoids materialising the full ``N x N`` distance matrix.
    """

    num_points = pos.shape[0]
    k = max(1, min(int(k), num_points - 1))
    if chunk_size is None:
        chunk_size = max(1, min(num_points, 4_000_000 // max(num_points, 1)))

    neighbours = []
    for start in range(0, num_points, chunk_size):
        stop = min(start + chunk_size, num_points)
        dist = torch.cdist(pos[start:stop], pos)
        rows = torch.arange(stop - start, device=pos.device)
        dist[rows, rows + start] = float("inf")
        neighbours.append(dist.topk(k, dim=-1, largest=False).indices)
    nbr = torch.cat(neighbours, dim=0)  # (N, k)

    src = nbr.reshape(-1)
    dst = torch.arange(num_points, device=pos.device).repeat_interleave(k)
    return make_bidirectional(torch.stack([src, dst], dim=0))


def make_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    edges = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edges = torch.unique(edges, dim=1)
    return edges[:, edges[0] != edges[1]]


def reverse_edge_permutation(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """For each edge ``e = (i, j)`` return the index of ``(j, i)``.

    Raises if the graph is not bidirectional.
    """

    src, dst = edge_index.to(torch.int64)
    keys = src * num_nodes + dst
    reverse_keys = dst * num_nodes + src
    sorted_keys, perm = torch.sort(keys)
    pos = torch.searchsorted(sorted_keys, reverse_keys).clamp(max=keys.numel() - 1)
    if not torch.equal(sorted_keys[pos], reverse_keys):
        raise ValueError("Graph is not bidirectional; call make_bidirectional() first.")
    return perm[pos]


def edge_geometry(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Edge features ``[x_src - x_dst, |x_src - x_dst|]`` as used by MeshGraphNets."""

    src, dst = edge_index
    rel = pos[src] - pos[dst]
    return torch.cat([rel, torch.linalg.norm(rel, dim=-1, keepdim=True)], dim=-1)


def lumped_cell_volume(pos: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Dual (median) control-volume size per node for a 2D triangle mesh.

    Each triangle gives one third of its area to each of its vertices. This is the
    ``V_i`` used by the flux-divergence decoder, so that ``sum_i V_i * du_i`` is the
    discrete total momentum change.
    """

    a, b, c = pos[cells[:, 0]], pos[cells[:, 1]], pos[cells[:, 2]]
    ab, ac = b - a, c - a
    area = 0.5 * torch.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    vol = pos.new_zeros(pos.shape[0])
    for col in range(3):
        vol.index_add_(0, cells[:, col], area / 3.0)
    return vol
