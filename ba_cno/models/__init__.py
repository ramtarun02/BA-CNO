"""Model zoo.

Benchmark models (no torch_geometric needed):
    MeshGraphNet  - published baseline (Pfaff et al. 2021)
    BACNO         - Boundary-Aware Conservative Neural Operator (this work)

Legacy GRaM challenge models live in ``ba_cno.models.graphsage`` and
``ba_cno.models.flux_operator`` and are imported from there directly.
"""

from .conservative_operator import BACNO, ConservativeFluxBlock, FluxDivergenceDecoder
from .meshgraphnet import MeshGraphNet


def build_model(name: str, node_in: int, edge_in: int, out_dim: int, **kwargs):
    name = name.lower()
    if name in ("mgn", "meshgraphnet"):
        return MeshGraphNet(node_in, edge_in, out_dim, **kwargs)
    if name in ("ba_cno", "bacno"):
        return BACNO(node_in, edge_in, out_dim, **kwargs)
    raise ValueError(f"Unknown model {name!r}")


__all__ = ["BACNO", "ConservativeFluxBlock", "FluxDivergenceDecoder", "MeshGraphNet", "build_model"]
