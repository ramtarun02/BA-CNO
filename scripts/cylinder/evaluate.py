"""Full-length rollout evaluation on CylinderFlow test trajectories.

Reports the MeshGraphNets accuracy metrics (RMSE-1 / -50 / -all, x1e-3) and the
physics-faithfulness metrics, each next to its ground-truth (CFD) value:

* divergence RMS of the velocity field (mass conservation)
* kinetic energy and total-momentum drift over the rollout
* Strouhal number from a wake probe (cross-stream velocity, 2D downstream)
* vorticity RMS in the wake (does the model keep or smear vortices?)

    python scripts/cylinder/evaluate.py --run outputs/cylinder/ba_cno_hard --max-trajectories 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ba_cno.data.cylinder import CylinderTrajectory
from ba_cno.metrics import physics as P
from ba_cno.models import build_model
from ba_cno.normalization import Normalizer
from ba_cno.rollout import rollout_cylinder


def load_run(run_dir: Path, device, which: str = "final.pt"):
    ckpt = torch.load(run_dir / which, map_location=device)
    cfg = ckpt["config"]
    norms = {}
    for k, state in ckpt["normalizers"].items():
        n = Normalizer(state["mean"].numel())
        n.load_state_dict(state)
        norms[k] = n.to(device)
    m = {k: v for k, v in cfg["model"].items() if k != "name"}
    node_in, edge_in = norms["node"].mean.numel(), norms["edge"].mean.numel()
    model = build_model(cfg["model"]["name"], node_in, edge_in, 3, **m).to(device)
    model.load_state_dict(ckpt["model"])
    return model, norms, cfg


def wake_probe_index(traj: CylinderTrajectory, diameters_downstream: float = 2.0) -> int:
    mesh = traj.mesh
    target = mesh.centre + torch.tensor([diameters_downstream * mesh.diameter, 0.0])
    return int(torch.argmin(torch.linalg.norm(mesh.pos - target, dim=-1)))


def physics_summary(vel: torch.Tensor, traj: CylinderTrajectory, probe: int) -> dict:
    mesh = traj.mesh
    ei, pos, vol = mesh.edge_index, mesh.pos, mesh.node_volume
    interior = mesh.node_type == 0
    stride = max(1, vel.shape[0] // 60)  # spatial-derivative metrics on ~60 snapshots
    div = torch.stack([P.divergence(vel[t], pos, ei)[interior].pow(2).mean().sqrt() for t in range(0, vel.shape[0], stride)])
    wake = interior & (pos[:, 0] > mesh.centre[0] + mesh.diameter)
    vort = torch.stack([P.vorticity_2d(vel[t], pos, ei)[wake].pow(2).mean().sqrt() for t in range(0, vel.shape[0], stride)])
    ke = P.kinetic_energy(vel, vol)
    mom = P.total_momentum(vel, vol)
    return {
        "divergence_rms_mean": float(div.mean()),
        "wake_vorticity_rms_mean": float(vort.mean()),
        "kinetic_energy": ke.tolist(),
        "momentum_x": mom[:, 0].tolist(),
        "strouhal": P.strouhal(vel[:, probe, 1], traj.dt, mesh.diameter, traj.inflow_speed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-trajectories", type=int, default=None)
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--save-rollouts", type=int, default=3, help="store full fields for this many trajectories")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = Path(args.run)
    device = torch.device(args.device)
    model, norms, cfg = load_run(run, device, args.checkpoint)
    root = Path(args.data_root or cfg["data"]["root"]) / args.split
    files = sorted(root.glob("traj_*.npz"))[: args.max_trajectories]

    per_traj = []
    for i, f in enumerate(files):
        traj = CylinderTrajectory(f)
        pred = rollout_cylinder(model, norms, traj, device=device).cpu()
        true = traj.velocity
        curve = P.rollout_rmse_curve(pred[1:], true[1:])
        probe = wake_probe_index(traj)
        rec = {
            "file": f.name,
            "rmse_1": float(curve[0]),
            "rmse_50": float(curve[min(49, len(curve) - 1)]),
            "rmse_all": float(curve[-1]),
            "rmse_curve": curve[::10].tolist(),
            "pred": physics_summary(pred, traj, probe),
            "true": physics_summary(true, traj, probe),
        }
        per_traj.append(rec)
        print(f"[{i + 1}/{len(files)}] {f.name} rmse_all={rec['rmse_all'] * 1e3:.2f}e-3 "
              f"St pred/true={rec['pred']['strouhal']:.3f}/{rec['true']['strouhal']:.3f}")
        if i < args.save_rollouts:
            np.savez_compressed(run / f"rollout_{args.split}_{f.stem}.npz", pred=pred.numpy(), true=true.numpy(),
                                pos=traj.mesh.pos.numpy(), cells=traj.mesh.cells.numpy())

    def agg(key):
        vals = np.array([r[key] for r in per_traj])
        return {"mean_x1e3": float(vals.mean() * 1e3), "std_x1e3": float(vals.std() * 1e3)}

    st_err = np.array([abs(r["pred"]["strouhal"] - r["true"]["strouhal"]) / r["true"]["strouhal"] for r in per_traj])
    summary = {
        "run": str(run),
        "num_trajectories": len(per_traj),
        "rmse_1": agg("rmse_1"),
        "rmse_50": agg("rmse_50"),
        "rmse_all": agg("rmse_all"),
        "strouhal_rel_error_median": float(np.nanmedian(st_err)),
        "divergence_rms_pred_over_true": float(np.mean([r["pred"]["divergence_rms_mean"] / r["true"]["divergence_rms_mean"] for r in per_traj])),
        "wake_vorticity_pred_over_true": float(np.mean([r["pred"]["wake_vorticity_rms_mean"] / r["true"]["wake_vorticity_rms_mean"] for r in per_traj])),
        "published_mgn_x1e3": {"rmse_1": 2.34, "rmse_50": 6.3, "rmse_all": 40.88},
    }
    (run / f"eval_{args.split}.json").write_text(json.dumps({"summary": summary, "per_trajectory": per_traj}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
