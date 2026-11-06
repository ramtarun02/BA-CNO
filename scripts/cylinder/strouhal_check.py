"""Day-1 sanity check: ground-truth Strouhal number vs the Schäfer & Turek benchmark.

Runs the evaluation's Strouhal pipeline (wake probe 2D downstream, cross-stream
velocity, Hann-windowed FFT peak) on the CFD ground truth only, and checks it two ways:

* internal: the FFT frequency must agree with zero-crossing counting on the
  late, developed half of the signal (validates the estimator);
* external: for trajectories close to the DFG 2D-2 case (Re ~ 100 *and*
  blockage D/H ~ 0.244) St is compared with the band 0.295-0.305.

    Re = U_mean * D / nu,   St = f * D / U_mean

U_mean is the bulk inflow speed, D the cylinder diameter measured from the mesh.
nu is not stored in the dataset; the default 1e-3 is the Schäfer-Turek fluid
(rho = 1). The channel height (0.41) matches Schäfer-Turek, but the length
(1.6 vs 2.2) and cylinder position differ, so the external comparison is
approximate. Blockage D/H varies between trajectories (~0.1-0.4) and shifts St
strongly, hence the blockage window.

    python scripts/cylinder/strouhal_check.py --split valid test --out outputs/cylinder/strouhal_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ba_cno.data.cylinder import CylinderTrajectory, NodeType, wake_probe_index
from ba_cno.metrics import physics as P

ST_BAND = (0.295, 0.305)  # Schäfer & Turek (1996), 2D-2, Re = 100, D/H = 0.1/0.41


def zero_crossing_frequency(signal: torch.Tensor, dt: float) -> float:
    x = signal.double().numpy()
    x = x - x.mean()
    up = np.where(np.diff(np.sign(x)) > 0)[0]
    return float((len(up) - 1) / ((up[-1] - up[0]) * dt)) if len(up) > 2 else float("nan")


def analyse(traj: CylinderTrajectory, nu: float, discard: int) -> dict:
    mesh = traj.mesh
    probe = wake_probe_index(mesh)
    signal = traj.velocity[:, probe, 1]
    u = traj.inflow_speed
    inflow = mesh.node_type == NodeType.INFLOW
    channel_h = float(mesh.pos[inflow, 1].max() - mesh.pos[inflow, 1].min())
    return {
        "re": u * mesh.diameter / nu,
        "st": P.strouhal(signal, traj.dt, mesh.diameter, u, discard=discard),
        "st_zero_crossing": zero_crossing_frequency(signal[signal.shape[0] // 2:], traj.dt) * mesh.diameter / u,
        "u_mean": u,
        "diameter": mesh.diameter,
        "blockage": mesh.diameter / channel_h,
        "centre_y": float(mesh.centre[1]),
        "probe_v_std": float(signal[discard:].std()),
        "probe_offset_d": float(torch.linalg.norm(mesh.pos[probe] - mesh.centre) / mesh.diameter),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/cylinder_flow")
    ap.add_argument("--split", nargs="+", default=["valid", "test"])
    ap.add_argument("--nu", type=float, default=1e-3)
    ap.add_argument("--discard", type=int, default=100, help="initial steps dropped before the FFT")
    ap.add_argument("--re-window", type=float, nargs=2, default=[80.0, 120.0], help="Re range compared with the band")
    ap.add_argument("--blockage-window", type=float, nargs=2, default=[0.22, 0.27], help="D/H range compared with the band")
    ap.add_argument("--min-amp", type=float, default=1e-3, help="probe v std below which the wake counts as steady")
    ap.add_argument("--out", default="outputs/cylinder/strouhal_check")
    args = ap.parse_args()

    rows = []
    for split in args.split:
        for f in sorted((Path(args.data_root) / split).glob("traj_*.npz")):
            rec = {"split": split, "file": f.name, **analyse(CylinderTrajectory(f, mmap=True), args.nu, args.discard)}
            rows.append(rec)
            print(f"{split}/{f.name} Re={rec['re']:7.1f} D/H={rec['blockage']:.3f} St={rec['st']:.3f} "
                  f"(zc {rec['st_zero_crossing']:.3f}) amp={rec['probe_v_std']:.3g}")
    if not rows:
        raise SystemExit("No trajectories found. Download and convert the data first.")

    shedding = [r for r in rows if r["probe_v_std"] >= args.min_amp]
    lo, hi = args.re_window
    blo, bhi = args.blockage_window
    near = [r for r in shedding if lo <= r["re"] <= hi and blo <= r["blockage"] <= bhi]
    st_near = np.array([r["st"] for r in near])
    est_gap = np.array([abs(r["st"] - r["st_zero_crossing"]) / r["st"] for r in shedding])
    summary = {
        "nu": args.nu,
        "num_trajectories": len(rows),
        "num_shedding": len(shedding),
        "re_range": [min(r["re"] for r in rows), max(r["re"] for r in rows)],
        "fft_vs_zero_crossing_rel_diff_median": float(np.nanmedian(est_gap)),
        "fft_vs_zero_crossing_rel_diff_p90": float(np.nanpercentile(est_gap, 90)),
        "re_window": [lo, hi],
        "blockage_window": [blo, bhi],
        "num_in_window": len(near),
        "st_in_window": sorted(round(float(s), 4) for s in st_near),
        "st_in_window_median": float(np.median(st_near)) if len(near) else None,
        "reference_band": list(ST_BAND),
    }
    summary["estimator_verdict"] = "PASS" if summary["fft_vs_zero_crossing_rel_diff_median"] < 0.01 else "FAIL"
    if near:
        ok = ST_BAND[0] - 0.01 <= summary["st_in_window_median"] <= ST_BAND[1] + 0.01
        summary["reference_verdict"] = "PASS" if ok else "MISMATCH"
    else:
        summary["reference_verdict"] = "NO_TRAJECTORIES_IN_WINDOW"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strouhal_check.json").write_text(json.dumps({"summary": summary, "trajectories": rows}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter([r["re"] for r in shedding], [r["st"] for r in shedding], c=[r["blockage"] for r in shedding], s=14, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="blockage D/H")
    ax.scatter([r["re"] for r in near], [r["st"] for r in near], s=40, facecolors="none", edgecolors="tab:red",
               label=f"compared (Re {lo:g}–{hi:g}, D/H {blo}–{bhi})")
    ax.axhspan(*ST_BAND, color="tab:red", alpha=0.25, label="Schäfer–Turek 2D-2 (Re=100)")
    ax.set_xlabel(f"Re = U_mean D / ν  (ν = {args.nu:g} assumed)")
    ax.set_ylabel("St = f D / U_mean (ground truth)")
    ax.set_title(f"CylinderFlow ground-truth Strouhal ({len(shedding)} shedding trajectories)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "strouhal_vs_re.png", dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
