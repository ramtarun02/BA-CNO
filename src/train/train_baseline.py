from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.datasets.gram_dataset import GRAMDataset, scan_dataset_files
from src.datasets.preprocessing import (
    PhaseAPreprocessor,
    fit_velocity_stats,
    load_velocity_stats,
    save_velocity_stats,
)
from src.datasets.splits import build_geometry_split, summarize_split
from src.models import BaselineGNN


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline GNN on GRaM samples.")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--subsample-points", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def _tensorize_sample(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor | None]:
    pressure = sample.get("pressure")
    target_pressure = None
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=np.float32)
        if pressure.shape[0] >= 10:
            target_pressure = pressure[5:]
        else:
            target_pressure = pressure[-5:]

    return {
        "input_features": torch.from_numpy(np.asarray(sample["input_features"], dtype=np.float32)).unsqueeze(0).to(device),
        "pos": torch.from_numpy(np.asarray(sample["pos"], dtype=np.float32)).unsqueeze(0).to(device),
        "target_velocity": torch.from_numpy(np.asarray(sample["velocity_out"], dtype=np.float32)).unsqueeze(0).to(device),
        "target_pressure": None
        if target_pressure is None
        else torch.from_numpy(target_pressure).unsqueeze(0).to(device),
    }


def _sample_metrics(pred_velocity: torch.Tensor, true_velocity: torch.Tensor) -> dict[str, float]:
    diff = pred_velocity - true_velocity
    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    return {
        "velocity_mae": float(mae),
        "velocity_rmse": float(rmse),
    }


def _iterate_epoch(
    model: BaselineGNN,
    dataset: GRAMDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    limit: int | None,
    pressure_loss_weight: float,
    seed: int,
) -> dict[str, float]:
    loss_fn = nn.HuberLoss(delta=1.0)
    is_train = optimizer is not None
    model.train(is_train)

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if limit is not None:
        indices = indices[:limit]

    total_loss = 0.0
    total_velocity_loss = 0.0
    total_pressure_loss = 0.0
    total_velocity_mae = 0.0
    total_velocity_rmse = 0.0
    num_samples = 0

    for idx in indices:
        processed = dataset[idx]
        batch = _tensorize_sample(processed, device)

        with torch.set_grad_enabled(is_train):
            outputs = model(batch["input_features"], batch["pos"])
            velocity_loss = loss_fn(outputs["velocity"], batch["target_velocity"])
            loss = velocity_loss
            pressure_loss_value = torch.tensor(0.0, device=device)

            if batch["target_pressure"] is not None and "pressure" in outputs:
                pressure_loss_value = loss_fn(outputs["pressure"], batch["target_pressure"])
                loss = loss + pressure_loss_weight * pressure_loss_value

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metrics = _sample_metrics(outputs["velocity"], batch["target_velocity"])
        total_loss += float(loss.item())
        total_velocity_loss += float(velocity_loss.item())
        total_pressure_loss += float(pressure_loss_value.item())
        total_velocity_mae += metrics["velocity_mae"]
        total_velocity_rmse += metrics["velocity_rmse"]
        num_samples += 1

    if num_samples == 0:
        raise ValueError("No samples were processed in the epoch.")

    return {
        "loss": total_loss / num_samples,
        "velocity_loss": total_velocity_loss / num_samples,
        "pressure_loss": total_pressure_loss / num_samples,
        "velocity_mae": total_velocity_mae / num_samples,
        "velocity_rmse": total_velocity_rmse / num_samples,
        "num_samples": float(num_samples),
    }


def _save_example_prediction(
    model: BaselineGNN,
    dataset: GRAMDataset,
    output_path: Path,
    device: torch.device,
) -> None:
    if len(dataset) == 0:
        return

    sample = dataset[0]
    batch = _tensorize_sample(sample, device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["input_features"], batch["pos"])

    payload: dict[str, Any] = {
        "pred_velocity": outputs["velocity"][0].detach().cpu().numpy(),
        "actual_velocity": np.asarray(sample["velocity_out"], dtype=np.float32),
        "pos": np.asarray(sample["pos"], dtype=np.float32),
        "airfoil_mask": np.asarray(sample["airfoil_mask"], dtype=np.bool_),
    }

    pressure = sample.get("pressure")
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=np.float32)
        payload["actual_pressure"] = pressure[5:] if pressure.shape[0] >= 10 else pressure[-5:]
    if "pressure" in outputs:
        payload["pred_pressure"] = outputs["pressure"][0].detach().cpu().numpy()

    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = _build_argparser().parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    train_cfg = config.get("baseline_gnn", {})

    seed = int(config["split"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(config["data"]["root"])
    records = scan_dataset_files(data_root)
    split = build_geometry_split(
        records,
        val_fraction=float(config["split"]["val_fraction"]),
        seed=seed,
        num_test_simulations=int(config["split"].get("num_test_simulations", 0)),
        test_simulation_ids=config["split"].get("test_simulation_ids"),
    )

    stats_path = Path(config["outputs"]["velocity_stats"])
    if not stats_path.exists():
        print("velocity_stats.json not found — computing from training split...", flush=True)
        raw_train = GRAMDataset(
            data_root,
            records=split["train_records"],
            transform=None,
            include_pressure=False,
        )
        velocity_stats = fit_velocity_stats(raw_train)
        save_velocity_stats(velocity_stats, stats_path)
        print(f"Saved velocity stats to {stats_path}", flush=True)
    else:
        velocity_stats = load_velocity_stats(stats_path)
    subsample_points = int(
        args.subsample_points
        or train_cfg.get("subsample_points")
        or config["preprocessing"].get("subsample_points", 1024)
    )

    preprocessor = PhaseAPreprocessor(
        coordinate_mode=config["preprocessing"]["coordinate_mode"],
        velocity_mean=velocity_stats["mean"],
        velocity_std=velocity_stats["std"],
        subsample_points=subsample_points,
        seed=seed,
        compute_boundary=bool(config["preprocessing"].get("compute_boundary_distance", True)),
        keep_airfoil=bool(config["preprocessing"].get("keep_airfoil", True)),
    )

    print(
        json.dumps(
            {
                "status": "initializing_baseline_gnn",
                "device": str(device),
                "subsample_points": subsample_points,
                "train_records": len(split["train_records"]),
                "val_records": len(split["val_records"]),
            }
        ),
        flush=True,
    )
    if device.type == "cpu" and subsample_points > 4096:
        print(
            f"Warning: subsample_points={subsample_points} on CPU can be slow; consider 2048-4096 for smoke runs.",
            flush=True,
        )

    train_dataset = GRAMDataset(
        data_root,
        records=split["train_records"],
        transform=preprocessor,
        include_pressure=bool(config["data"].get("include_pressure", True)),
    )
    val_dataset = GRAMDataset(
        data_root,
        records=split["val_records"],
        transform=preprocessor,
        include_pressure=bool(config["data"].get("include_pressure", True)),
    )

    sample_input_dim = train_dataset[0]["input_features"].shape[-1]
    model = BaselineGNN(
        in_dim=int(sample_input_dim),
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        num_layers=int(train_cfg.get("num_layers", 3)),
        k_neighbors=int(train_cfg.get("k_neighbors", 8)),
        output_steps=int(train_cfg.get("output_steps", 5)),
        predict_pressure=bool(train_cfg.get("predict_pressure", True)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(args.epochs or train_cfg.get("epochs", 5))
    train_limit = args.train_limit if args.train_limit is not None else train_cfg.get("train_limit")
    val_limit = args.val_limit if args.val_limit is not None else train_cfg.get("val_limit")
    pressure_loss_weight = float(train_cfg.get("pressure_loss_weight", 0.05))

    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "outputs/baseline_gnn"))
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}...", flush=True)
        train_metrics = _iterate_epoch(
            model,
            train_dataset,
            optimizer=optimizer,
            device=device,
            limit=int(train_limit) if train_limit is not None else None,
            pressure_loss_weight=pressure_loss_weight,
            seed=seed + epoch,
        )
        val_metrics = _iterate_epoch(
            model,
            val_dataset,
            optimizer=None,
            device=device,
            limit=int(val_limit) if val_limit is not None else None,
            pressure_loss_weight=pressure_loss_weight,
            seed=seed,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_velocity_loss": train_metrics["velocity_loss"],
            "train_pressure_loss": train_metrics["pressure_loss"],
            "val_loss": val_metrics["loss"],
            "val_velocity_loss": val_metrics["velocity_loss"],
            "val_pressure_loss": val_metrics["pressure_loss"],
            "val_velocity_mae": val_metrics["velocity_mae"],
            "val_velocity_rmse": val_metrics["velocity_rmse"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_dim": int(sample_input_dim),
                        "hidden_dim": int(train_cfg.get("hidden_dim", 64)),
                        "num_layers": int(train_cfg.get("num_layers", 3)),
                        "k_neighbors": int(train_cfg.get("k_neighbors", 8)),
                        "output_steps": int(train_cfg.get("output_steps", 5)),
                        "predict_pressure": bool(train_cfg.get("predict_pressure", True)),
                    },
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "split_summary": summarize_split(split),
                },
                output_dir / "best_model.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "config": str(config_path),
        "device": str(device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "subsample_points": subsample_points,
        "split_summary": summarize_split(split),
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _save_example_prediction(model, val_dataset, output_dir / "val_prediction_example.npz", device)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
