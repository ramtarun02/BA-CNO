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
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.datasets.graph_dataset import GRAMGraphDataset
from src.datasets.gram_dataset import scan_dataset_files
from src.datasets.preprocessing import PhaseAPreprocessor, load_velocity_stats
from src.datasets.splits import build_geometry_split, summarize_split
from src.models import BaselineGNN


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline GNN on GRaM samples.")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument(
        "--subsample-points",
        type=int,
        default=None,
        help="Number of points to keep per sample. Use 0 to disable subsampling and keep all points.",
    )
    parser.add_argument("--node-batch-size", type=int, default=None)
    parser.add_argument("--eval-node-batch-size", type=int, default=None)
    parser.add_argument("--num-hops", type=int, default=None)
    parser.add_argument("--graph-cache-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. Defaults to config value or auto-detection (CUDA -> MPS -> CPU).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def _resolve_device(requested_device: str | None) -> torch.device:
    requested = str(requested_device or "auto").lower()
    mps_backend = getattr(torch.backends, "mps", None)

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")

    if requested == "mps":
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "MPS was requested, but Apple Silicon GPU support is unavailable in this PyTorch environment."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option: {requested_device!r}")


def _sample_metrics(pred_velocity: torch.Tensor, true_velocity: torch.Tensor) -> dict[str, float]:
    diff = pred_velocity - true_velocity
    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    return {
        "velocity_mae": float(mae),
        "velocity_rmse": float(rmse),
    }


def _resolve_subsample_points(
    cli_value: int | None,
    train_cfg: dict[str, Any],
    preprocessing_cfg: dict[str, Any],
) -> int | None:
    raw_value: Any
    if cli_value is not None:
        raw_value = cli_value
    elif "subsample_points" in train_cfg:
        raw_value = train_cfg.get("subsample_points")
    else:
        raw_value = preprocessing_cfg.get("subsample_points", 1024)

    if raw_value is None:
        return None

    value = int(raw_value)
    return None if value <= 0 else value


def _iter_target_node_batches(
    num_nodes: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"node batch size must be positive, got {batch_size}")

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(num_nodes, generator=generator)
    else:
        order = torch.arange(num_nodes)

    return [order[start : start + batch_size] for start in range(0, num_nodes, batch_size)]


def _build_subgraph_batch(
    graph: Data,
    target_nodes: torch.Tensor,
    num_hops: int,
    device: torch.device,
) -> dict[str, Any]:
    subset, edge_index, mapping, _ = k_hop_subgraph(
        target_nodes.cpu(),
        num_hops=num_hops,
        edge_index=graph.edge_index,
        relabel_nodes=True,
    )

    batch: dict[str, Any] = {
        "x": graph.x[subset].to(device),
        "pos": graph.pos[subset].to(device),
        "edge_index": edge_index.to(device),
        "target_velocity": graph.y_velocity[subset].to(device),
        "seed_mapping": mapping.to(device=device, dtype=torch.long),
    }
    batch["target_pressure"] = (
        graph.y_pressure[subset].to(device) if hasattr(graph, "y_pressure") else None
    )
    return batch


def _iterate_epoch(
    model: BaselineGNN,
    dataset: GRAMGraphDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    limit: int | None,
    pressure_loss_weight: float,
    seed: int,
    node_batch_size: int,
    num_hops: int,
) -> dict[str, float]:
    loss_fn = nn.HuberLoss(delta=1.0)
    is_train = optimizer is not None
    model.train(is_train)

    graph_indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(graph_indices)
    if limit is not None:
        graph_indices = graph_indices[:limit]

    total_loss = 0.0
    total_velocity_loss = 0.0
    total_pressure_loss = 0.0
    total_velocity_mae = 0.0
    total_velocity_rmse = 0.0
    num_target_nodes = 0
    num_node_batches = 0

    for graph_offset, graph_idx in enumerate(graph_indices):
        graph = dataset[graph_idx]
        effective_batch_size = min(int(node_batch_size), int(graph.num_nodes))
        target_batches = _iter_target_node_batches(
            int(graph.num_nodes),
            batch_size=effective_batch_size,
            seed=seed + graph_offset,
            shuffle=is_train,
        )

        for target_nodes in target_batches:
            batch = _build_subgraph_batch(graph, target_nodes, num_hops=num_hops, device=device)

            with torch.set_grad_enabled(is_train):
                outputs = model(batch["x"], pos=batch["pos"], edge_index=batch["edge_index"])
                seed_mapping = batch["seed_mapping"]
                pred_velocity = outputs["velocity"][seed_mapping]
                true_velocity = batch["target_velocity"][seed_mapping]
                velocity_loss = loss_fn(pred_velocity, true_velocity)
                loss = velocity_loss
                pressure_loss_value = torch.tensor(0.0, device=device)

                if batch["target_pressure"] is not None and "pressure" in outputs:
                    pred_pressure = outputs["pressure"][seed_mapping]
                    true_pressure = batch["target_pressure"][seed_mapping]
                    pressure_loss_value = loss_fn(pred_pressure, true_pressure)
                    loss = loss + pressure_loss_weight * pressure_loss_value

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            metrics = _sample_metrics(pred_velocity.detach(), true_velocity.detach())
            batch_nodes = int(seed_mapping.numel())
            total_loss += float(loss.item()) * batch_nodes
            total_velocity_loss += float(velocity_loss.item()) * batch_nodes
            total_pressure_loss += float(pressure_loss_value.item()) * batch_nodes
            total_velocity_mae += metrics["velocity_mae"] * batch_nodes
            total_velocity_rmse += metrics["velocity_rmse"] * batch_nodes
            num_target_nodes += batch_nodes
            num_node_batches += 1

    if num_target_nodes == 0:
        raise ValueError("No target nodes were processed in the epoch.")

    return {
        "loss": total_loss / num_target_nodes,
        "velocity_loss": total_velocity_loss / num_target_nodes,
        "pressure_loss": total_pressure_loss / num_target_nodes,
        "velocity_mae": total_velocity_mae / num_target_nodes,
        "velocity_rmse": total_velocity_rmse / num_target_nodes,
        "num_samples": float(num_target_nodes),
        "num_node_batches": float(num_node_batches),
    }


def _predict_graph(
    model: BaselineGNN,
    graph: Data,
    device: torch.device,
    num_hops: int,
    eval_node_batch_size: int,
) -> dict[str, torch.Tensor]:
    pred_velocity = torch.empty_like(graph.y_velocity, device="cpu")
    pred_pressure = torch.empty_like(graph.y_pressure, device="cpu") if hasattr(graph, "y_pressure") else None

    model.eval()
    with torch.no_grad():
        for target_nodes in _iter_target_node_batches(
            int(graph.num_nodes),
            batch_size=min(eval_node_batch_size, int(graph.num_nodes)),
            seed=0,
            shuffle=False,
        ):
            batch = _build_subgraph_batch(graph, target_nodes, num_hops=num_hops, device=device)
            outputs = model(batch["x"], pos=batch["pos"], edge_index=batch["edge_index"])
            seed_mapping = batch["seed_mapping"]
            pred_velocity[target_nodes] = outputs["velocity"][seed_mapping].detach().cpu()
            if pred_pressure is not None and "pressure" in outputs:
                pred_pressure[target_nodes] = outputs["pressure"][seed_mapping].detach().cpu()

    result: dict[str, torch.Tensor] = {"velocity": pred_velocity}
    if pred_pressure is not None:
        result["pressure"] = pred_pressure
    return result


def _save_example_prediction(
    model: BaselineGNN,
    dataset: GRAMGraphDataset,
    output_path: Path,
    device: torch.device,
    num_hops: int,
    eval_node_batch_size: int,
) -> None:
    if len(dataset) == 0:
        return

    graph = dataset[0]
    outputs = _predict_graph(
        model,
        graph,
        device=device,
        num_hops=num_hops,
        eval_node_batch_size=eval_node_batch_size,
    )

    payload: dict[str, Any] = {
        "pred_velocity": outputs["velocity"].permute(1, 0, 2).numpy(),
        "actual_velocity": graph.y_velocity.permute(1, 0, 2).cpu().numpy(),
        "pos": graph.pos.cpu().numpy(),
        "airfoil_mask": graph.airfoil_mask.cpu().numpy(),
    }
    if hasattr(graph, "y_pressure"):
        payload["actual_pressure"] = graph.y_pressure.transpose(0, 1).cpu().numpy()
    if "pressure" in outputs:
        payload["pred_pressure"] = outputs["pressure"].transpose(0, 1).numpy()

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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = _resolve_device(args.device or train_cfg.get("device", "auto"))

    data_root = Path(config["data"]["root"])
    records = scan_dataset_files(data_root)
    split = build_geometry_split(
        records,
        val_fraction=float(config["split"]["val_fraction"]),
        seed=seed,
        num_test_simulations=int(config["split"].get("num_test_simulations", 0)),
        test_simulation_ids=config["split"].get("test_simulation_ids"),
    )

    velocity_stats = load_velocity_stats(config["outputs"]["velocity_stats"])
    subsample_points = _resolve_subsample_points(
        args.subsample_points,
        train_cfg=train_cfg,
        preprocessing_cfg=config["preprocessing"],
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

    k_neighbors = int(train_cfg.get("k_neighbors", 8))
    num_layers = int(train_cfg.get("num_layers", 3))
    num_hops = int(args.num_hops or train_cfg.get("num_hops", num_layers))
    node_batch_size = int(args.node_batch_size or train_cfg.get("node_batch_size", 2048))
    eval_node_batch_size = int(
        args.eval_node_batch_size or train_cfg.get("eval_node_batch_size", max(node_batch_size, 4096))
    )
    graph_cache_dir = args.graph_cache_dir or train_cfg.get(
        "graph_cache_dir", "outputs/baseline_gnn/graph_cache"
    )
    undirected_graph = bool(train_cfg.get("undirected_graph", False))

    print(
        json.dumps(
            {
                "status": "initializing_baseline_gnn",
                "device": str(device),
                "subsample_points": subsample_points if subsample_points is not None else "all",
                "train_records": len(split["train_records"]),
                "val_records": len(split["val_records"]),
                "node_batch_size": node_batch_size,
                "num_hops": num_hops,
            }
        ),
        flush=True,
    )
    if device.type == "mps":
        print("Using Apple Silicon GPU via PyTorch MPS.", flush=True)
        if subsample_points is None or subsample_points > 8192:
            print("Large point clouds on MPS will build k-NN on CPU first to avoid GPU memory spikes.", flush=True)
    if device.type == "cpu" and subsample_points is not None and subsample_points > 4096:
        print(
            f"Warning: subsample_points={subsample_points} on CPU can be slow; consider 2048-4096 for smoke runs.",
            flush=True,
        )

    train_dataset = GRAMGraphDataset(
        data_root,
        records=split["train_records"],
        transform=preprocessor,
        include_pressure=bool(config["data"].get("include_pressure", True)),
        k_neighbors=k_neighbors,
        graph_cache_dir=graph_cache_dir,
        undirected=undirected_graph,
    )
    val_dataset = GRAMGraphDataset(
        data_root,
        records=split["val_records"],
        transform=preprocessor,
        include_pressure=bool(config["data"].get("include_pressure", True)),
        k_neighbors=k_neighbors,
        graph_cache_dir=graph_cache_dir,
        undirected=undirected_graph,
    )

    sample_input_dim = int(train_dataset[0].x.shape[-1])
    model = BaselineGNN(
        in_dim=sample_input_dim,
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        num_layers=num_layers,
        k_neighbors=k_neighbors,
        output_steps=int(train_cfg.get("output_steps", 5)),
        predict_pressure=bool(train_cfg.get("predict_pressure", True)),
        dropout=float(train_cfg.get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
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
            node_batch_size=node_batch_size,
            num_hops=num_hops,
        )
        val_metrics = _iterate_epoch(
            model,
            val_dataset,
            optimizer=None,
            device=device,
            limit=int(val_limit) if val_limit is not None else None,
            pressure_loss_weight=pressure_loss_weight,
            seed=seed,
            node_batch_size=eval_node_batch_size,
            num_hops=num_hops,
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
                        "num_layers": num_layers,
                        "k_neighbors": k_neighbors,
                        "output_steps": int(train_cfg.get("output_steps", 5)),
                        "predict_pressure": bool(train_cfg.get("predict_pressure", True)),
                        "dropout": float(train_cfg.get("dropout", 0.1)),
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
        "node_batch_size": node_batch_size,
        "eval_node_batch_size": eval_node_batch_size,
        "num_hops": num_hops,
        "graph_cache_dir": str(graph_cache_dir),
        "undirected_graph": undirected_graph,
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _save_example_prediction(
        model,
        val_dataset,
        output_dir / "val_prediction_example.npz",
        device,
        num_hops=num_hops,
        eval_node_batch_size=eval_node_batch_size,
    )

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
