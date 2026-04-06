from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.datasets.gram_dataset import GRAMDataset, scan_dataset_files
from src.datasets.preprocessing import PhaseAPreprocessor, fit_velocity_stats, save_velocity_stats
from src.datasets.splits import build_geometry_split, save_split_manifest, summarize_split


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase A data smoke runner.")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    data_root = config["data"]["root"]
    records = scan_dataset_files(data_root)
    split = build_geometry_split(
        records,
        val_fraction=float(config["split"]["val_fraction"]),
        seed=int(config["split"]["seed"]),
    )

    split_output = config["outputs"]["split_manifest"]
    save_split_manifest(split, split_output)

    train_dataset = GRAMDataset(data_root, records=split["train_records"])
    stats = fit_velocity_stats(train_dataset, include_targets=True)
    save_velocity_stats(stats, config["outputs"]["velocity_stats"])

    preprocessor = PhaseAPreprocessor(
        coordinate_mode=config["preprocessing"]["coordinate_mode"],
        velocity_mean=stats["mean"],
        velocity_std=stats["std"],
        subsample_points=int(config["preprocessing"]["subsample_points"]),
        seed=int(config["split"]["seed"]),
        compute_boundary=bool(config["preprocessing"]["compute_boundary_distance"]),
    )

    example = preprocessor(train_dataset[0])

    payload = {
        "config": str(config_path),
        "split_summary": summarize_split(split),
        "velocity_mean": stats["mean"].tolist(),
        "velocity_std": stats["std"].tolist(),
        "example_feature_shape": list(example["input_features"].shape),
        "example_num_points": int(example["pos"].shape[0]),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
