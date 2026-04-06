from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .gram_dataset import SampleRecord, records_to_json, scan_dataset_files


def build_geometry_split(
    records: Iterable[SampleRecord],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Split by full simulation IDs to avoid geometry leakage."""

    records = list(records)
    simulation_ids = sorted({record.simulation_id for record in records})
    if not simulation_ids:
        raise ValueError("No records were provided.")

    rng = random.Random(seed)
    shuffled_ids = simulation_ids[:]
    rng.shuffle(shuffled_ids)

    if len(shuffled_ids) == 1:
        train_ids = shuffled_ids
        val_ids: list[int] = []
    else:
        n_val = max(1, int(round(len(shuffled_ids) * val_fraction)))
        n_val = min(n_val, len(shuffled_ids) - 1)
        val_ids = sorted(shuffled_ids[:n_val])
        train_ids = sorted(shuffled_ids[n_val:])

    train_records = [record for record in records if record.simulation_id in set(train_ids)]
    val_records = [record for record in records if record.simulation_id in set(val_ids)]

    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "train_simulation_ids": train_ids,
        "val_simulation_ids": val_ids,
        "train_records": train_records,
        "val_records": val_records,
    }


def summarize_split(split: dict[str, Any]) -> dict[str, Any]:
    train_ids = set(split["train_simulation_ids"])
    val_ids = set(split["val_simulation_ids"])
    overlap = sorted(train_ids & val_ids)
    return {
        "seed": split["seed"],
        "val_fraction": split["val_fraction"],
        "num_train_records": len(split["train_records"]),
        "num_val_records": len(split["val_records"]),
        "num_train_simulations": len(train_ids),
        "num_val_simulations": len(val_ids),
        "overlap_simulation_ids": overlap,
    }


def save_split_manifest(split: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": split["seed"],
        "val_fraction": split["val_fraction"],
        "train_simulation_ids": split["train_simulation_ids"],
        "val_simulation_ids": split["val_simulation_ids"],
        "train_records": records_to_json(split["train_records"]),
        "val_records": records_to_json(split["val_records"]),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a geometry-aware split manifest.")
    parser.add_argument("--data-root", type=str, default="warped-ifw")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exploration/geometry_split.json",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    records = scan_dataset_files(args.data_root)
    split = build_geometry_split(records, val_fraction=args.val_fraction, seed=args.seed)
    save_split_manifest(split, args.output)
    print(json.dumps(summarize_split(split), indent=2))


if __name__ == "__main__":
    main()
