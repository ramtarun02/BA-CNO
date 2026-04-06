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
    num_test_simulations: int = 0,
    test_simulation_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Split by full simulation IDs to avoid geometry leakage."""

    records = list(records)
    simulation_ids = sorted({record.simulation_id for record in records})
    if not simulation_ids:
        raise ValueError("No records were provided.")
    if num_test_simulations < 0:
        raise ValueError("num_test_simulations must be non-negative.")

    all_simulation_ids = set(simulation_ids)
    requested_test_ids = sorted({int(sim_id) for sim_id in (test_simulation_ids or [])})
    missing_test_ids = sorted(set(requested_test_ids) - all_simulation_ids)
    if missing_test_ids:
        raise ValueError(f"Unknown test simulation IDs requested: {missing_test_ids}")

    rng = random.Random(seed)
    shuffled_ids = simulation_ids[:]
    rng.shuffle(shuffled_ids)

    if requested_test_ids:
        test_ids = requested_test_ids
        if num_test_simulations and len(test_ids) != num_test_simulations:
            raise ValueError(
                "num_test_simulations must match the number of explicit test_simulation_ids."
            )
    elif num_test_simulations:
        n_test = min(int(num_test_simulations), len(shuffled_ids) - 1)
        if n_test <= 0:
            raise ValueError("At least one simulation must remain outside the test split.")
        test_ids = sorted(shuffled_ids[:n_test])
    else:
        test_ids = []

    test_id_set = set(test_ids)
    remaining_ids = [sim_id for sim_id in shuffled_ids if sim_id not in test_id_set]
    if not remaining_ids:
        raise ValueError("No simulations remain after assigning the test split.")

    if len(remaining_ids) == 1:
        train_ids = sorted(remaining_ids)
        val_ids: list[int] = []
    else:
        n_val = max(1, int(round(len(remaining_ids) * val_fraction)))
        n_val = min(n_val, len(remaining_ids) - 1)
        val_ids = sorted(remaining_ids[:n_val])
        train_ids = sorted(remaining_ids[n_val:])

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    train_records = [record for record in records if record.simulation_id in train_id_set]
    val_records = [record for record in records if record.simulation_id in val_id_set]
    test_records = [record for record in records if record.simulation_id in test_id_set]

    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "num_test_simulations": len(test_ids),
        "train_simulation_ids": train_ids,
        "val_simulation_ids": val_ids,
        "test_simulation_ids": test_ids,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
    }


def summarize_split(split: dict[str, Any]) -> dict[str, Any]:
    train_ids = set(split["train_simulation_ids"])
    val_ids = set(split["val_simulation_ids"])
    test_ids = set(split.get("test_simulation_ids", []))
    overlap_train_val = sorted(train_ids & val_ids)
    overlap_train_test = sorted(train_ids & test_ids)
    overlap_val_test = sorted(val_ids & test_ids)
    return {
        "seed": split["seed"],
        "val_fraction": split["val_fraction"],
        "num_train_records": len(split["train_records"]),
        "num_val_records": len(split["val_records"]),
        "num_test_records": len(split.get("test_records", [])),
        "num_train_simulations": len(train_ids),
        "num_val_simulations": len(val_ids),
        "num_test_simulations": len(test_ids),
        "overlap_simulation_ids": sorted((train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)),
        "overlap_train_val": overlap_train_val,
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test,
    }


def save_split_manifest(split: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": split["seed"],
        "val_fraction": split["val_fraction"],
        "num_test_simulations": split.get("num_test_simulations", len(split.get("test_simulation_ids", []))),
        "train_simulation_ids": split["train_simulation_ids"],
        "val_simulation_ids": split["val_simulation_ids"],
        "test_simulation_ids": split.get("test_simulation_ids", []),
        "train_records": records_to_json(split["train_records"]),
        "val_records": records_to_json(split["val_records"]),
        "test_records": records_to_json(split.get("test_records", [])),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a geometry-aware split manifest.")
    parser.add_argument("--data-root", type=str, default="warped-ifw")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test-simulations", type=int, default=0)
    parser.add_argument("--test-simulation-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exploration/geometry_split.json",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    records = scan_dataset_files(args.data_root)
    split = build_geometry_split(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_test_simulations=args.num_test_simulations,
        test_simulation_ids=args.test_simulation_ids,
    )
    save_split_manifest(split, args.output)
    print(json.dumps(summarize_split(split), indent=2))


if __name__ == "__main__":
    main()
