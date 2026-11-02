from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional at Phase A
    torch = None

    class Dataset:  # type: ignore[override]
        """Light fallback so the loader remains usable without PyTorch installed."""

        pass


SAMPLE_FILE_RE = re.compile(
    r"(?P<simulation_id>\d+)_(?P<window_id>\d+)-(?P<variant_id>\d+)\.npz$"
)


@dataclass(frozen=True)
class SampleRecord:
    """Metadata parsed from one dataset file name."""

    file_path: str
    simulation_id: int
    window_id: int
    variant_id: int

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name


ReturnMode = Literal["numpy", "torch"]


def parse_sample_filename(file_path: str | Path) -> SampleRecord:
    """Parse a file name like `1021_1-0.npz` into structured metadata."""

    path = Path(file_path)
    match = SAMPLE_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Unrecognized sample name format: {path.name!r}. "
            "Expected '{simulation_id}_{window_id}-{variant_id}.npz'."
        )

    return SampleRecord(
        file_path=str(path),
        simulation_id=int(match.group("simulation_id")),
        window_id=int(match.group("window_id")),
        variant_id=int(match.group("variant_id")),
    )


def scan_dataset_files(data_root: str | Path) -> list[SampleRecord]:
    """Scan a directory of `.npz` files and return sorted sample records."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No `.npz` files found under: {root}")

    return [parse_sample_filename(path) for path in files]


class GRAMDataset(Dataset):
    """Dataset loader for the GRaM transient airfoil point-cloud benchmark."""

    def __init__(
        self,
        data_root: str | Path,
        records: Optional[Sequence[SampleRecord]] = None,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        include_pressure: bool = True,
        return_mode: ReturnMode = "numpy",
        mmap_mode: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.records = list(records) if records is not None else scan_dataset_files(self.data_root)
        if limit is not None:
            self.records = self.records[:limit]

        self.transform = transform
        self.include_pressure = include_pressure
        self.return_mode = return_mode
        self.mmap_mode = mmap_mode

        if self.return_mode == "torch" and torch is None:
            raise ImportError(
                "return_mode='torch' requires PyTorch. Install the `training` extra first."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = self.load_sample(record)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.return_mode == "torch":
            sample = _to_torch(sample)
        return sample

    def load_sample(self, record: SampleRecord) -> dict[str, Any]:
        """Load one sample and convert the airfoil indices into a boolean mask."""

        with np.load(record.file_path, allow_pickle=False, mmap_mode=self.mmap_mode) as data:
            keys = set(data.files)

            pos = np.asarray(data["pos"], dtype=np.float32)
            idcs_airfoil = np.asarray(data["idcs_airfoil"], dtype=np.int64)
            velocity_in = np.asarray(data["velocity_in"], dtype=np.float32)
            velocity_out = np.asarray(data["velocity_out"], dtype=np.float32)
            pressure = None
            if self.include_pressure and "pressure" in keys:
                pressure = np.asarray(data["pressure"], dtype=np.float32)

            time_values = None
            if "t" in keys:
                time_values = np.asarray(data["t"], dtype=np.float32)

        airfoil_mask = np.zeros(pos.shape[0], dtype=bool)
        if idcs_airfoil.size:
            airfoil_mask[idcs_airfoil] = True

        return {
            "pos": pos,
            "airfoil_mask": airfoil_mask,
            "idcs_airfoil": idcs_airfoil,
            "velocity_in": velocity_in,
            "velocity_out": velocity_out,
            "pressure": pressure,
            "time": time_values,
            "metadata": {
                "file_path": record.file_path,
                "file_name": record.file_name,
                "simulation_id": record.simulation_id,
                "window_id": record.window_id,
                "variant_id": record.variant_id,
                "num_points": int(pos.shape[0]),
                "num_airfoil_points": int(airfoil_mask.sum()),
            },
        }


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a small, JSON-serializable summary for quick inspection."""

    pressure = sample.get("pressure")
    return {
        "file_name": sample["metadata"]["file_name"],
        "simulation_id": sample["metadata"]["simulation_id"],
        "window_id": sample["metadata"]["window_id"],
        "variant_id": sample["metadata"]["variant_id"],
        "num_points": sample["metadata"]["num_points"],
        "num_airfoil_points": sample["metadata"]["num_airfoil_points"],
        "pos_shape": list(sample["pos"].shape),
        "velocity_in_shape": list(sample["velocity_in"].shape),
        "velocity_out_shape": list(sample["velocity_out"].shape),
        "pressure_shape": list(pressure.shape) if pressure is not None else None,
    }


def records_to_json(records: Sequence[SampleRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def _to_torch(value: Any) -> Any:
    if torch is None:
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    return value


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect GRaM `.npz` samples.")
    parser.add_argument("--data-root", type=str, default="warped-ifw")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    records = scan_dataset_files(args.data_root)
    dataset = GRAMDataset(args.data_root, records=records, limit=args.limit)

    payload = {
        "num_files": len(records),
        "num_simulations": len({record.simulation_id for record in records}),
        "examples": [summarize_sample(dataset[idx]) for idx in range(len(dataset))],
    }
    print(json.dumps(payload, indent=args.indent))


if __name__ == "__main__":
    main()
