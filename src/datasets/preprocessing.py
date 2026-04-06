from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional fallback
    cKDTree = None


def normalize_positions(
    pos: np.ndarray,
    mode: str = "center_scale",
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Normalize coordinates either by centering or center+global scale."""

    pos = np.asarray(pos, dtype=np.float32)
    center = pos.mean(axis=0, keepdims=True)
    centered = pos - center

    if mode == "none":
        scale = np.array([1.0], dtype=np.float32)
        normalized = pos.copy()
    elif mode == "center":
        scale = np.array([1.0], dtype=np.float32)
        normalized = centered
    elif mode == "center_scale":
        radius = np.linalg.norm(centered, axis=1).max()
        scale = np.array([max(float(radius), eps)], dtype=np.float32)
        normalized = centered / scale
    else:
        raise ValueError(f"Unsupported coordinate normalization mode: {mode}")

    stats = {
        "center": center.squeeze(0).astype(np.float32),
        "scale": scale.astype(np.float32),
        "mode": np.array([mode]),
    }
    return normalized.astype(np.float32), stats


def normalize_velocity(
    velocity: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply per-component normalization to velocity arrays shaped `(..., 3)`."""

    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return ((velocity - mean) / np.maximum(std, eps)).astype(np.float32)


def fit_velocity_stats(
    samples: Iterable[dict[str, Any]],
    include_targets: bool = True,
    max_samples: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Fit global velocity statistics using the training split only."""

    sum_vec = np.zeros(3, dtype=np.float64)
    sum_sq_vec = np.zeros(3, dtype=np.float64)
    count = 0

    for idx, sample in enumerate(samples):
        if max_samples is not None and idx >= max_samples:
            break

        arrays = [sample["velocity_in"]]
        if include_targets:
            arrays.append(sample["velocity_out"])

        for array in arrays:
            flat = np.asarray(array, dtype=np.float64).reshape(-1, 3)
            sum_vec += flat.sum(axis=0)
            sum_sq_vec += np.square(flat).sum(axis=0)
            count += flat.shape[0]

    if count == 0:
        raise ValueError("Cannot fit velocity statistics on an empty sample set.")

    mean = sum_vec / count
    variance = np.maximum(sum_sq_vec / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count": np.array([count], dtype=np.int64),
    }


def compute_boundary_distance(
    pos: np.ndarray,
    airfoil_mask: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Compute nearest-airfoil distance for each point."""

    pos = np.asarray(pos, dtype=np.float32)
    airfoil_mask = np.asarray(airfoil_mask, dtype=bool)
    airfoil_points = pos[airfoil_mask]

    if airfoil_points.size == 0:
        return np.full(pos.shape[0], np.inf, dtype=np.float32)

    if cKDTree is not None:
        tree = cKDTree(airfoil_points)
        distances, _ = tree.query(pos, k=1)
        return distances.astype(np.float32)

    distances = np.empty(pos.shape[0], dtype=np.float32)
    for start in range(0, pos.shape[0], chunk_size):
        stop = min(start + chunk_size, pos.shape[0])
        diff = pos[start:stop, None, :] - airfoil_points[None, :, :]
        distances[start:stop] = np.linalg.norm(diff, axis=-1).min(axis=1)
    return distances


def deterministic_subsample_indices(
    num_points: int,
    target_points: int,
    seed: int = 0,
    priority_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick a reproducible subset while preserving both boundary and fluid coverage."""

    if target_points >= num_points:
        return np.arange(num_points, dtype=np.int64)

    rng = np.random.default_rng(seed)

    if priority_mask is None:
        indices = rng.choice(num_points, size=target_points, replace=False)
        return np.sort(indices.astype(np.int64))

    priority_mask = np.asarray(priority_mask, dtype=bool)
    priority_indices = np.flatnonzero(priority_mask)
    other_indices = np.flatnonzero(~priority_mask)

    if priority_indices.size == 0:
        indices = rng.choice(num_points, size=target_points, replace=False)
        return np.sort(indices.astype(np.int64))

    if other_indices.size == 0:
        chosen = rng.choice(priority_indices, size=target_points, replace=False)
        return np.sort(chosen.astype(np.int64))

    observed_ratio = priority_indices.size / num_points
    target_priority = int(round(target_points * observed_ratio))

    min_priority = min(priority_indices.size, max(64, target_points // 8))
    max_priority = min(priority_indices.size, max(min_priority, int(target_points * 0.5)))
    target_priority = int(np.clip(target_priority, min_priority, max_priority))
    target_priority = min(target_priority, target_points - 1)

    target_other = target_points - target_priority
    if target_other > other_indices.size:
        target_other = other_indices.size
        target_priority = target_points - target_other

    sampled_priority = rng.choice(priority_indices, size=target_priority, replace=False)
    sampled_other = rng.choice(other_indices, size=target_other, replace=False)
    merged = np.concatenate([sampled_priority, sampled_other])
    return np.sort(merged.astype(np.int64))


def subsample_sample(
    sample: dict[str, Any],
    target_points: int,
    seed: int = 0,
    keep_airfoil: bool = True,
) -> dict[str, Any]:
    """Subsample all point-indexed arrays consistently."""

    num_points = sample["pos"].shape[0]
    priority_mask = sample["airfoil_mask"] if keep_airfoil else None
    indices = deterministic_subsample_indices(
        num_points=num_points,
        target_points=target_points,
        seed=seed,
        priority_mask=priority_mask,
    )

    index_lookup = np.full(num_points, -1, dtype=np.int64)
    index_lookup[indices] = np.arange(indices.shape[0], dtype=np.int64)

    new_sample = dict(sample)
    new_sample["pos"] = sample["pos"][indices]
    new_sample["airfoil_mask"] = sample["airfoil_mask"][indices]
    new_sample["idcs_airfoil"] = np.flatnonzero(new_sample["airfoil_mask"]).astype(np.int64)
    new_sample["velocity_in"] = sample["velocity_in"][:, indices, :]
    new_sample["velocity_out"] = sample["velocity_out"][:, indices, :]

    if sample.get("pressure") is not None:
        new_sample["pressure"] = sample["pressure"][:, indices]

    if sample.get("boundary_distance") is not None:
        new_sample["boundary_distance"] = sample["boundary_distance"][indices]

    metadata = dict(sample.get("metadata", {}))
    metadata["num_points_before_subsample"] = int(num_points)
    metadata["num_points"] = int(indices.shape[0])
    metadata["num_airfoil_points"] = int(new_sample["airfoil_mask"].sum())
    metadata["subsample_seed"] = seed
    new_sample["metadata"] = metadata
    new_sample["subsample_indices"] = indices
    new_sample["index_lookup"] = index_lookup
    return new_sample


def build_input_features(
    sample: dict[str, Any],
    include_pressure: bool = False,
) -> np.ndarray:
    """Create per-point Phase A features for later baseline experiments."""

    features = [
        np.asarray(sample["pos"], dtype=np.float32),
        sample["airfoil_mask"].astype(np.float32)[:, None],
    ]

    if sample.get("boundary_distance") is not None:
        features.append(np.asarray(sample["boundary_distance"], dtype=np.float32)[:, None])

    velocity_history = np.asarray(sample["velocity_in"], dtype=np.float32).transpose(1, 0, 2)
    features.append(velocity_history.reshape(sample["pos"].shape[0], -1))

    pressure = sample.get("pressure")
    if include_pressure and pressure is not None:
        features.append(np.asarray(pressure[:5], dtype=np.float32).T)

    return np.concatenate(features, axis=1).astype(np.float32)


@dataclass
class PhaseAPreprocessor:
    """Small reusable preprocessing bundle for Phase A experiments."""

    coordinate_mode: str = "center_scale"
    velocity_mean: Optional[np.ndarray] = None
    velocity_std: Optional[np.ndarray] = None
    subsample_points: Optional[int] = None
    seed: int = 0
    compute_boundary: bool = True
    keep_airfoil: bool = True

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        processed = dict(sample)
        processed["metadata"] = dict(sample.get("metadata", {}))

        if self.subsample_points is not None:
            processed = subsample_sample(
                processed,
                target_points=self.subsample_points,
                seed=self.seed,
                keep_airfoil=self.keep_airfoil,
            )

        processed["pos"], coord_stats = normalize_positions(
            processed["pos"],
            mode=self.coordinate_mode,
        )
        processed["coord_stats"] = coord_stats

        if self.compute_boundary:
            processed["boundary_distance"] = compute_boundary_distance(
                processed["pos"],
                processed["airfoil_mask"],
            )

        if self.velocity_mean is not None and self.velocity_std is not None:
            processed["velocity_in"] = normalize_velocity(
                processed["velocity_in"],
                mean=self.velocity_mean,
                std=self.velocity_std,
            )
            processed["velocity_out"] = normalize_velocity(
                processed["velocity_out"],
                mean=self.velocity_mean,
                std=self.velocity_std,
            )
            processed["velocity_stats"] = {
                "mean": np.asarray(self.velocity_mean, dtype=np.float32),
                "std": np.asarray(self.velocity_std, dtype=np.float32),
            }

        processed["input_features"] = build_input_features(processed)
        return processed

    def fit(self, samples: Iterable[dict[str, Any]], include_targets: bool = True) -> "PhaseAPreprocessor":
        stats = fit_velocity_stats(samples, include_targets=include_targets)
        self.velocity_mean = stats["mean"]
        self.velocity_std = stats["std"]
        return self


def save_velocity_stats(stats: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: np.asarray(value).tolist() for key, value in stats.items()}
    output_path.write_text(json.dumps(serializable, indent=2))
    return output_path


def load_velocity_stats(stats_path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(stats_path).read_text())
    return {
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
        "count": np.asarray(payload.get("count", [0]), dtype=np.int64),
    }
