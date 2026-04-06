

# Boundary-aware point-cloud operator experiments for transient airfoil flow prediction.

GRaM-PINO Challenge 

## Phase A status

This repository now includes the initial data foundation for the experiment plan:

- project scaffold under `src/`, `configs/`, `notebooks/`, and `outputs/`
- dataset indexing and loading for `warped-ifw/*.npz`
- geometry-aware train/validation splits by full simulation ID
- preprocessing utilities for normalization, boundary distance, and debug subsampling
- a starter notebook for data exploration

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m src.datasets.gram_dataset --data-root warped-ifw --limit 3
```

## Data assumptions

Each `.npz` sample is expected to contain:

- `pos`: `(N, 3)` point coordinates
- `idcs_airfoil`: airfoil point indices
- `velocity_in`: `(5, N, 3)` past velocity frames
- `velocity_out`: `(5, N, 3)` target future velocity frames
- `pressure`: optional `(10, N)` pressure field
