# BA-CNO: Boundary-Aware Conservative Neural Operator

A graph neural surrogate for fluid flow whose message passing is built like a finite-volume scheme. Edge messages are **antisymmetric fluxes** (`F_ij = −F_ji`) weighted by symmetric learned face areas. An optional **flux-divergence decoder** writes the velocity update as `du_i = (Σ_j G_ij + boundary sources) / V_i`, so the interior cannot create or destroy discrete momentum.

The idea started in the [GRaM @ ICLR 2026 challenge](https://gram-competition.github.io) (transient flow around F1-style front-wing elements). This repo now tests it against **established CFD benchmarks with published baselines**, and judges it on accuracy *and* on physics a CFD engineer would check.

> Status: benchmark study in progress. See [`docs/STUDY_PLAN.md`](docs/STUDY_PLAN.md) for the locked plan. Results will be added here once they exist.

## Benchmarks

| Benchmark | Physics | Baseline compared against | Physics checks |
|-----------|---------|---------------------------|----------------|
| CylinderFlow (DeepMind) | 2D unsteady laminar vortex shedding | MeshGraphNets (Pfaff et al. 2021) | Strouhal number, momentum/energy drift, divergence, wake vorticity |
| AirfRANS (NeurIPS 2022) | 2D steady RANS airfoils, Re 2–6 M | GraphSAGE / GUNet / MLP / PointNet | C_L, C_D, Cp, Cf, boundary-layer profiles; Reynolds / AoA / unseen-geometry OOD |
| Turbulent channel (phase 2) | wall-bounded turbulence | Lee & Moser DNS | u⁺(y⁺), Reynolds stresses, spectra |

## Layout

```
ba_cno/
  graph.py                 mesh/k-NN graphs, reverse-edge map, dual volumes, scatter ops
  models/
    meshgraphnet.py        reference MeshGraphNets
    conservative_operator.py  BA-CNO (flux_mode: none|soft|hard, decoder: pointwise|flux_divergence)
    graphsage.py, flux_operator.py   original GRaM challenge models (legacy)
  metrics/physics.py       LSQ gradients, divergence, vorticity, energy, momentum, Strouhal, wall forces
  data/
    tfrecord.py            TensorFlow-free reader for MeshGraphNets datasets
    cylinder.py            CylinderFlow adapter
    gram/                  GRaM challenge data pipeline (legacy)
  rollout.py, normalization.py
scripts/cylinder/          download.sh, convert.py, strouhal_check.py, train.py, budget.py, evaluate.py
scripts/gram/              original challenge training scripts
configs/cylinder/          the ablation ladder (mgn, ba_cno_soft, ba_cno_hard, ba_cno_none_fluxdec, ba_cno_hard_fluxdec)
tests/test_core.py         conservation, antisymmetry, gradient and parser tests
```

## Quick start (CylinderFlow)

```bash
pip install -e ".[dev]"
python -m pytest tests -q

# 'train:50' fetches only the first ~50 train trajectories (~750 MB of 13.6 GB); use 'train' on Colab
bash scripts/cylinder/download.sh data/cylinder_flow/raw valid test train:50
python scripts/cylinder/convert.py --raw data/cylinder_flow/raw --out data/cylinder_flow --splits valid test
python scripts/cylinder/convert.py --raw data/cylinder_flow/raw --out data/cylinder_flow --splits train --max 50

# Ground-truth Strouhal sanity check (Day 1)
python scripts/cylinder/strouhal_check.py --split valid test

# Mac smoke run. batch_size=1: MPS keeps one compiled graph per batch shape, so mixed-mesh batches leak memory
python scripts/cylinder/train.py --config configs/cylinder/ba_cno_hard_fluxdec.yaml \
  --override train.max_steps=600 train.batch_size=1 data.max_trajectories=20 output_dir=outputs/smoke/ba_cno_hard_fluxdec
python scripts/cylinder/evaluate.py --run outputs/smoke/ba_cno_hard_fluxdec --max-trajectories 2
```

Training resumes automatically from `<output_dir>/last.pt`, which is useful on Colab and Kaggle. GPU runs (timing, step budget, training, evaluation) go through [`notebooks/03_cylinder_colab.ipynb`](notebooks/03_cylinder_colab.ipynb); `scripts/cylinder/budget.py` turns the measured it/s into one step budget for all configs.

## References

- Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia. *Learning Mesh-Based Simulation with Graph Networks.* ICLR 2021.
- Bonnet, Mazari, Cinnella, Gallinari. *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions.* NeurIPS 2022 Datasets & Benchmarks.
- Schäfer, Turek. *Benchmark computations of laminar flow around a cylinder.* 1996.
- Lee, Moser. *Direct numerical simulation of turbulent channel flow up to Re_τ ≈ 5200.* JFM 2015.
