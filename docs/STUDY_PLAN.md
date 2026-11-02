# BA-CNO benchmark study: plan and locked spec

*Boundary-Aware Conservative Neural Operator: does a finite-volume inductive bias make a graph neural surrogate more physically faithful?*

Origin: GRaM @ ICLR 2026 challenge (transient flow around F1-style front-wing elements, `scripts/gram/`). The challenge is over and its test set and metric were never public. This study takes the core idea and tests it on benchmarks with published baselines and trusted CFD reference data.

## 1. Research question and hypotheses

> When the message passing in a graph surrogate is made to behave like finite-volume fluxes (antisymmetric, conservative, with boundary-only sources), is the surrogate (a) as accurate as published baselines, (b) more physically faithful, and (c) better at generalising to unseen geometry and flow conditions?

| ID | Hypothesis | Measured by |
|----|------------|-------------|
| H1 accuracy | BA-CNO matches or beats MeshGraphNets / AirfRANS baselines at equal parameter count and training budget | RMSE-1/50/all (Cylinder); field MSE, surface p MSE, C_D/C_L error, Spearman ρ (AirfRANS) |
| H2 physics | Conservative fluxes reduce non-physical behaviour in long rollouts | momentum/energy drift, divergence RMS, wake vorticity retention, Strouhal error; surface Cp, Cf, boundary-layer profiles |
| H3 generalisation | The gain grows out of distribution | AirfRANS Reynolds-OOD and AoA-OOD tasks, plus an unseen-airfoil-family split; Cylinder test set (unseen cylinder size/position/inflow) |
| H4 mechanism | The benefit comes from the conservation structure, not extra parameters | ablation ladder below, with the same latent size and depth everywhere. Parameter counts are reported, since the face weights add about 5% and the flux decoder about 20%. If a variant wins, a width-matched MGN is run to rule out capacity as the cause |

## 2. Model ladder (identical 128-latent, 15-layer encode-process-decode)

| Config | Processor messages | Decoder | Purpose |
|--------|-------------------|---------|---------|
| `mgn` | free | pointwise | published baseline, retrained under our budget |
| `ba_cno_soft` | free + antisymmetry penalty | pointwise | original GRaM idea |
| `ba_cno_hard` | antisymmetric by construction, symmetric face weights | pointwise | conservation as structure, not penalty |
| `ba_cno_none_fluxdec` | free | flux-divergence | isolates the decoder |
| `ba_cno_hard_fluxdec` | antisymmetric | flux-divergence | full model |

The flux-divergence decoder writes the velocity update as `du_i = (Σ_j G_ij + b_i S_i) / V_i` with `G_ij = −G_ji`, and sources only on boundary nodes. Total discrete momentum `Σ V_i du_i` can then change only through the boundaries. This is guaranteed to round-off and is checked in `tests/test_core.py`.

**Wording discipline:** this is discrete conservation on the graph induced by the mesh, using lumped dual volumes. It does not mean the network solves the Navier–Stokes equations, and it does not enforce ∇·u = 0. Divergence is measured, not imposed.

## 3. Benchmark 1: CylinderFlow (week 1)

- **Data:** DeepMind MeshGraphNets `cylinder_flow`. COMSOL incompressible NS, 1000/100/100 trajectories × 600 steps, dt = 0.01, ~1.9k nodes. Cylinder size and position and the inflow vary between trajectories.
- **Protocol (as Pfaff et al. 2021):** one-step training on `v_{t+1} − v_t` with N(0, 0.02) input noise on fluid nodes; full 599-step rollout on the 100 test trajectories; inflow and wall nodes reset to ground truth.
- **Published reference (MGN, ×10⁻³):** RMSE-1 2.34 ± 0.12, RMSE-50 6.3 ± 0.7, RMSE-all 40.88 ± 7.2 (trained for ~10M steps). Our runs use an equal but smaller budget, so the **retrained `mgn` is the fair baseline**. The published number is a reference ceiling, and we say so explicitly.
- **Physics metrics** (each reported next to the CFD value): Strouhal number from a wake probe 2D downstream; momentum and kinetic-energy drift over the rollout; interior divergence RMS; wake vorticity RMS (do vortices survive or get smeared?).
- **Sanity check of the data against classical CFD:** the domain follows the Schäfer & Turek (1996) DFG benchmark. For their 2D-2 case (Re = 100), the reference is St ≈ 0.295–0.305. Check that the ground-truth St from our wake probe lands in this band for trajectories near Re ≈ 100. This validates the metric pipeline before it is used to judge the models.

## 4. Benchmark 2: AirfRANS (week 2)

- **Data:** AirfRANS (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks). 1000 2D incompressible steady RANS (k-ω SST) simulations of NACA 4- and 5-digit airfoils, Re 2–6 M, AoA −5° to 15°. The `airfrans` Python package (download is ~9 GB processed).
- **Tasks:** `full` (800/200), `reynolds` (train Re 3–5 M → test 2–3 M and 5–6 M), `aoa` (train −2.5° to 12.5° → test the extremes, near stall).
- **Unseen geometry:** every AirfRANS test airfoil is already a different NACA shape from the training set. We also add our own harder split, **train on 4-digit, test on 5-digit**, clearly labelled as ours.
- **Model change:** steady problem, so a pointwise decoder. The processor keeps the hard antisymmetric fluxes. The graph is mesh connectivity or k-NN on subsampled points (decided on Day 8 after a timing test).
- **Metrics:** official ones (volume MSE per field, surface p MSE, C_D and C_L relative error, Spearman ρ_D and ρ_L). Near-wall ones on top: Cp(x/c) curves, skin friction Cf(x/c) via `Simulation.wallshearstress`, and boundary-layer velocity profiles at x/c = 0.3 and 0.7 via `Simulation.boundary_layer`. These are compared against the RANS reference.
- **Caveat to cite correctly:** the AirfRANS main-paper tables were trained with plain MSE because of an implementation error, and the corrected results are in their Appendix N. We compare against the appendix numbers **and** retrain GraphSAGE under our own budget.

## 5. Phase 2 (after the 2 weeks): near-wall turbulence

Turbulent channel DNS (JHTDB or a small self-generated Re_τ ≈ 180 box). Rollouts are compared against Lee & Moser (2015) and Moser–Kim–Mansour (1999): u⁺ vs y⁺, Reynolds-stress profiles, and energy spectra. This tests the GRaM organisers' claim that high-frequency content is the hard part.

## 6. Compute plan

| Stage | Where | What |
|-------|-------|------|
| Prototype | MacBook (MPS/CPU) | `pytest`, conversion, 50-trajectory subsets, 1–2k-step smoke runs, eval on 5 trajectories |
| Main runs | Colab/Kaggle GPU | every config at the same step budget; auto-resume from `last.pt` across sessions |

The step budget is fixed on Day 2 from a measured it/s on the Colab GPU (default 300k). If time allows, 3 seeds for `mgn` and the best BA-CNO variant; otherwise 1 seed, with that limitation stated.

## 7. Two-week timeline

| Day | Work | Done when |
|-----|------|-----------|
| 1 | Download and convert Cylinder valid/test plus 50 train trajectories; run tests; St sanity check vs Schäfer–Turek | GT St pipeline validated |
| 2 | Smoke-train all 5 configs on the Mac; Colab notebook; time it/s; fix the budget | budget locked |
| 3–5 | Colab runs of the 5 configs (in parallel across sessions) | `final.pt` for each |
| 6 | Full 100-trajectory evaluation; figures: RMSE curve, St scatter, momentum drift, vorticity snapshots | Cylinder results table |
| 7 | Write Cylinder section; decide go/no-go on claims | draft section |
| 8 | AirfRANS download, adapter, timing test, graph choice | loader + metrics run on GT |
| 9–11 | Train `graphsage`/`mgn`, `ba_cno_hard` on full, reynolds, aoa (+ 4→5-digit split) | checkpoints |
| 12 | AirfRANS evaluation incl. Cp, Cf, BL profiles | results table + figures |
| 13 | README results section, reproducibility pass, clean repo | repo ready |
| 14 | Portfolio case study draft (only claims the data supports) | article draft |

## 8. Deliverables

1. Clean GitHub repo `ramtarun02/BA-CNO`: package, configs, tests, Colab notebook, results tables, figures.
2. Portfolio case study article, **only if** the results give a real insight. A clear negative result ("the conservation bias doesn't help accuracy but does fix momentum drift", or "it doesn't help at all, and here is why") is still a valid, honest case study.

## 9. Risks

- **Budget gap to published MGN:** mitigated by retraining MGN under the same budget.
- **Flux decoder slower or less stable:** watch the loss scale; target normalisation is scale-only for this decoder (centring would add a fake source).
- **AirfRANS memory (~180k points per sim):** subsample as the AirfRANS paper does, or use a coarser graph.
- **Over-claiming:** every physics number is reported next to its CFD value. No "physics-informed" language beyond what the tests prove.
