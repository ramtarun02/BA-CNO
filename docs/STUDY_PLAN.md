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

## 10. Progress log

### Day 1 (2026-11-02): data, tests, Strouhal check

- **Data:** valid (100) and test (100) downloaded and converted. Train uses the first 50 trajectories, fetched as a byte range (`download.sh ... train:50`). The reader skips the truncated last record.
- **Tests:** `pytest` 7/7 pass, including new tests for partial-download truncation and memory-mapped loading.
- **Bug fixed, bulk inflow speed:** `CylinderTrajectory.inflow_speed` averaged the inflow nodes. Those cluster near the walls, so the result was ~30% below the true bulk velocity (1.02 vs 1.47 on test/0). It now integrates the parabolic profile (trapezoid rule, no-slip ends). Absolute St and Re were previously inflated by ~45%; model-vs-truth St ratios were unaffected.
- **Domain:** 1.6 × 0.41 channel (height matches Schäfer–Turek, length is 1.6 not 2.2). D/H ranges ~0.10–0.39. With ν = 1e-3 assumed, Re ranges 14–211. 190/200 trajectories shed.
- **Strouhal check** (`scripts/cylinder/strouhal_check.py`, output `outputs/cylinder/strouhal_check/`):
  - *Estimator: PASS.* FFT peak vs zero-crossing count on the developed half: median relative difference 0.18%, p90 1.3%.
  - *Against Schäfer–Turek: MISMATCH.* The 5 trajectories with Re 80–120 and D/H 0.22–0.27 give St = 0.242–0.261 (median 0.245), against the 0.295–0.305 band. Across all D/H, St rises with blockage, and the high-blockage trajectories (D/H > 0.3) do reach 0.30–0.35. A first wider window without a blockage filter looked like a pass (median 0.290), but only because of that blockage effect.
  - *Open question:* the most likely cause is that ν ≠ 1e-3, so the Re labels are off; the shorter channel and off-centre cylinders are secondary candidates. ν could not be recovered from the upstream pressure gradient (the sign is inconsistent with Poiseuille near the inlet). **Consequence:** St is used only as a *relative* metric (prediction vs ground truth on the same trajectory). That comparison is valid whatever ν is. No absolute St–Re claim is made against Schäfer–Turek.

### Day 2 (2026-11-02): smoke runs, Colab, timing

- **Memory, Colab:** the full train split is 13.6 GB of float32, more than free-Colab RAM. Trajectories are now memory-mapped from the uncompressed `.npz` (`mmap_npz`), so only the meshes stay in RAM.
- **Memory, Mac:** with `batch_size=2`, MPS OOMs after ~500 steps. MPS keeps one compiled graph per batch shape, and mixed-mesh batches give O(trajectories²) shapes; driver memory grew ~40 MB/step and `empty_cache()` doesn't help. Mac smoke runs therefore use `batch_size=1` and 20 trajectories. CUDA runs keep `batch_size=2`. `train.py` warns about this.
- **Smoke runs** (600 steps, MPS, bs=1): all 5 configs train, save, and evaluate end to end (`outputs/smoke/`). The loss falls for each. The rollout metrics at 600 steps carry no information.

| config | params | Mac it/s (bs=1) |
|---|---|---|
| mgn | 2,333,059 | 4.11 |
| ba_cno_soft | 2,457,874 | 2.58 |
| ba_cno_hard | 2,457,874 | 3.09 |
| ba_cno_none_fluxdec | 2,556,948 | 3.88 |
| ba_cno_hard_fluxdec | 2,556,948 | 2.99 |

  Parameter overhead vs MGN: +5.3% (face weights), +9.6% (flux decoder). `ba_cno_soft` is the slowest, because of the antisymmetry penalty pass.
- **Colab:** `notebooks/03_cylinder_colab.ipynb` covers setup, the timing cell, and resumable training and evaluation with checkpoints on Drive. `scripts/cylinder/budget.py` sets the budget so that the slowest config fits the hours available.
- **Budget: NOT YET LOCKED.** It must come from the Colab GPU timing cell, not from Mac numbers. For reference, the Mac numbers give 175k steps at 20 h per config.

### Day 2 (cont., 2026-11-07): compute moved to RunPod, budget locked at 1M steps

- **Compute:** RunPod Secure RTX 4090 ($0.89/h, pod `6xz8jq6lecbiot`, US-NC-1). Community 4090/5080 stock ran out at placement. All 5 configs train **concurrently** on the one GPU via `scripts/cylinder/run_pod.sh`. Data is on the pod's local disk; checkpoints are on `/workspace`.
- **Timing** (5 concurrent runs, batch 2, CUDA): `mgn` 10.1, `ba_cno_none_fluxdec` 10.0, `ba_cno_hard` 9.7, `ba_cno_hard_fluxdec` 9.7, `ba_cno_soft` 4.7 it/s (26.4 alone). Aggregate ≈ 44 it/s, ~1.7× running the configs one after another.
- **Budget locked: 1,000,000 steps per config** (1/10 of the published MGN budget; ~3.3 epochs of the 599k training pairs). Estimated 34 h wall-clock, ≈ $30. Projections were 5M ≈ 171 h / $152 and 10M ≈ 341 h / $304. 1M was chosen for the reasons below; extra budget goes to seeds (and a 300k point) rather than more steps.
  - A longer run can't simply be continued later, because the LR schedule decays over `max_steps`.
  - Inductive-bias gains may shrink at very long budgets.
  - Seeds give error bars.
- **Validation during training:** every 20k steps, one-step loss on 64 fixed valid pairs plus 50-step rollout RMSE on 2 valid trajectories. `best.pt` is selected on validation RMSE-50, never on test.
- **Bugs found while scaling up:**
  - `rollout_cylinder` built node features on the CPU, which crashed GPU rollouts (and would have crashed GPU evaluation).
  - Memory-mapping 1,000 trajectories needs ~2,000 file descriptors (mmap dups each fd), against a 1,024 default; the dataset now raises the soft limit.
  - The pod's Python is externally managed (PEP 668).
