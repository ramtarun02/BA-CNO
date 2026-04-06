
## Research objective

Can a point-cloud neural operator with **finite-volume-inspired local flux updates** outperform a black-box spatiotemporal predictor on transient aerodynamic rollout? 

A concrete hypothesis set could be:

- H1: Local flux-style message passing improves generalization across different airfoil arrangements.
- H2: Boundary-aware processing helps because the airfoil surface is explicitly marked by `idcs_airfoil` and the task assumes no-slip conditions.
- H3: Multi-horizon training with residual rollout improves 5-step prediction quality over direct unconstrained decoding.


## Problem formulation

Each sample contains positions `pos` of shape `(100k, 3)`, airfoil surface indices `idcs_airfoil`, 5 past velocity frames `velocity_in`, and 5 target future frames `velocity_out`; pressure is also included but described as being provided for completeness. The benchmark spans 181 simulated geometries with one to three differently sized airfoils at random relative positions and pitch angles, and submissions are scored on a held-out test split using an undisclosed accuracy metric based on similarity to the true future velocity fields.

Formally, define the model as a geometry-conditioned operator

$$
\hat{U}_{t+1:t+5} = \mathcal{G}_\theta(X, B, U_{t-4:t})
$$

where $X \in \mathbb{R}^{N \times 3}$ are sampled positions, $B$ encodes boundary information from `idcs_airfoil`, and $U_{t-4:t}$ are the 5 input velocity frames. Because the data are not given on the original CFD mesh, your “governing mechanism” should be framed as a learned discrete transport rule on an induced graph over sampled points, not as an exact finite-volume solver. 

## Method design

Use a 4-stage model: feature construction, graph building, operator blocks, and rollout head. This is practical because the data are already subsampled to 100k points.

### 1. Feature construction

For each point $i$, create a feature vector with:

- Coordinates $x_i = (x,y,z)$.
- Binary airfoil flag $b_i$, set to 1 if $i \in \texttt{idcs\_airfoil}$.
- Distance-to-boundary proxy, computed as distance to the nearest airfoil point.
- Recent velocity history $u_i^{t-4},\dots,u_i^t$, flattened or temporally encoded.
- Optional pressure features from the 5 input times only, used in ablations because pressure is available in the sample structure. 

A minimal feature tensor is therefore shape `(N, 3 + 1 + 1 + 15)` if you use coordinates, flag, boundary distance, and 5 velocity frames with 3 components each. This gives the model explicit access to geometry, boundary, and temporal information instead of forcing it to infer everything from raw velocity alone. 

### 2. Graph construction

Build a geometric graph over points using either k-NN in Euclidean space or radius neighborhoods. Since each sample has 100k points, start with a **downsampled training graph** for tractability, for example 8k–20k nodes during early experiments, then scale up only after the pipeline works. 

Use at least two graph types in ablation:

- Spatial k-NN graph: simple and robust.
- Hybrid graph: k-NN plus extra edges from each non-airfoil point to a few nearest airfoil points.
The second option gives stronger boundary coupling, which is important because the benchmark explicitly encodes the airfoil surface as a subset of sampled points under a no-slip condition. 


### 3. Finite-volume-inspired operator block

Your core block should mimic a conservative local update:

$$
h_i^{(\ell+1)} = h_i^{(\ell)} + \phi_{\text{self}}(h_i^{(\ell)}, g_i) + \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\,\phi_{\text{flux}}(h_i^{(\ell)}, h_j^{(\ell)}, x_i-x_j, g_i, g_j)
$$

where $h_i$ is the latent state, $g_i$ contains geometry features, and $\alpha_{ij}$ is a normalization weight. This block is “finite-volume-inspired” because it updates each point from local exchange terms and a self term, resembling flux divergence plus local source correction. 

To make it more mechanistic, split the edge update into directional pieces:

- Relative geometry term: $x_j-x_i$.
- Relative state term: $h_j-h_i$.
- Boundary interaction term: function of $b_i, b_j$, and boundary distance.
- Learned face weight or pseudo-area weight: scalar gate from an MLP.
This structure makes the edge message closer to a learned numerical flux than ordinary message passing. 

Add these regularization ideas:

- Antisymmetry loss, encouraging $F_{ij} + F_{ji} \approx 0$ away from boundaries.
- Boundary no-slip penalty on predicted velocity magnitude at airfoil points.
- Smoothness on low-energy regions only, so you do not wash out high-frequency wakes that the organizers flag as crucial. 


### 4. Temporal prediction head

Use one of two rollout strategies:

- Direct multi-horizon head: predict all 5 future frames in one shot.
- One-step recurrent head: predict one frame, append it, and roll forward 5 times.
Since the task is fixed to 5 input and 5 output frames, the direct head is a good baseline, while the recurrent head better matches the idea of a learned discrete time-stepping mechanism. 

My recommendation is staged:

1. Start with direct 5-frame prediction for speed and stability.
2. Move to recurrent residual rollout once the model learns sensible local dynamics.
This reduces engineering risk while preserving the research direction. 

## Loss design

Because the hidden evaluation metric is undisclosed, your safest plan is to optimize several accuracy-sensitive losses that target both amplitude and structure of the velocity field. The competition states only that evaluation measures similarity between predicted and ground-truth 3D velocity fields, so you should hedge against multiple plausible metrics. 

Use a composite loss:

$$
\mathcal{L} =
\lambda_1 \mathcal{L}_{\text{vel}}
+ \lambda_2 \mathcal{L}_{\text{multi-step}}
+ \lambda_3 \mathcal{L}_{\text{boundary}}
+ \lambda_4 \mathcal{L}_{\text{flux}}
+ \lambda_5 \mathcal{L}_{\text{spectral}}
$$

Recommended terms:

- $\mathcal{L}_{\text{vel}}$: pointwise L1 or Huber loss on velocity vectors over all future frames.
- $\mathcal{L}_{\text{multi-step}}$: later horizons weighted slightly more to improve rollout consistency.
- $\mathcal{L}_{\text{boundary}}$: penalty on nonzero velocity at airfoil points.
- $\mathcal{L}_{\text{flux}}$: antisymmetry or local conservation penalty in latent or decoded space.
- $\mathcal{L}_{\text{spectral}}$: frequency-sensitive loss to preserve fine wake structure. 

A very practical first weighting is:

- 1.0 velocity loss
- 0.5 late-horizon weighting
- 0.2 boundary loss
- 0.05 flux regularizer
- 0.1 spectral loss
Then tune after observing whether predictions are too smooth, unstable, or boundary-violating. These weights are heuristic, but the need for multi-horizon and boundary sensitivity follows directly from the 5-step forecasting setup and the stated no-slip condition. 


## Experimental program

Your experiments should be organized into phases, not a single giant attempt. Since the benchmark allows all available data to be used for training and holds evaluation on a hidden test split, your private validation scheme becomes critical. 

### Phase A: dataset understanding

Goals:

- Verify tensor shapes and sample consistency.
- Inspect point-cloud geometry.
- Check whether point sampling is fixed or varies across windows within a simulation.
- Compute velocity magnitude ranges and per-axis statistics.
- Measure fraction of airfoil points and boundary-distance distribution.
This phase is necessary because the number of airfoil points varies across samples, and that variation likely matters for both masking and loss weighting. 


### Phase B: strong baseline

Train a simple but credible baseline before any FVM mechanism:

- Input: coordinates + airfoil mask + 5-frame velocity history.
- Graph: k-NN.
- Model: 4–6 graph message-passing layers.
- Head: direct prediction of 5 future frames.
- Loss: plain L1 or Huber.
This establishes whether your fancy mechanism actually helps and gives you an ablation anchor. 


### Phase C: boundary-aware baseline

Add:

- Distance-to-airfoil feature.
- Stronger weighting of errors near boundary.
- Optional explicit overwrite or soft projection toward zero on airfoil points.
This phase tests whether geometry/boundary encoding alone already yields most of the gains. Since the dataset explicitly supplies airfoil indices and assumes no-slip on the airfoil surface, this is a high-value modification. 


### Phase D: flux-inspired operator

Replace generic message passing with the flux block. Compare:

- Standard symmetric messages.
- Directional flux messages.
- Directional messages + antisymmetry regularization.
- Directional messages + recurrent rollout.
This is your main scientific contribution phase. 


### Phase E: multi-scale refinement

Add:

- Two neighborhood scales.
- Coarse-to-fine encoding, for example far-field and near-boundary branches.
- Optional spectral loss or residual high-frequency head.
This phase is directly motivated by the organizers’ statement that low-frequency flow is easier and the key challenge is the high-frequency component. 


## Validation protocol

Because the official metric is hidden, build a validation suite that tracks multiple failure modes. The official test split is held out, and all public data can be used for training, so your own internal split must be geometry-aware rather than random-window-only. 

Use two validation splits:

- **Random window split**: easy sanity check.
- **Geometry split**: hold out entire simulation IDs or geometry families.
The second one is more meaningful because the competition spans varying airfoil arrangements, and you want genuine operator generalization to unseen geometry rather than memorization of local trajectory fragments. 

Track these metrics:

- Mean L1/L2 velocity error over all points and times.
- Error by horizon, from step 1 to step 5.
- Error on airfoil points, near-boundary shell, and far-field points.
- Kinetic-energy or velocity-magnitude distribution mismatch.
- High-frequency proxy, such as error after subtracting a local spatially smoothed field. 


## Risks and mitigations

Your biggest risks are memory, oversmoothing, and false “physics” claims. Each sample has 100k points and 10 total time steps, so end-to-end full-resolution graph training may be too expensive at first; begin with aggressive point subsampling, neighborhood caching, and patchwise training. 

Specific risks:

- Oversmoothing: common in graph models, especially harmful because high-frequency flow is the stated challenge. 
- Boundary leakage: the model predicts nonzero velocity on airfoil points despite the no-slip assumption. 
- Rollout drift: recurrent prediction accumulates error across 5 future steps. 
- Spurious conservation story: sampled points are not the original mesh control volumes, so phrase your contribution carefully as an induced-graph approximation. 

Mitigations:

- Residual connections and limited graph depth.
- Frequency-aware or gradient-aware loss.
- Boundary-aware features and penalties.
- Teacher forcing or scheduled sampling for recurrent rollout.
- Conservative wording in the paper and README. 


## Step-by-step build

Below is the implementation procedure I would follow in order.

### Step 1: Set up the repository

Create a clean project layout:

- `data/`
- `src/datasets/`
- `src/models/`
- `src/losses/`
- `src/train/`
- `src/eval/`
- `configs/`
- `notebooks/`
- `outputs/`
Keep experiment configuration separate from model code so you can iterate quickly and reproduce ablations cleanly.


### Step 2: Write a dataset loader

Implement a loader that reads one sample and returns:

- `pos` `(N, 3)`
- `airfoil_mask` `(N,)`
- `velocity_in` `(5, N, 3)`
- `velocity_out` `(5, N, 3)`
- optional `pressure` `(10, N)`
The sample format is explicitly listed in the challenge document, and the number of airfoil points may vary per simulation, so always convert `idcs_airfoil` into a boolean mask at load time. 

Also store the sample ID split into simulation ID and window index. That will help you create geometry-aware train/validation partitions instead of accidentally leaking windows from the same simulation across splits. 

### Step 3: Build preprocessing

Add preprocessing functions for:

- Coordinate normalization, either global or per-sample.
- Velocity normalization, preferably using training-set mean/std.
- Boundary distance computation using nearest airfoil point.
- Point subsampling for debug runs, for example 4096 or 8192 points.
This step matters because 100k-point full-resolution training will be slow, and early bugs are easier to catch on small subsets. 


### Step 4: Create visualization notebooks

Before training anything, plot:

- 3D point clouds with airfoil points highlighted.
- Velocity magnitude at each input/output frame.
- Boundary-distance histograms.
- Per-sample and per-simulation statistics.
This helps verify that your masks and normalization are correct and lets you spot degenerate samples early.


### Step 5: Implement a baseline graph model

Start with a simple architecture:

1. Point encoder MLP on per-point input features.
2. k-NN graph builder.
3. 4–6 graph residual blocks.
4. Output head producing 5 future velocity vectors per point.
Do not start with recurrent rollout or custom flux logic; first prove the pipeline can learn the task at all. The challenge itself suggests graph neural networks as a suitable model family, so this is fully aligned with the benchmark. 

### Step 6: Train the baseline on small subsets

Use a tiny setting first:

- 4k–8k points/sample.
- Batch size 1–2.
- 20–30 epochs.
- Huber loss on all 5 future frames.
Your success criterion is not leaderboard quality yet; it is getting decreasing validation loss and physically sensible qualitative predictions.


### Step 7: Add boundary-aware features

Extend the model input with:

- Airfoil mask.
- Distance to airfoil.
- Optionally, nearest-airfoil direction vector.
Then add a boundary penalty on predicted velocity at airfoil points. This is low-risk and highly justified because the challenge encodes the surface explicitly and states the no-slip boundary condition. 


### Step 8: Implement flux-style message passing

Create a new message block:

1. Compute edge features from $h_i, h_j, x_j-x_i, b_i, b_j$.
2. Pass through a flux MLP to get edge flux $F_{ij}$.
3. Aggregate incoming fluxes per node.
4. Add self-update term.
5. Apply residual latent update.
At first, keep this as a drop-in replacement for the generic message-passing block so ablations remain clean.

### Step 9: Add conservative regularization

Implement:

- Antisymmetry penalty $||F_{ij} + F_{ji}||$.
- Optional neighborhood flux-sum penalty away from boundaries.
- Mild boundary correction term.
These are regularizers, not hard constraints. Since you are not on the original CFD mesh, soft enforcement is more honest and usually more stable.


### Step 10: Compare direct vs recurrent rollout

Run two models:

- Direct model predicts all 5 steps at once.
- Recurrent model predicts 1 step repeatedly for 5 steps.
For recurrent training, begin with teacher forcing, then optionally mix in scheduled sampling. The benchmark is specifically multi-step future prediction, so horizon-wise error curves will tell you which strategy is more robust. 


### Step 11: Add high-frequency preservation

If predictions are too smooth, add one or more of:

- Spectral loss on voxelized or neighborhood-interpolated velocity fields.
- Gradient loss using graph finite differences.
- Residual decomposition head: low-frequency base + high-frequency correction.
This is directly motivated by the organizers’ warning that high-frequency flow estimation is the main challenge. 


### Step 12: Scale resolution

Only after the model works on subsampled points should you scale toward larger point sets. At this stage, consider:

- Farthest-point or stratified subsampling.
- Local patches around airfoils plus sparser far-field support.
- Hierarchical graph blocks.
- Mixed precision and gradient checkpointing.
The final winning model may still not use all 100k points uniformly; intelligent allocation near the airfoil and wake may be better.


### Step 13: Run ablations

Your ablation table should include:

- Baseline GNN.
-     + boundary features.
-     + flux block.
-     + flux regularizer.
-     + recurrent rollout.
-     + high-frequency loss.
-     + multi-scale neighborhoods.
This is essential if you want your idea to look ordered and convincing rather than like a bundle of tricks.


### Step 14: Build submission pipeline

Create inference code that:

- Loads test samples.
- Applies exactly the same preprocessing.
- Outputs predictions in the required file format.
- Logs runtime and memory.
Since submissions are made through the competition repository and scored on a hidden split, clean reproducible inference code matters almost as much as the model itself. 


## Suggested timeline

Use a compressed but realistic plan because the deadline is April 22, 2026 AoE. The challenge page states that date explicitly, so your plan should prioritize a strong validated baseline in the first few days and reserve the last stretch for ablations and submission hardening. 

A practical schedule:

- Days 1–2: data loader, visualization, split logic, normalization.
- Days 3–4: baseline graph model and first training runs.
- Days 5–6: boundary-aware variant and better validation.
- Days 7–9: flux-inspired operator implementation.
- Days 10–11: recurrent rollout and high-frequency loss.
- Days 12–13: scale-up, ensemble ideas, and submission packaging.
- Final days: retrain best configs and prepare write-up. 


## What to write in the proposal

Your method description should say something like this: the model learns a geometry-conditioned point-cloud neural operator whose latent update is structured as local directional flux exchange over an induced geometric graph, with explicit boundary-aware conditioning from airfoil surface points and residual multi-step rollout for transient airflow forecasting. That language is faithful to the benchmark’s operator-learning framing, its point-sampled dataset structure, and its no-slip aerodynamic setting. 

A strong title would be: **Boundary-Aware Conservative Neural Operator on Induced Point Graphs for Transient Airfoil Flow Prediction**. It is specific, practical, and avoids overclaiming exact recovery of the finite-volume solver while still expressing your main idea. 

Would you like the next step to be a concrete PyTorch project skeleton with module names, tensor shapes, and pseudocode for each class?

<div align="center">⁂</div>

 : GRaM-Competition-ICLR-2026-_-Competition-track-of-the-GRaM-workshop.pdf

