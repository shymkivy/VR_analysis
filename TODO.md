# VR analysis — TODO

Persistent task list for the VR / calcium-imaging decoding work. Mirrors
the in-session task tracking; long-form rationale lives in `notes.txt`.

Status legend: `[ ]` pending · `[~]` in progress · `[x]` completed · `[!]` blocked

---

## Approaches roadmap (2026-05-18, see `notes.txt`)

- [ ] **#1** — Encoding model: monitor → neural
  - Regress neural ensemble scores on monitor features (per-component R²
    across algos/k). Tells which ensembles are visually driven.
- [ ] **#2** — Symmetric dim-red on monitor frames + CCA vs neural latents
  - Run PCA/NMF on `two_mon_frames`; CCA between monitor latents and
    neural latents. Symmetric, no encoding/decoding direction.
- [ ] **#3** — Event-triggered averages on neural ensembles
  - PSTH around object-enters-FOV / leaves-FOV / reward / lick. Closest
    to Tier-3 biology overlays.
- [~] **#4** — Behavior-supervised dim-red (CEBRA / dPCA-style)
  - In `VR_ca_cebra.py`. Build features → fit CEBRA-agg + CEBRA-pix →
    decoding A/B. See decoding-diagnostics section below.

---

## PCA decoding diagnostics (current focus)

- [~] **#5** — PCA decoding sweep — per-feature R² vs n_neural_PCs
- [ ] **#6** — CEBRA per-feature decoding diagnostic (mirror of #5)
- [ ] **#7** — Restrict-targets decoding ablation
  - Drop low-variance / nonstationary targets and see if mean R² recovers.

- [~] **#46** — Regression + shuffled-regression harness hardening (2026-06-03)
  - [x] Shuffle null fix: `f_circshift_rates` (shared `f_analysis.py`) gained
        `min_shift` + `rng` kwargs; was `randint(0,T)` on the global RNG so
        `shuffle_seed` was ignored. All call sites now pass a seeded rng.
        `min_shift` reverted to 0 (2026-06-12): a nonzero min biases the null by
        excluding near-aligned shuffles (anti-conservative); uniform shifts are
        the unbiased null. Propagated to the dim-red / ensemble path
        (`f_shuffle_data`) which the original fix had missed.
  - [x] `standardize_emb` (train-fold z-score of embedding cols) — stops PC1
        from dominating KNN distance / Ridge penalty; makes the n_pcs sweep
        honest. On by default in the sweep + null cells.
  - [x] `pca_within_fold` — fit PCA on train rows only inside each fold
        (leak-free embedding axes). Opt-in (slower).
  - [x] `independent_target_shifts` — per-column shifts in `circshift_target`
        for a decorrelated per-feature null (opt-in).
  - [x] Unified the three blocked-CV helpers into one
        `f_blocked_cv_r2` in `functions/f_cebra_helpers.py` (2026-06-03).
        The bar-plot cell, the per-feature PCA sweep (`f_cv_r2_single` now a
        thin wrapper), and the decoding-vs-n_pcs cell (`_cv_r2_multi` wrapper)
        all delegate to it — single implementation, no drift. Verified it
        reproduces all three old helpers exactly (standardize=False, embargo=0).
        `f_detrend_col` moved to `f_feature_helpers.py` (shared).
  - [x] Propagated embargo / train-only standardize / detrend to the bar-plot
        cell (`bar_*` knobs) and the decoding-vs-n_pcs cell (`*_dec` knobs).
        standardize now ON by default in both (was off) — shifts their numbers.
  - [x] RidgeCV α: `decoder='ridgecv'` option in the unified helper + sweep
        (`ridgecv_alphas` knob) + decoding cell (`decoder_dec`). Tunes α by
        LOO-CV inside each train fold — fairer linear ceiling than fixed α=1.
  - [ ] Next: empirical per-feature p-values (1+#{null≥real})/(n+1) for the
        sweep/null cells (already in the detrend×embargo grid; port back).
        NaN policy is now explicit (per-column for single-output via the sweep;
        any-column-row for multi-output — documented in f_blocked_cv_r2).

---

## Open issue: adversarial nonstationarity in top pix-PCs (#8)

Real R² for pix-PC1/PC2 falls far below shuffled-control null. Slow
session-spanning structure in those PCs creates a misleading train-test
relationship that the decoder picks up and then applies wrong on the held-
out block. Full diagnosis in `notes.txt` 2026-05-18.

- [~] **#8** — Address adversarial nonstationarity in top pix-PC decoding
      *(blocked by #9, #10, #11)*
  - [~] **#9** — Diagnostic: plot top pix-PCs vs time + per-fold R² breakdown
  - [x] **#10** — Fix: `detrend_sigma_sec` knob to lowpass-subtract targets
        — implemented 2026-06-03 in the PCA decoding sweep cell of
        `VR_ca_cebra.py`. NaN-aware Gaussian high-pass `f_detrend_col`
        applied to each target before CV; null cell inherits the detrended
        targets so real/null stay matched. Also added `embargo_sec`
        (purge gap around each test block) to cut autocorrelation leakage.
  - [x] **#11** — Follow-up: try detrending the neural embedding too
        — `detrend_embedding` knob (precomputed-PCA path) added alongside #10.
  - [x] Detrend × embargo sweep cell added 2026-06-03 (after the shuffled-
        control cell in `VR_ca_cebra.py`). Grids detrend σ (incl. OFF) × embargo,
        with a circshift_target null + one-sided empirical p per cell. Outputs a
        feature×combo ΔR² heatmap (`*` = real>null @ p<0.05) and per-focus-feature
        (pix_PC1/2) real-vs-null-band line plots. Validated on synthetic data:
        reproduces the real<<null pathology at σ=off and its collapse-to-chance
        after detrend.
  - [x] Block-selective detrend (2026-06-03): `f_resolve_detrend_sigma` +
        `detrend_blocks` knobs in the sweep / bar / decoding cells and
        `grid_detrend_blocks` in the grid. Rationale: slow agg/position targets
        carry real signal in their slow component, so high-passing them removes
        SIGNAL; only pix (PC1/2 drift) should be detrended. Grid σ defaults
        bumped to [None, 15, 30, 60] s — behavior is slow, σ must be large.
        First real run: with global detrend, agg R² variance largest at σ=off
        (detrend killing slow real signal); embargo=10 slightly below embargo=0
        (healthy — removes autocorr inflation). **Re-run with detrend_blocks=
        {'pix': ~30} to confirm pix-PC1/2 recover without penalizing agg.**

---

## Open issue: empty-frame encoding ambiguity (#12)

When `presence=0`, all per-side agg features collapse to 0 — numerically
ambiguous with "object dead center," and creates a sparse signal that
drags R² negative for low-variance features. CEBRA-side: forces all
"empty-frame" neural states to embed together.

- [ ] **#12** — Address empty-frame encoding ambiguity *(blocked by #13–#16)*
  - [ ] **#13** — Option A: per-side presence mask during decoding
  - [~] **#14** — Option B: NaN-fill empty frames + filter (CURRENT DEFAULT)
  - [ ] **#15** — Option C: any-monitor presence mask (less strict)
  - [ ] **#16** — Option D: sentinel value for absent (e.g. lat = −10 rad)

---

## Open issue: multi-object collapse in agg features (#17)

When multiple objects are in FOV simultaneously, the agg representation
reports one summary per frame, losing per-object info and (in `'nearest'`
mode) jumping discontinuously at object swaps.

- [ ] **#17** — Address multi-object collapse in agg feature representation
      *(blocked by #18, #19, #20)*
  - [x] **#18** — Option (a): salience-weighted mean across in-FOV objects
        — implemented as `obj_collapse='salience_mean'` in `f_aggregate_monitor_features`
  - [x] **#19** — Option (b): per-object slot representation (full identity)
        — implemented as `f_aggregate_monitor_features_per_obj` + `build_per_obj`
        knob; each original object id gets its own channel slot, shared across
        monitors. More thorough than top-N: every obj is tracked, not just nearest.
  - [ ] **#20** — Option (c): angular histogram across the monitor

---

## Feature expansion: directional / motion channels (#21)

Position alone doesn't decode from neural; classical motion-selective /
looming-tuned cells fire on directional motion at a receptive-field
location. Add velocity-like channels in a new helper (`f_motion_features`)
so motion vs position can be ablated cleanly.

- [ ] **#21** — Add directional/motion features to monitor representation
      *(blocked by #22, #23, #24)*
  - [x] **#22** — Layer 1: object screen velocity per side
        — implemented as `f_motion_features` in `f_cebra_helpers.py` + `build_motion`
        knob. 6/7 channels per side: d_lat, d_vert, d_dist, [d_ang_size],
        motion_mag, motion_dir_sin, motion_dir_cos.
  - [x] **#23** — Layer 2: mouse self-motion channels (world-frame d_dist, d_phi)
        — `f_self_motion_features` + `build_self_motion` knob (2026-05-21).
        Block name `'self_mot'`, 2 channels (`self_d_dist`, `self_d_phi`).
        Uses `np.unwrap` on `phi` before differentiating.
  - [ ] **#24** — Layer 3: full optical flow on pixel movie
        *(now folded into #49 — flow on a retinotopic grid; see PLAN_monitor_features.md)*

---

## Joint multi-dim decoding methods (#25)

Per-feature decoding from independent linear regressions misses
information about joint structure across output dimensions. Three
joint-decoding methods queued.

- [ ] **#25** — Add joint multi-dim decoding methods *(blocked by #26, #27, #28)*
  - [x] **#26** — Option 1: multi-output Ridge + joint-R² metric
        — added `multioutput=` kwarg to `f_blocked_cv_r2` and `_cv_r2_multi`,
        plus script knobs `bar_multioutput` and `multioutput_dec`.
        `'variance_weighted'` weights per-column R² by variance so
        high-variance dims dominate (avoids drag from low-variance
        binary columns like presence). (2026-05-21)
  - [ ] **#27** — Option 2: Reduced-Rank Regression (RRR)
  - [x] **#28** — Option 3: PLS Regression
        — added to real-vs-reconstructed cell as `decoder_rec='pls'`,
        with `n_components_pls` knob. CCA also added as `decoder_rec='cca'`
        (similar joint method, optimizes corr instead of cov).

---

## Pix-side motion features (Levels A / B / C)

Three approaches to capture motion in the pixel representation, mirroring
agg-side `X_mot`. Listed in order of effort.

- [x] **#29** — Level A: `d/dt` of pix-PC traces (`X_pix_mot`).
      Implemented 2026-05-21 inside `build_feature_blocks`. Block name
      `'pix_mot'`, n_pix_pca channels (`d_pix_PC1`, ...). Optional
      smoothing via `pix_motion_smooth_sigma`. No NaN-masking.
- [ ] **#30** — Level B: PCA on pixel diff frames.
      Computes frame-to-frame diff first, then PCA → motion-specific PCs
      rather than static-scene PCs. More expressive than A.
- [ ] **#24 (= Level C)** — Full optical flow on pixel movie (Farneback +
      PCA on flow field). Most expressive, heaviest. Already listed under
      #21's tree.

---

## Temporal-pattern decomposition of pix-PC traces

PCA on already-PCA'd pix-PCs is a no-op (already decorrelated). To find
recurring temporal patterns, need methods that explicitly bring time in.

- [ ] **#31** — Time-lagged PCA on pix-PC traces.
      Stack `[pix_pc(t), pix_pc(t-τ), …]` and PCA. Top components are
      k-step sequence templates.
- [ ] **#32** — DMD (Dynamic Mode Decomposition) on pix-PC traces.
      Fit linear operator A: pix_pc(t+1) ≈ A·pix_pc(t). Eigendecomposition
      gives oscillatory/decaying spatial-temporal modes.

---

## Visual-cortex-style monitor input representations (#47)

Current monitor features decode weakly (position barely above chance). `agg`
(object-center) and `pix` (PCA on raw intensity) are both far from how visual
cortex encodes the scene — which is LOCATION-selective and DIRECTION-selective
(oriented edges at retinotopic locations, with motion). Add feature banks built
on the filter → nonlinearity → spatial pool → compress → linear-decode pipeline,
applied to `two_mon_frames` (already ~101×101 per monitor in angular coords).
Each is a new `built_blocks` block behind a `build_*` knob, PCA-compressed,
decoded through the existing harness. Full design + per-tier builder specs in
`PLAN_monitor_features.md`; rationale/literature in `notes.txt` 2026-06-06.

- [ ] **#47** — Visual-cortex-style monitor representations *(blocked by #48–#53)*
  - [~] **#48** — Tier 1: retinotopic grid pooling (`f_grid_features`, block
        `'grid'`). Per-cell occupancy / contrast / edge energy. Location
        selectivity; cheapest; composes with all later tiers. **Do first.**
        Extractor + registry + `build_visual_blocks` driver implemented &
        tested in `functions/f_visual_features.py` (2026-06-06). TODO: wire
        into VR_ca_cebra.py build cell (`built_blocks.update(...)`) + sweep.
        Per-cell `stats` are now a selectable subset (occupancy/mean/edge);
        `n_pca=None` → raw block. Recon + generic trace/stack viewers added
        (2026-06-11).
    - [ ] **#54** — Expand per-cell grid stats (more feature TYPES per cell).
          Current edge stat is orientation-BLIND (mean |∇| only). Candidates:
          - **Edge direction / orientation**: per-cell dominant gradient
            orientation (structure-tensor angle) + anisotropy/coherence, or
            an orientation histogram (a few oriented-edge bins per cell).
            Cheaper proxy for the Gabor tier (#50); keeps location selectivity.
          - **Wavelet / multi-scale**: per-cell wavelet-energy bands (Haar /
            Daubechies or steerable pyramid) for scale + orientation tuning —
            a lighter alternative to the full Gabor bank.
          - **Speed / velocity**: per-cell motion magnitude+direction. Overlaps
            the flow tier (#49); decide whether to keep motion in `grid` (one
            combined block) or leave it to `flow`. (See PLAN_monitor_features.md.)
          Each is a new entry in the grid `stats` tuple (or its own extractor)
          so it composes with the existing occupancy/mean/edge channels.
  - [ ] **#49** — Tier 2: optical-flow on the grid (`f_flow_features`, block
        `'flow'`). Farneback dense flow → per-cell speed, dir sin/cos,
        divergence (local looming). Direction-of-edge-motion. Realizes
        **#24 / Level C**. Likely winner if cells are location+direction tuned.
  - [ ] **#50** — Tier 3: Gabor energy bank + grid (`f_gabor_features`, block
        `'gabor'`). Orientation × SF complex-cell energy = V1 simple/complex.
  - [ ] **#51** — Tier 4a: motion-energy model (Adelson–Bergen, block `'meng'`).
        x–y–t oriented filters → principled direction+speed tuning.
  - [ ] **#52** — Tier 4b: pretrained-CNN front-end (block `'cnn'`). Early-conv
        feature maps; data-driven ceiling / benchmark.
  - [ ] **#53** — Tier 5 supporting channels: local looming (flow divergence),
        DoG center-surround SF/contrast, oriented-edge/HOG proxy, 1-D angular
        histogram (= **#20**, degenerate grid). Fold into tiers as needed.

---

## Code organization

- [x] **#34** — Modular feature generation: `built_blocks` registry + `combine_blocks` helper.
      Implemented 2026-05-21. Build cell now a single call to
      `build_feature_blocks(...)` in `functions/f_feature_helpers.py`
      (renamed from f_cebra_helpers.py). Every consumer cell migrated to
      `target_blocks` / `sweep_target_blocks` / `target_blocks_dec` /
      `target_blocks_bar`. CEBRA fit cell driven by `cebra_supervisions`
      dict → `cebra_models` dict; downstream cells iterate over it.
      Legacy aliases (X_agg, X_pix, X_mot, cebra_agg, cebra_pix) kept.

- [x] **#35** — Lick + reward feature block.
      Implemented 2026-05-21. `f_lick_reward_features` in
      `f_feature_helpers.py` + `build_lick_reward` knob. Block name
      `'beh'`, 2 channels (`lick`, `rew`). Optional smoothing via
      `lick_reward_smooth_sigma`. Nearest-neighbor resampling so raw
      event spikes aren't smeared during behavior→imaging interp.
      `lr_data` built once in VR_ca_cebra.py via `f_proc_lick_rew`.

- [x] **#55** — mov_data built on the imaging-frame grid, delay-corrected
      (2026-06-11). `f_proc_movement` with `frame_times` → `mov_data['time']`
      IS the imaging clock + `delay_corrected` flag. Removes the 10 Hz→imaging
      double-resample (`pulse_delay=0` downstream) and renders movies at the
      imaging rate. Velocity via `np.gradient(x, beh_t)`; event binning shifted
      by `ev_shift` in `f_proc_lick_rew`/`f_get_monitor_coords`; `- bh_pulse_delay`
      plot sites stripped. Behavior-only path unchanged.

- [x] **#56** — Feature-block visualization tooling. `f_feature_viz.py`:
      `f_reconstruct_feature_movie`, `f_resolve_recon_n_pcs` (variance-fraction
      PCs), `f_stack_recon_over_original`. `f_plot_block_traces` in `f_decoding.py`
      (per-channel viewer for any `built_blocks` entry). Driven by the recon cell.

---

## Embedding similarity diagnostics

- [x] **#33** — Procrustes distance between embeddings.
      Implemented 2026-05-21 in `VR_ca_cebra.py`. Pairwise disparity
      matrix over PCA-3 + every model in `cebra_models` + aligned overlay
      scatter for one chosen pair via `proc_align_pair`. Strips
      rotation/reflection/scaling ambiguity so two embeddings can be
      compared on shape alone. 0 = identical geometry, 1 = totally
      different.

---

## Ensemble extraction port (#36) — MATLAB → Python

Port of the MATLAB `ensemble_analysis_YS` pipeline (NMF + CV + threshold
+ statistical checks). Compute helpers in `functions/f_ensembles.py`;
visuals in `functions/f_ensemble_plots.py`; orchestration cells appended
to `VR_ca_dimred.py`; CEBRA-latent variant cell appended to
`VR_ca_cebra.py`. See `notes.txt` 2026-06-01.

- [~] **#36** — Ensemble extraction port *(blocked by #37–#42)*
  - [x] **#37** — Method A: CV grid (smooth_SD × num_comp) with
        leave-neuron-out test error. `f_cv_estimate_grid` +
        `f_cv_estimate_one` + `f_dred_test_lno`.
  - [x] **#38** — Method B: auto-num-comp via shuffle PCA eigenvalues.
        `f_estimate_dim_corr`.
  - [x] **#39** — Method C: threshold-based extraction. `f_ens_get_thresh`
        (`signal_z`, `shuff` modes; `signal_clust_thresh` deferred) +
        `f_apply_thresh`.
  - [x] **#40** — Method D: cluster-based extraction.
        `f_filter_cells_by_shuf_corr` + `f_cluster_cells` +
        `f_clust_ens_scores` + `f_extract_clust`.
  - [x] **#41** — MATLAB-style plots: `f_plot_cv_grid`,
        `f_plot_dim_estimate`, `f_plot_raster_mean`,
        `f_plot_trial_indicator`, `f_plot_ensemble_deets`,
        `f_plot_comp_scatter`, `f_plot_ens_overview`.
  - [~] **#42** — Behavior-clamped exploration *(blocked by #43–#45)*
    - [x] **#43** — Branch (a): ensemble extraction over CEBRA latents.
          New cell in `VR_ca_cebra.py`; treats embedding as scores +
          ridge-fit coeffs, runs `f_ens_get_thresh` + `f_apply_thresh`.
    - [x] **#44** — Branch (b): behavior-residualized NMF.
          `f_residualize_on_behavior` + new cell in `VR_ca_dimred.py`.
    - [x] **#45** — Branch (c): constrained NMF sketch (`f_NMF_constrained`)
          with H_clamp = behavior block. Cell in `VR_ca_dimred.py`.
          First-pass implementation — defer hyperparameter sweeps to
          a follow-up issue.

---

## Conventions

- This file is the authoritative project TODO. The in-session task tool
  mirrors it during a working session.
- When a parent issue is blocked by sub-options, the parent stays open
  until all sub-tasks resolve (one with `[x]` and the others triaged).
- Long-form rationale and dated discussion entries go in `notes.txt`.
- Code references point at `VR_ca_cebra.py`, `VR_ca_dimred.py`,
  `VR_ca_analysis.py`, and `VR_analysis.py`.
