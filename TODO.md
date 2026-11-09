# VR analysis — TODO

Persistent task list for the VR / calcium-imaging decoding work. Mirrors
the in-session task tracking; long-form rationale lives in `notes.txt`.

Status legend: `[ ]` pending · `[~]` in progress · `[x]` completed · `[!]` blocked

---

## NEXT UP (2026-11-06)

- [ ] **#78** — ONE feature builder + one interface for VR_feature_search and VR_ca_cebra/encoding
      (modular pipeline, piece 3). Plan: `PLAN_feature_builder_merge.md`. Two parallel stacks today
      (f_feature_configs→f_iter_grid_cells vs f_vis_sets_to_plan→_build_grid_block). CONFIRMED silent bug:
      a single-SF entry builds 8 ori / Nyquist in the search but 4 ori / ds=4 in cebra. Also exact vs
      randomized layer-1 SVD, cebra holds all raw inputs in RAM. Steps: fix single-SF → move the streaming
      builder to f_feature_source (pluggable raw source) → cebra's builder wraps it → one spec builder
      `f_feature_plan` → f_feature_search one path + tidy (legacy builders to f_feature_legacy.py) → cebra
      feature cell → lazy render. Decisions open (old-scorer knobs, legacy, vis_sets→feat_sets, pix).

- [x] **#77** — ONE encoder config + ONE encoding sweep wrapper (modular pipeline, piece 2). Plan:
      `PLAN_encoding_config.md`. Defaults = VR_ca_cebra's enc_params (ridge α 1e3, lags 0–500 ms,
      interleaved 5-fold, 150 joint feature PCs, standardize, target σ=6 frames). Config defaults at the
      top of f_encoding.py with every option documented + unknown keys raise + tag; `f_enc_target`;
      `f_encode_sweep` (axes: encoder keys / feature sets & procs / targets; 1-D + 2-D; tidy table + line
      plot); feature search + VR_encode_scan encode through f_run_encoder. Today the three scorers differ
      (search: α 1, 200 ms only, σ 0, contiguous folds). Open: double reduction (n_feature_pcs on top of
      feat_sets procs), embargo 0 vs 0.5 s.

- [x] **#76** — ONE ensemble-generation config reused by every script (first piece of the modular
      pipeline, TARGETS). Plan: `PLAN_ensemble_config.md`. `f_default_ens_config(**overrides)` (stages
      neural/dimred/apply/stats/select, defaults = cebra's tuned values, unknown keys raise) +
      `f_ens_neural_input` + `f_ensembles_from_config` (stats-only re-run via `prior`) + `f_ens_targets` +
      optional per-dataset cache keyed by config hash. Parity first (each script's current settings →
      identical ens_table), then port cebra / VR_encode_scan / VR_tuning_scan / VR_feature_tuning and add
      ensembles as encode targets in the feature search. Defaults = VR_ca_cebra's current values
      (z_thresh membership, NMF input σ=6, sparse NMF l1 0.5 / alpha_W 1e-4, seed 42, hclust sort).
      + NEW SCRIPT `VR_ensembles.py`: load one dataset (calcium only) → ensembles from the config →
      cebra's ensemble plots (scores, overview raster, LNO ECDF, CV trace, FOV map; opt-in dim-red sweep
      + NMF consensus) → optional save to the ensemble cache. Cell map in the plan file.
      DONE 2026-11-06: config + cache + VR_ensembles.py; parity exact; ported VR_ca_cebra (old cells and
      duplicate plots removed), VR_feature_search (Cell 1b targets), VR_encode_scan, VR_tuning_scan,
      VR_feature_tuning. Real default config built for L/11_2_25 (10/15 sig).

- [ ] **#75** — Cross-dataset FEATURE SCAN: score ONE feature config (from VR_feature_search's
      `feat_sets`) for encoding across ALL datasets. Plan A (recommended): new thin driver
      `VR_feature_scan.py` looping datasets → `fsearch.f_feature_search(session, configs, …)` so scoring
      is IDENTICAL to the single-dataset search. Needs: (1) `f_score_representations` option to return
      per-cell r/R² (currently computed then discarded, f_dataset_scan.py ~247) for cortex maps;
      (2) resumable store keyed (mouse, dset, config), mirroring f_encode_scan `_save_atomic`;
      (3) `f_render_clear` between datasets. Plan B (alt): port VR_encode_scan.py to the vis_sets plan
      path — keeps its store/models/ensembles/plots but scores differently (within-fold proc, 0–500 ms
      FIR vs search's proc_before_cv + 200 ms), so results won't line up with the search.
      Cost: conv_autosave only covers L_11_2_25 → every other dataset re-renders + convolves (gabor GPU
      ~3 min/bank; fft CPU ~3–14 min per SF level). Open: which feature, encode-only vs +decode, which
      mice/datasets. Old-feature store `encode_scan/enc_7_18_26` is not comparable.

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
  - [x] **#50** — Tier 3: Gabor energy bank + grid (`f_gabor_features`, block
        `'gabor'`). Orientation × SF complex-cell energy = V1 simple/complex.
        Done 2026-06-16: quadrature bank (even DC-removed), √(even²+odd²) energy,
        FFT-conv (T-chunked) → grid mean-pool; `f_gabor_bank` for viz; registered
        + wired into `VR_ca_encoding.py` build cell (n_pca=20). A/B vs agg/pix open.
  - [x] **#51** — Tier 4a: motion-energy model (Adelson–Bergen, block `'meng'`).
        x–y–t oriented filters → principled direction+speed tuning. Done
        2026-06-17: complex spatial Gabor × complex temporal Gabor, |·|² opponent
        energy, halo-chunked; `f_motion_energy_features`. Also added optional
        `'phase'` stat to `'gabor'` (cosφ/sinφ → velocity via FIR). Both wired
        into encoder+decoder visual_specs. A/B vs agg/pix/flow open.
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

## Strategic review 2026-06-13 (PROJECT_GOAL.md + REVIEW_2026-06)

Reframe (see `PROJECT_GOAL.md`): the project is a BIDIRECTIONAL stimulus↔neural
map; the ENCODING arm (stimulus→calcium per neuron) is the primary deliverable
and the opto-target source, and it does not yet exist. Decoding (current
pipeline) is the validation arm. `REVIEW_2026-06_decoding_gaps_and_tasks.md`
P-numbers mapped to real IDs below. Verify-checklist results recorded in
`notes.txt` 2026-06-13.

- [~] **#57** — Encoding arm (REVIEW P1-1) — **PRIMARY**. New `VR_ca_encoding.py`
      + `functions/f_encoding.py` (compute) + `f_encoding_plots.py` (visuals),
      mirroring the f_decoding split. Per-neuron ridge from a LAGGED feature
      design matrix (built_blocks visual + self_mot + beh); locomotion is a
      PREDICTOR not a target. Score = held-out per-neuron correlation,
      ceiling-normalized (#58). Output a tuning table (cell → best feature
      block / retinotopic location-direction) = seed of the opto generator
      (#64). Variance-partition nested models for the areal gradient.
  - [x] **v1 minimal validation pass** (2026-06-13): `f_encoding.py`
        (`f_lag_embed`, `f_encode_per_neuron`, `f_encode_null`, `f_encode_summary`)
        + `VR_ca_encoding.py`. Short causal FIR (lags via `f_lags_from_ms`),
        multi-output ridge through `f_blocked_cv_r2`, per-neuron OOF Pearson r vs
        circshift null. Run on a dark-condition session. NEXT: variance partition,
        tuning table.
  - [~] **P2 sub-plan** (`../reserrch_repo/SUBPLAN_P2_encoding_model.md`, 2026-07-03):
        §4 audit run (notes 2026-07-03). **D1 + D5 built + synthetic-gated** (not yet on a
        real session): `f_encode_fit_weights` (weights = RF) + `f_rf_pixel_frames` /
        `f_rf_grid_location` + `f_encoding_plots.f_plot_rf` (canary viewer);
        `f_encode_ceiling_fraction` (r/r_full). Pixel-RF + ceiling cells in `VR_ca_encoding.py`.
        Synthetic STRF recovery passes (blob + peak lag). Added STA baseline (`f_encode_sta`).
        **D2 built + gated** (2026-07-03): `f_encode_fit_smooth` (Tikhonov/Laplacian smoothness),
        `f_encode_smooth_cv`, `f_encode_tune_sigma_lambda` (σ×λ joint CV). Gate step 3 passes
        (smooth beats ridge at low N: corr 0.57 vs 0.17). D2 driver cells in `VR_ca_encoding.py`.
        **D2 nested guard added** (2026-07-03, P2 §7): `f_encode_tune_sigma_lambda` now SELECTS
        (σ,λ) on a leading time split and REPORTS the winner on a held-out tail (`report_frac`,
        `sel_r`/`report_r`) — no hyperparameter-selection optimism; synthetic-verified.
        **self_mot fwd/lat/yaw decomposition added** (2026-07-03, P2a-9): `f_self_motion_decomp`
        (heading projection: fwd=dx·cosφ+dz·sinφ, lat=−dx·sinφ+dz·cosφ, yaw=dφ) wired via the
        `self_motion_decomp` knob in `build_feature_blocks`/`f_pipeline` (default off = old 2-ch);
        synthetic-verified (mag preserved, lat≈0 forward-only). Required precursor to the P2b-5 partition.
        **D3 banded ridge + variance partition built + gated** (2026-07-03, P2a-5/P2b-1): `BandedRidge`
        (per-band λ via per-band column-rescaled Ridge; `predict_band`) in f_decoding + `decoder='banded'`
        threaded through `make_decoder`/`f_blocked_cv_r2`; `f_encode_banded`/`f_encode_tune_bands`
        (per-band α joint-grid, nested select/held-out-report)/`f_encode_variance_partition` (drop-one
        on JOINT model → visual-unique/behavior-unique/shared R²). Gate step 4 passes (noise band
        unique≈0, correct vis>beh ordering, shared≈0 for independent bands; penalty == closed-form
        banded solve). D3 driver cell in `VR_ca_encoding.py`.
        Audit → `../reserrch_repo/AUDIT_P2_findings.md` (flags: self_mot=speed+yaw; #62 CEBRA leak).
        NEXT: view real RFs (build mode); run D3 partition on real data (self_mot decomp on). Deferred:
        D4 task-event band + capture-offset regressor/D6 per-cell lags/D7 dual-readout (→P5)/D8/D9.
      *Working ceiling stance (2026-06-13, supersedes #58):* use the FULL-feature
      model r (predict each cell from ALL blocks, best model) as the per-cell
      RELATIVE ceiling — judge a feature/model by how much of r_full it recovers.
      Soft positive controls = `self_mot`/running (strongly drives mouse V1) +
      reward/lick timing (the self_mot-only sanity gate already does this). Null =
      circshift. Caveat: r_full conflates feature-incompleteness with cell noise,
      so it is a relative anchor, NOT an absolute "X% explained".
- [?] **#58 (PARKED / maybe)** — Event-aligned noise ceiling (REVIEW P0-3).
      Deferred 2026-06-13: it's really a TUNING analysis in disguise — you must
      pick which event defines each cell's "signal", which is cell-type-dependent
      and circular (assumes the tuning to measure the ceiling). Speculative +
      interpretation-heavy for free foraging. Revisit only if the r_full relative
      ceiling proves insufficient. (Idea: align to object FOV-entry / looming /
      capture, split-half reliability = ceiling, weighted to transition epochs.)
      *Superseded 2026-06-16:* user prefers a SHUFFLE TEST over a noise ceiling.
      Done — `f_encode_null_dist` (N circshift nulls) + `f_encode_signif` (per-cell
      one-sided p, BH-FDR q, null-referenced z) in `f_encoding.py`; significance
      cell in `VR_ca_encoding.py`. Rank opto candidates by z among FDR-sig cells.
- [ ] **#59** — Transient vs sustained regimes (REVIEW P1-4). Onset/transient
      channels (FOV-entry, motion-onset, looming-rate) + sustained presence; an
      adaptation/history term (time-since-onset or exp trace of recent local
      stimulus energy); STRATIFY decode/encode eval by regime (running vs
      stationary, object moving vs static) — never only pooled. Extends the
      `motion`/`self_mot` blocks (#21).
- [~] **#60** — FIR / lag-embedding (REVIEW P2-1). Lag-augment the design
      matrix: predict feature(t) from neural over [t−Δ,t+Δ] (decode) and
      activity(t) from features over causal [t−Δ,t] (encode). Sweep Δ ~0.1–1.5 s.
      The linear baseline an RNN must beat.
      - ENCODE side: done (f_lag_embed + f_encode_per_neuron, VR_ca_encoding.py).
      - DECODE side: done 2026-06-19 — f_lag_embed/f_lags_from_ms moved to
        f_decoding (shared base); f_blocked_cv_r2 gains `lags` (lag-after-PCA on
        the emb path, end rows masked, OOF idx unchanged); `decode_lags_ms`
        (bidirectional ±300 ms) wired into VR_ca_cebra.py bar/sweep/null/recon/
        diagnostic. Incompatible with pca_within_fold (raises).
- [ ] **#61** — RNN encoder (REVIEW P2-2) — only after #60. Reuse the
      `RNN_scripts` / `f_RNN_dred` path; compare to FIR-linear on the SAME
      ceiling-normalized metric.
- [ ] **#62** — Decoder-correctness remainder (REVIEW P0-1; most of P0-1 already
      done — see notes 2026-06-12). Optional per-fold CEBRA refit (nested-CV) or
      drop the CEBRA decodability claim; raise CEBRA `out_dim` to 8–32 where an
      embedding feeds a decoder (3 only for scatter); surface pooled-OOF R²
      alongside per-fold in the headline cells. Links #46, #56.
- [ ] **#63** — Horizon + ground/sky channels (REVIEW P0-2, DOWNGRADED). Add the
      terrain→sky horizon edge (render and/or a 'horizon elevation per monitor'
      scalar) + global ground/sky luminance split. Skip detailed grass texture
      (low acuity; ground flow ≈ redundant with self_mot). The dark-condition
      session is the real input-fidelity fix / clean debug dataset.
- [ ] **#64** — Opto pattern generator (REVIEW P3-2) — endgame. From #57: for a
      specified stimulus event emit predicted population vector; rank cells by
      (predicted drive × encoding reliability × ensemble stability); intersect
      targetable cells → ranked ≤50-cell holographic list.
- [ ] **#65** — CV / cross-session ensemble identity (REVIEW P3-1). Extend
      `f_ensemble_extract` (#42) from cross-seed STABILITY to temporal-CV
      membership (same cells co-fire in held-out time) + cross-session identity
      (opto targets must re-identify on a new day).

- [~] **#66** — Cross-dataset decoding + encoding quality (region × session).
      Run enc/dec quality over ALL do_proc datasets; compare by brain REGION and
      separate region effects from per-session quality (some sessions just
      perform better). Plot SUMMARY scores only, not per-dataset figures.
  - [~] **Fast across-dataset TRIAGE** `VR_dataset_scan.py` + `functions/f_dataset_scan.py`
        (2026-07-11): loops mice × do_proc datasets, loads calcium + CHEAP behavior
        vectorization (NO render, NO feature cache) → per-dataset decode R² (neural→self_mot,
        neural→agg) + encode r (agg+self_mot→neural) + covariates (n_cells/T/participation-
        ratio/agg-coverage) → ranked table + bars. Leak-free within-fold PCA; NaN target rows
        drop. Purpose = pick which datasets deserve the full feature build + sweep. Core metric
        wiring synthetic-verified; NOT yet run on real data. Ranks by stimulus/self-motion
        coupling (triage proxy, not the gabor/meng result). Region metadata into the table = TODO.
      *Architecture — precompute-and-cache features (decouple render from analysis):*
  - [~] **Feature-generation script** `VR_feature_gen.py` + helper
        `functions/f_feature_cache.py` (`f_save_features` / `f_load_features` /
        `f_features_path` / `f_features_exist`, joblib + gen-config staleness
        check) — DONE 2026-06-24. `VR_feature_gen.py` loops MICE × do_proc
        datasets: render + `build_feature_blocks` + `build_visual_blocks` → save
        `built_blocks` + frame_t + side_tags + meta + gen_config (staleness). Cache
        FEATURES only (neural stays in data_ca); `SAVE_MOVIE` flag optionally adds
        two_mon_frames. Informative names `<mouse>_<dset>_<GEN_TAG>_features.joblib`
        (`tag` added to `f_features_path`/`f_features_exist`) + `<tag>_config.json`
        provenance sidecar (`f_write_config_json`). Config = an INLINE `cfg` dict in
        the script (Spyder-edited; module/JSON config forms tried then dropped).
  - [x] Analysis scripts (encoding / cebra) get a `load_cached_features` knob →
        `f_load_features(...)` instead of the inline build cell (2026-06-24: an
        `if 0:` "load cached features" cell in each, run INSTEAD of render+build).
        Keep the inline build for single-dataset feature-method DEV.
  - [x] **Pipeline modularization (phases 1-3, 2026-06-26)**: `functions/f_pipeline.py`
        — `f_load_session` / `f_vectorize_session` / `f_acquire_features` /
        `f_build_features` / `f_render_session` factor the load/build/render glue;
        wired into cebra/encoding/dimred + VR_feature_gen (helpers → f_feature_gen.py).
        Load/build junction replaced the `if 0:` cell (cebra: `load_cached_features`
        switch in cell 1). `f_resolve_feature_file` tag = FILTER + index. VR_feature_gen
        gained DSETS/MAX_DATASETS/OVERWRITE + tqdm; `example_frames` saved every cache.
  - [x] **DECODING cross-dataset (2026-06-29)**: `f_run_decode_sweep`/`f_run_decode_null`
        (shared single=multi compute) in f_decoding.py; `f_decode_cache.py`; cebra
        save-decode cell (`DECODE_TAG`, F:/VR/decode); `VR_decode_analysis.py` loads
        decode caches → tidy DataFrame → per-block/region summary plots. Region meta
        still None in the save (below).
  - [ ] **ENCODING cross-dataset (NEXT, PRIMARY)**: mirror the decode pipeline — shared
        `f_run_encode`, `f_encode_cache.py`, save cell in VR_ca_encoding.py,
        `VR_encode_analysis.py`, + the tuning table (cell → best block / retinotopy → #64).
  - [ ] Per-dataset score compute already returns dicts (f_encode_*, f_blocked_cv_r2)
        — loop them over the cached features into a results DataFrame (row per
        dataset or dataset×cell) with metadata (region, n_cells, SNR, running frac).
  - [ ] Identify the **region metadata source**: per-session label (a `VR_data.xlsx`
        column?) vs per-cell from retinotopic registration
        (`retinotopic_mapping_analysis.py` field-sign areamaps). Per-session label
        is the pragmatic first cut; per-cell is the rigorous version (later).
  - [ ] Cross-dataset summary plots ONLY (not per-dataset figures): median/quantile
        per-neuron encoding r per dataset grouped/colored by region; frac>null;
        decoding R² per key feature per region; session-quality covariates
        (n_cells, mean SNR, running frac) alongside to disentangle region vs quality.
      *Defer (after encoding matures):* ceiling-normalized (#58) cross-dataset
      scores; per-cell region (retinotopic registration); variance-partition by
      region; full all-models comparison across datasets.

- [x] **#67** — Model factory consolidation (2026-06-13). Lifted PLS/CCA out of
      the recon cell's hand-rolled loop into the shared `make_decoder` +
      `f_blocked_cv_r2` (new `f_resolve_n_components` per-fold rank clamp;
      single-output 2D-pred ravel). Encoding (`f_encoding` + `VR_ca_encoding.py`)
      and the decode bar/sweep/decoding-vs-n_pcs cells now select
      ridge/ridgecv/knn/pls/cca via one path; recon cell de-duplicated.
      Extends #28.

- [~] **#68** — Feature-selection groundwork: least features for best decode/encode
      (2026-07-01; extends #47/#50/#51/#66). Reframe: separate Chain-1 (feature→MOVIE
      fidelity, neuron-free) from Chain-2 (activity↔features). Working basis = gabor
      MAGNITUDE (form) + MENG (motion); phase is a hypothesis to falsify, not a default.
      Rationale in notes.txt 2026-07-01.
  - [x] **Chain-1 built** — `functions/f_feature_recon.py` + `VR_feature_recon.py`:
        reconstruct a mouse-acuity-LOWPASS target movie from each feature set
        (mag / phase / meng / mag+meng / all / pix) via blocked-CV ridge → OOF
        variance-weighted R², motion-stratified + example frames. Compiles +
        synthetic-tested; NOT yet run on real data.
  - [~] **Run Chain-1** on a real session (2026-07-03 progress): gabor MAGNITUDE and
        PHASE both reconstruct the acuity-lowpass movie WELL; MENG stays at very low R²
        so far. Next = keep optimizing meng (via the sweep harness below) to get a usable
        motion feature representation — the target is meng good enough to serve as a
        DECODING TARGET for the neuronal (Chain-2) plans, not just Chain-1 fidelity.
        Still TODO: sweep the acuity cutoff (σ≈0.3–0.5 px at 0.5 cpd — render already ≈
        acuity, so also try coarser toward the grid scale); reconstruct vs LAGGED mag so
        meng's motion isn't double-counted; dark-condition session.
  - [ ] **Test A** (neural, Chain-2): phase unique Δr given mag+meng via
        `f_encode_block_importance` — if ≈0, drop phase.
  - [ ] **Test B** (duality): coarse-grid+phase vs fine-grid-no-phase — same encoding r?
        → keep whichever is fewer features.
  - [ ] Match meng speed bands (`tf`/`λ`) to the MEASURED on-screen object/optic-flow
        velocity distribution rather than guessing.
  - [~] The feature-PARAMETER × score sweep (grid density / #orient / #SF / #PCs / lags /
        decoder) — HARNESS BUILT in `functions/f_feature_optim.py` + `VR_feature_optim.py`
        (2026-07-02): extract superset once (`f_extract_gabor_multigrid` /
        `f_extract_meng_multigrid`, conv ONCE → pool all grids), subset cheaply, score
        through Chain-1 recon (`f_sweep_recon`/`f_sweep_marginal`). Gabor + meng cells
        share the feature-agnostic engine; meng judged on HIGH-motion R². Use this to push
        meng R² up (target-quality for Chain-2).
  - [~] **NEURAL-scored feature-param search** (2026-07-18): the Chain-2 sibling of the recon
        sweep above. `functions/f_feature_search.py` + `VR_feature_search.py`: on ONE dataset, loop
        candidate feature `configs` (plan-based → f_resolve_features, cache-backed — no reconvolution)
        and score each by ENCODE + DECODE (f_score_representations: decode R² neural→feature + encode r
        feature→neural) → ranked scorecard. `f_variant_configs` = the separate/combined/raw-PCA
        comparison from the pca_feat caches; combinations = derived gaborS+mengS; fresh ds/n_pca/filter
        sweeps = plan overrides. `enc_n_jobs` caps encode cores (CPU/crash guard). Compiles; NOT yet
        run on real data. NEXT: run on 1 dataset, pick the best feature repr for encode+decode, then
        extend to a joint encode+decode objective / marginal-axis sweep.
  - [ ] **Optimize meng** (own task; low R² as of 2026-07-03). Goal = a motion feature
        set good enough to be a Chain-2 DECODING TARGET, not just Chain-1 fidelity. Sweep
        via the meng cell of `VR_feature_optim.py`: `tf` (temporal-freq / speed bands),
        `λ`/#SF, #orient, grid, #PCs, decoder — scored on HIGH-motion R². Match speed bands
        to the MEASURED on-screen object/optic-flow velocity distribution (see item below)
        rather than guessing. Check meng vs LAGGED-mag so mag's Δ over time isn't stealing
        meng's motion variance. Compare meng against the flow tier (#49) once flow exists.
  - [~] **Sampled (point-sampling) extraction method** (2026-07-11): a SECOND gabor/meng method
        that POINT-samples the dense response on a lattice every `sample_ds` px (vs grid mean-pool;
        ds=1 → full conv) + optional `presmooth_sigma`. In `f_visual_features` (`pool='sample'`,
        `f_subsample_grid_block` for conv-once/derive-coarser, `build_visual_blocks` `type` override);
        `f_dataset_scan.f_score_representations` gives one decode R² + encode r per method (compact
        format); driver cells in `VR_ca_cebra.py` (build + compare gabor/meng-sampled vs pix), viz
        panel cell in `VR_feature_viz.py`. Compiles + synthetic-gated; NOT yet run on real data.
        GOAL: does point-sampled gabor/meng beat straight pixel-PCA (both directions)?
  - [~] **Convolved-feature repository** (2026-07-11): persist the DENSE per-filter convolved maps
        ONCE per dataset so resolution/param sweeps LOAD+subsample instead of re-convolving (the slow
        part). `functions/f_conv_repo.py` (uint16-quantized per-filter .npy + manifest, INCREMENTAL —
        add orientations/wavelengths/TFs without recomputing; mmap+subsample loader → block).
        `VR_feature_gen.py` `GEN_MODE='conv_repo'` (+ `conv_specs`) builds the repo; `VR_ca_cebra.py`
        sampled-methods cell `methods_source='conv_repo'` loads+samples+scores (no re-conv). gabor mag
        + meng opponent. Compiles + synthetic-gated end-to-end; NOT yet run on real data. Deferred:
        gabor phase in repo, percentile quant, standalone optimize script.
  - [ ] **Optic flow (#49) NOT yet touched** — the flow tier (`f_flow_features`, block
        `'flow'`, Farneback → per-cell speed/dir/divergence) is registered but has never
        been run/extracted or scored through Chain-1. Bring it into the sweep alongside meng
        as the second motion representation; the two are the candidates for a motion target.

REVIEW items folded into existing tasks (no new ID):
- P1-2 (object target definedness / presence-location hurdle model) → **#12**.
  Presence already AUC-scored; presence-in-R² leak fixed 2026-06-12.
- P1-3 (motion-energy over flow; flow degenerate on silhouettes) → amends
  `PLAN_monitor_features.md` + reorders **#50/#51** ahead of **#49**.

---

## Gabor / visual-feature representation comparison (2026-07-23)

Goal: find the LEANEST visual feature set that maximizes encode+decode, not RF fidelity — all scored
through `f_feature_search` / `f_score_representations` (decode R² neural→feature + encode r feature→neural).
Two established constraints frame the sweeps: (a) MAGNITUDE features need ≥3 orientations (0°/90° give
IDENTICAL responses to 45° vs 135° → unlearnable by any regressor; squaring makes orientation tuning
order-2 → ~3-4 is the Nyquist floor, 8+ is redundant/PCA-collapsed); (b) PHASE may need only ~2 (test);
(c) DECODE is per-static-frame so motion/speed likely unneeded — motion matters mainly for ENCODE.

- [ ] **#69** — Gabor baseline "up to speed". Validate the recent fixes end-to-end on real video:
  quadrature magnitude √(even²+odd²) (already correct), luminance DC channel, centered Nyquist sampling
  (`_sample_idx`), `samp_factor`, and the obj_size WIDTH fix (x now full width — REGENERATE all caches).
  Verify with `VR_feature_kernel_plot.py` (kernel sizes/sampling to scale). Establishes the magnitude
  baseline. Sweep `n_orient` (2/3/4/8) → find the encode/decode plateau (expect 2≪4≈8).
- [ ] **#74** — SIGNED QUADRATURE gabors (`stats=('quadrature',)` → gabQe + gabQo), added 2026-07-25.
  The even/odd filter responses themselves — no magnitude taken, no normalization. Since re = |R|cosφ
  and ro = |R|sinφ this carries magnitude AND phase JOINTLY (|R| = √(re²+ro²) recoverable), so it is a
  strict superset of #69's magnitude and #70/#71's phase, in 2 channels instead of 3. Three reasons it
  should be the reference point for the whole #69–#71 comparison:
    (a) INFORMATION — {re,ro} and {mag,cosφ,sinφ} are the same information (bijection away from mag=0),
        so nothing is lost; quadrature just uses the better-conditioned basis.
    (b) LINEAR MODELS — the combination a ridge would need from mag+phase is the PRODUCT mag·cosφ, which
        a linear model cannot form; mag·cosφ IS re. So for ridge encode/decode, mag+phase is strictly
        worse-conditioned than quadrature while carrying identical information. mag+phase is only a
        genuinely DIFFERENT hypothesis under a nonlinear decoder (`make_decoder` knn/pls/cca).
    (c) CONDITIONING — no division, so empty screen regions contribute ~0 rather than a full-amplitude
        arbitrary unit vector. Also means the FFT conv path applies at full parity (see f_conv_gpu
        FFT_WITH_PHASE, which has to stay off for gabPc/gabPs).
  Also the natural LINEAR gabor stage for the encoding arm (#57): a ridge on signed (re,ro) can
  represent a phase-sensitive oriented RF; magnitude cannot, and normalized phase is contrast-blind.
  Not phase-invariant → keep Nyquist λ/2 sampling, don't coarsen. Implemented in f_gabor_features +
  f_conv_gpu._gab_stat_torch (no new conv — re/ro already computed). NOT yet in f_feature_optim's
  separate bank builder (`f_build_gabor_banks`, `if 'phase' in stats`) — add if that path is used.
- [ ] **#70** — magnitude + NORMALIZED phase. `stats=('magnitude','phase')` → gabM + gabPc + gabPs.
  Test whether adding phase (position/polarity) lifts encode/decode over magnitude alone. NOTE: phase
  here is UNIT-modulus (re/mag, ro/mag) — magnitude divided out. Pair it WITH gabM (never phase alone
  for a linear model, see #71). Given #74(b), expect this to match or trail quadrature under ridge; run
  it against a NONLINEAR decoder if you want it to test something quadrature doesn't.
- [ ] **#71** — phase-only gabors. gabPc/gabPs without magnitude; sweep orientations (test the "phase needs
  only 2" hypothesis). Probes whether position/phase alone carries the decodable signal. CAUTION
  (measured 2026-07-24): re/mag EXPLODES as |R|→0 and the gate is an absolute `mag > 1e-6`, so over an
  empty screen these channels emit unit-magnitude values driven by float rounding rather than stimulus —
  two conv engines that both shipped disagree at corr 0.9917, and gpu-direct vs gpu-fft at 0.86. So a
  chunk of phase-only "signal" is engine noise. Fix the gate (scale-RELATIVE threshold) before trusting
  this result, or read it via #74 instead.
- [ ] **#72** — optical flow / motion, ENCODE-focused (decode likely needs no speed). The `flow` feature
  (Lucas-Kanade) exists but is degenerate on filled silhouettes (uniform interiors → only edge normal-flow).
  Compare `flow` vs `meng` vs phase-at-lags. NOTE: a LINEAR (ridge) decoder can't extract Δφ from cos/sin
  phase (needs the bilinear product across lags — exactly what meng precomputes); so phase+lags→motion needs
  a NONLINEAR decoder (`make_decoder` knn/pls/cca) or unwrapped phase. Test both.
- [ ] **#73** — FFT-spectrum features. IMPLEMENTED 2026-07-26 as `f_visual_features.f_fft_features`
  (`kind='fft'`, driven by `fsearch.f_pyramid_grid('fft', …)` / `fft_pyr_sets` in VR_feature_search);
  NOT yet run on a real session. A gabor response at (x₀,y₀,θ,λ) IS one Fourier coefficient of the
  Gaussian-windowed patch at frequency (cosθ/λ, sinθ/λ) — so the bank is a lossy subsample of the patch
  FFT, and this extractor keeps the whole lattice instead. Constant-Q preserved by ONE PATCH SIZE PER
  LEVEL (patch tied to λ), so it is a pyramid, not a single constant-bandwidth FFT.
  Same window (`sigma_scale·λ`), same support (`ceil(3σ)`, capped per axis at H−1/W−1 — exact), same
  centres (`_sample_idx`) as the gabor bank, and binned back onto the bank's (ori × SF) sectors with
  gabor naming (`fftM`/`fftQe`/`fftQo`_o{io}sf{isf}_r{iy}c{ix}) → at `n_sf_sub=n_ori_sub=1` the columns
  align index-for-index with the gabor pyramid, so any fft-over-gabor gap is exactly the intra-sector
  frequency spread the bank discards. `n_sf_sub`/`n_ori_sub` > 1 subdivide sectors = strictly richer.
  MEASURED: the equivalence holds to **corr 0.9998** when the target frequency lands on a lattice point,
  and the dominant error is LATTICE QUANTIZATION, not the DC-removal/broadband effect first assumed —
  `frac = |N/λ − round(N/λ)|` vs agreement gives **corr −0.88** (frac<0.15 → 0.96, frac>0.35 → 0.39).
  Hence `patch_snap=True` (default): nudge the patch ±15% so the target lands on a lattice point, which
  lifted mean agreement 0.961 → **0.985** with every config at frac<0.15. Requires gamma=1 (an
  anisotropic window is orientation-dependent, so one patch FFT cannot serve the bank) and
  `reduce='sample'`. Remaining: run against the gabor pyramid in one scorecard (#69/#74), and a GPU
  backend only if profiling calls for it.

---

## Conventions

- This file is the authoritative project TODO. The in-session task tool
  mirrors it during a working session.
- When a parent issue is blocked by sub-options, the parent stays open
  until all sub-tasks resolve (one with `[x]` and the others triaged).
- Long-form rationale and dated discussion entries go in `notes.txt`.
- Code references point at `VR_ca_cebra.py`, `VR_ca_dimred.py`,
  `VR_ca_analysis.py`, and `VR_analysis.py`.
