# VR_analysis — project conventions

Analysis of calcium imaging from mice navigating a virtual environment
with moving objects. Spyder-based, cell-by-cell exploratory workflow.

## Key files

- `VR_analysis.py` — behavior-only session analysis.
- `VR_ca_analysis.py` — calcium imaging + monitor vectorization. Builds
  `mov_data`, `vec_data_l/r`, `two_mon_frames`.
- `VR_ca_dimred.py` — unsupervised neural dim-red (PCA + NMF sweep).
- `VR_ca_cebra.py` — behavior-supervised dim-red (CEBRA) and decoding
  sweep (per-feature R² × n_neural_PCs, with shuffled control).
- `VR_feature_viz.py` — visualize the DECODING features: reconstruct the image-space
  blocks (pix/grid/flow/gabor/meng, gabor mag+phase, per-patch phase-quiver) back to
  monitor movies vs the ORIGINAL input. Cells: load → Original-movie (cached or render
  via f_render_session) → OVERVIEW (all blocks vs original at the saved `example_frames`,
  fast via `frame_idx` subset recon) → per-block deep-dive → gabor mag+phase → per-patch
  quiver. Each cell plots frames inline (vs original) and only saves the TIFF when
  `make_movie`; `save_orig` then also writes the paired `..._ORIG.tif` (same frames +
  `mov_tag`). To raise fidelity, increase a block's grid/n_pca in visual_specs and
  REGENERATE. Refocused 2026-06-30 (dropped the dense full-res optic-flow cell — not a
  decoding feature). Moved out of VR_ca_cebra.py 2026-06-29 (cebra keeps `if 0` refs).
- `VR_feature_recon.py` — Chain-1 groundwork (created 2026-07-01): reconstruct the
  monitor MOVIE from feature blocks, NO neural data — measures each feature set's
  stimulus-coding fidelity (the offline ceiling before neural decode/encode). Cells:
  load → truncate to a frame SUBSET → render+extract (gabor mag+phase / meng / pix) →
  build a mouse-acuity-smoothed TARGET movie (lowpass) → blocked-CV ridge from each
  feature set (mag / phase / meng / mag+meng / all / pix) to the target PCs → OOF R²
  (overall + high/low motion) → example recon frames vs target at motion frames. Compute
  in f_feature_recon.py. Purpose: quantify how much `meng` adds over gabor magnitude and
  whether phase is redundant given mag+meng (notes.txt 2026-07-01). Extended 2026-07-02:
  a `dsets` selector makes ONE video decoder from 1 dataset (as before) or POOLED across
  many — the map is neuron-free/session-invariant, so pooling gives cross-session data.
  len(dsets)>1 → per-session raw extract → f_pool_sessions (shared feature+movie PCA) →
  LEAVE-SESSION-OUT CV; per-session held-out R² is the generalization headline.
- `VR_feature_optim.py` — parameter sweep to OPTIMIZE the gabor video decoder per dataset
  (TODO #68, created 2026-07-02). Render + extract a gabor bank ONCE, then sweep pooled
  recon R²: n_pca (nested truncation) / lags / decoder (all re-score, cheap) marginally
  (default) or joint; grid/#orient/SF axes supported (re-extract) but off in run 1. Separate
  GABOR (pooled R²) and MENG (extra `tf` axis; judged on HIGH-motion R²) sweep cells share the
  render+target + the feature-agnostic engine in f_feature_optim.py. NO neural data.
- `VR_decode_analysis.py` — compare decoding ACROSS datasets (loads only). Loads
  the per-dataset decode caches (f_decode_cache) cebra writes via its save cell,
  builds a tidy table (mouse/dset/region/block/feature/n_pcs/r2/r2_null/delta), and
  plots per-block plateau R² over datasets, a feature×dataset heatmap, real−null Δ,
  and an optional region×block view. Uses the SHARED f_run_decode_sweep numbers.
- `VR_video_recon.py` — synthetic monitor video reconstruction +
  terrain rendering + all related debugging cells (orientation stripes,
  terrain-height labels, mouse-vs-terrain delta plots, chunk-overlap
  diagnostic, NN-vs-bilinear lookup, Phase-1/2/3 anchor diagnostics).
  Self-contained; load `bh_data` and render here, not in the CEBRA
  script.
- `functions/f_functions.py` — production rendering toolkit only.
    - loaders: `f_load_bh_data_all`, `f_proc_movement`,
      `f_get_monitor_coords`, `f_comp_FOV_adj`, `f_add_phase`.
      `f_proc_movement` builds `mov_data` directly on the imaging-frame
      grid, delay-corrected, when `frame_times` is passed (2026-06-11):
      `mov_data['time']` IS the imaging clock and
      `mov_data['delay_corrected']` is True. See Cross-script deps.
    - rendering: `f_angles_to_movie` (legacy), `f_angles_to_movie_v2`
      (filled / antialiased / alpha-composited / optional per-object +
      depth output), `f_render_terrain` (heightmap ground-plane render,
      optional depth output), `f_composite_with_depth` (Z-buffer
      composite of terrain + objects), `f_add_terrain_to_monitor`
      (one-call wrapper: terrain render + Z-composite for one monitor,
      use behind a `render_terrain` bool), `f_save_mon_movie`.
    - world-coord math: `f_terrain_world_coords` (single source of
      truth for the cell→world projection — shared with diagnostics).
      Orientation kwargs (chunk_pitch, chunk_centered, cell_size,
      flip_x/z, swap_xz). For this rig: chunk_pitch=122,
      chunk_centered=True, cell_size=1.0, flip_z=True. Terrain has
      a 3-cell overlap per side (verify with `f_diag_chunk_overlap`
      from the diagnostics module).
- `functions/f_render_diagnostics.py` — debug + sanity helpers
  separated out from `f_functions.py` on 2026-05-20. Imports from
  `f_functions` (one-way). Imported mainly by `VR_video_recon.py`.
    - terrain coord sanity plots: `f_plot_terrain_layout`,
      `f_plot_terrain_mouse_alignment`,
      `f_plot_terrain_object_alignment`,
      `f_terrain_coord_diagnostic`, `f_plot_mouse_vs_terrain`,
      `f_diag_chunk_overlap`, `f_check_obj_coord_system`,
      `f_run_terrain_diagnostics` (master wrapper).
    - chunk-0 hand-rolled lookups (alignment debugging):
      `f_diag_chunk0_mouse_height`, `f_diag_chunk0_mouse_height_bilinear`,
      `f_diag_chunk0_offset_sweep`, `f_diag_chunk0_offset_sweep_2d`,
      `f_diag_chunk0_offset_sweep_2d_bilinear`.
    - rendered-frame debug: `f_add_orientation_stripes`,
      `f_plot_monitor_frame`, `f_plot_obj_terrain_heights`,
      `f_plot_monitor_outline`.
    - per-object angle / distance over-time helpers:
      `f_plot_lateral_over_time`, `f_plot_vertical_over_time`,
      `f_plot_dist_over_time` (and the `*_over_time2` variants).
- `functions/f_feature_helpers.py` — generic feature-engineering helpers
  used by any decoding/dim-red pipeline. Holds monitor-feature aggregators
  (`f_aggregate_monitor_features`, `f_aggregate_monitor_features_per_obj`),
  motion features (`f_motion_features`, `f_self_motion_features` — velocities
  via `np.gradient(x, beh_t)`, non-uniform-grid safe), clock alignment
  (`f_resample_to_imaging`), z-score (`f_zscore`), high-pass detrend
  (`f_detrend_col` — NaN-aware Gaussian high-pass for #8/#10;
  `f_resolve_detrend_sigma` — block-selective σ lookup so pix can be
  high-passed while slow agg/position targets stay raw), and the
  orchestrator `build_feature_blocks` that builds the full `built_blocks`
  dict in one call. Moved out of `f_cebra_helpers.py` on 2026-05-21.
- `functions/f_cebra_helpers.py` — CEBRA + decoding helpers:
  `make_cebra_supervision` (registry → NaN-free z-scored supervision matrix),
  `f_run_cebra` (CEBRA fit/transform wrapper), and `f_blocked_cv_r2` — the
  unified blocked-CV decoding helper (single/multi-output, NaN-aware, with
  embargo, train-only standardize, within-fold PCA, RidgeCV; added 2026-06-03,
  replaces the three drifted inline CV functions in `VR_ca_cebra.py`).
  Extended 2026-06-04: `return_predictions=True` returns time-ordered OOF
  predictions (for trace/scatter diagnostics); `task='classification'`/`'auto'`
  scores binary targets (presence) with a mirrored classifier + ROC-AUC
  instead of R² (R² is misleading for base-rate-skewed binaries); estimator
  construction delegated to `f_decoding.make_decoder`. Extended 2026-07-02:
  `fold_scheme='contiguous'` (default, byte-identical) | `'interleaved'` (+`n_blocks`)
  — interleaved round-robins small blocks into folds so no fold lands entirely on a flat
  epoch (which tanks per-fold R²); contiguous stays the honest whole-epoch default. Plus
  `fold_ids` (per-frame group array) → leave-group-out CV, e.g. leave-session-out across
  pooled datasets (`res['fold']` then = session id → per-session held-out R²).
- `functions/f_decoding.py` — decoding toolkit, compute + plots in one module
  (created 2026-06-04 by extracting repeated machinery out of `VR_ca_cebra.py`).
    - compute: `f_imaging_fs`, `make_decoder` (shared ridge/ridgecv/knn factory
      + classification twins), `f_shuffle_neural` (per-cell circshift null),
      `f_pca_prefix`, `f_apply_block_detrend`, `f_is_binary_col`,
      `build_target_columns` / `build_target_matrix` (registry → per-column or
      concatenated targets, with binary/presence split + block-selective
      detrend), `legacy_target_blocks`.
    - sweep orchestration (2026-06-29): `f_run_decode_sweep(built_blocks, Y_neu,
      frame_t, cfg)` — the SHARED per-dataset decode compute (per-feature blocked-CV
      R²/AUC × n_neural_PCs), factored from the cebra sweep cell so single-dataset
      (cebra) and cross-dataset (VR_decode_analysis) use identical code; returns
      r2_mat/auc_mat + targets/group_rows/pca_neu_full. `f_run_decode_null(sweep, …)`
      — shuffle-control null reusing the sweep's targets+PCA → r2_mat_null/auc_mat_null.
    - plots (all return None so a trailing call doesn't echo a Figure into the
      Spyder console): `f_plot_decode_bars`, `f_plot_decode_sweep_summary`,
      `f_plot_decode_real_vs_null`, `f_plot_feature_heatmap`,
      `f_plot_sweep_lines`, `f_plot_oof_trace_scatter`,
      `f_plot_perfeature_null_grid`, `f_plot_grid_delta_heatmap`,
      `f_plot_grid_focus_lines`, `f_plot_real_vs_shuffle_line`,
      `f_plot_input_raster` (agg+pix), `f_plot_block_traces` (generic
      per-channel viewer for ANY built_blocks entry), `f_plot_pred_scatter`,
      `f_plot_pred_traces`.
- `functions/f_decode_cache.py` — per-dataset DECODE-results cache (joblib),
  mirror of f_feature_cache (2026-06-29): `f_save_decode` / `f_load_decode` /
  `f_decode_path` / `f_decode_exist` / `f_list_decode_files` / `f_resolve_decode_file`
  (file > tag > index). Stores the r2/auc matrices + shuffle null + `(block,name)`
  labels + decode cfg + meta; NO y arrays / neural. `<mouse>_<dset>_<DECODE_TAG>_decode.joblib`.
- `functions/f_visual_features.py` — visual-cortex-style monitor feature
  extractors (TODO #47/#48). `f_grid_features` (per-cell occupancy/mean/edge,
  selectable `stats`), `f_flow_features` (per-cell speed/dir/div optical flow),
  `f_grid_bounds`, registry `VISUAL_FEATURE_REGISTRY`, and the
  `build_visual_blocks` driver (per-side extract → resample → optional PCA via
  `n_pca`; merged into `built_blocks`). `n_pca=None` → raw block. The PCA-application
  half is factored into `f_apply_block_pca(X, raw_names, fname, n_pca)` (grouped/int/float
  n_pca; shared with the pooled feature-recon path so grouped-gabor PCA can't drift).
- `functions/f_feature_viz.py` — reconstruct an image-space monitor movie from
  a feature block to SEE what it keeps (created 2026-06-09).
  `f_reconstruct_feature_movie` (pix: inverse-PCA; grid/flow: cell-paint by a
  channel), `f_resolve_recon_n_pcs` (int / variance-fraction / None →
  component count), `f_stack_recon_over_original` (recon over the matched
  original movie, resampled to the imaging clock), `f_plot_movie_frames` /
  `f_plot_movie_frames_compare` (plot example frames inline; compare original vs
  feature frames at the same timepoints). Driven by `VR_feature_viz.py`.
- `functions/f_feature_recon.py` — Chain-1 (feature→MOVIE reconstruction, neuron-free
  fidelity ceiling; created 2026-07-01). `f_truncate_session` (slice a session to N
  frames so build runs on a subset — f_build_features forbids max_frames + the full
  movie is ~8 GB), `f_deg_per_sample`/`f_acuity_sigma_px` (monitor FOV geometry →
  mouse-acuity Gaussian σ per axis), `f_lowpass_movie` (per-monitor spatial lowpass),
  `f_make_target` (target movie: 'lowpass' [default; pix stays fair] / 'rank' [drop pix]
  / 'raw', + target PCA), `f_assemble_features` ((block, name-prefix) column select —
  'gabor_gabM' mag / 'gabor_gabP' phase / meng / pix), `f_motion_score`, `f_recon_score`
  /`f_recon_compare` (blocked-CV ridge feature→movie-PCs via `f_blocked_cv_r2`, OOF
  variance-weighted R², motion-stratified), `f_recon_frames_at` + `f_plot_recon_bars`.
  Driven by `VR_feature_recon.py`. Rationale in notes.txt 2026-07-01. Extended 2026-07-02:
  `f_pc_sweep`/`f_plot_pc_sweep` (recon R² vs #feature-PCs, nested truncation),
  `f_recon_diagnose` (per-fold vs POOLED R² — explains the flat-fold blowup),
  `f_plot_motion_traces`/`f_plot_pred_overlay`; and MULTI-SESSION pooling
  `f_accumulate_session` (per-session raw extract + movie) + `f_pool_sessions`
  (concat + shared feature PCA via `f_apply_block_pca` + shared movie target + session_ids;
  motion zeroed at session joins) → ONE video decoder across datasets, scored leave-session-out.
- `functions/f_feature_optim.py` — feature-PARAMETER sweep to optimize decoding (TODO #68).
  Efficient extraction: `f_extract_gabor_multigrid` (conv ONCE, pool to every grid),
  `f_build_banks` (generic per-side extract+resample; `f_build_gabor_banks` wraps it — meng
  reuses `f_build_banks`), `f_gabor_subset`/`f_orient_indices` (column subset), nested-PCA
  truncation. TWO scoring paths share this: RECONSTRUCTION (feature→movie, Chain-1) via a
  FEATURE-AGNOSTIC engine `f_sweep_recon` (configs → n_pca×lags×decoder, scored by
  `f_feature_recon.f_recon_score`) + `f_sweep_marginal` (one axis at a time, joint if given
  multiple) + `f_plot_sweep_marginal` (line for numeric axes, bar for lags/decoder) + gabor
  adapter `f_gabor_configs`; and the parked NEURAL path (`f_sweep_gabor_params`/`f_score_block`).
  MENG reuses the SAME engine: `f_extract_meng_multigrid` (Adelson–Bergen, byte-compat with
  f_motion_energy_features, conv+temporal ONCE → pool all grids) + `f_build_meng_banks` +
  `f_meng_configs` (subset orientation/SF/temporal-freq); generic `_subset_cols`/`_parse_meng_name`
  (f_gabor_subset now uses `_subset_cols` too). Sweep gained the `tf` axis; meng is judged on
  HIGH-motion R² (it can't rebuild static form). Driven by `VR_feature_optim.py`.
- `functions/f_encoding.py` — encoding arm (TODO #57): `f_lag_embed` /
  `f_lags_from_ms` (short causal FIR design), `f_encode_per_neuron` (per-neuron
  blocked-CV via `f_blocked_cv_r2`, returns OOF Pearson r + R²),
  `f_encode_null` (circshift target null), `f_encode_block_importance`
  (drop-one unique + block-alone marginal r), `f_encode_summary`. Model choice
  (ridge/ridgecv/knn/pls/cca) threads through to `make_decoder`.
- `functions/f_feature_cache.py` — per-dataset feature cache (TODO #66):
  `f_save_features` / `f_load_features` / `f_features_path` / `f_features_exist`
  (joblib; `tag`=GEN_TAG in the filename) + `f_write_config_json` (provenance
  sidecar) + `f_list_feature_files` / `f_resolve_feature_file` (scan a features dir
  for a dataset's caches; pick one by filename > tag > index) + `f_pick_frames` (n
  evenly-spaced example frames). Caches `built_blocks` so render+extract runs once;
  stores the generation config for staleness detection. Neural stays in data_ca; raw
  movies skipped unless SAVE_MOVIE, but a few ORIGINAL `example_frames` (+ indices) are
  saved regardless so viz can show the input next to a feature recon without re-rendering.
- `functions/f_pipeline.py` — cross-script orchestration glue (TODO #66 follow-up,
  2026-06-26). Three layers, all wired into VR_ca_cebra.py / VR_ca_encoding.py
  (+ f_load_session into VR_ca_dimred.py, + the build pair into VR_feature_gen.py):
    - `f_load_session(...)` (phase 1) — the "session prelude": load calcium + behavior,
      pick est1, run the cheap vectorization (delegates to `f_vectorize_session`),
      compute the clock → a `session` dict (keys incl. `bh` = bh_data[n_dset]).
      `behavior=False` → calcium-only fast path (behavior keys None). Owns the rig
      defaults (`DEFAULT_CAM_PARAMS` / `DEFAULT_CAIM_KW`). NO neural smoothing/sort, NO
      render (per-script / in f_build_features).
    - `f_vectorize_session(est1, bh, cam_params, ...)` — cheap mov_data/lr_data/
      vec_data_l/r/clock for ONE dataset, no LOADING (so VR_feature_gen.py can load
      once + vectorize per dataset).
    - `f_acquire_features(source, dset_name, build_fn=…, …)` (phase 2) — build/load
      junction: `source='build'` calls `build_fn` (returns built_blocks/frame_t/
      side_tags/movie) + optional autosave; `source='cache'` resolves a cache
      (file > tag > index) + loads. cebra/encoding pass
      `build_fn=lambda: f_build_features(session, cfg)`.
    - `f_build_features(session, cfg, ...)` (phase 3) — the SHARED render+extract
      (f_render_session → side-select → build_feature_blocks + build_visual_blocks).
      Single source of truth used by cebra/encoding's build_fn AND VR_feature_gen.py's
      batch loop. `cfg` is a FLAT dict (also the gen_config).
    - `f_render_session(session, cfg)` — the render half (f_angles_to_movie_v2 [+terrain]
      → two_mon_frames), factored out so VR_feature_viz.py can REGENERATE the raw movie
      from the session when a cache lacks it (SAVE_MOVIE was off).
- `functions/f_feature_gen.py` — batch helpers for VR_feature_gen.py (2026-06-29):
  `f_build_one_dataset` (vectorize + flatten nested cfg → `f_build_features`) and
  `f_resolve_dsets` (cfg['DSETS'] → list of do_proc indices). Compute-only; the batch
  loop stays inline in the script.
- `VR_feature_gen.py` — batch build+save script (TODO #66): loops MICE × do_proc
  datasets, builds 1-by-1 via `f_feature_gen.f_build_one_dataset` (→ f_build_features),
  saves caches, frees the render between datasets. Config = an inline `cfg` dict
  (`GEN_TAG` versions caches; `SAVE_MOVIE`; `DSETS` selector — None=all;
  `MAX_DATASETS` caps total builds this run, cache-skips don't count). VR_ca_cebra.py
  loads via the `load_cached_features` switch in cell 1 (build-or-load junction: build
  path renders+extracts and optionally
  saves; load path resolves a cache by file/tag/index — set knobs in cell 1).
- `functions/f_utils.py` — shared utils (`f_load_caim_data`).
- `functions/f_ensembles.py` — NMF/PCA dim-red wrappers + ensemble
  extraction pipeline ported from MATLAB `ensemble_analysis_YS`
  (2026-06-01). Compute-only; visuals in `f_ensemble_plots.py`. Four
  methods: CV grid (`f_cv_estimate_grid`, `f_cv_estimate_one`,
  `f_dred_test_lno`), auto-num-comp via shuffle PCA eigenvalues
  (`f_estimate_dim_corr`), threshold extraction (`f_ens_get_thresh`,
  `f_apply_thresh`), cluster extraction (`f_filter_cells_by_shuf_corr`,
  `f_cluster_cells`, `f_extract_clust`). Single entry
  `f_ensemble_extract`. Behavior-clamped helpers:
  `f_residualize_on_behavior`, `f_NMF_constrained` (sketch).
  Also owns the dim-red wrappers (`f_NMF`, `f_PCA`, `f_sparsePCA`,
  `f_mini_batch_sparsePCA`, `f_dred_add_error`, `f_hoyer_sparsity`,
  `f_component_stability`) — moved here from `VR_ca_dimred.py` so the
  CV / shuffle loops can call them without circular imports.
- `functions/f_ensemble_plots.py` — MATLAB-style ensemble visuals:
  `f_plot_cv_grid` (port of `f_plot_cv_error_3D`),
  `f_plot_dim_estimate` (Method B companion),
  `f_plot_raster_mean` (sorted raster + linked mean trace),
  `f_plot_trial_indicator` (coloured bands above raster),
  `f_plot_ensemble_deets` (per-ensemble 4-panel detail figure),
  `f_plot_comp_scatter` (2D/3D coloured by ensemble),
  `f_plot_ens_overview` (convenience: sorted raster + per-ensemble
  loop, mirrors the MATLAB driver's end-of-script figure set).

## Project tracking

- `PROJECT_GOAL.md` — north star (stable). The project is a BIDIRECTIONAL
  stimulus↔neural map; the ENCODING arm (stimulus→calcium per neuron,
  ceiling-normalized → opto target) is the PRIMARY deliverable and does not yet
  exist — current pipeline only decodes (the validation arm). Endgame: a ranked
  ≤50-cell holographic stimulation list. Encoding arm = TODO #57.
- `REVIEW_2026-06_decoding_gaps_and_tasks.md` — strategic gaps + P0–P3 plan
  (external review, verify-checklist confirmed 2026-06-13). Mapped to TODO
  #57–#65 (strategic-review section).
- `TODO.md` — authoritative task status with options/sub-issues.
- `notes.txt` — dated lab-notebook entries with rationale for
  methodology decisions.
- `PLAN_monitor_features.md` — deep design plan for visual-cortex-style
  monitor input representations (retinotopic grid / optical flow / Gabor
  energy / motion energy / CNN). Tracked as TODO #47. Re-prioritize:
  motion-energy (#50/#51) over flow (#49) — flow degenerate on silhouettes.

## Workflow conventions

- **Spyder cell-based:** scripts are run cell-by-cell (`#%%` markers),
  not top-to-bottom. `if 0:` / `if 1:` blocks are toggle-by-selection
  exploratory variants, not dead code.
- **Preserve old code:** commented-out blocks of prior computations are
  reference material — don't delete during edits.
- **Append to notes.txt:** when the user says "save to notes," append a
  timestamped entry with reasoning preserved.

## Environment quirks

- NumPy < 2.0: use `np.arctan2` / `np.arccos` / `np.arcsin` (not the
  short-name aliases — they don't exist in this env).
- Windows paths; PowerShell shell. Python via Spyder kernel.

## Cross-script data dependencies

When running `VR_ca_cebra.py` or `VR_ca_dimred.py`, the data loading
cells reload `data_ca`/`bh_data` themselves — they don't require
`VR_ca_analysis.py` to have run first. Behavior-only `VR_analysis.py`
uses `f_load_bh_data_all` with `data_ca=None` (no alignment computed).

**mov_data clock (2026-06-11):** with imaging data, `mov_data['time']` IS
the imaging clock (`== est1['frame_times']`), delay-corrected, length T_img;
`mov_data['delay_corrected']` is True. So: pass `pulse_delay=0` to
`build_feature_blocks` when delay_corrected (the `f_resample_to_imaging`
calls become identities); do NOT subtract `bh_pulse_delay` from
`mov_data['time']` in plots; event times (`rewards.Time`, `reward_time`)
are raw behavior clock and need an `ev_shift` before comparing to
`mov_data['time']` (handled in `f_proc_lick_rew` / `f_get_monitor_coords`).
Behavior-only sessions have no `bh_pulse_delay` → `delay_corrected=False`,
uniform 10 Hz `interp_step` grid.

## Open methodological issues

Tracked as parent tasks with sub-options in `TODO.md`:
- #8  — adversarial nonstationarity in top pix-PCs (decoding side).
- #12 — empty-frame encoding ambiguity (agg features).
- #17 — multi-object collapse choice (agg features).

Each parent is blocked by 3-4 sub-options to try. Resolve one of the
sub-options before claiming the parent done.
