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
  construction delegated to `f_decoding.make_decoder`.
- `functions/f_decoding.py` — decoding toolkit, compute + plots in one module
  (created 2026-06-04 by extracting repeated machinery out of `VR_ca_cebra.py`).
    - compute: `f_imaging_fs`, `make_decoder` (shared ridge/ridgecv/knn factory
      + classification twins), `f_shuffle_neural` (per-cell circshift null),
      `f_pca_prefix`, `f_apply_block_detrend`, `f_is_binary_col`,
      `build_target_columns` / `build_target_matrix` (registry → per-column or
      concatenated targets, with binary/presence split + block-selective
      detrend), `legacy_target_blocks`.
    - plots (all return None so a trailing call doesn't echo a Figure into the
      Spyder console): `f_plot_decode_bars`, `f_plot_decode_sweep_summary`,
      `f_plot_decode_real_vs_null`, `f_plot_feature_heatmap`,
      `f_plot_sweep_lines`, `f_plot_oof_trace_scatter`,
      `f_plot_perfeature_null_grid`, `f_plot_grid_delta_heatmap`,
      `f_plot_grid_focus_lines`, `f_plot_real_vs_shuffle_line`,
      `f_plot_input_raster` (agg+pix), `f_plot_block_traces` (generic
      per-channel viewer for ANY built_blocks entry), `f_plot_pred_scatter`,
      `f_plot_pred_traces`.
- `functions/f_visual_features.py` — visual-cortex-style monitor feature
  extractors (TODO #47/#48). `f_grid_features` (per-cell occupancy/mean/edge,
  selectable `stats`), `f_flow_features` (per-cell speed/dir/div optical flow),
  `f_grid_bounds`, registry `VISUAL_FEATURE_REGISTRY`, and the
  `build_visual_blocks` driver (per-side extract → resample → optional PCA via
  `n_pca`; merged into `built_blocks`). `n_pca=None` → raw block.
- `functions/f_feature_viz.py` — reconstruct an image-space monitor movie from
  a feature block to SEE what it keeps (created 2026-06-09).
  `f_reconstruct_feature_movie` (pix: inverse-PCA; grid/flow: cell-paint by a
  channel), `f_resolve_recon_n_pcs` (int / variance-fraction / None →
  component count), `f_stack_recon_over_original` (recon over the matched
  original movie, resampled to the imaging clock). Driven by the recon cell in
  `VR_ca_cebra.py`.
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

- `TODO.md` — authoritative task status with options/sub-issues.
- `notes.txt` — dated lab-notebook entries with rationale for
  methodology decisions.
- `PLAN_monitor_features.md` — deep design plan for visual-cortex-style
  monitor input representations (retinotopic grid / optical flow / Gabor
  energy / motion energy / CNN). Tracked as TODO #47.

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
