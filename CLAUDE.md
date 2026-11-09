# VR_analysis — project conventions

Analysis of calcium imaging from mice navigating a virtual environment
with moving objects. Spyder-based, cell-by-cell exploratory workflow.

## Key files

`SCRIPTS.md` = one-line purpose index of every script + the main pipeline (feature_search →
ca_encoding → ca_cebra). Keep it updated when adding a script.

- `VR_analysis.py` — behavior-only session analysis.
- `VR_ca_analysis.py` — calcium imaging + monitor vectorization. Builds
  `mov_data`, `vec_data_l/r`, `two_mon_frames`.
- `VR_ca_dimred.py` — unsupervised neural dim-red (PCA + NMF sweep).
- ENSEMBLES in VR_ca_cebra (2026-11-06): settings = `ens_cfg = enspipe.f_default_ens_config(...)` and
  `dimred_params = ens_cfg['dimred']` (ONE source for ensembles + neural cleanup); the old consensus /
  APPLY / STATISTICS / LNO-ECDF / CV-trace cells are ONE load-or-fit cell (`f_ensembles_cached`, cache
  `<features_dir>/<mouse>/ens_cache/`, shared with VR_ensembles.py + VR_feature_search Cell 1b) giving
  ens_out / ens_table / ens_targets; old cells kept commented. Downstream (ASSIGN, encode-over-inputs,
  LNO scatter, tuning) unchanged. A cell_ids check warns if ens_cfg['neural'] ≠ cebra's Y_neu_raw prep.
  Same shared config + cache now in VR_encode_scan / VR_tuning_scan / VR_feature_tuning (2026-11-06; their
  old per-script settings differed: no NMF sparsity, cv_consensus, σ=0 in encode_scan → 0/15 sig on
  11_2_25 vs 10/15 with the shared config). Store tags bumped (enc_11_6_26 / tun_11_6_26); the stores
  record ens_params = {tag, cfg, which}. VR_feature_search Cell 1b: `use_targets` (cells / ens_all /
  ens_sig) → f_feature_search(targets=…), each block built once and scored per target.
- `VR_ensembles.py` — ONE dataset's NMF ensembles from the SHARED ensemble config (2026-11-06,
  PLAN_ensemble_config.md / TODO #76): `f_default_ens_config` (sections neural/dimred/apply/stats/select,
  defaults = VR_ca_cebra's settings, unknown keys raise) → `f_ensembles_cached` (load from
  `<features_dir>/ens_cache/<mouse>_<dset>_<cfgtag>_ens.joblib` or fit+stats+save) → cebra's ensemble
  plots → `f_ens_targets` (ens_all / ens_sig traces for encode/decode). All in f_ensemble_pipeline.py;
  `f_ensembles_from_config(prior=…)` re-runs only stats when the fit sections are unchanged.
- `VR_ca_cebra.py` — behavior-supervised dim-red (CEBRA) and decoding
  sweep (per-feature R² × n_neural_PCs, with shuffled control).
  FEATURE CONFIG (2026-07-26, reformatted 2026-07-27): the visual half of the feature-plan cell is a
  `vis_sets` dict in the SAME self-contained form as VR_feature_search.py's `feat_sets` — a config that
  wins the search pastes over unchanged. BOTH sides now read it through the ONE canonical reader
  `f_feature_source.f_resolve_feat_sets` (reserved '_common' defaults → per-entry override → `kind`
  validation), so a dict cannot mean two different things; `fsrc.f_vis_sets_to_plan` is the plan-side
  twin of `fsearch.f_feature_configs`, and cell names match exactly (verified for gabor/meng/fft/pix ×
  every marginalize). `kind='pix'` works here too → the raw-movie block with layer1+layer2 run
  back-to-back, matching f_pix_config. NB '_common' is a defaults entry, NOT a block: read the names
  off the returned plan (`_vis_plan`), never off `vis_sets`, when extending `feature_list`.
  `feature_list` still selects the NON-visual blocks (agg / pix /
  motion / pix_mot / self_mot / beh / obj_vis) and the legacy cached gaborS/mengS via
  `feat_pca_variant` — all of it resolves in the ONE `f_resolve_features` call (one shared render).
  Downstream, a BANK name addresses all its cells: the encoder cell's `feature_block=['gabmag']` and
  the decode sweep's `sweep_target_blocks` expand via `f_expand_blocks` to every `<bank>:<cell>`
  block (concatenated for encode); an explicit `'gabmag:sf0.01016'` scores ONE level, and legacy exact
  names pass through untouched. Bank vs explicit-cell-list gives bit-identical scores. CAVEAT the
  script-side `sweep_target_blocks` pre-filter must match `<bank>:` prefixes too, else it drops a bank
  name before f_expand_blocks ever sees it (fixed 2026-08-05).
  THREE-SCRIPT PARITY (2026-08-05): VR_feature_search / VR_ca_cebra / VR_ca_encoding now all build
  through `feature_plan` + `vis_sets` + `f_resolve_features`, and MUST agree on the knobs that bind the
  shared conv_autosave store — `edge_min_frac` (0.45), `conv_autosave_tag` ('auto1'), render 150×101,
  terrain off. `edge_min_frac` is the trap: it is absent from `f_default_build_cfg` (so it defaults to
  None = guard OFF), it changes the columns a bank emits (at cyc_fov=1 it drops 1 of only 2
  point-samples), and it is PART OF THE AUTOSAVE KEY — MEASURED, off vs 0.45 gives
  `..._1161497c.npz` vs `..._22eeaba9.npz`, so a mismatched script silently re-convolves the whole bank
  instead of reloading. It was set only in VR_feature_search until 2026-08-05.
  SINGLE-SF FIX (2026-11-09): f_vis_sets_to_plan now sends single-SF entries through f_pyramid_specs too (they
  used f_grid_block → 4 ori / ds=4 vs the search's 8 / Nyquist, silently). Pyramid entries unchanged.
  `f_feature_source.DEFAULT_MARGINALIZE = ('ori',)` is the canonical fallback when an entry names no
  `marginalize`; `f_vis_sets_to_plan`, `f_pyramid_grid` and `f_pyramid_block` all read it. Before that
  the plan side defaulted to `()` while the search defaulted to `('ori',)` — the SAME dict gave one
  cell per filter in cebra and one per SF level in the search, silently (same class as the 07-27 drift
  bug, same function pair).
  `vis_sets` ships ONE entry per feature type at identical `magcomb2pca` params (cyc_fov=(1,2,4,8,16),
  odegs=(0,45,90), marginalize='all', layer2 pca 50 → one `<name>:all` block at 50 dims each):
  gabmag / gabquad / fftmag / fftquad / mengopp, so only the REPRESENTATION differs. NB raw widths
  differ before the PCA (magnitude 1 channel per filter, quadrature 2, meng ×tf).
  BUGFIX 2026-07-27: `marginalize='all'` pooled everything in the search but NOTHING in the plan-side
  `_build_grid_block` (it normalized 'wl'→'sf' but never expanded the 'all' shortcut). Canonical
  `f_feature_source.f_marg_axes` now serves both — `f_feature_search._marg_axes` delegates to it.
  BUGFIX 2 (2026-07-27, same class): `f_grid_block`/`f_specs_block` STORED the spec via a bare
  `tuple(marginalize)`, so a BARE STRING 'all' became ('a','l','l') → three unknown axes → nothing
  pooled, silently (one `<name>:all` block became 15 per-filter blocks). Latent while cebra passed the
  tuple `('all',)`; matching the search's bare string exposed it. `f_marg_spec` is now the canonical
  spec→tuple normalizer (bare string = ONE axis) and all four store sites use it.
- `VR_ca_encoding.py` — the ENCODING arm (#57): predict each neuron's deconvolved rate proxy from
  stimulus + self-motion, per-neuron blocked-CV ridge with a short causal FIR vs a per-cell circshift
  null. Also the P2 spine drivers (D1 RF weights/STA, D2 smoothness + σ-CV, D3 banded ridge +
  visual/behavior variance partition). PORTED 2026-08-05 to the same feature structure as
  VR_ca_cebra / VR_feature_search: `feature_list` (non-visual) + a `vis_sets` dict (visual banks) →
  `f_proc_feature_plan` + `f_vis_sets_to_plan` → ONE `f_resolve_features` call, replacing the old
  inline `visual_specs` + flat `cfg` + `f_acquire_features` junction (kept commented below the new
  cell). `enc_blocks` therefore takes BANK names (`'gabmag'`, `'mengopp'`) which `f_encode_per_neuron`
  expands via `f_expand_blocks`. NB the gabor-bank sanity cell still uses the OLD grid-pooled idiom
  (grid=(6,6), n_orient=4, wavelengths=(6,12)) — it convolves directly and is independent of the plan,
  so it runs, but it no longer describes the bank the script builds; `VR_feature_kernel_plot.py` is the
  current kernel/sampling check.
- `VR_feature_viz_cache.py` — visualize the DECODING features from a BLOCK CACHE (joblib):
  reconstruct the image-space blocks (pix/grid/flow/gabor/meng, gabor mag+phase, per-patch
  phase-quiver, per-orientation panel) back to monitor movies vs the ORIGINAL input. Cells:
  load → Original-movie (cached or render via f_render_session) → OVERVIEW (all blocks vs
  original at the saved `example_frames`, fast via `frame_idx` subset recon) → per-block
  deep-dive → gabor mag+phase → per-patch quiver → per-orientation panel. Each cell plots
  frames inline and only saves the TIFF when `make_movie`; `save_orig` writes the paired
  `..._ORIG.tif`. Split from VR_feature_viz.py 2026-07-12 (was moved out of VR_ca_cebra.py
  2026-06-29; refocused 2026-06-30).
- `VR_feature_viz_conv_repo.py` — visualize / analyze a CONV_REPO (dense per-filter maps,
  2026-07-12). Cells: load repo → sampled built_blocks (`f_load_conv_blocks`, with kinds/
  wavelengths/odegs/tfreqs load filters) → (util) convert repo → browsable HDF5 (`f_conv_repo_to_h5`)
  → ACTIVITY diagnostic (`act_source='blocks'` = `f_feature_viz.f_block_activity` reuses the loaded
  blocks, NO re-read, honours the load filters, keyed by SF/TF INDEX; `'repo'` = `f_conv_activity`
  re-reads the dense maps for exact full-res + wavelength/tf VALUES, `frame_ds` speed knob) →
  per-orientation feature-map panel on the sampled blocks (no original — repo has no movie).
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
- `VR_feature_search.py` — feature-PARAMETER search on ONE dataset, scored by ENCODE + DECODE
  (created 2026-07-18; the neural-scored SIBLING of VR_feature_optim.py's movie-recon sweep).
  Load one dataset (f_load_session) → neural target (f_make_neural_target = VR_ca_cebra's Y_neu
  recipe) → loop candidate feature `configs`, each REGENERATED FRESH on the GPU (force_build=True
  bypasses the cache so the search spans the full param space; the monitor render runs ONCE, shared
  across configs, only the conv re-runs) → score each with f_dataset_scan.f_score_representations
  (decode R² neural→feature + encode r feature→neural) → ranked table + scorecard. The
  separate/combined/raw-PCA question is `f_granularity_configs` (n_pca int=combined | ('filter',n)=
  separate | None=raw); combinations = derived blocks (gaborS+mengS); ds/sf_cyc_deg/odegs/tf_hz vary
  via the extraction spec. `f_variant_configs`+force_build=False = the cache-load path instead. Compute
  in f_feature_search.py. Encode CPU is bounded by the BLAS thread-cap env vars atop the script
  (f_encode_per_neuron is serial). Eventual goal: optimize feature params jointly over encode+decode.
- `VR_decode_analysis.py` — compare decoding ACROSS datasets (loads only). Loads
  the per-dataset decode caches (f_decode_cache) cebra writes via its save cell,
  builds a tidy table (mouse/dset/region/block/feature/n_pcs/r2/r2_null/delta), and
  plots per-block plateau R² over datasets, a feature×dataset heatmap, real−null Δ,
  and an optional region×block view. Uses the SHARED f_run_decode_sweep numbers.
- `VR_dataset_scan.py` — ACROSS-DATASET quality triage (created 2026-07-11, #66). Loops
  mice × do_proc datasets, loads calcium + CHEAP behavior vectorization (NO render, NO
  feature cache), scores each session's encode/decode quality vs the cheap ground-truth
  stimulus (agg + self_mot) → ranked table + bars. Picks which datasets deserve the full
  feature build + sweeps. Compute in `functions/f_dataset_scan.py`; shared home for future
  cross-dataset encode/decode comparisons.
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
  motion features (`f_motion_features`, `f_self_motion_features` — collapsed
  speed+yaw; `f_self_motion_decomp` — heading-projected fwd/lat/yaw for the
  encoding partition, P2a-9, via `self_motion_decomp=True`; velocities
  via `np.gradient(x, beh_t)`, non-uniform-grid safe), clock alignment
  (`f_resample_to_imaging`), z-score (`f_zscore`), high-pass detrend
  (`f_detrend_col` — NaN-aware Gaussian high-pass for #8/#10;
  `f_resolve_detrend_sigma` — block-selective σ lookup so pix can be
  high-passed while slow agg/position targets stay raw), the UNIFIED
  processing primitive `f_proc(X, steps, fit=None)` (2026-07-20 — ordered
  step-list `[('norm',zscore|standardize|total_var|first_sv), ('pca',n|frac|
  (frac,cap)), ('whiten',True)]`, each optional, ANY order; fit=None fits +
  returns per-step params, fit=those TRANSFORMS a test fold leak-free; used at
  feature-grid layer1/2 AND within-fold on the decode/encode predictor via
  f_blocked_cv_r2 `pred_proc`), `f_num_tag(x, sig=4)` (short round-trip-safe
  %g numeric tag for names/filenames — the shared SF/TF formatter used by the
  feature-grid cell names and conv-autosave filenames, 2026-07-24), and the
  orchestrator `build_feature_blocks` that builds the full `built_blocks` dict
  in one call. Moved out of `f_cebra_helpers.py` on 2026-05-21.
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
  pooled datasets (`res['fold']` then = session id → per-session held-out R²). Plus
  `bands`/`band_alphas` (2026-07-03) → `decoder='banded'` per-band ridge (emb path only;
  columns survive standardize so bands stay aligned).
- `functions/f_decoding.py` — decoding toolkit, compute + plots in one module
  (created 2026-06-04 by extracting repeated machinery out of `VR_ca_cebra.py`).
    - compute: `f_imaging_fs`, `make_decoder` (shared ridge/ridgecv/knn/pls/cca/**banded**
      factory + classification twins), `BandedRidge` (per-band λ via per-band column
      rescale of Ridge(α=1); `predict_band` = additive per-band contribution — D3),
      `f_shuffle_neural` (per-cell circshift null),
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
- `functions/f_dataset_scan.py` — fast per-dataset encode/decode quality triage (#66,
  2026-07-11). `f_scan_dataset` (one leak-free blocked-CV fit each: decode R² neural→self_mot
  + neural→agg via within-fold PCA, encode r agg+self_mot→neural; + covariates n_cells/T/
  participation-ratio/agg-coverage), `f_build_cheap_blocks` (agg+self_mot+beh, no render),
  `f_participation_ratio`, `f_count_dsets`, `f_plot_scan`. Triage proxy (agg+self_mot only),
  not the full gabor/meng result. Driven by `VR_dataset_scan.py`. Also `f_score_representations`
  (2026-07-11): compact per-METHOD encode+decode score, one row per representation — the method
  analogue of f_scan_dataset; used by the sampled-feature comparison cells in VR_ca_cebra.py. Emits
  a FAMILY of metrics per block (2026-07-21): DECODE (neural→feature) decode_r2 (var-weighted, the
  default sort) / decode_r2_uniform (col-equal) / decode_r2_med (median col) / decode_validfrac;
  ENCODE (feature→neural, per-cell) encode_r_med / encode_r_mean / encode_r2_med / encode_r2_mean /
  encode_frac_sig (frac cells r>r_sig). Plus the per-unit DISTRIBUTION of each median metric —
  `{decode_r2,encode_r,encode_r2}_{min,p5,p25,p75,p95,max}` (over feature cols for decode, cells for
  encode) via `_dist_stats`. Optional POPULATION encode (`encode_neural_proc`=f_proc step-list e.g.
  [('norm','zscore'),('pca',20)] | int/float→[('pca',n)] | None): reduces the per-cell OOF true/pred to
  population dims with that proc (fit on true OOF, applied to pred; `_pop_encode`) → variance-weighted (by
  each dim's true variance) `encode_pop_r2` + `encode_pop_r` (+uniform + per-dim R² dist `encode_pop_{...}`);
  None → NaN.
  Plot keys 'pop_r2'/'pop_r2u'/'pop_r'. `_decode_r2` returns the dict (computed from the SAME OOF preds, no extra
  CV). SIZE MONITOR (2026-07-27): every scored row also carries `_block_size` — `n_dim` (width actually
  decoded) / `raw_dim` (width BEFORE the layer/PCA proc, from the block's `raw_dim`; == n_dim when no
  reduction ran) / `n_frames` / `n_cells_kept` / `block_mb` / `decode_peak_gb`, plus the VARIANCE the
  kept components explain: `var_kept` (block/layer2 PCA `explained_variance_ratio_.sum()`),
  `l1_var_kept` (the per-filter layer1 PCAs pooled, WIDTH-WEIGHTED by each filter's input columns —
  set by `f_iter_grid_cells`, 1.0 when layer1 ran no PCA, NaN when it had no proc at all) and
  `var_kept_total` = their product = fraction of the RAW block variance still present. The verbose
  per-block line appends `[T×n_dim of raw_dim L1var..% var..%, ..MB, decode~..GB]`. For this to work
  the non-grid path in f_feature_search keeps the fitted `pca` + pre-proc `raw_dim` on the block (so
  pix reports too). MEASURED on the real finest-SF gabor level (3780 cols): layer2 pca 50 → 92.2%,
  100 → 96.6%, 200 → 99.0%. `DECODE_PEAK_FACTOR = 3.7` is the MEASURED peak
  of `_decode_r2` ÷ (T·n_col·8 B) — the target is cast to float64, then the per-fold OOF lists, their
  concatenation and the time-order reindex all coexist; linear in n_col (3.70 at 500/1000/2000/4000), so
  it extrapolates. Tabulate with `f_feature_search.f_feature_sizes(res, ram_gb=…)`. f_feature_search's `f_plot_feature_search` picks a panel metric via `decode_metric`/`encode_metric`
  (keyed by DECODE_METRICS/ENCODE_METRICS, each → col+title+dist-prefix) and `show_pct=True` overlays the
  distribution as whiskers (5–95 line, 25–75 box, ◆ min/max) — exposes sparse encoders the median hides.
- `functions/f_visual_features.py` — visual-cortex-style monitor feature
  extractors (TODO #47/#48). `f_grid_features` (per-cell occupancy/mean/edge,
  selectable `stats`), `f_flow_features` (per-cell speed/dir/div optical flow),
  `f_grid_bounds`, registry `VISUAL_FEATURE_REGISTRY`, and the
  `build_visual_blocks` driver (per-side extract → resample → optional PCA via
  `n_pca`; merged into `built_blocks`). `n_pca=None` → raw block. The PCA-application
  half is factored into `f_apply_block_pca(X, raw_names, fname, n_pca)` (grouped/int/float
  n_pca; shared with the pooled feature-recon path so grouped-gabor PCA can't drift). `_pca_fit` also
  takes a `(frac, cap)` tuple = variance-fraction capped at `cap` comps (2026-07-17); `f_pca_group_spec
  (raw_names, granularity, n_pca)` builds the grouped-PCA prefix dict from sampled-block column names —
  `'filter'` (per o×SF×TF, L‖R combined) or `'orient'` (per o) — driving the pca_feat gen granularities.
  `f_pca_group_cols(raw_names, pool_axis, n_pca)` (2026-07-19) is the POOL-AWAY grouping keyed by EXPLICIT
  column indices (handles the non-leading sf/tf axes that startswith can't): `pool_axis` collapses that
  axis, groups by the non-pooled o/sf/tf ('none'=per filter, 'ori', 'sf'/'wl', 'tf', ('sf','tf')=per-o,
  'all'=one group); f_apply_block_pca gained a `('pool',axis,n)` triple dispatching to it (drives the
  f_feature_search two-layer streaming builder).
  EDGE GUARD (2026-07-28): `f_edge_axis_frac` (per-axis on-screen fraction of a filter's envelope
  energy, analytic erf over the same ±3σ support the extractors build) + `f_edge_keep_mask` (2-D mask
  = outer product of the two axes, since the envelope is isotropic; matches measured kernel energies
  to 0.0%). `edge_min_frac` (default `EDGE_MIN_FRAC`=0.45, None=off) drops point-samples keeping less
  than that fraction on screen — threaded through f_gabor/f_motion_energy/f_fft_features, applied at
  the naming loop, and settable run-wide via `build_cfg['edge_min_frac']` (per-spec wins). Needed
  because `_sample_idx` walks UP from the centre-congruent offset, so when (n−1−off) divides by ds the
  lattice ends ON the final pixel with no mirrored partner: at cyc_fov=16, x=4 keeps 91% but x=149
  only 56%. At 0.45 on the 150×101 render it drops cyc_fov=1's x=149 (leaving just the centred filter)
  and cyc_fov=16's two corners; raise to ~0.6 to also drop the whole flush column/rows. The centre
  sample survives at every level. CAVEAT the surviving set is RAGGED (not a full (ny,nx) lattice), so
  `f_feature_viz._grid_from_names` / `f_subsample_grid_block` cannot render or re-stride a guarded
  block. The autosave key includes the threshold, so off / 0.45 / 0.60 never share a cached block.
  `f_fft_global_features` (2026-07-29) — kind `'fftglobal'`, the PATCH-FREE limit of f_fft_features:
  plain `rfft2` of the whole frame, NO window, so perfect frequency resolution and ZERO localization —
  position is carried entirely in the PHASE. Same (ori × SF) targets via `_fft_sector_map`, but ONE
  value per target and names with NO `_r{iy}c{ix}` (there is no lattice; f_pca_group_cols tolerates it,
  the image-space viewers cannot render it). MEASURED: stats='magnitude' is shift-invariant to 0.0e+00
  under a 1-px shift while 'quadrature' moves 4.4e-01 — the clean demonstration that magnitude discards
  position and Re/Im keeps it. The contrast partner for the gabor bank.
  `f_wavelet_features` (2026-07-29) — kind `'wavelet'`, separable 2-D wavelet (numpy; no pywt in this
  env). `family` ∈ haar/db2/db4 via `_WAVELET_TAPS` + the QMF relation; `decimate=True` → DWT,
  critically sampled (VERIFIED: coefficients == 3·Σ(H/2^j)(W/2^j) exactly, and one level preserves
  energy to ratio 1.000000, i.e. orthonormal) but SHIFT-VARIANT (measured 2.02% change in per-frame
  |coef| under a 1-px shift); `decimate=False` → SWT/a-trous with dilated kernels, full resolution per
  level, redundant but shift-INVARIANT (0.00%). The 3 detail subbands LH/HL/HH ARE the orientations
  (io 0..2) — there is NO free orientation axis, so `odegs` must be None or length 3 and `n_orient`
  must be 3, else it RAISES. `f_wavelet_level_for(wl_px)` = round(log2 λ)−1 maps the pyramid's SF axis
  to a DWT level, so cyc_fov (1,2,4,8,16) → levels (6,5,4,3,2) and a cyc_fov list selects levels
  directly (one level per call, matching gabor/fft). Level is the `sf{isf}` axis, so marginalize /
  layer procs / f_score_grid work unchanged. Names `wv{LH|HL|HH}_o{io}sf{isf}_r{iy}c{ix}`.
  ADDING A KIND (2026-07-29): the checklist of every list that needs the name — beyond this registry —
  is a comment above `VISUAL_FEATURE_REGISTRY`. Missing item 2 (`_VISUAL_SOURCES`) is how 'fft' shipped
  broken. Kind-specific knobs now ride through `f_pyramid_grid`/`f_pyramid_block` via `**kind_kw`, and
  `f_pyramid_specs` VALIDATES them against that kind's `KIND_KNOBS`, so a typo raises rather than
  silently becoming an ignored extractor kwarg.
  `f_fft_features` (2026-07-26, #73) — windowed patch-FFT spectrum binned onto the GABOR BANK'S OWN
  AXES. A gabor response at (x₀,y₀,θ,λ) IS one Fourier coefficient of the Gaussian-windowed patch at
  f=(cosθ/λ, sinθ/λ), so the bank is a lossy subsample of the patch FFT; this keeps the whole lattice.
  Constant-Q preserved via ONE PATCH SIZE PER LEVEL (patch tied to λ → a pyramid, not one
  constant-bandwidth FFT); ONE wavelength per call. Shares geometry with the gabor path so the two are
  comparable: σ=sigma_scale·λ, half-width ceil(3σ) capped per axis at H−1/W−1 (exact — beyond that is
  all zero pad, same argument as f_conv_gpu._crop_kernel), centres from `_sample_idx` (identical to
  pool='sample'), borders zero-padded. `_fft_sector_map` bins the rfft2 half-plane (which covers
  orientation [0,180) exactly like the bank) by angular midpoint edges + GEOMETRIC radial edges, DC
  never binned, out-of-range dropped. Names `fftM`/`fftQe`/`fftQo`_o{io}sf{isf}_r{iy}c{ix} reuse the
  gabor convention, so f_pca_group_cols / marginalize / layer1-2 procs / f_iter_grid_cells / f_score_grid
  work UNCHANGED, and at n_sf_sub=n_ori_sub=1 columns align index-for-index with the gabor pyramid.
  stats: 'magnitude' = sector-mean |F|; 'quadrature' = Re/Im at the NEAREST coefficient (complex values
  can't be sector-averaged — phase cancels). `n_sf_sub`/`n_ori_sub`>1 subdivide sectors (richer, breaks
  the 1:1 index match). rfft2 phase is corner-referenced and is de-ramped to the patch CENTRE so Re/Im
  mean the same as gabQe/gabQo. `patch_snap=True` (default, `_snap_patch`) nudges the patch ±15% so the
  target frequency lands on a lattice point: MEASURED, that quantization — not DC-removal — dominates
  fft-vs-gabor agreement (corr(frac, agreement) = −0.88; snapping lifted mean 0.961 → 0.985, and at
  frac≈0.03 agreement is 0.9998, confirming the one-coefficient claim). Requires gamma=1 (anisotropic
  windows are orientation-dependent) and reduce='sample'. CPU only (no GPU backend; build_visual_blocks
  correctly withholds `backend=` from fft). SPEEDUP 2026-07-30 (a cyc_fov=4 unit took 11360 s / 3.2 h):
  three fixes, ~18x measured, numerics unchanged (max rel diff 3e-7, inside the uint16 autosave quant).
  (1) `scipy.fft.rfft2(..., workers=-1)` instead of `np.fft.rfft2` — numpy always computes in DOUBLE
  and single-threaded, so that one call was 6.60 s/chunk vs 0.19 s (complex64 + threads) = 35x alone.
  (2) index BOTH lattice axes at once, `pw[:, rows[:,None], cols_i[None,:]]` — the two-step
  `pw[:,rows][:,:,cols_i]` first materializes a (t,ny,W,Ny,Nx) intermediate, 7.7 GB vs 0.36 GB at
  cyc_fov=4, 6x slower. (3) skip the phase de-ramp when only magnitude is asked (|F·ph| == |F|, so it
  was a full-spectrum complex multiply for nothing) and index the sector BEFORE `abs` (26 of 6441
  coefficients). Now ~3-14 min per conv-unit over both monitors at T=140k. Driven by
  `fsearch.f_pyramid_grid('fft', …)` / a `kind='fft'` entry in VR_feature_search's `feat_sets`.
  BUGFIX 2026-07-27: 'fft' was missing from `f_feature_source._VISUAL_SOURCES` (and so from the derived
  `_RENDER_SOURCES`), so an fft input got no `vis_specs` entry, never triggered the monitor render, and
  died in `_extract_input` with "build did not produce source 'fft'" — the fft path had never actually
  run. Everything else already handled it (`_visual_spec_from_input`, `_VISUAL_PARAM_KEYS`,
  VISUAL_FEATURE_REGISTRY, f_conv_autosave.AUTOSAVE_KINDS). The reason it appeared to work earlier is
  that those configs were built through `f_pyramid_grid('gabor', …)`, i.e. gabor banks under fft names.
  SECOND extraction method (2026-07-11): `f_gabor_features`/`f_motion_energy_features` gained
  `pool='grid'`(default, mean-pool)|`'sample'` (POINT-sample the dense response every `sample_ds`
  px → grid-shaped output `r{iy}c{ix}` over ⌈H/ds⌉×⌈W/ds⌉; ds=1 → full conv) + `presmooth_sigma`
  (blur movie before conv). Shared reducer `_make_reducer`; `f_subsample_grid_block(X,names,factor)`
  derives coarser resolutions from ONE fine extraction (exact stride composition, no re-conv).
  `build_visual_blocks` `type` override: a spec KEY can be an arbitrary block name mapping to a
  registry extractor via `type` (so several configs → distinct named blocks). Driven by the
  sampled-methods cells in VR_ca_cebra.py (build + `f_score_representations` compare vs pix).
- `functions/f_feature_viz.py` — reconstruct an image-space monitor movie from
  a feature block to SEE what it keeps (created 2026-06-09).
  `f_reconstruct_feature_movie` (pix: inverse-PCA; grid/flow: cell-paint by a
  channel), `f_resolve_recon_n_pcs` (int / variance-fraction / None →
  component count), `f_stack_recon_over_original` (recon over the matched
  original movie, resampled to the imaging clock), `f_plot_movie_frames` /
  `f_plot_movie_frames_compare` (plot example frames inline; compare original vs
  feature frames at the same timepoints). `_grid_from_names` (2026-07-11) infers the (ny,nx)
  lattice from `r{iy}c{ix}` names so POINT-SAMPLED blocks render with the right shape (not the
  stale params grid). Driven by `VR_feature_viz_cache.py` / `VR_feature_viz_conv_repo.py`.
  Driver helpers factored out of the cache script (2026-07-12, parameterized): `f_orig_for_side`,
  `f_orig_frames` (original at frame_idx from example_frames/orig_mov), `f_frame_indices`,
  `f_save_movie_pair` (feature TIFF + paired `_ORIG.tif`); the cache script keeps thin
  state-binding adapters (`_orig_frames`/`_frame_indices`/`_save_with_orig`). FEATURE-KERNEL OVERLAY
  (2026-07-22, verify gabor/meng size + sampling vs the monitor): `f_kernel_geom` (normalized spec →
  wl_px/σ/support/ds/n_samples via the SAME f_deg_per_px/f_ds_from_wl as extraction), `f_plot_filter_kernel`
  (mark ALL sample locations as points + draw the actual gabor TO SCALE — `_gabor_pair` even/odd |
  `_gauss_env` env, the SAME builders — at the sample nearest screen center, out to ±sigma_extent·σ),
  `f_kernel_gallery` (raw bank kernels). Driven by
  `VR_feature_kernel_plot.py` (build specs via f_pyramid_specs = same as the search; ONE figure per
  orientation×SF filter + size/sampling readout).
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
- `functions/f_feature_search.py` — feature-PARAMETER search on ONE dataset scored by ENCODE +
  DECODE (created 2026-07-18; neural-scored sibling of f_feature_optim's recon sweep). `f_make_neural_target`
  (est1 → Y_neu, VR_ca_cebra recipe: drop-silent→smooth→peak-norm), `f_granularity_configs` (fresh-build
  separate/combined/raw PCA comparison for one block+extraction), `f_variant_configs` (the cache-load
  analogue → …_pcasep/…_pcacomb/…_pcaraw cache_tags, force_build=False only), `f_feature_search` (render
  ONCE then loop configs → f_resolve_features [`only`=score blocks, deps auto-pulled; force_build=True →
  every block origin='build', GPU, cache bypassed, pre-rendered movie reused] →
  f_dataset_scan.f_score_representations → tidy ranked table; one bad config never kills the sweep].
  RAW extractions are MEMOIZED across configs (`_raw_block`/`_reduce_raw`,
  keyed by `_extraction_sig`) → comb/sep/raw + prep variants of one extraction share ONE GPU conv (render
  once, conv once, only PCA+prep re-run); derived combos / cache loads use the full resolve).
  `f_pix_config(proc)` = the raw-movie pixel BASELINE config (the real visual input) — the reference to
  compare the gabor/fft/meng grids against. SIMPLIFIED 2026-07-27: NO pix-specific processing stage. The
  block is always built RAW (`n_pix_pca=None`, no `pix_pre_proc`) and the ENTIRE ordered `proc` step-list
  is applied by the ordinary block-proc path (f_feature_search line ~707, which honours ('pca',n)), i.e.
  pix runs through the SAME f_proc stages as a bank's layer1/layer2. `proc=[]` → raw pixels untouched.
  Name derived from the ordered steps, so order stays visible (pix:zscore_pca300 = correlation-PCA vs
  pix:pca300_zscore = z-scored PCs). The build flattens the movie to (T, H·W·n_side) float32 either way,
  so moving the PCA out of the build costs no extra peak there; it does make f_resample_to_imaging
  interpolate ~30k raw columns instead of a few PCs (column-chunked in float32 → bounded, just slower).
  Previously the list was SPLIT at the first int ('pca',n) into pix_pre_proc / n_pix_pca / post-PCs —
  removed, since it made pix the one kind whose reduction lived outside layer1/layer2.
  FEATURE-SET FORMAT (2026-07-27): `f_feature_configs(feat_sets, strict=True, tag_suffix=None, **common)`
  is the ONE declarative entry point — a single self-contained dict where a reserved `'_common'` key holds
  the knobs every set inherits and each other key is one set `{'kind': 'gabor'|'fft'|'meng'|'pix', …}`.
  The dict is READ by the canonical `f_feature_source.f_resolve_feat_sets` (defaults→override→validate,
  returning (name, tag, kind, knobs)), SHARED with the plan side `f_vis_sets_to_plan` so the same dict
  means the same thing in VR_feature_search and VR_ca_cebra. `PYRAMID_KINDS`/`KIND_KNOBS` live there too.
  The dict KEY becomes the config tag → '<kind>:specs:<key>' (pix → 'pix:<key>'). Dispatch: 'pix' →
  f_pix_config, else f_pyramid_grid(kind, tag=key, **knobs) (pass-through, so hand-built config lists
  still work). pix takes the same layer1_proc/layer2_proc as every kind (one block, no filter axis → the
  two run back-to-back); only the AXIS knobs (cyc_fov/odegs/marginalize/stats) are withheld from it.
  PRECEDENCE entry > `common` kwarg > '_common'. '_common' holds the DEFAULTS and a one-off VARIANT is
  just an entry repeating the knob (e.g. `layer2_proc=[]` next to the pca-100 default = a raw twin) — no
  second pass and no selector list. The kwarg layer is for PROGRAMMATIC sweeps (many values × every set
  in one expression, each pass given a `tag_suffix` so names don't collide); an entry that pins the swept
  knob keeps its own value. `strict` raises
  on an unknown `kind` and on a per-entry knob that kind IGNORES (tf_hz/output on gabor|fft, stats/
  n_sf_sub on meng, axis knobs on pix) — the guard for the silent bug where a pyr_sets3 of 'fft'/'meng'
  entries was built via f_pyramid_grid('gabor', …), scoring gabor banks under fft/meng names.
  COMPONENT SWEEP (2026-07-29, cell 3c): `f_pca_tag(n)` names one ('pca', n) spec distinctly across all
  four forms — int `pca50` / float `pcav0.95` / (frac,cap) `pcav0.99c200` / None-or-0 `pcaraw` — so a
  count of 50 and a fraction of 0.5 can't collide in the scorecard (also used by `_pix_step_tag`).
  Sweep = one `f_feature_configs` pass per entry with `tag_suffix=f_pca_tag(n)`; conv_autosave reloads
  the conv so only the PCA + scoring re-run. `f_plot_pca_sweep(res, …)` = 2×2 panels — metric vs `n_dim`
  (log x) and vs `var_kept`, × decode/encode — with ONE LINE PER FEATURE SET via `_sweep_series`, which
  strips the trailing `_pca…` tag off `config` and groups on what remains, so gabor/fft/meng/wavelet
  overlay. Integer entries sweep the n_dim axis, fraction entries sweep the var_kept axis; mixing both
  shows where the curve saturates in each currency. VERIFIED: int gives exactly n dims, float gives
  var_kept ≥ the request, (frac,cap) never exceeds cap, and var_kept is monotone in n_dim.
  SWEEP SPEEDUPS (2026-11-06): `f_feature_search(scores=('encode',))` skips the decode fit (columns NaN;
  same knob on f_score_representations); a grid cell byte-identical to one already scored in the call
  (blake2b of X) reuses that row — e.g. sweep values past a set's full width; layer1 pca/svd now fits ONE
  exact eigen-decomposition per filter per call (`_l1_decomp`, cached in raw_cache) and slices it per
  value (`_l1_nested`, same k/identity rules as _pca_fit/_svd_fit; verified identical k on 30 cases).
  Exact > sklearn's randomized TruncatedSVD where singular values nearly tie (measured 374-col filter,
  n=64: σ64/σ65=1.003, randomized subspace off by sin 0.32, exact hits the ideal top-64 energy). Cell 3c
  in VR_feature_search = ONE component sweep: sweep_layer layer1|layer2 × sweep_score encode|decode →
  `f_plot_comp_sweep` line plot, clipped past full width (`l1_dim_in` column = width before layer1).
  BUGFIX 2026-11-06: grid blocks with a layer1 reduction but no layer2 PCA carried the RAW column names
  (e.g. 20 names on a 6-col block) — both f_iter_grid_cells and f_feature_source._build_grid_block. Now
  `_layer1_names` gives each filter its raw names if layer1 kept the width, else '<filter>_PC1..k';
  `raw_names` = the layer2-input columns, matching `raw_dim`.
  `f_feature_sizes(res, sort_by='decode_peak_gb', ram_gb=None)` = the DATASET-SIZE monitor: one row per
  scored block (n_frames × n_dim, raw_dim, `reduction` = raw/n_dim, `l1_var%` / `var%` / `varTot%` =
  the variance the kept components explain at layer1 / layer2 / overall ('-' where that layer had no
  PCA), block_mb, decode_peak_gb, + a ok/tight/OVER `fits` verdict when ram_gb is given). Reads shapes back off `res` rather than a live
  registry BECAUSE the grid build streams — each cell is scored then freed, so the blocks never coexist.
  build_feature_blocks gained `pix_pre_proc=None` + `n_pix_pca=None`→raw-pixel block (both default off →
  cebra/dataset_scan unchanged). PROC-BEFORE-CV (2026-07-21): f_feature_search scores with
  f_score_representations `proc_before_cv=True` — ALL procs (feature layer/pix + neural_proc) applied
  GLOBALLY, the CV split only divides, folds add NO within-fold normalization (decode neural predictor =
  `f_proc(Yk, neural_proc)` once; encode feature predictor = the globally-proc'd block as-is; via empty
  within-fold pred_proc). `proc_before_cv=False` (default) keeps the leak-free within-fold path
  (cebra/dataset_scan byte-identical). Trade-off: global procs are target-blind but fit on all timepoints.
  `f_plot_feature_search` (decode+encode scorecard bars). FEATURE-GRID builder (2026-07-20, ONE param
  builder — replaced the earlier 3 two-layer builders): a config with a `'grid'` key uses `f_build_grid`
  (memory-safe streaming). `f_feature_grid(kind, marginalize=(...), sf_cyc_deg, tf_hz, n_orient|odegs, ds,
  layer1_proc, layer2_proc, tag=None, ...)` (`odegs`=explicit orientation ANGLES in degrees, e.g. (0,90)
  → those exact orientations + sets n_orient; None → n_orient evenly-spaced in [0,180). The build
  extractors f_gabor_features/f_motion_energy_features gained an `odegs` kwarg [thetas=deg2rad(odegs)],
  forwarded via _VISUAL_PARAM_KEYS; GPU path already took thetas. `tag` suffixes the config name so a
  PROC sweep — same bank+marginalize, different layer1/2 procs — doesn't collide in the scorecard; driver
  cell 2 loops marg_sets × proc_sets; CAVEAT each proc condition RE-CONVOLVES, conv not yet shared):
  the bank's filters = an ori×wl(SF)×tf grid; `marginalize` ⊆
  {'ori','wl'/'sf','tf'} POOLS those axes, the REMAINING axes form a param matrix → one scored CELL per
  remaining-axis combo (marginalize=()→per filter, {'ori'}→per (sf,tf), {'ori','wl'}→per tf, all→whole
  bank). `layer1_proc` (per individual filter) + `layer2_proc` (per cell) are UNIFIED f_proc step-lists
  (legacy n_pca_layer1/n_pca_layer2/input_norm/prep fold in). `f_iter_grid_cells` (GENERATOR) streams: conv
  ONE (sf,tf) at a time (orientation always batched), group per filter via `f_pca_group_cols('none')` +
  `f_proc(layer1)`, FREE raw, route filters to cells by the non-marginalized axes, and YIELD each cell
  (concat + `f_proc(layer2)`) the moment its LAST contributing conv-unit is done (`_has_later_contributor`).
  f_feature_search INTERLEAVES: score each yielded cell (single-block f_score_representations) → print
  progress → FREE it. Peak RAM ≈ one conv raw + the currently-open cells (conv-local marginalize=()/{'ori'}
  → ONE conv-unit's cells → bounded even with NO pca; pooled sf/tf → until the last contributor). Verified
  interleaved == build-all (Δ=0). `f_grid_shape(g)`→(#cells,#conv-units); thin `f_build_grid` = dict(gen)
  for tests. Cell names carry PHYSICAL values on every axis — orientation in DEGREES (io·180/n_orient,
  e.g. 'o45_sf0.13_tf2.0'), not the raw filter INDEX. `f_score_grid(res,metric)` parses those → ori(deg)/
  sf/tf long df to pivot into the 3D score matrix. UNIFIED PROCESSING (`f_feature_helpers.f_proc`, 2026-07-20):
  an ordered step-list `[('norm',zscore|standardize|total_var|first_sv), ('pca',n|frac|(frac,cap)),
  ('whiten',True)]` (each optional; fit/transform split for leak-free re-apply) used at grid layer1/2 AND
  within-fold on the decode/encode predictor — `f_score_representations` `neural_proc`/`feature_proc` →
  `f_blocked_cv_r2`'s optional `pred_proc` (default None = byte-identical old standardize+PCA, so
  cebra/encoding/decoding untouched). PROGRESS OUTPUT (2026-07-20): clean hierarchy — one `[render] …s`
  line, per config `[i/N] name  <descriptor>` + `build: M cells from U conv-units in …s` + per-cell
  `[j/M] cell dec/enc …s` (via `f_score_representations(verbose=True)`) + `done in …s`; the deep build ⏱
  prints in f_build_features/f_render_session/build_visual_blocks are now `verbose`-gated (DEFAULT True →
  cebra/gen/viz unchanged; the search passes verbose=False to silence them). A config =
  {name, blocks, plan overrides, source overrides}; plan `n_pca` is the sep/combined/raw switch on a
  FRESH build (int | ('filter'|'orient',n) | None — the granularity tuple form added to
  f_visual_features.f_apply_block_pca). Fresh GPU regen (not cache) is the DEFAULT. `f_build_features`/
  `f_resolve_features` gained a `prerendered` movie passthrough so the render is shared across configs.
  `feat_prep` axis (`f_prep_configs`): None | 'zscore' | 'whiten' (via f_resolve_features whiten=
  _whiten_block) — scored with proc_before_cv=True (no within-fold standardization of the predictor), so
  feat_prep moves ENCODE (ridge penalty depends on column scale) as well as DECODE (features=target,
  variance-weighted R²). [Corrected 2026-11-06: an earlier note said ~a no-op for ENCODE; that held only
  for the within-fold path.] Driven by `VR_feature_search.py`. EXTENDED TIMING (2026-07-24): a `timing=False`
  flag ADDITIVE over `verbose` — flushed timestamped START + duration markers per MAJOR step (render, per
  conv-unit, finalize/PCA, build+pca [pix raw-extract vs PCA split], decode, encode split in
  f_score_representations `timing=`) so progress is readable live while monitoring CPU; default off =
  unchanged output. CONV AUTOSAVE (2026-07-24): `_raw_block` gained an SSD reuse hook (build_cfg
  ['conv_autosave']) — reload the raw pooled gabor/meng block if a matching recipe×geometry block exists
  (via `f_conv_autosave`), else build + save; PRE proc/marginalize so proc variants of one bank reload
  instead of reconvolving. SF NAMING (2026-07-24): `f_iter_grid_cells` now formats sf/tf via
  `f_num_tag` (cell names / labels / timing prints) so cyc_fov-derived long floats read `sf0.01016` not
  `sf0.010162601626016261`; the raw float stays in the streaming key (contributor match), formatted only
  at the name join; `f_score_grid` re-parses the short names (display-only, round-trip-safe).
- `functions/f_encoding.py` — encoding arm (TODO #57): `f_lag_embed` /
  `f_lags_from_ms` (short causal FIR design), `f_encode_per_neuron` (per-neuron
  blocked-CV via `f_blocked_cv_r2`, returns OOF Pearson r + R²),
  `f_encode_null` (circshift target null), `f_encode_block_importance`
  (drop-one unique + block-alone marginal r), `f_encode_summary`. Model choice
  (ridge/ridgecv/knn/pls/cca) threads through to `make_decoder`. P2 sub-plan D1/D5
  (2026-07-03): `f_encode_fit_weights` (per-neuron ridge on ALL rows → the WEIGHTS =
  receptive field, reshaped (d_stim,n_lags,n_kept)), `f_rf_pixel_frames` (→ n_lags×H×W
  spatiotemporal pixel RF), `f_rf_grid_location` (|W| pooled over channels → location map),
  `f_encode_ceiling_fraction` (per-cell r/r_full, #57 relative ceiling), `f_encode_sta`
  (parameter-free reverse-correlation RF baseline, same shape → same viewer). D2 (smoothness
  + σ-CV): `f_encode_fit_smooth` (generalized Tikhonov ‖y−Xw‖²+wᵀPw, P = graph-Laplacian over
  spatial neighbors + adjacent lags → legible RF where plain ridge is salt-and-pepper),
  `f_encode_smooth_cv` (blocked-CV OOF r for the smooth solver), `f_encode_tune_sigma_lambda`
  (JOINT deconv-σ × smoothness-λ search by encoding r — NESTED guard: selects on a leading
  split, reports `sel_r`/`report_r` on a held-out tail, P2 §7). Shared `_encode_design` behind
  fit_weights/sta/smooth. D3 banded ridge + variance partition (2026-07-03): `_band_columns`
  ((band→[blocks]) → per-design-column band ids, lag-major tile), `f_encode_banded` (per-neuron
  banded-ridge CV, `sel_range` slab for the nested split), `f_encode_tune_bands` (per-band α
  joint-grid select/held-out-report), `f_encode_variance_partition` (drop-one on the JOINT
  banded model → per-cell visual-unique / behavior-unique / shared R²; group blocks into
  'visual'+'behavior' bands for the P2b-5 read). Uses `BandedRidge` (f_decoding).
  ENCODER CONFIG (2026-11-06, PLAN_encoding_config.md / TODO #77): `_enc_defaults()` at the TOP of
  f_encoding.py documents every key + its options; defaults = VR_ca_cebra's enc_params (ridge α 1e3,
  lags 0–500 ms, interleaved 5-fold, 150 joint feature PCs, standardize; + target_sigma_frames 6 /
  target_peak_norm for the coming f_enc_target). `f_default_encoder_params(**overrides)` now RAISES on
  unknown keys (it silently accepted typos); `f_enc_config_tag`. VR_dataset_scan pins its old values.
  `f_enc_target(est1, cfg)` → per-cell target via `f_utils.f_neural_rate` (drop silent → smooth σ → peak-
  norm), the ONE recipe also behind f_make_neural_target + f_ens_neural_input (bit-identical refactor).
  `f_encode_sweep(enc_cfg, axes, targets, built_blocks)` = the general encoding sweep (encoder keys /
  'lag_ms' / 'target', 1-D or N-D product, every target by default) → tidy df + per-unit r; plot with
  `f_encoding_plots.f_plot_encode_sweep`. 1-D result == f_encoder_sweep (parity-tested).
  VR_feature_search ENCODES with this shared config since 2026-11-06: `f_feature_search(enc_cfg=…)` →
  f_run_encoder per scored block (n_feature_pcs=None = no double reduction), same encode_* columns + pop
  scores + enc_tag; Cell 1 enc_cfg + Y_neu from f_enc_target. Encode numbers are NOT comparable to runs
  before this (old scorer: α 1, 200 ms only, σ 0).
  VR_ca_cebra's encoding sweep cell runs f_encode_sweep (targets cells/ens_*, optional 2nd axis, line plots
  + the old panel view); VR_encode_scan encodes via `f_encode_scan.f_encode_one_cfg` with enc_cfg
  (n_feature_pcs=None). All three scripts now share one encoder model (TODO #77 done 2026-11-06).
- `functions/f_encoding_plots.py` — encoding visuals (P2·D1, 2026-07-03): `f_plot_rf`
  (a fitted RF `(n_lags,H,W)` as a frame-per-lag weight-movie, diverging 0-centered) +
  `f_plot_rf_cells` (cells×lags grid). The RF viewer is the build's canary. Driven by the
  D1 cells in `VR_ca_encoding.py`.
- `functions/f_feature_cache.py` — per-dataset feature cache (TODO #66):
  `f_save_features` / `f_load_features` / `f_features_path` / `f_features_exist`
  (joblib; `tag`=GEN_TAG in the filename) + `f_write_config_json` (provenance
  sidecar) + `f_list_feature_files` / `f_resolve_feature_file` (scan a features dir
  RECURSIVELY for a dataset's caches; pick one by filename > tag > index) + `f_pick_frames` (n
  evenly-spaced example frames). All block caches live in ONE `cache/` subfolder (2026-07-12,
  parallel to `conv_repo/`): `<features_dir>/cache/<mouse>_<dset>_<GEN_TAG>_features.joblib`; the
  resolver rglobs so legacy flat/per-dataset files still load. Caches `built_blocks` so render+extract runs once;
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
    - PIX GATE (2026-07-28): `build_pix` was `pix_frames is not None`, i.e. ON whenever the render
      ran — so a gabor/meng-only build still flattened the movie to (T, H·W·n_side) float32 and PCA'd
      it (11.7 GB / ~40-67 s at T=96k) on EVERY `_raw_block` call, per conv-unit. `_resolve_build` now
      sets `cfg['build_pix'] = 'pix' in srcs` and f_pipeline honours it (default True for direct
      callers). This was the 66 s that followed an `autosave HIT` — the autosave itself was fine.
    - `f_build_features(session, cfg, ...)` (phase 3) — the SHARED render+extract
      (f_render_session → side-select → build_feature_blocks + build_visual_blocks).
      Single source of truth used by cebra/encoding's build_fn AND VR_feature_gen.py's
      batch loop. `cfg` is a FLAT dict (also the gen_config).
    - `f_render_session(session, cfg)` — the render half (f_angles_to_movie_v2 [+terrain]
      → two_mon_frames), factored out so VR_feature_viz_cache.py can REGENERATE the raw movie
      from the session when a cache lacks it (SAVE_MOVIE was off).
- `functions/f_conv_repo.py` — dense per-filter CONVOLVED-feature repository (fast decoder
  optimization; created 2026-07-11, #68). Persists the DENSE convolved maps ONCE per dataset so
  resolution/param sweeps LOAD+subsample instead of re-convolving. `f_gabor_map` / `f_meng_map`
  (one dense (T,H,W) map per orientation×wavelength[×TF], reuse f_visual_features privates),
  uint16 quant (`_quant_u16`/`_dequant_u16`, per-filter min/max). Storage = per-dataset repo dir
  with ONE uint16 (T,H,W) `.npy` per side×filter — PHYSICAL specs name maps
  `<stat>_<side>_o{odeg}_sf{cyc/deg}[_tf{Hz}].npy` (e.g. `gabM_L_o0_sf0.13.npy` /
  `mengopp_L_o0_sf0.13_tf2.npy`; `_filter_fname`), legacy pixel specs use `_wl{px}[_tf{tag}]`; the
  filename is only an on-disk key + label (selection/loading use manifest fields). `f_conv_repo_retag`
  migrates an old physical repo's px-named files → the clean sf/tf names. odeg = orientation in DEGREES.
  Plus `manifest.json` (params + quant) + `frame_t.npy`. Repo I/O
  `f_conv_repo_dir` / `f_conv_filter_specs` / `f_build_conv_repo` (INCREMENTAL — skips filter files
  already in the manifest) / `f_list_conv_filters` / `f_conv_repo_meta` / `f_conv_frame_t` /
  `f_load_conv_map` / `f_sample_conv_to_block` (mmap → subsample-before-dequant → (X,names) raw
  block, RAM-safe; `ds` = scalar | (h,w) tuple | CALLABLE `ds(entry)` for PER-FILTER strides).
  `f_ds_from_wl(wl_px, factor)` = Nyquist-matched stride (magnitude map is phase-invariant → smooth on
  σ=sigma_scale·λ, so ds≈0.5·λ_px); feature-plan sample inputs take `ds`=int/(h,w)/`'nyquist'` (+`ds_factor`).
  meng `temporal_support`='auto' → `f_auto_support(tf_hz, fs, n_cycles, support_min, support_max)` =
  per-tf CONSTANT-CYCLE support (≈n_cycles·fs/tf, odd, capped) mirroring the spatial gabor's σ∝λ →
  uniform temporal bandwidth (low tf → long window, capped by support_max). The scan groups meng by
  (θ,wl,output) so per-tf support keeps sharing one spatial conv; f_meng_maps_gpu takes {tf:support}.
  The SHARED consumer `f_load_conv_blocks(repo, ds_list, n_pca, kinds,
  wavelengths, odegs, tfreqs, select)` → built_blocks-compatible sampled blocks (`gaborS_ds*`/
  `mengS_ds*`), with field limits to LOAD ONLY a subset (kinds=('gabor',) / wavelengths=(6,) /
  odegs=(0,90) / tfreqs=(0.15,); shared `_conv_filter` — a tfreqs limit only applies to entries that
  HAVE a tfreq, so it never drops gabor). Used by the scripts (each has a `smp_repo_tag`/`conv_tag`
  knob for the repo GEN_TAG): VR_ca_cebra.py (`methods_source='conv_repo'`) + VR_ca_encoding.py
  (opt-in merge cell → add gaborS_ds*/mengS_ds* to enc_blocks), VR_feature_viz_conv_repo.py
  (dedicated repo viz), VR_feature_recon.py (`feat_source='conv_repo'` — renders the movie for the
  target, loads features from the repo). `f_conv_activity(repo, select, reduce)` → per-filter response magnitude over time
  (frame-mean over space) + one time×space number per filter (df + ts) — the diagnostic for CHOOSING
  orientations / wavelengths / temporal freqs (`frame_ds` reads every Nth frame → I/O ~N× less, the
  speed knob since it otherwise reads every dense map in full); plotted by `f_feature_viz.f_plot_conv_activity`
  (grouped by odeg/wavelength/tfreq) in VR_feature_viz_conv_repo.py. `f_conv_repo_to_h5` converts a
  repo → browsable HDF5. ~1 GB/filter/side at num_samp=101.
  A converter to HDF5/TIFF for browsing can be added later if needed. `f_build_conv_repo` gained a
  `backend='cpu'|'gpu'` (+`device`) knob (2026-07-16) → GPU path via f_conv_gpu at the `_spec_maps`
  seam (parity-tested; `cfg['conv_backend']` in VR_feature_gen.py). `f_conv_filter_specs_phys` +
  `f_deg_per_px` (2026-07-16): build specs from PHYSICAL units — gabor SF in cycles/degree
  (`sf_cyc_deg`) + meng TF in Hz (`tf_hz`) — converted to px/frame via deg/px (hFOV/vFOV + render
  resolution × sample_ds) and fs; stamps sf_cyc_deg/wl_deg/tf_hz onto each spec so the scan's filt_df
  + tuning readout report preferred SF/TF in cyc/deg + Hz (resolution-independent). NOTE: rig FOV
  corrected 2026-07-16 to FOV_axis='horizontal'/FOV_deg=98.4 (→ vFOV 66.18°) in DEFAULT_CAM_PARAMS +
  VR_feature_gen; existing conv_repos (built at square 101×101) must be regenerated.
- `functions/f_conv_gpu.py` — torch (GPU) gabor / motion-energy convolution, CPU-parity with
  f_conv_repo (created 2026-07-16). Reuses the numpy kernel builders `f_visual_features._gabor_pair`
  / `_temporal_quad` as the single source of truth for the MATH; only the conv engine differs.
  `f_gabor_map_gpu` / `f_meng_maps_gpu` (dense (T,H,W) tensors, meng SHARES the spatial gabor across
  tfreqs) + reduce-fused `f_gabor_activity_gpu` (batches filters as conv2d out-channels → whole-
  monitor magnitude, never materializes a full dense map) / `f_meng_activity_gpu` + `_spec_maps_gpu`
  (the f_build_conv_repo GPU backend). Parity gotcha handled: torch conv2d/conv1d is cross-
  correlation, so kernels are FLIPPED to match scipy's true convolution (else meng opponent negates);
  verified allclose (corr≈1.0, rel~1e-6) incl. sign. STREAMS over time: the movie stays on CPU (native
  uint8), only a `chunk_t`-frame slice (+temporal halo for meng) is on the GPU (`_to_chunk`) — peak VRAM
  ∝ chunk_t·filt_batch·H·W, INDEPENDENT of session length (a full 140k-frame movie is 8.5 GB → OOM if
  resident; streamed peak ~5 GB gabor / 1.5 GB meng). All functions return CPU arrays. Also the
  feature-BUILD-path GPU backend (2026-07-17): `f_gabor_pooled_gpu` / `f_meng_pooled_gpu` — drop-ins
  for f_visual_features.f_gabor_features / f_motion_energy_features's inner conv. Reached via
  `build_cfg['conv_backend']='gpu'` → build_visual_blocks(backend=…) → the extractors' `backend='gpu'`
  (much faster than the CPU fftconvolve which also materializes the full ~8.5 GB dense response).
  SPATIAL ENGINE (2026-07-24): every spatial conv goes through `_spatial_conv` = exact drop-in for
  `F.conv2d(x,w,padding=pad)` that (a) `_crop_kernel`s away kernel rows/cols which can only ever hit the
  ZERO PAD (exact — with σ=0.5λ a cyc_fov=1 gabor is 453² on a 101×150 render, i.e. LARGER than the frame;
  cropping to 2H−1 × 2W−1 = 201×299 is free and also shrinks the FFT) and (b) dispatches to direct conv2d
  for small kernels or `_fft_conv2d` for large (`FFT_MIN_K`=48; `backend='direct'|'fft'` forces). Direct
  conv2d costs K² per pixel — a cost the CPU path NEVER paid (it used scipy.fftconvolve), so the original
  "GPU" port silently traded an O(HW log HW) algorithm for O(HW·K²). `_fft_conv2d` = rfft2 → pointwise →
  irfft2, ONE forward transform shared across all filters of a conv-unit (same SF → same kernel size),
  kernels flipped BACK (a frequency multiply is TRUE convolution; conv2d is correlation — without this the
  antisymmetric ODD gabor and the meng opponent come out NEGATED), and sub-chunked over frames internally
  (`FFT_BUDGET_BYTES`) since linear-conv padding inflates 101×150 → ~315×448. Measured 8.1× on the 5-SF
  pyramid (~23 min → ~3 min per bank over 100k frames × 2 monitors). Magnitude parity vs the CPU reference
  IMPROVES ~100× (3.9e-7 vs 3.6e-5 — the reference is itself an FFT); meng opponent 1.2e-5, corr 1.0.
  `FFT_WITH_PHASE=False` (default) keeps gabor PHASE on the direct path: `re/mag` is gated by an ABSOLUTE
  `mag > 1e-6`, so over a BLACK region (true response exactly 0) direct gives exact zeros → phase 0 while
  ANY FFT engine gives ~1e-5 rounding → gate opens → arbitrary unit vector (point-sampled phase corr drops
  to 0.86). Pre-existing, not new: cpu(scipy) vs gpu(direct) ALREADY disagree on phase (corr 0.9917) — the
  GATE is the bug. A scale-relative gate would fix it for all engines; that is a feature-semantics change,
  to be made deliberately with TODO #70/#71, then set FFT_WITH_PHASE=True. Temporal conv1d unchanged
  (support is short). Parity/bench scripts: scratchpad `test_fft_conv_parity.py`, `test_phase_zero_region.py`,
  `bench_fft_conv.py`.
  GPU-NATIVE stat+pool (2026-07-24): the magnitude/phase/even/luminance + meng opponent/both AND the
  spatial pool now run ON THE GPU (`_gab_stat_torch` per token; `_make_reducer_torch` = torch reducer,
  byte-parity with `f_visual_features._make_reducer`: grid = 0/1-assignment matmuls summing the EXACT
  `f_grid_bounds` reduceat segments ÷ per-cell area [handles non-divisible edge cells]; sample = centered
  strided index), so only the small pooled (t,ny,nx) crosses to CPU — NOT the full-res (t,H,W) per filter
  (that per-filter `.cpu()` + numpy stat+pool was the ~900s bottleneck: GPU idle, all cores pegged).
  The pooled fns now take `tokens` + `pool/grid/sample_ds` (not numpy `chan`/`reduce_fn`); assembly in the
  extractors is unchanged. Parity verified vs the CPU numpy path: torch-CPU EXACT (~1e-7) for mag/phase/
  even/lum × opponent/both × grid(incl. non-divisible)/sample; on CUDA gabor's batched conv2d picks a
  TF32 algo (~2e-4 scale-rel, negligible for standardized features, and PRE-EXISTING — the conv2d was
  always on GPU), meng conv stays ~1e-7 (disabling `torch.backends.cudnn/cuda.matmul.allow_tf32` → 1e-7).
- `functions/f_conv_autosave.py` — SSD autosave for CONVOLVED+POOLED gabor/meng/fft blocks (created
  2026-07-24). HOOK MOVED 2026-07-26: the active hook is now SPEC-LEVEL inside
  `f_visual_features.build_visual_blocks` (`autosave=` kwarg ← `f_autosave_ctx`), the ONE point every
  build path funnels through (f_resolve_features → f_build_features → build_visual_blocks), so
  VR_feature_search AND VR_ca_cebra share one store — a bank convolved by the search reloads in cebra.
  Keyed by `f_spec_fname` on the RESOLVED extractor params (which already bake in geometry: sf_cyc_deg
  → `wavelengths` px, ds='nyquist' → `sample_ds`) × sides × movie shape × #imaging frames, skipping
  perf-only knobs (`_KEY_SKIP` = backend/device/chunk_t/verbose) so a cpu/gpu or chunk-size change
  reuses the block. `AUTOSAVE_KINDS`=('gabor','meng','fft'). Cached PRE-PCA and pre-proc → ONE saved
  block serves every n_pca / proc / marginalize variant. Opt in with `build_cfg['conv_autosave']=True`
  (+ `conv_autosave_tag`); the root defaults to the source `features_dir` (injected by
  f_resolve_features), so callers need only the flag. Verified: MISS→write, HIT→reload at rel ~5e-6
  (per-column uint16), key sensitive to sample_ds/wavelengths/stats and blind to chunk_t, ~50× faster
  than rebuilding. The legacy input-spec-keyed `f_autosave_locate` is kept for direct callers but is no
  longer called by `f_feature_search._raw_block` (which now keeps only its in-run memo); stores written
  under the old `<mouse>_<dset>_<tag>_<geomhash>` dirs are orphaned by the new
  `<mouse>_<dset>_<tag>` layout and can be deleted. Persists the RAW (n_pca=None) pooled block that
  `f_feature_search._raw_block` builds so a later config sharing the same bank but differing only in
  proc/marginalize (e.g. the pyr_sets mag/mag2pca/mag1norm variants — all same conv, different
  layer1/layer2) RELOADS instead of re-convolving (the in-run raw_cache frees each conv per unit to
  bound RAM, so this store is the cross-config/cross-run reuse). Save point is PRE proc/marginalize, so
  ONE block serves every variant. `f_autosave_locate(build_cfg, session, inputs)` → (dir, path,
  geom_meta): keyed by extraction RECIPE (`inputs` minus the non-conv 'proc'/'input_proc') × the pixel
  REALIZATION (lat/vert_samp, deg_per_px, fs, sigma_scale, gamma, side — physical inputs are
  geometry-independent so these MUST bind the key) → `<features_dir>/conv_autosave/<mouse>_<dset>_<geom_
  tag>/<human-tokens>_<recipe_hash>.npz` (human tokens via `f_feature_helpers.f_num_tag`). PER-COLUMN
  uint16 quant (`_quant_u16_cols`/`_dequant_u16_cols` — a pooled block mixes filters/stats of different
  scale, so one per-array min/max would lose precision; rel err ~1e-5, negligible pre-standardize) +
  shared `frame_t.npy` + `manifest.json` (provenance). `f_autosave_load` → (block, frame_t) shaped like
  _raw_block's return or None (miss/unreadable → rebuild); `f_autosave_put` writes atomically. Opt-in via
  `build_cfg['conv_autosave']=True` (+ `conv_autosave_tag` namespacing); gabor/meng ONLY (pix has no conv
  and its raw pixel block is ~30k cols). f_feature_search threads the store root from `base_source` into
  build_cfg and the `timing` flag → `autosave HIT`/`WROTE` prints. To share across different ds/render
  size use a conv_repo (dense) instead.
- `functions/f_feature_scan.py` — wide feature-parameter SCAN engine + pluggable SCORER (created
  2026-07-16). Splits GENERATION from SCORING so encode/decode scorers reuse the same generation.
  `f_scan_conv_activity(movies, side_tags, specs, backend='gpu', device, filt_batch, reduce,
  combine_sides, transform_meng, standardize, scan_file, data, store_neural, ...)` →
  (X_stim (T,n_filt), filt_df): convolves a wide gabor/meng grid on the fly (gabor batched
  `filt_batch` at a time = the GPU resource knob; meng grouped by (orientation,wavelength) sharing
  the spatial conv), reduces each to its whole-monitor magnitude regressor (magnitude-only — no dense
  storage), sorts columns by (odeg,wavelength,tfreq); `backend='cpu'` scipy fallback (`n_workers`
  threads). RESUMABLE SINGLE-FILE STORE (`scan_file`, joblib): rewritten ATOMICALLY after every batch
  → crash-safe; a re-run SKIPS filters already present (keyed by filter identity) and only convolves
  NEW ones — widen `specs` freely, old columns are reused byte-for-byte. The file is SELF-CONTAINED —
  `data` (dset identity + frame_t/fs + active_cells/cell_ids + dimred_params/seed, plus Y_neu_raw +
  ens_out/ens_table when `store_neural=True`) is embedded so another script reloads + scores with NO
  session reload. Readers (safe mid-scan): `f_scan_load` (→ X_stim/filt_df, specs subset + optional
  standardize), `f_scan_load_data` (→ the embedded bundle), `f_scan_progress` (n_done / per-kind /
  remaining-vs-target), `f_plot_scan_coverage` (orientation×wavelength done-grid). DATASET STATS:
  `scan_conv_activity(spatial_mean=True)` also records per-filter TIME-MEAN spatial maps (L‖R — where
  activity sits on the monitor); `f_scan_stats` → per-filter magnitude distribution (mean/std/max +
  percentiles of the whole-monitor regressor, the conv-repo-style magnitude analysis), `f_scan_spatial`
  → the stored spatial maps. Plots: `f_plot_scan_magnitude` (magnitude vs SF/orientation/TF, mean +
  p25–p75 band — how the bank responds across the grid), `f_plot_scan_traces` (magnitude vs frame by
  SF/TF, conv-repo-style), `f_plot_scan_spatial` (overall + per-group time-mean maps — where activity
  sits). Contextualize tuning vs where/how often a feature appeared on screen. Driven by
  VR_feature_tuning_progress.py's DATASET STATS cell. The scan file lives in `<features_dir>/scan/`
  (PARALLEL to conv_repo/), versioned by a `gen_tag` via `f_scan_path` (`<dset>_<gen_tag>_<lat>x<vert>
  [_ds..]_scan.joblib`) — scan + progress scripts build the same path from the shared tag; gen_tag
  (+ dset_name) is embedded and CHECKED on resume/load. `f_nyquist_limits(cam_params, lat, vert, fs,
  sample_ds, temporal_support)` prints/returns the resolvable sf_cyc_deg (+wavelength) / tf_hz for a
  render×downsample+fs combo. `f_scan_conv_activity` verbose prints per-batch + total convolve vs
  disk-save timing (filt/s) for profiling render/ds/batch choices.
  `f_save_scan_results`/`f_load_scan_results` persist the SCORER output for later re-plot without
  recompute — the fitted ridge/STA RESPONSE `W` (n_filter,n_lags,n_units) + per-unit tables/marginals
  + ALL params (scan reduce/transform/standardize, tuning method/lags/sources, ensemble-extraction
  params, seed); VR_feature_tuning.py writes `<..>_tuning.joblib` next to the scan file. `f_score_scan(X_stim,
  filt_df, neural, scorer='tuning'|'encode'|'decode', sources=
  ('cells','ensembles'), ens_out, ens_table, ...)` → {source: result}; scorer='tuning' (v1) reuses
  f_tuning (f_select_tuning_units → f_reverse_corr_tuning → f_tuning_readout) per source;
  'encode'/'decode' are stubbed extension points (raise NotImplementedError; will reuse
  f_score_representations / f_run_encoder / f_run_decode_sweep). `f_filter_activity_maps` re-convolves
  a unit's preferred filter over the movie (no repo) → per-side location (corr) + STA maps for the
  monitor-space verify. Driven by VR_feature_tuning.py.
- `VR_feature_tuning.py` — WIDE gabor/meng feature-tuning SCAN, per cell & per ensemble (created
  2026-07-16). Unlike VR_ca_cebra.py (loads a known-good feature set for encode/decode), this SURVEYS
  a wide filter grid: load → neural prep → ensembles (via f_ensemble_pipeline.f_run_ensembles) →
  render movie once → GPU scan (f_scan_conv_activity, `scan_filt_batch` knob) → tuning per cell AND
  per ensemble (f_score_scan scorer='tuning') → plots (population/marginals/lags/example heatmaps) →
  SIGNIFICANT-CELLS cell (f_tuning.f_tuning_significance = per-cell circshift shuffle null on peak
  |tuning|, BH-FDR → refit + plot the sig subset only) → monitor-space verify (re-convolve preferred
  filter) → cell↔ensemble summary. Tuning plots (f_encoding_plots): f_plot_tuning_heatmap gained
  `axis='sf'|'tf'|'auto'` (orientation × spatial- OR temporal-freq, other axis pinned to pref);
  f_plot_tuning_marginals_all = per-unit marginal overlay (one line/ensemble, `normalize` shape vs
  raw). Cell-based. The SCAN
  cell writes a resumable `scan_file` (embeds the dataset + neural + ensembles) — stop/resume freely,
  widen the grid without rescanning old filters.
- `VR_feature_tuning_progress.py` — watch/analyze a running (or finished) scan file from a SEPARATE
  process (created 2026-07-16). Points at the same `scan_file`; since it's rewritten atomically the
  reader sees a consistent snapshot mid-scan. Prints progress + `f_plot_scan_coverage`, then reloads
  the embedded neural/ensemble bundle (`f_scan_load` + `f_scan_load_data`) and scores the PARTIAL scan
  (tuning so far) with no session reload.
- `functions/f_encode_scan.py` — CROSS-DATASET per-neuron ENCODING sweep (created 2026-07-17). Loop
  mice × datasets and, for EACH neuron, score its OOF R² (+Pearson r) regressing onto each feature BLOCK
  (agg/self_mot/pix/gaborS+mengS) under each MODEL (ridge/cca) — NO ensembles. Resumable, tag-versioned
  joblib store mirroring f_feature_scan: `f_encode_scan_path(root,tag)` → `<root>/encode_scan/<tag>_
  encode_scan.joblib`; `_save_atomic` (temp+os.replace), skip-done keyed by `_enc_key(mouse,dset,block,
  model)`. State = `config` + `results[key]` (per-combo r2/r/keep over n_active, NaN where dropped) +
  `datasets[dkey]` (identity + `cell_ids` into est1 rows; NO locations) + `target` grid. Compute:
  `f_encode_prepare` (build Y_enc [drop-silent→smooth→peak-norm, est1 order] + features via
  f_resolve_features ONCE per dataset) → `f_encode_one` (one block×model via `f_encode_per_neuron`,
  scatters kept-cell R² back to n_active). `f_encode_scan_load`/`f_encode_scan_progress`/
  `f_encode_scan_table` (tidy mouse/dset/SOURCE/unit/block/model/r2/r/sig). SOURCE dimension (2026-07-17):
  units are raw `cells` AND/OR `ensembles` — `_enc_key` includes source; ensembles extracted per dataset
  via f_ensemble_pipeline.f_run_ensembles (FAST: `sig_method='cv'`, `n_shuffle=0` = no bootstrap; plots
  OFF), each ensemble score-trace encoded like a unit (which='all' keeps r2 for every ensemble);
  `f_ensemble_record` stores a compact per-dataset ensemble record (sig/cv_r2/members) in
  `datasets[dkey]['ens']`, and the table's `sig` column flags significant ensembles (NaN for cells).
  Locations/registration live in `functions/f_fov_register.py` (2026-07-17) and are RE-EXPORTED here:
  `f_footprint_centroids` (est1['A']/dims center-of-mass, order='F'), `f_fov_homography`/`f_apply_homography`
  (numpy DLT), `f_load_fov_landmarks` (VR_data.xlsx tp_fov_locs→wf_locs + wf_im), `f_neuron_widefield_coords`,
  `f_rot_xy`/`f_rot_for_plot` (rotate image+coords together), `f_plot_neuron_scores_fov`. Also `f_bg_image`
  (reconstruct the 2p structural background b@mean(f) = blood vessels, for an orientation sanity-check),
  `f_warp_to_widefield` (perspective-warp a 2p-space image into the widefield frame via inverse homography +
  scipy map_coordinates, `order`=1 bilinear|0 nearest, NaN outside FOV), and `f_footprint_label_map` (paint
  each cell's A footprint into a 2p image → per-pixel max-intensity cell LABEL + intensity, transpose-aware →
  warp to project ROIs, not just centroids). f_encode_scan keeps the score-table plots `f_plot_score_hist`/
  `f_plot_block_model_compare`. Driven by VR_encode_scan.py (write) + VR_encode_scan_plots.py (read) +
  VR_register_2pfov_cells.py (project cell ROIs onto the widefield map: loads all 2p datasets → footprint
  centroids/footprints → homography → widefield atlas, colored by dataset or by the encode-scan R²; sister to
  VR_register_2pfov.py). Knobs: `rot_k` rotates the output; `cell_mode='dots'|'roi'` (centroid markers vs
  warped A footprints, `roi_thresh`/`roi_alpha`); `show_cells`; `show_bg`+`bg_darkness`/`bg_alpha`/`bg_cmap`
  (warped vessel overlay); `show_date` (label each FOV with its dset_name under the top edge); `transpose_2p`
  (reconcile order='F' A/b reshape with the tp_im TIFF landmark orientation — applied to centroids + bg +
  footprints together). SECTION 4 colors neurons by score from EITHER store (`score_src='encode'` →
  f_encode_scan R² for enc_block×enc_model | `'tuning'` → f_tuning_scan `tun_metric` for `tun_feat_set`),
  joined by (dset_name, neuron); `bright_by_score` maps score→per-neuron alpha (brightness) so strong cells
  pop on the widefield (works in dots + roi modes; `score_cmap`/`score_clip`/`alpha_floor`).
- `VR_encode_scan.py` — SCRIPT 1 (compute+save, 2026-07-17): loop `mice × f_resolve_dsets(DSETS)`, skip a
  dataset if all block×model combos done else `f_load_session` → `f_encode_prepare` → per-combo
  `f_encode_one` → `f_store_put` + `_save_atomic` after EVERY combo (crash-safe/resumable). Knobs:
  `ENC_TAG`, `blocks`, `models`, `model_kw`, `lags_ms`, `sigma_frames`, `embargo_sec`, `conv_backend`
  ('gpu' builds gabor/meng on GPU — much faster than CPU). Bad datasets logged+skipped. Reuses the
  pca_feat caches like cebra (no reconvolution): `feat_pca_variant` ('pcacomb'/'pcasep'/…) points src_m's
  cache_tag at `<tag>_<variant>` so gaborS/mengS LOAD; `feat_whiten` load-time whiten; `feat_combine`
  composes gaborS+mengS from the loaded blocks via `f_encode_prepare(combine=…)` → `f_combine_loaded_pca`
  (composite names are dropped from the resolve `need` and their source blocks loaded instead).
- `VR_encode_scan_plots.py` — SCRIPT 2 (read+spatial plots, 2026-07-17): load the store + `f_encode_scan_table`,
  reload each est1 (behavior=False), VERIFY n_cells_total/dset_name vs the stored record, compute footprint
  centroids, register to the mouse widefield map via VR_data.xlsx landmarks, then plot per-neuron R² vs 2D
  location per (block,model) overlaid across a mouse's datasets + R² distributions + ridge-vs-cca/block-vs-block.
- `functions/f_tuning_scan.py` — cross-dataset feature-TUNING sweep (reverse-correlation / STA; the tuning
  analogue of f_encode_scan; created 2026-07-17). Per cell AND ensemble, reverse-correlate activity against
  the CACHED gabor/meng features (per orientation × temporal-freq) + other blocks, NO reconvolution.
  Resumable tag-versioned store mirroring f_encode_scan, keyed on (mouse,dset,source,feat_set):
  `f_tuning_scan_path`/`f_new_store`/`f_tuning_scan_load`/`_tun_key`/`f_tuning_store_put`/`f_tuning_store_set_dataset`/
  `f_tuning_scan_progress`/`f_tuning_scan_table`. `DEFAULT_FEAT_SETS` maps a feat_set → cache blocks (meng =
  [mengS,mengS2,mengS3] → 4 orient × 3 tf matrix; gabor=[gaborS]; pix/agg/self_mot). Core:
  `f_block_regressors(built_blocks, feat_set, block_names)` → (X_stim (T,n_filter), filt_df) — per per-o×tf
  PCA group a whole-monitor magnitude regressor via the cheap PCA-inverse column-mean
  `X[:,xslice]@pca[g].components_.mean(1)+pca[g].mean_.mean()` (verified == `_recon_raw` col-mean; handles
  pca None / whitened); non-visual blocks use their columns directly. `f_tuning_one` → `f_reverse_corr_tuning`
  (method='sta') + `f_tuning_readout` (visual) → COMPACT result (peak-lag `resp` (n_filter,n_unit) + readout
  table + marginals; no full W×lags). `f_block_monitor_maps` = per-filter time-mean L‖R monitor maps (context,
  stored/dataset). `f_result_to_readout`/`f_population_matrix` reconstruct viewer inputs. Reuses
  f_encode_scan.f_encode_prepare (Y+features+ensembles) + f_ensemble_record + the fov re-exports.
- `VR_tuning_scan.py` — SCRIPT 1 (compute+save, 2026-07-17): loop mice × datasets, skip fully-done, load
  features from the pca_feat cache (`feat_pca_variant='pcasep'` — per-orientation×tf groups; NO render/conv),
  extract ensembles once (FAST cv-sig, n_shuffle=0, plots off), per (source,feat_set) `f_tuning_one` →
  `f_tuning_store_put` + `_save_atomic` after every combo. Knobs: `TUN_TAG`, `sources`, `feat_sets`,
  `tune_method` ('sta'), `lags_ms`, `sigma_frames`, `feat_pca_variant`/`feat_whiten`, ensemble params.
- `VR_tuning_scan_plots.py` — SCRIPT 2 (read+plot, 2026-07-17): load store + `f_tuning_scan_table`; population
  orientation×tf matrix per visual feat_set (`f_population_matrix`), per-unit/ensemble marginals + example
  response heatmaps (`f_result_to_readout` → `f_plot_tuning_marginals_all`/`f_plot_tuning_heatmap`), monitor-
  average frames (`f_plot_tuning_monitor` on the stored per-filter maps), cell↔ensemble comparison, opt-in
  significant-cells (re-run `f_tuning_significance`) + widefield projection colored by pref_odeg/peak.
- `functions/f_feature_gen.py` — batch helpers for VR_feature_gen.py (2026-06-29):
  `f_build_one_dataset` (vectorize + flatten nested cfg → `f_build_features`) and
  `f_resolve_dsets` (cfg['DSETS'] → list of do_proc indices). `f_build_conv_repo_one_dataset`
  (2026-07-11) = the GEN_MODE='conv_repo' path (vectorize → `f_render_session` → `f_build_conv_repo`;
  no block build). `f_build_pca_feat_one_dataset` (2026-07-17) = the GEN_MODE='pca_feat' path:
  convolve → PCA → save ONLY the small reduced blocks (the dense conv_repo is too big on disk). Renders
  ONCE, builds RAW point-sampled gabor/meng blocks from `cfg['pca_specs']` (via f_resolve_features build
  path — physical sf_cyc_deg/tf_hz → px + GPU conv), then PCA-compresses at TWO independent granularities
  to SEPARATE tagged caches: `SAVE_COMBINED` → `_pcacomb` (one PCA per INPUT — all orientations, L‖R
  combined, whole-block f_apply_block_pca); `SAVE_SEPARATED` → `_pcasep` (one PCA per FILTER
  o×SF×TF, L‖R, via `f_visual_features.f_pca_group_spec`). `n_pca_combined`/`n_pca_separated` = int |
  float(0,1) | `(frac,cap)`. Saved blocks are UNWHITENED (whiten is load-time). Compute-only; batch loop
  inline in the script.
- `VR_feature_gen.py` — batch build+save script (TODO #66): loops MICE × do_proc
  datasets, builds 1-by-1 via `f_feature_gen.f_build_one_dataset` (→ f_build_features),
  saves caches, frees the render between datasets. Config = an inline `cfg` dict
  (`GEN_TAG` versions caches; `SAVE_MOVIE`; `DSETS` selector — None=all;
  `MAX_DATASETS` caps total builds this run, cache-skips don't count). `GEN_MODE='blocks'`
  (default) | `'conv_repo'` (dense per-filter maps via `cfg['conv_specs']`, incremental per-filter) |
  `'pca_feat'` (convolve→PCA→save small reduced blocks via `cfg['pca_specs']` + `SAVE_COMBINED`/
  `SAVE_SEPARATED` + `n_pca_combined`/`n_pca_separated`; two granularities → `_pcacomb`/`_pcasep` caches).
  VR_ca_cebra.py loads via the `load_cached_features` switch in cell 1 (build-or-load junction: build
  path renders+extracts and optionally
  saves; load path resolves a cache by file/tag/index — set knobs in cell 1). The sampled-methods
  cell there has `methods_source='extract'|'conv_repo'` (conv_repo = load maps + subsample, no re-conv).
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
- `functions/f_ensemble_pipeline.py` — reusable ensemble-workflow orchestration glue
  (created 2026-07-16), the orchestration layer over f_ensembles / f_ensemble_plots so
  any script gets the whole workflow from one import (`import f_ensemble_pipeline as
  enspipe`). Four drivers: `f_ensemble_apply` (fit dim-red + threshold → ens_out; the
  expensive FIT, keeps a pristine `ens_list_apply` copy), `f_ensemble_stats` (membership
  choice + per-ensemble CV score/significance → tidy ens_table with the extraction-idx→
  true-neuron-ID map via `cell_ids`; MUTATES ens_out in place with the chosen membership;
  cheap to re-run), `f_run_ensembles` (apply→stats→optional plots in one call; `plots`
  knob + `fov` dict for the FOV map), `f_neural_source` (ens_out/ens_table or a dim-red
  mode → the reassigned decoder/encoder input Y_neu: raw / svd_recon / components /
  reliable_cells / ens_components / ens_cells). Code moved verbatim from the VR_ca_cebra.py
  APPLY/STATISTICS/ASSIGN cells (which now just set param knobs + call these). Does NOT own
  dimred_params / cell_ids / the neural target's smoothing — those stay script-side.

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
