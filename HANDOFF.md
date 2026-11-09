# HANDOFF — VR_analysis (session ending 2026-07-27)

Snapshot of what changed this session and where to pick up. See `CLAUDE.md` for the
always-current module map and `notes.txt` (2026-07-25 / 07-26 / 07-27 entries) for dated
rationale. Prior handoffs (2026-07-24 and earlier) are kept below — still valid.

## Big picture (this session)
Fixed the feature-build speed properly, added two new visual representations, and wired the
whole VR_feature_search feature-config into VR_ca_cebra so a representation that wins the
search can be moved straight into the encoding arm (#57). Everything compiles and is unit-
tested; **none of it has run on a real session** — that is the next action.

## What changed (2026-07-25 → 07-27)
1. **FFT convolution on the GPU** (`f_conv_gpu._spatial_conv`, all 6 spatial sites). The GPU was
   *not* underused — `nvidia-smi` showed 100% (Task Manager's default graph is the 3D engine;
   CUDA is on Compute_0). The problem was algorithmic: the CPU path used `scipy.fftconvolve`
   (K-independent) while the GPU port used direct `F.conv2d` (∝K²), and at `sigma_scale=0.5` a
   `cyc_fov=1` gabor is **453×453 on a 101×150 frame** — bigger than the image, ~75% of the bank's
   cost. Now: `_crop_kernel` (exact, drops rows/cols that only hit zero pad, 453²→201×299) +
   `_fft_conv2d` (one forward rfft2 shared across a conv-unit's filters). **8.1× measured**
   (~23 min → ~3 min per bank over 100k frames × 2 monitors); magnitude parity vs the CPU
   reference *improved* ~100×.
2. **Signed-quadrature gabors** (#74) — `stats=('quadrature',)` → `gabQe`/`gabQo`. Strict superset
   of magnitude and of normalized phase (|R| recovered exactly), 2 channels not 3, no division.
   Under ridge, mag+phase needs the product `mag·cosφ` — which *is* `re` — so quadrature is the
   linear-model-friendly way to keep phase.
3. **FFT-spectrum features** (#73) — `f_fft_features`, one patch size per pyramid level, binned back
   onto the gabor bank's own axes so columns align index-for-index. Confirmed a gabor response IS
   one Fourier coefficient (corr **0.9998** on-lattice); the dominant error is lattice quantization
   (corr(frac, agreement) = −0.88), fixed by `patch_snap=True` (0.961 → 0.985).
4. **cebra integration** — `f_vis_sets_to_plan` (plan-side twin of `f_pyramid_grid`, so configs
   paste over with *identical* cell names) + `f_expand_blocks` (a bank name expands to all its
   `<bank>:<cell>` blocks; bit-identical to listing them). Cell 401's visual half is now a
   `vis_sets` dict; `feature_list` still selects agg/pix/motion/pix_mot/self_mot/beh. Cell 926's
   encoder takes a bank name.
5. **conv_autosave moved** to `build_visual_blocks` — the one point every build path funnels
   through, so the search and cebra now **share one store** (a bank convolved by one reloads in the
   other). Re-keyed on resolved extractor params, cached pre-PCA/pre-proc.
6. **`marginalize='all'` drift fix (silent bug)** — the search expanded the `'all'` shortcut to
   every axis; the plan-side `_build_grid_block` did not, so the *same config* gave ONE block in the
   search and 15 per-filter blocks in cebra with no error. Root cause was two copies of one rule, so
   `f_feature_source.f_marg_axes` is now canonical and `f_feature_search._marg_axes` delegates.
   Worth knowing because the failure mode is silent: an encoder quietly fed a different
   representation than the one requested.
7. **Comparable configs staged.** `vis_sets` in cebra now has ONE entry per feature type at
   identical `magcomb2pca` params (`gabmag` / `gabquad` / `fftmag` / `fftquad` / `mengopp`), each
   building to a single `<name>:all` block at 50 dims. `VR_feature_search` gained the quadrature
   counterparts (`quadcomb2pca` / `quadcomb1norm2pca` in the active `pyr_sets2`; `quad` / `quad2pca`
   / `quad1norm2pca` in `pyr_sets`). CAVEAT: the pca-50 makes them fair, but raw widths differ
   before it — magnitude 1 channel/filter, quadrature 2, meng ×tf.

## ⚠️ Before the next run
- **Restart the kernel** — `f_conv_gpu` and friends must be re-imported for the FFT path to exist,
  and the `_thread_cap` env vars are still import-time only.
- **The old autosave store is orphaned.** New layout is `<mouse>_<dset>_<tag>`; the old
  `<mouse>_<dset>_<tag>_<geomhash>` dirs under `F:/VR/features/L/conv_autosave/` are dead disk —
  delete them. First run repopulates.
- In `VR_feature_search.py`, line ~264 (`configs = pyr_configs2`) overrides the assembled list on
  line ~262, so `fft_pyr_configs` won't run until that's changed.

## Next steps (priority order)
1. **Run one real scorecard** — the configs are already staged (quadrature entries in `pyr_sets` /
   `pyr_sets2`, `fft_pyr_configs` built), so this is just picking the list on the `configs =` line.
   Validates the FFT conv path, quadrature and fft features in one pass and answers #69/#73/#74.
   Watch the `timing=True` per-conv-unit lines: the low-SF units should collapse from minutes to
   seconds.
2. **Run cell 401 → 475 → 926 in cebra** on the same config — first real integration check of the
   feature plumbing and the encoding arm on a search-chosen representation.
3. Then the deferred items, in rough order of value: meng's **temporal conv1d** is still direct and
   is likely meng's new bottleneck (4 convs per (o,sf,tf), support up to 91 frames); a **runtime
   thread cap** (`torch.set_num_threads` + `threadpool_limits`) that works on a warm kernel;
   `FFT_MIN_K` 48 → ~24 (FFT still won at K=31); meng `output='both'` vs `'opponent'` (#75, not yet
   written); quadrature branches in `f_feature_optim.f_build_gabor_banks` and `f_feature_viz`;
   a GPU backend for `f_fft_features` if profiling asks for it.
4. Still open from earlier: the **P2 encoding spine D1→D2→D3 has never run on real data** (#57), and
   the `obj_size['x']` width fix means all render/agg-derived caches are stale.

---

# HANDOFF — VR_analysis (session ending 2026-07-24)

Snapshot of what changed this session and where to pick up. See `CLAUDE.md` for the
always-current module map and `notes.txt` (2026-07-24 entry) for dated rationale. Prior
handoffs (2026-07-10 and earlier) are kept below — still valid.

## Big picture (this session)
Made `VR_feature_search.py` (the neural-scored feature-param search, TODO #69–#73) fast +
robust after it was crashing / running ~900 s per gabor bank. Four opt-in, parity-preserving
changes + one diagnosis. All compile; unit-tested (parity, autosave round-trip, SF round-trip)
in the `rnn` conda env (numpy 1.26 + torch + CUDA). NOT yet run end-to-end on a real session —
that's the next action.

## What changed (2026-07-24)
1. **GPU-native stat+pool** (`f_conv_gpu.f_gabor_pooled_gpu`/`f_meng_pooled_gpu`, + 2 call sites
   in `f_visual_features`). The old "GPU" path only did conv2d/conv1d on the GPU, then moved the
   FULL-RES (t,H,W) response to CPU per filter for numpy magnitude/opponent + pool — the ~900 s
   bottleneck (GPU idle, all cores pegged). Now magnitude/phase/even/luminance + meng
   opponent/both AND the spatial pool run on the GPU (`_gab_stat_torch`, `_make_reducer_torch`);
   only the small pooled (t,ny,nx) crosses to CPU. Grid pool = exact reduceat-segment matmul
   (non-divisible edge cells handled); sample = strided index. Parity: torch-CPU EXACT ~1e-7;
   CUDA gabor ~2e-4 from PRE-EXISTING conv2d TF32 (negligible for standardized features).
2. **Conv autosave** (`f_conv_autosave.py` new, hook in `f_feature_search._raw_block`). Reload
   the RAW pooled gabor/meng block from SSD if a matching recipe×geometry block exists, else
   build + save. Saved PRE proc/marginalize → configs sharing a bank but differing only in proc
   (pyr_sets mag/mag2pca/mag1norm) reload instead of reconvolving. Opt-in
   `build_cfg['conv_autosave']` (+ `conv_autosave_tag`); per-column uint16; gabor/meng only.
3. **SF naming** (`f_feature_helpers.f_num_tag`, applied in `f_iter_grid_cells`). cyc_fov floats
   now print/label as `sf0.01016` not `sf0.010162601626016261`; round-trip-safe (f_score_grid).
4. **Extended timing** (`timing=False` in `f_feature_search` + `f_score_representations`).
   Flushed timestamped START/duration per major step (render/conv-unit/finalize-PCA/build+pca/
   decode/encode) to watch the slow step live against CPU/GPU.

## ⚠️ Restart the kernel before the next run
The BLAS thread cap (`_thread_cap` atop the script, now 14) is IMPORT-TIME ONLY — a warm Spyder
kernel (numpy/torch/MKL already imported) ignores it, which is the likely crash cause (all
threads pegged). A cold kernel makes it apply. torch also has its own pools not covered by the 4
env vars; a runtime cap (`torch.set_num_threads` + `threadpoolctl.threadpool_limits`) would apply
even warm — OFFERED, not yet added. Cell 1 currently has `conv_autosave=True`/`tag='auto1'`,
Cell 3 `timing=True`.

## Next steps (this session's priority)
1. **Restart kernel, run `VR_feature_search.py` end-to-end** on L (base_tag `gen_7_24_26`): first
   real-data run of the GPU-native path + autosave. First config convolves+saves; the rest
   (mag2pca/mag1norm/…) should show `autosave HIT` and skip conv. Confirm GPU utilization up,
   wall-clock down, scores sane.
2. If still oversubscribing CPU, add the runtime thread cap block (torch.set_num_threads +
   threadpool_limits) near the top of the script.
3. Continue the #69–#73 gabor/visual-feature comparison (leanest set maximizing encode+decode):
   n_orient plateau (#69), +phase (#70), phase-only (#71), meng/flow motion (#72).

---

# HANDOFF — VR_analysis (session ending 2026-07-10)

Snapshot of what changed this session (2026-07-03 → 07-10) and where to pick up.
See `CLAUDE.md` for the always-current module map and `notes.txt` for dated rationale.
The 2026-07-01 section (feature/decode/viz pipelines) is kept below — still valid, still unrun.

## Big picture (this session)
Built the **P2 encoding spine** (the PRIMARY deliverable, TODO #57) against the researcher
sub-plan `../reserrch_repo/SUBPLAN_P2_encoding_model.md`. Everything below is SYNTHETIC-GATED
and compiles but is **NOT yet run on a real session** — running D1→D2→D3 on real data is the
next action.

## What was built / changed (2026-07-03 → 07-10)
- **D2 nested/held-out guard** (P2 §7): `f_encode_tune_sigma_lambda` selects (σ,λ) on a leading
  time split and reports the winner on a held-out tail (`report_frac`, `sel_r`/`report_r`).
- **self_mot fwd/lat/yaw decomposition** (P2a-9): `f_self_motion_decomp` (heading projection)
  wired via a default-off `self_motion_decomp` knob in `build_feature_blocks`/`f_pipeline`;
  old 2-channel speed+yaw block kept.
- **D3 banded ridge + variance partition** (P2a-5/P2b-1): `BandedRidge` (per-band λ) in
  `f_decoding` + `predict_band`; `decoder='banded'` threaded through `make_decoder`/`f_blocked_cv_r2`;
  `f_encode_banded`/`f_encode_tune_bands`/`f_encode_variance_partition` in `f_encoding`; new D3
  driver cell (groups blocks into 'visual' vs 'behavior' bands). Gate step 4 passes.
- **Gabor sanity cell**: added a to-scale kernel overlay on the monitor frame + grid.
- Prior (this session, earlier): D1 pixel-RF weights + STA + RF viewer, D5 relative ceiling.

## Next steps (this session's priority)
1. **Run D1→D2→D3 on a real session** (build mode, `self_motion_decomp=True`) — view real RFs
   + the visual/behavior variance partition. First real-data check for the encoding arm.
2. **D4** — task-event band (lick/reward/capture lag-embedded FIR) + capture/offset visual
   regressor (the other required behavior band).
3. **Neural-objective feature sweep** — swap the `f_sweep_recon` scorer from feature→movie
   (Chain-1) to a neural scorer (encoding / decoding / composite neuron→movie); `f_feature_optim`
   has a parked neural path. Also: optimize **meng** (low Chain-1 R²) as a Chain-2 target;
   **optic flow #49 untouched**.

---

# HANDOFF — VR_analysis (session ending 2026-07-01)

Snapshot of what changed the prior session (2026-06-25 → 07-01) and where to pick up.
See `CLAUDE.md` for the always-current module map and `notes.txt` for dated rationale.

## Big picture
The feature pipeline is now fully modular: **generate once → cache → load** across all
analysis scripts, plus a matching **decode-results** pipeline and a refocused
**feature-visualization** script. The PRIMARY deliverable (encoding arm, TODO #57) is
still the north star; its cross-dataset pipeline is the main next step.

## What was built / changed

### 1. `functions/f_pipeline.py` — cross-script orchestration (NEW, phases 1–3)
- `f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx, cam_params=None,
  behavior=True, …)` → `session` dict (calcium + behavior + cheap vectorization + clock).
  `behavior=False` = calcium-only. Owns rig defaults `DEFAULT_CAM_PARAMS`/`DEFAULT_CAIM_KW`.
- `f_vectorize_session(est1, bh, cam_params, …)` — the cheap vectorization for ONE dataset,
  no loading (so batch can load-once/vectorize-per-dataset).
- `f_acquire_features(source, dset_name, build_fn=…, …)` — build (calls `build_fn`, +optional
  autosave) OR load a cache (resolve by file>tag>index). Also stashes example_frames.
- `f_build_features(session, cfg, …)` — SHARED render+extract (calls `f_render_session` →
  side-select → `build_feature_blocks` + `build_visual_blocks`). `cfg` = flat dict = gen_config.
- `f_render_session(session, cfg)` — render half only (objects [+terrain] → two_mon_frames),
  factored so VR_feature_viz can regenerate the raw movie on demand.
- Wired into: `VR_ca_cebra.py`, `VR_ca_encoding.py`, `VR_ca_dimred.py` (f_load_session),
  and `VR_feature_gen.py` (f_vectorize_session + f_build_features).

### 2. Feature cache + batch generator
- `functions/f_feature_cache.py`: `f_save_features`/`f_load_features`/`f_features_path`/
  `f_list_feature_files`/`f_resolve_feature_file` (**tag is now a FILTER + index**, since
  GEN_TAG is shared by all datasets) + `f_pick_frames`. Now stores `example_frames`
  (+`example_frame_idx`): a few ORIGINAL rendered frames saved EVEN WHEN SAVE_MOVIE=off.
- `functions/f_feature_gen.py` (NEW): `f_build_one_dataset` (→ f_build_features),
  `f_resolve_dsets`.
- `VR_feature_gen.py`: batch build+save, `DSETS`/`MAX_DATASETS`/`OVERWRITE`/`N_EXAMPLE_FRAMES`,
  1-by-1 with gc + tqdm progress bar. Config = inline nested `cfg` dict.

### 3. Decode-results pipeline (NEW)
- `functions/f_decoding.py`: `f_run_decode_sweep(built_blocks, Y_neu, frame_t, cfg)` +
  `f_run_decode_null(sweep, …)` — the SHARED per-dataset decode compute (per-feature
  blocked-CV R²/AUC × n_neural_PCs, real + shuffle null), factored from the cebra cells.
- `functions/f_decode_cache.py` (NEW): mirror of f_feature_cache for decode results
  (r2/auc matrices + null + labels + cfg + meta; no y/neural).
- `VR_ca_cebra.py`: sweep + null cells call the shared fns; new **save-decode cell**
  (`save_decode`, `DECODE_TAG`, `F:/VR/decode/<mouse>/`).
- `VR_decode_analysis.py` (NEW): loads decode caches across datasets → tidy DataFrame →
  per-block plateau-R² over datasets, feature×dataset heatmap, real−null Δ, region view.

### 4. `VR_feature_viz.py` — refocused on DECODING-feature reconstructions (heavily reworked)
Purpose: visualize the features we decode from (pix/grid/flow/gabor/meng) vs the original.
- Loads a cache; **Original-movie** cell (cached or `f_render_session`); **OVERVIEW** cell
  (all blocks vs original at the saved example_frames, labels show PC counts); **per-block
  deep-dive**; **merged gabor cell**.
- **Merged gabor cell** (was two cells): rows = pix recon (labeled with PC count) / gabor
  mag / gabor phase; optional **per-patch quiver overlay on any/all rows** (`overlay_quiver`,
  `overlay_on`); frames auto-picked at **motion** (`frame_mode` = motion_spread / motion_event
  / even / saved).
- Frame-economical: reconstructs ONLY displayed/saved frames via `frame_idx` (never the 96k
  movie); quiver flow only at shown frames via `f_gabor_flow_at` (local windows).
- New helpers in `f_feature_viz.py`: `f_plot_movie_frames` / `f_plot_movie_frames_compare`
  (compare gained a GENERIC quiver overlay: pass any motion-field dict `{vx,vy,speed,mag,grid}`),
  `f_gabor_flow_at`, `f_pick_motion_frames` (rank by ‖Δ pix‖). Quiver render fixed:
  magnitude gate + global fixed arrow scale (speed ref over GATED cells) + `arrow_frac`/
  `arrow_width`. Removed `f_plot_quiver_frames` and the dense full-res optic-flow cell.

## Known state / caveats
- The "original" in viz is the **pix reconstruction** (rank-`n_pix_pca`, e.g. 50 PCs) unless
  the true movie is cached (SAVE_MOVIE) or rendered (do_render); the only TRUE originals in a
  cache are the ~8 `example_frames` (evenly spaced). At motion-selected frames → pix recon.
- Only NEW caches (built after this session) carry `example_frames`. Old ones don't.
- All touched files compile (`py_compile`); helpers unit-tested on synthetic data. NOT run
  end-to-end on real data this session — first real run of cebra build/sweep or feature_gen
  is the integration check (likeliest snag = a `cfg` key).

## Next steps (priority order)
1. **Encoding cross-dataset pipeline (PRIMARY, #57)** — mirror the decode pipeline for the
   encoding arm: a shared `f_run_encode`, `f_encode_cache.py`, save cell in
   `VR_ca_encoding.py`, `VR_encode_analysis.py`; the **tuning table** (cell → best block /
   retinotopy) feeds the opto generator (#64). `f_encoding.py` compute already exists.
2. **Run the decode loop end-to-end**: cebra per dataset with `save_decode=True` →
   `VR_decode_analysis.py`.
3. Optional: default `mov_tag` naming, region metadata into decode/feature meta, `#62`
   (pooled-OOF R²), motion signal options in `f_pick_motion_frames` (motion/meng blocks).
