# Plan — visual-cortex-style monitor input representations

Deep plan for replacing/augmenting the current monitor (visual-stimulus)
features with representations closer to how visual cortex encodes the scene:
**location-selective, edge/orientation-selective, and direction-selective.**
Tracked as TODO **#47** (sub-tasks #48–#53). Rationale + literature framing in
`notes.txt` 2026-06-06.

Status legend: `[ ]` pending · `[~]` in progress · `[x]` done

---

## 0. Motivation (the gap)

The two current monitor representations sit at opposite wrong extremes:

- **`agg`** — one object-CENTER point per object (lat/vert/dist). Discards
  spatial extent, edges, retinotopic location, and local motion. "Location" is
  a single coordinate, not an activation map.
- **`pix`** — PCA on raw rendered intensity. Top PCs are global brightness /
  "anything on screen" / L-vs-R balance — the slow, drift-prone modes behind
  #8. Linear PCA on raw pixels is not aligned to any cortical computation; it
  never extracts oriented edges, local motion, or spatial frequency.

Neither speaks the V1 vocabulary (oriented edges × retinotopic location ×
direction × spatial frequency), and the decoder is linear (ridge) while cortex
applies nonlinearities (rectify, square) BEFORE its linear readout.

## 1. Organizing principle

Mirror the canonical encoding-model pipeline
(Hubel–Wiesel → Adelson–Bergen motion energy → Gallant-lab encoding models):

> **filter → nonlinearity → spatial/temporal pooling → compress → linear decode**

Put the nonlinear visual front-end into the FEATURES, then the existing linear
blocked-CV decoder can read it. Substrate is ideal: `two_mon_frames` is already
~**101×101 per monitor in angular coordinates** (pixel ≈ visual degree;
vFOV ≈ 65.9°), crisp synthetic silhouettes, known geometry.

## 2. Shared infrastructure (build once, reuse for every tier)

All tiers are new **feature blocks** in the existing architecture — no new
decoding machinery:

- **Builders** live in `functions/f_feature_helpers.py` (compute) following the
  `f_motion_features` / `f_aggregate_monitor_features` pattern. Heavy/optional
  visual ops (cv2 flow, gabor banks) may get their own module
  `functions/f_visual_features.py` if `f_feature_helpers.py` gets crowded.
- **Registration:** each builder wired into `build_feature_blocks(...)` behind a
  `build_<name>` knob; result lands in `built_blocks['<name>']` with `{'X',
  'names', ...}`. Selectable via `sweep_target_blocks` / `target_blocks_*`,
  decoded through `f_blocked_cv_r2` (incl. AUC for any binary channels).
- **Clock:** compute on the behavior-clock movie (temporally ordered, needed for
  flow/motion-energy), THEN resample to imaging frame times with the existing
  `f_resample_to_imaging` — same as `pix` does today.
- **Compression:** every filter bank is high-dimensional. Compress per block
  with PCA (like `pix`, `n_<name>_pca` knob) OR keep the spatial grid coarse
  (e.g. 6×6) so raw dims stay manageable. Always report cumulative explained
  variance.
- **Shared spatial grid:** a single retinotopic tiling helper (Tier 1) that
  every later tier pools INTO — so "location selectivity" composes with edges
  (Tier 3) and motion (Tiers 2/4).

### Cross-cutting concerns (apply to all tiers)
- **Dimensionality vs CV.** Bigger feature sets revive the negative-R² /
  overfitting dynamics → always PCA-compress + run the blocked CV + shuffle
  null + OOF/train-vs-test diagnostics already built. (`f_blocked_cv_r2`,
  sweep, null, per-fold, OOF cells all transfer unchanged.)
- **Drift (#8).** Motion/edge-motion features are differentiated/rectified →
  inherently high-pass → expected to be LESS drift-prone than raw `pix`.
- **Per-monitor vs concatenated.** Default: compute per monitor (L/R), then
  concatenate channels (mirrors `agg`/`motion` per-side layout). Keep a side
  tag in channel names.
- **Mouse specifics.** Low acuity (~0.5 cyc/deg) and area-dependent SF/TF tuning
  (V1 vs AL/PM/RL) → use a SMALL multi-scale bank (2–3 spatial frequencies),
  don't over-resolve.
- **NaN policy.** Empty frames (no object, no terrain) → features are genuinely
  ~0 (not NaN) for edge/flow banks (there's no edge). Decide per builder; most
  are defined everywhere (unlike `agg`).

---

## 3. The tiers (implementation order)

### Tier 1 — Retinotopic grid pooling  (#48)  [cheap, do first]
Convert "object center" into a population "what's in each part of the visual
field." Tile each monitor into an `(ny, nx)` grid; per cell compute simple
statistics.
- **Builder:** `f_grid_features(frames, grid=(ny,nx), stats=(...), side_tag)`.
- **Channels per cell (start simple):** occupancy (mean filled fraction), mean
  contrast/intensity, optional local edge energy (Sobel magnitude). 
- **Block name:** `'grid'`. Dims ≈ `ny*nx*len(stats)*n_sides` → PCA-compress if
  large.
- **Why first:** the single biggest conceptual fix (location selectivity),
  trivial to implement, composes with every later tier.
- **Read:** if `'grid'` decodes >> `'agg'`, spatial layout matters beyond the
  center coordinate.

### Tier 2 — Optical-flow on a retinotopic grid  (#49; realizes #24 / Level C)
Direction-of-motion of edges/objects at each location — the stated target.
- **Builder:** `f_flow_features(frames, grid, side_tag)`. Dense flow
  (`cv2.calcOpticalFlowFarneback`) between consecutive behavior frames → pool
  the (vx, vy) field into the Tier-1 grid.
- **Channels per cell:** speed = √(vx²+vy²), direction sin/cos (no ±π wrap),
  and **divergence** (∂vx/∂x+∂vy/∂y) as a LOCAL looming/expansion channel.
- **Block name:** `'flow'`. Supersedes the global `motion` block's spatial
  blindness; complements it (keep both).
- **Deps:** `opencv-python` (cv2). Heavier than Tier 1, very interpretable.
- **Read:** direction-selective cells → `'flow'` should beat both `'agg'` and
  `'motion'`.

### Tier 3 — Gabor energy bank + grid  (#50)  [true V1 simple/complex cells]
Oriented edges at locations and spatial frequencies.
- **Builder:** `f_gabor_features(frames, n_orient, sf_list, grid, energy=True,
  side_tag)`. Convolve with a Gabor bank (orientations × SFs); **complex-cell
  energy** = √(even² + odd²) of each quadrature pair (phase-invariant); pool
  into the grid.
- **Channels per cell:** energy per (orientation × SF). Keep small:
  e.g. 4 orientations × 2–3 SFs.
- **Block name:** `'gabor'`. Dims large → PCA-compress (`n_gabor_pca`).
- **Deps:** `skimage.filters.gabor_kernel` (or scipy conv). 
- **Variants:** simple-cell (linear, phase-sensitive) vs complex-cell (energy);
  a cheap proxy = oriented-derivative / HOG per grid cell.
- **Read:** orientation/edge tuning → `'gabor'` lifts R² over `'grid'`.

### Tier 4a — Motion-energy model (Adelson–Bergen)  (#51)  [most principled motion]
Spatiotemporal oriented filters (quadrature pairs in x–y–t) → joint direction-
AND speed-selective energy. Gold-standard model of direction-selective cortex
("moving edges" done right).
- **Builder:** `f_motion_energy_features(frames, n_orient, n_speed, grid)`.
- **Block name:** `'meng'`. Heaviest hand-built bank; PCA-compress.
- **Do if** Tier-2 flow already shows motion helps and we want the principled
  ceiling for direction tuning.

### Tier 4b — Pretrained-CNN front-end  (#52)  [data-driven ceiling/benchmark]
Early conv-layer feature maps of a pretrained vision (or motion) network as
features. Goal-driven CNNs are the current best predictors of visual cortex
(Yamins, Bashivan). Most expressive, least interpretable, heaviest.
- **Builder:** `f_cnn_features(frames, model, layer, pool)`.
- **Block name:** `'cnn'`. Use as an UPPER-BOUND benchmark: how much do the
  hand-built banks leave on the table?
- **Deps:** torch + a pretrained model.

### Tier 5 — Supporting channels  (#53)  [fold into tiers as needed]
- Local **looming** = flow divergence (already in Tier 2).
- **Spatial-frequency / contrast**: DoG (center-surround, retina/LGN front-end)
  multi-scale channels; the Gabor SF axis already covers SF.
- **Oriented-edge / HOG** cheap proxy for Tier 3.
- 1-D **angular histogram** across the monitor (#20) = a degenerate (1×nx) grid.

---

## 4. Sequencing & success criteria

1. Build Tier 1 (`grid`) → A/B vs `agg` in the sweep.
2. Build Tier 2 (`flow`) → A/B vs `agg`/`motion`. **Most likely winner** if the
   population is location+direction selective (the working hypothesis).
3. Build Tier 3 (`gabor`) → does oriented-edge structure add over occupancy?
4. Tier 4 (`meng` / `cnn`) only if 1–3 show motion/edges help — to estimate the
   ceiling.

Success = a monitor-feature block whose blocked-CV R² (and real-vs-null Δ)
clears the `agg`/`pix` baseline on the features that currently decode near/below
chance, with the OOF scatter slope ≈ 1 (genuine, not drift-misled).

## 5. Relationship to existing TODO items
- **#24** (optical flow, Level C) is realized by **Tier 2 (#49)**.
- **#30** (PCA on pixel diff-frames, Level B) is a cheaper motion proxy —
  complementary, keep as an option.
- **#21/#22/#23** (agg-side `motion`, `self_mot`) stay — global/object-relative
  velocity; the new blocks add the spatial (retinotopic) dimension they lack.
- **#20** (angular histogram) = degenerate Tier 1.

## 5b. Implementation architecture (modular)

Built for plug-in extensibility so feature types can be chosen à la carte and a
visualization module can introspect any of them.

### (i) Extractor interface — `functions/f_visual_features.py`
Every feature type is a pure spatial/spatiotemporal op with one signature:
```
f_<name>_features(movie, **params) -> (X, names)
    movie : (T, H, W) one monitor, behavior clock, temporally ordered
    X     : (T, d) features on the same clock ; names : d base channel names
```
No clock/PCA/per-side logic inside extractors. **Done:** `f_grid_features`
(Tier 1) + shared `f_grid_bounds` (the grid every pooling tier reuses).

### (ii) Feature registry — seed of the feature-SELECTION module
`VISUAL_FEATURE_REGISTRY = {name: {fn, defaults, desc}}`. Choosing feature types
= naming them in a `specs` dict with optional param overrides. Tiers 2–4 add one
entry each; nothing else changes. **Done** (holds `'grid'`).

### (iii) Build driver — `build_visual_blocks(movies, side_tags, beh_t, frame_t,
pulse_delay, specs, default_n_pca=...)`
Per type: run extractor per side → concat (side tag in names) → resample to
imaging clock (`f_resample_to_imaging`) → optional per-block PCA. Returns the
SAME dict shape as `build_feature_blocks` (`{'X','names','pca','raw_dim',...}`)
so the script merges via `built_blocks.update(build_visual_blocks(...))`. **Done.**

### (iv) Integration into VR_ca_cebra.py  [not wired yet]
In the build cell, after `build_feature_blocks(...)`, add:
```
visual_specs = {'grid': {'grid': (6,6), 'n_pca': 20}}   # add 'flow', 'gabor', ...
built_blocks.update(build_visual_blocks(
    [left_mon_frames, right_mon_frames] (or per `side`), side_tags,
    beh_t, frame_t, pulse_delay, visual_specs))
```
Then `'grid'` (etc.) are selectable in every `*_target_blocks` knob and decode
through the existing harness. (Deferred — wire after review; Spyder-buffer care.)

### (v) Eventual unified feature-selection module
Long-term: fold the existing hardcoded `build_feature_blocks` blocks (agg / pix /
motion / self_mot / pix_mot / beh / per_obj) into the same registry+spec pattern,
so ALL feature types — behavioral and visual — are chosen through one
`specs`-driven selector instead of a wall of `build_*` booleans. Migrate
incrementally; keep back-compat aliases.

### (vi) Feature-ANALYSIS / visualization module  [planned — `f_feature_plots.py`]
Introspect any block (visual or behavioral) from `built_blocks`:
- `f_plot_feature_traces(block)` — channels over time (shared x with raster).
- `f_plot_feature_corr(block)` — channel correlation matrix + dendrogram.
- `f_plot_feature_variance(block)` — variance / PCA explained-variance spectrum.
- `f_plot_grid_layout(block, grid)` — render the retinotopic tiling and paint
  each cell by a chosen channel/time (grid/flow/gabor); for flow, quiver of the
  per-cell velocity; for gabor, the orientation×SF energy per cell.
- `f_plot_filter_bank(...)` — show the Gabor/motion-energy kernels themselves.
- `f_plot_example_frames(movie, extractor, t)` — raw frame next to its filtered/
  pooled output, to sanity-check what each feature "sees."
Compute/plot split mirrors f_ensembles/f_ensemble_plots and f_decoding's helpers.

## 6. Open questions / user ideas
- (placeholder) User has additional ideas on edge / edge-motion representation —
  fold in before finalizing the Tier-2/3 builder design.
- Resolution of the grid (6×6? 8×8?) and # SFs/orientations — tune for mouse
  acuity vs dimensionality.
- Per-monitor vs single fused visual field (the two monitors are ±45° — could
  stitch into one panoramic angular map).
