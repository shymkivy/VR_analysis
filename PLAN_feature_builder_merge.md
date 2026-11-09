# PLAN — one feature builder + one interface for VR_feature_search and VR_ca_cebra (2026-11-06)

Third piece of the modular pipeline (after the shared ensemble config and the shared encoder config).
Goal: the SAME feature-set dict builds the SAME blocks through the SAME code in every script, with the
same call shape on both sides, and the feature-search build code is tidied while doing it.

## 1. What exists today — two parallel stacks

| layer | feature search (f_feature_search.py) | cebra / encoding (f_feature_source.py) |
|---|---|---|
| dict reader | `f_resolve_feat_sets` (shared ✓) | `f_resolve_feat_sets` (shared ✓) |
| dict → spec | `f_feature_configs` → `f_pyramid_grid` → `f_feature_specs` (+ `f_pix_config`) | `f_vis_sets_to_plan` → `f_pyramid_specs`+`f_specs_block` (multi-SF) or `f_grid_block` (single SF) |
| spec format | config `{'name', 'grid': {kind, inputs, marginalize, layer1_proc, layer2_proc}}`; pix: `{'name','blocks','plan','proc'}` | plan block `{'origin', 'inputs', 'grid': {kind, marginalize, layer1_proc, layer2_proc}}`; pix: `{'origin','inputs','proc'}` |
| duplicates | `f_feature_grid` ≈ `f_grid_block`; `f_pyramid_grid` ≈ `f_pyramid_block`; `f_feature_specs` ≈ `f_specs_block` | |
| grid builder | `f_iter_grid_cells`: STREAMS one conv-unit at a time (raw freed per unit), exact cached layer-1 decomposition, `_layer1_names`, yields cells | `_build_grid_block`: ALL inputs extracted in ONE `f_build_features` call (all raw in RAM at once), layer-1 via `f_proc` (randomized SVD), returns all cells |
| raw source | build only (`_raw_block` → a one-block `f_resolve_features` per unit → `build_visual_blocks` + conv_autosave) | build / conv_repo / `from_block` (`_extract_input`) |
| orchestration | `f_feature_search`: THREE paths (grid / memoizable non-grid / full resolve), score memo, targets, enc_cfg | `f_resolve_features`: origin groups (build / cache / repo / derived) |
| shared ✓ | conv_autosave store, edge_min_frac 0.45, 150×101 render, cell names (f_num_tag), DEFAULT_MARGINALIZE, f_marg_spec / f_marg_axes, KIND_KNOBS validation, encoder config | |

## 2. Divergences found (same dict → different features)
1. **Single-SF entries — CONFIRMED silent bug.** `{'g': dict(kind='gabor', sf_cyc_deg=0.13)}`:
   search → 8 orientations, Nyquist sampling (via f_pyramid_specs); cebra → **4 orientations, fixed
   ds=4** (f_grid_block's own defaults). Pyramid entries (cyc_fov) are identical — why it never showed.
2. **Layer-1 pca/svd**: search = exact cached decomposition (2026-11-06); cebra = randomized SVD →
   ~1e-4 differences, larger where singular values nearly tie. Current cebra vis_sets use
   layer1_proc=None, so not yet hit.
3. **pix**: search = kind='pix' + layer procs; cebra's `feature_list 'pix'` = the old n_pix_pca path
   (vis_sets kind='pix' already matches the search).
4. **Memory model**: cebra holds every raw input of every bank at once; the search's peak is one
   conv-unit. A big pyramid that fits in the search can OOM in cebra.
5. Track record: the 2026-07-27 marginalize bug and the 2026-11-06 names bug each had to be fixed in BOTH builders.

## 3. Target design
```
feat_sets dict ──► f_resolve_feat_sets ──► ONE spec builder ──► plan-block specs {name: block}
   (same dict in both scripts)                (f_feature_plan)        │
                                                                      ▼
                         ONE grid builder: f_iter_grid_cells(block, raw_source, cache) — streaming
                           raw_source = build (conv_autosave) | conv_repo | from_block
                                │                                   │
                   feature search: score each cell, free it   cebra: collect cells → built_blocks
```
- **One spec format = the plan block** (it already carries origin / repo / from_block). A search config
  becomes `{'name': <tag>, 'block': <plan block>}`; pix is just another plan block.
- **One spec builder** `f_feature_plan(feat_sets, **common)` (in f_feature_source) — every bank entry goes
  through f_pyramid_specs (fixes divergence 1); used by BOTH `f_feature_configs` (search) and cebra/encoding
  (replacing f_vis_sets_to_plan, kept as an alias).
- **One grid builder** moved to f_feature_source: the search's streaming `f_iter_grid_cells`, with a
  pluggable raw source (build per conv-unit, repo, from_block) and the per-call cache (raw blocks,
  layer-1 decompositions). `_build_grid_block` becomes `dict(f_iter_grid_cells(...))` (fixes 2 and 4).
- **Same call shape on both sides**:
  ```python
  feat_sets = {'_common': {...}, 'gabmag': {...}, ...}            # identical dict in both scripts
  # search:  configs = fsearch.f_feature_configs(feat_sets)       → scored cell by cell
  # cebra:   plan    = fsrc.f_feature_plan(feat_sets)             → built_blocks via f_resolve_features
  ```
  (`vis_sets` in cebra/encoding renamed `feat_sets`, old name accepted.)
- Optional: `feature_sets.py` — named dicts both scripts import (a search winner is referenced, not pasted).

## 4. Tidy-up of the feature-search build code (folded into the steps)
- **DONE 2026-11-07 — agg gated like pix**: every conv-unit fetch rebuilt the unused `agg` block (1.53 s of
  a 1.7 s autosave-hit fetch, profiled). `build_feature_blocks(build_agg=True)` + f_build_features passes
  cfg['build_agg'] + `_resolve_build` sets it to 'agg' in srcs (agg still computed when motion needs its
  presence columns). Checks: visual block identical, fetch 1.36 → 0.06 s; agg identical; motion identical.
- Remaining review findings (2026-11-07): `_raw_block` → 4-layer one-block plan per conv-unit; render
  required on autosave hits (shape only needed); 3 code paths in f_feature_search; two unit constructions
  in f_iter_grid_cells (inputs vs legacy homogeneous bank); one cache dict mixing raw + l1fit entries;
  orientation parsed from names by regex; bulky inline timing prints.
- **One path in f_feature_search**: every config is a plan block (grid or pix) → the same stream; drop the
  memoizable non-grid path and the full-resolve path (they only serve pix and the legacy configs).
- **`_raw_block`** calls build_visual_blocks directly (with conv_autosave) instead of a whole one-block
  f_resolve_features per conv-unit.
- **Caches made explicit**: `raw_cache` currently mixes raw-block string keys and ('l1fit', …) tuples →
  one small cache object {'raw', 'l1fit'} passed through.
- **Legacy config builders** (`f_feature_grid`, `f_granularity_configs`, `f_variant_configs`,
  `f_prep_configs`, `f_build_grid`, `_reduce_raw`, `f_pix_config`'s separate plan path) → moved to
  `f_feature_legacy.py` (still importable; only the opt-in example cells 3b / 4 use them).
- **Old-scorer knobs** on f_feature_search (`feat_prep`, `feat_pre_pca_zscore`, `scorer_standardize`,
  `feature_proc`, `encode_neural_proc`, `lags_ms`/`r_sig` for encode) → encode comes from enc_cfg only;
  keep the decode knobs until a decoder config exists. (Decision below.)
- **Lazy render**: when every conv-unit is an autosave hit, use placeholder movies (shape only) — no
  render at all; render on the first miss / pix (discussed 2026-11-06).

## 5. Steps (each verified before the next)
1. **DONE 2026-11-09** — f_vis_sets_to_plan routes EVERY bank entry through f_pyramid_specs + f_specs_block
   (f_grid_block no longer used there). Checks: all 25 existing/pyramid/pix entries identical; single-SF
   gabor/fft/meng entries now == the search (8 ori / Nyquist instead of 4 / ds=4).
   Original step: **Fix divergence 1 now** (small): single-SF entries go through f_pyramid_specs on the plan side.
   PARITY: every pyramid entry's inputs unchanged; single-SF entries now == the search's.
2. **Move the streaming builder** into f_feature_source with a pluggable raw source + explicit cache;
   the search imports it. PARITY: search blocks byte-identical to today (grid + layer-1 svd + names).
3. **`_build_grid_block` → wrapper** over it (repo / from_block sources included). PARITY: cebra's current
   vis_sets blocks identical; any layer-1 pca/svd entry differs only by the exact-vs-randomized amount
   (measured and reported); peak RAM measured before/after.
4. **One spec builder** `f_feature_plan`; `f_feature_configs` = names + plan blocks; f_pyramid_grid /
   f_feature_specs / f_grid_block / f_pyramid_block become thin wrappers. PARITY: config names + inputs
   identical for the current feat_sets / vis_sets dicts.
5. **f_feature_search → one path**; `_raw_block` direct; legacy builders → f_feature_legacy.py; old-scorer
   knobs removed (per decision). PARITY: Cell 3 / 3c tables identical on the test sets.
6. **Cebra / VR_ca_encoding feature cell**: `feat_sets` name + `f_feature_plan`; legacy feature_list
   names (gaborS/mengS/gaborG, feat_pca_variant) as comments; pix via a feat_sets entry (or keep the old
   pix block deliberately). Optional feature_sets.py.
7. **Lazy render** (shape-only placeholders on autosave hits).

## 6. Decisions needed
- **Old-scorer knobs** (step 5): remove from f_feature_search, or keep as a legacy switch? VR_dataset_scan
  and cebra's "Score feature representations" cell still use f_score_representations directly — they are
  unaffected either way.
- **Legacy builders**: move to f_feature_legacy.py (proposed) or delete outright?
- **Naming**: rename cebra/encoding `vis_sets` → `feat_sets` (proposed, old name accepted)?
- **cebra 'pix'**: switch to a feat_sets kind='pix' entry, or keep the old n_pix_pca pix block?
- **Shared feature_sets.py**: wanted now, or later?
