# PLAN — one encoder config + one encoding sweep, reused everywhere (2026-11-06)

Second piece of the modular pipeline (after the shared ensemble config, PLAN_ensemble_config.md).
Goal: every script encodes with the SAME model, set from ONE config with documented defaults
(= VR_ca_cebra's tuned values), and every encoding sweep goes through ONE wrapper.

## Why
Today there are three encoding scorers with different settings, so the same feature set gets
different encoding r depending on the script:

| setting | VR_feature_search (f_score_representations) | VR_ca_cebra (enc_params → f_run_encoder) | VR_encode_scan (f_encode_one) |
|---|---|---|---|
| model | ridge, alpha 1 (library default) | ridge, alpha 1000 | ridge, alpha 1000 |
| lags | 200 ms only | 0–500 ms (6 lags) | 0–500 ms (6 lags) |
| folds | 5 contiguous | 5 interleaved | 5 interleaved |
| embargo | 0.5 s | 0 s | 0.5 s |
| feature standardize | off (procs before CV) | on | on |
| feature reduction | feat_sets layer1/2 procs | 150 PCs, joint | n_feature_pcs 200 (cache) |
| neural target | σ=0, peak-norm | σ=6 frames, peak-norm | σ=0, peak-norm |

## What already exists (f_encoding.py, used by cebra)
- `f_default_encoder_params(**overrides)` — flat config dict (~25 keys). NB silently accepts typos.
- `f_run_encoder(cfg, built_blocks, Y, frame_t, frame_mask)` → per-neuron scores (r_med / r2_med /
  frac_sig) + population scores (r2_vw / rv / procrustes / rsa) from one fit; per-neuron arrays in
  `out['res']`. Joint feature PCA is fit on the whole session (mild leak, pipeline-wide stance).
- `f_encoder_sweep(cfg, sweep_var, values, …)` — vary ONE key (+ virtual 'lag_ms').
- `f_encode_over_inputs` — same encoder over cells / reliable / ensembles / ens_cells.

## 1. The config — defaults = VR_ca_cebra's enc_params (2026-11-06)
`enc_cfg = enc.f_default_encoder_params(**overrides)`; defaults moved to the TOP of f_encoding.py,
every key with all its options commented (same treatment as _ens_defaults). Unknown keys RAISE.
`f_enc_config_tag(cfg)` → short hash for stores/labels.

```python
{
  # target (NEW section — today it is each script's own Y_neu recipe)
  'target_sigma_frames': 6,        # Gaussian σ on spikes (frames); cebra Y_neu = 6
  'target_peak_norm':    True,
  'target_smooth_ms':    0,        # extra smoothing inside the fit (existing key)
  # features
  'feature_block':    ['gabmag'],  # bank name(s) / block(s); a list is concatenated
  'n_feature_pcs':    150,         # None = all columns | int = top-k
  'feature_pca':      'joint',     # 'per_block' | 'joint' | 'joint_postlag'
  'covariate_blocks': [],          # e.g. ['self_mot']
  'standardize':      True,
  # model
  'decoder':          'ridge',     # ridge | ridgecv | elasticnet | lasso | huber | lad | knn | pls | cca
  'ridge_alpha':      1e3,
  'l1_ratio':         1e-4,        # elasticnet only
  'ridgecv_alphas':   None, 'k_nn': 15, 'n_components': 20,
  'roughness_lambda': 0, 'roughness_order': 2,
  'n_neural_pcs':     None,        # None = per-neuron | int = reduced-rank target
  # CV
  'lags_ms':          [0, 100, 200, 300, 400, 500],
  'n_folds':          5, 'fold_scheme': 'interleaved', 'n_blocks': None, 'embargo_sec': 0.0,
  # scores
  'scores':           ['r_med', 'r2_med', 'frac_sig', 'r2_vw', 'rv', 'procrustes', 'rsa'],
  'r_sig':            0.1, 'rsa_subsample': 2000, 'seed': 42,
}
```

## 2. Neural target from the config
`f_enc_target(est1, cfg)` → Y (T, n_cells) + cell_ids, the one spikes→rate recipe (reuses
f_ens_neural_input's code without the hclust sort). Ensemble targets keep coming from
`f_ens_targets` (shared ensemble config). Targets dict = {'cells', 'ens_all', 'ens_sig'}.

## 3. The sweep wrapper
`f_encode_sweep(enc_cfg, axes, targets, built_blocks=None, feat_sets=None, session=None, …)`
→ tidy DataFrame, one row per (axis values × target [× feature set]) with every score + n_valid +
config tag; per-unit r kept in a side dict for maps / significance.

| axis kind | examples | reruns per value |
|---|---|---|
| encoder key | ridge_alpha, lags_ms, lag_ms (virtual), decoder, n_feature_pcs, fold_scheme, target_smooth_ms | fit only (features built once) |
| feature | feat_sets entry; layer1/layer2 size; pre/post procs (= today's Cell 3c, via f_proc_sweep) | features (conv from cache) + fit |
| target | cells / ens_all / ens_sig | fit only |

- 1-D (`axes={'ridge_alpha': [...]}`) and 2-D (`axes={'x': ('n_feature_pcs', [...]), 'line': ('target', [...])}`)
  → `f_plot_encode_sweep(df, metric)`: x = inner axis, one line per outer value (same style rules as
  f_plot_comp_sweep: colour = feature set, line style = target, point labels, saturation clipping).
- Feature-axis adapter: the streamed grid cells become a one-block `built_blocks` per cell →
  `f_run_encoder` with `feature_block=[cell]`; reuses the conv cache, layer1 decomposition cache and
  identical-block reuse already in f_feature_search.

## 4. Where it plugs in
- VR_ca_cebra: `enc_params = enc.f_default_encoder_params(...)` overrides only; sweep cell → wrapper.
- VR_feature_search: ENCODE via f_run_encoder with enc_cfg (Cell 3 scorecard + Cell 3c sweep become
  wrapper calls). DECODE stays on f_score_representations until a decoder config gets the same
  treatment (cebra already has f_default_decoder_params / f_decoder_sweep).
- VR_encode_scan: f_encode_one → f_run_encoder; store records enc tag + cfg.
- Later: runner + resumable store over datasets × configs (→ #75 cross-dataset feature scan).

## Steps (each verified before the next)
Status 2026-11-06: STEP 1 DONE — defaults at the top of f_encoding.py (every key + options), unknown keys
raise, f_enc_config_tag. Parity: cebra's enc_params cell == new defaults; VR_dataset_scan pinned to its
old values (unchanged); f_run_encoder identical old vs new module on a real block (L/11_2_25).
STEP 2 DONE — f_utils.f_neural_rate = the ONE spikes→rate recipe; f_encoding.f_enc_target(est1, cfg) uses it,
and f_make_neural_target + f_ens_neural_input now call it too (bit-identical before/after; ensemble cache
still loads). f_enc_target σ6 == cebra's Y_neu (column order aside), σ0 == the feature search's target.
STEP 3 DONE — enc.f_encode_sweep(enc_cfg, axes, targets, built_blocks, …) → (tidy df, per-unit r);
axes = encoder keys | 'lag_ms' | 'target' (implicit: every target), any number of axes (product);
encp.f_plot_encode_sweep(df, metric, x, line) with p25–p75 band, auto log-x, list-valued x. PARITY:
1-D ridge_alpha and lag_ms sweeps == f_encoder_sweep exactly (17 scores/point) on L/11_2_25.
STEP 4 DONE — f_feature_search(enc_cfg=…) encodes every scored block with f_run_encoder under the shared
config (feature_block=[block], n_feature_pcs=None), filling the same encode_* columns + encode_r2_vw/rv/
procrustes/rsa + enc_tag; decode unchanged. VR_feature_search Cell 1 builds enc_cfg + Y_neu via
f_enc_target (σ6), Cells 3/3c pass enc_cfg. PARITY: encode cols == f_run_encoder direct; enc_cfg=None ==
old module; decode unchanged. Example: fftmag2_4 layer1 svd8 → old scorer r_med 0.075, shared 0.127.
STEP 5 DONE — cebra's encoding sweep cell → f_encode_sweep (enc_sweep_targets cells/ens_all/ens_sig,
optional enc_line_axis 2nd axis, enc_plot_metrics line plots, old per-score panel view kept for 1 axis +
cells). VR_encode_scan → enc_cfg (n_feature_pcs=None; blocks pre-reduced by the plan) +
f_encode_scan.f_encode_one_cfg (== f_encode_one at the same settings, per cell); its target recipe →
f_neural_rate (max diff 0.0). Store tag enc_11_6_26 (not yet written). #77 DONE.
1. Defaults to the top of f_encoding.py + docs + unknown-key raise + tag. PARITY: defaults ==
   cebra's enc_params cell exactly; `f_run_encoder` output unchanged on a real block.
2. `f_enc_target`; PARITY vs cebra's Y_neu (σ=6) and the feature search's (σ=0 override).
3. Wrapper for encoder-key + target axes (1-D/2-D) + plot; PARITY: 1-D result == f_encoder_sweep.
4. Feature-axis adapter; feature search Cell 3/3c encode through it (numbers WILL change — new model).
5. Port cebra's sweep cell, VR_encode_scan; bump the scan store tag.

## Decisions (resolved: defaults = cebra) and open points
- DOUBLE REDUCTION — RESOLVED 2026-11-06: feature-axis sweeps (and any encode of a feat_sets cell) run
  with n_feature_pcs=None, so the feature set's own layer1/layer2 processing alone sets the dims; the
  150-PC default applies only to encoder-key sweeps on raw banks (cebra's usage).
- Feature-search encode numbers change (alpha 1→1000, 1→6 lags, σ 0→6, standardize on). Old runs
  are not comparable to new ones.
- Cost: 6 lags × up to 150 features = 900 design columns; slower than the current 1-lag scorer —
  measure in step 4.
- Embargo: cebra 0 s vs scan 0.5 s. Cebra default (0) per this plan; 0.5 s is safer against
  autocorrelation leakage between folds — flag for review.
