# PLAN — one ensemble-generation config, reused everywhere (2026-11-06)

First piece of the modular pipeline (TARGETS). Goal: a dataset gives the SAME ensembles in every
script, from one config dict you pass in, and the config that produced them is recorded.

## Why
All four callers already use the shared engine (`f_ensemble_pipeline`: apply → stats), but each
passes its own settings, so the same dataset yields different ensembles:

| | VR_ca_cebra | VR_encode_scan | VR_tuning_scan / VR_feature_tuning |
|---|---|---|---|
| input smoothing | σ=6 frames | σ=0 | σ=6 |
| NMF sparsity | l1_ratio 0.5, alpha_W 1e-4 | none | none |
| membership | z_thresh | cv_consensus | cv_consensus |
| ensembles used | sig only | all (+ sig flag) | all / sig |

Call sites: VR_ca_cebra.py:195/290/320, VR_encode_scan.py:84/192, VR_feature_tuning.py:94/111,
VR_tuning_scan.py:87/154.

## The config
`ens_cfg = enspipe.f_default_ens_config(**overrides)` — a dict with one section per stage, so a
change re-runs only the stages it affects. Defaults = VR_ca_cebra's CURRENT values (copied from its
cells at lines 133–137 / 195–219 / 287–312 on 2026-11-06 — not imported: importing the script would
run it).

```python
ens_cfg = {
  'neural': {'sigma_frames': 6, 'peak_norm': True,                  # cebra 133-134
             'sort': 'hclust', 'hclust_kw': {'standardize': True, 'metric': 'kl',
                                             'method': 'average', 'similarity_transform': 'auto'}},
  'dimred': f_default_dimred_params(                                  # cebra 195-219
             method='nmf', num_comp=15, smooth_ms=0, normalize='norm_mean_std',
             l1_ratio=0.5, beta_loss='kullback-leibler', alpha_W=1e-4, alpha_H='same',
             max_iter=5000, tol=1e-4, alpha=1.0, k_folds=5, fold_scheme='interleaved',
             n_blocks=None, norm_scope='global', cv_frames=5000,
             scores=['test_err', 'lno_r2_med', 'lno_r2_pool'], r_reliable=0.05),   # seed → 42 default
  'apply':  {'thresh_mode': 'signal_z', 'signal_z': 2.0},             # cebra 287-288
  'stats':  {'cv_method': 'pca', 'membership': 'z_thresh', 'consensus_frac': 0.5,  # cebra 305-312
             'sig_method': 'cv', 'sig_alpha': 0.05, 'cv_sig_thresh': 0.0, 'n_shuffle': 50},
  'select': {'which': 'sig', 'trace': 'scores'},                      # cebra encode cells use 'sig'
}
```
- Overrides: nested (`dimred={'num_comp': 12}`) or a flat key routed to its section
  (`num_comp=12`). Unknown keys RAISE (same guard as KIND_KNOBS — no silently ignored typo).
- `f_ens_config_tag(cfg)` → short stable hash, written into every store/cache/plot title.

## Functions (all in functions/f_ensemble_pipeline.py — no new module)
1. `f_default_ens_config(**overrides)` — build + validate.
2. `f_ens_neural_input(est1, cfg)` — drop silent → smooth → peak-norm → (X (T,N), cell_ids). One
   recipe; replaces the copies in cebra / f_encode_prepare / feature_tuning / f_make_neural_target.
3. `f_ensembles_from_config(X, cfg, cell_ids, fs, prior=None)` → `{ens_out, ens_table, cfg, tag}`.
   Calls the existing f_ensemble_apply / f_ensemble_stats. `prior` = an earlier result: if only the
   'stats' section changed, skip the NMF refit (keeps cebra's cheap re-run workflow).
4. `f_ens_targets(ens, cfg)` → `{'ens_all': (Y, labels), 'ens_sig': (Y, labels)}` via the existing
   selector (f_select_tuning_units / f_neural_source) — what encode/decode/tuning consume.
5. Optional disk cache `f_ens_cache_load/save(features_dir, mouse, dset, tag)` →
   `<features_dir>/ens_cache/<mouse>_<dset>_<tag>.joblib` (ens_out + ens_table + cfg). NMF at
   5000 iters is minutes per dataset; any script with the same config reloads instead of refitting.

## Steps (each verified before the next)
Status 2026-11-06: steps 1–3 DONE. Parity EXACT for cebra and VR_encode_scan settings on 11_2_25
(scores/coeffs/members/ens_table identical; cebra cfg 10/15 sig vs encode_scan cfg 0/15). Cache tested
(load 0.09 s, mismatch → refit). VR_ensembles.py run end to end. Feature search: Cell 1b `use_targets`
+ f_feature_search(targets=…) DONE (targets=None identical to before). Cebra PORTED (one load-or-fit cell; downstream
verified). Cache root = per-mouse features dir for all scripts. VR_encode_scan / VR_tuning_scan /
VR_feature_tuning PORTED (f_ensembles_cached; stores record ens_tag + cfg; store tags bumped to
enc_11_6_26 / tun_11_6_26 so old per-script ensembles never mix in). #76 DONE.
1. Write the functions. PARITY: with the config set to each script's CURRENT settings, the result
   must equal that script's ens_table / scores exactly (same seed) — proves the wrapper changes nothing.
2. Decide the defaults (below), then save the cache layer.
3. Feature search: new Cell 1b → ensembles from the config → `targets` for the sweep (cells /
   ens_all / ens_sig as separate lines). Needs the scorer to accept several targets (pipeline step 2).
4. Port VR_ca_cebra (keep apply/stats as separate cells via `prior`), VR_encode_scan,
   VR_tuning_scan, VR_feature_tuning. Old param blocks stay commented as reference.
5. Stores record `ens_tag` + the full cfg (VR_encode_scan already stores ens_params — extend it).

## New script: `VR_ensembles.py` — load ONE dataset, build + inspect its ensembles from the config
The dedicated place to tune and look at ensembles; every other script then just consumes the config.
Calcium-only load (`f_load_session(..., behavior=False)`) — no render, no features. Reuses the
existing plot functions (f_ensemble_plots) — no new plotting code for starters.

| Cell | Content | Source in cebra |
|---|---|---|
| 1 | paths, mouse/dset, `ens_cfg = f_default_ens_config(...)` (print it + its tag) | 195 |
| 2 | load session (calcium only) → `f_ens_neural_input` → X, cell_ids | 125–150 |
| 3 | (opt-in) dim-red sweep: num_comp × beta_loss/method by CV-LNO → `f_plot_sweep_curves` | 220–244 |
| 4 | (opt-in) NMF consensus stability → `f_plot_nmf_stability` | 255–275 |
| 5 | **ensembles from config** (cache load or fit) → ens_out, ens_table; print the table | 278–320 |
| 6 | per-ensemble scores / significance → `f_plot_ens_scores` | 326 |
| 7 | sorted raster + one panel per ensemble → `f_plot_ens_overview` | 330 |
| 8 | reliability: cell LNO-R² ECDF with ensembles overlaid → `f_plot_lno_ceiling` | 340–350 |
| 9 | per-ensemble CV reconstruction trace → `f_plot_ens_cv_trace` | 355–363 |
| 10 | FOV map of significant-ensemble members → `f_plot_ensemble_fov` | 336 |
| 11 | (opt-in) save to the ensemble cache so every other script reloads these exact ensembles | — |

Later additions (not for starters): ensemble activity vs behavior (needs behavior=True), compare two
configs side by side (membership overlap, matched components), across-dataset summary.
Step order: write after step 2 (needs the config + cache); add to SCRIPTS.md under the pipeline.

## Decisions (resolved 2026-11-06: defaults = cebra's current values)
- membership `z_thresh`, NMF input σ=6, seed 42, hclust sort — all as in cebra now.
- Caveat kept for later: `z_thresh` scores members on the data that chose them, so cv_r2 (and the
  cv-based significance) is optimistic; `membership='cv_consensus'` is one override away.
- hclust order: checked in the parity step whether it changes the NMF result (init depends on row
  order); kept as the `neural.sort` option either way.
- The scan scripts (VR_encode_scan / tuning scans) will CHANGE results when ported, since their current
  settings differ (σ=0 or 6, no sparsity, cv_consensus). Old stores stay tagged with their old config.
