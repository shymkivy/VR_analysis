# -*- coding: utf-8 -*-
"""
VR_encode_scan.py — SCRIPT 1: cross-dataset per-neuron ENCODING sweep (compute + save).

Loop mice × datasets and, for EACH neuron, score how well it regresses onto each feature BLOCK
(agg / self_mot / pix / gaborS+mengS) under each MODEL (ridge / cca) — per-neuron OOF R² (+ Pearson r).
Results accumulate into ONE tag-versioned, RESUMABLE joblib store (f_encode_scan), rewritten atomically
after EVERY (dataset, block, model) combo, so stop/resume never loses progress and widening
blocks/models only computes the new combos. The store keeps ONLY neuron indices + scores + dataset
identity (NO locations) — VR_encode_scan_plots.py loads locations separately and joins by dset_name.

No ensembles. Compute in functions/f_encode_scan.py (reuses f_encode_per_neuron + f_resolve_features).
Spyder cell-based (#%% markers). Created 2026-07-17.
"""

import sys
import os
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

_thread_cap = 6                      # cap BLAS threads before numpy import
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(_thread_cap)

try:                                 # UTF-8 stdout so emoji in library prints (⏱) can't crash on Windows cp1252
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except Exception:
    pass

from f_pipeline import f_load_session
from f_feature_source import f_default_feature_plan, f_default_build_cfg, f_default_source
from f_feature_gen import f_resolve_dsets
import f_encoding as enc
import f_ensembles as ens
import f_ensemble_pipeline as enspipe
import f_encode_scan as es


#%% =========================================================================
# SECTION 1 · SETUP — mice / datasets / paths / tag
# ==========================================================================
mice         = ['L']                 # mouse ids to sweep
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
data_dir_of  = lambda m: 'F:/VR/data_proc/' + m
feat_root    = 'F:/VR/features/'      # shared root; the encode_scan/ store folder sits here

ENC_TAG      = 'enc_11_6_26'          # versions the store file (bump to start a fresh sweep)
                                     # enc_7_18_26 = old per-script ensembles (not the shared config)
DSETS        = None                   # None/'all' = all do_proc | list of n_dset | {mouse: [...]}


#%% SECTION 2 · CONFIG — sources, blocks, models, encoder + ensemble knobs, feature plan
sources      = ('cells', 'ensembles')     # unit sources to encode: raw cells AND/OR ensembles
blocks       = ['agg', 'pix', 'motion', 'self_mot', 'beh', 'gaborS', 'mengS', 'gaborS+mengS']   # each scored ALONE, per unit
# ENCODER = the shared encoder config (f_encoding; defaults = VR_ca_cebra's enc_params — ridge α 1e3,
# lags 0–500 ms, interleaved 5-fold, standardize, target σ 6; every key documented in _enc_defaults()).
# n_feature_pcs=None: the blocks arrive already reduced (n_feature_pcs below → feature plan).
# Override by name, e.g. f_default_encoder_params(n_feature_pcs=None, embargo_sec=0.5).
enc_cfg      = enc.f_default_encoder_params(n_feature_pcs=None)
models       = ['ridge']          # , 'cca'                    # per-unit model sweep (→ enc_cfg 'decoder')
model_kw     = {                                              # extra encoder keys per model
                #'cca':   {'n_components': 30},
                }
sigma_frames = enc_cfg['target_sigma_frames']     # neural target smoothing (from the encoder config)
print(f'encoder config {enc.f_enc_config_tag(enc_cfg)}')
n_feature_pcs    = 200          # cap feature PCs per visual block (loaded → keep top-N; derived → joint-PCA size); None = use cache's saved count

# ENSEMBLES (only if 'ensembles' in sources): the SHARED ensemble config (f_ensemble_pipeline; defaults
# = VR_ca_cebra, every parameter documented in _ens_defaults()). Per dataset they LOAD from
# <feat_root><mouse>/ens_cache/ (built by VR_ensembles.py / cebra) or are fit + saved there.
# Override by name, e.g. f_default_ens_config(num_comp=12, membership='cv_consensus').
ens_cfg   = enspipe.f_default_ens_config()
ens_which = 'all'                         # 'all' = every ensemble (sig flag stored) | 'sig' = significant only
print(f'ensemble config {enspipe.f_ens_config_tag(ens_cfg)}')

# feature acquisition — features are per-dataset, so f_encode_prepare builds them ONCE per dataset
# (build pix/agg/self_mot + gabor/meng), then every cell × block × model combo for that dataset is
# scored on that single build (no per-model rebuild).
conv_backend = 'gpu'                       # 'gpu' (torch, fast — like the tuning scan) | 'cpu' (scipy)
conv_device  = 'cuda'                      # torch device for conv_backend='gpu'
# point-sample stride for the BUILT gaborS/mengS blocks. RAM scales ∝ 1/ds²: at 150×101 over a full
# session ds=2 → ~30k columns → a ~11 GB float32 block (22 GB if float64) → OOM; ds=4 → ~7.6k cols.
# The gabor MAGNITUDE map is smooth (Nyquist stride ≈0.5·λ ≈6 px), so ds≥2 loses little. Lower to 2
# only if you have the RAM (and want finer retinotopy).
samp_ds      = 4
feature_plan = f_default_feature_plan()
for _bn in ('gaborS', 'mengS', 'gaborS+mengS', 'gaborS+mengS2'):   # override sampled-visual ds
    for _inp in feature_plan.get(_bn, {}).get('inputs', []):
        if _inp.get('reduce') == 'sample':
            _inp['ds'] = samp_ds
# n_feature_pcs → per-block PC budget. On a LOADED cache block (pix/gaborS/mengS) the plan's n_pca is an
# UPPER-BOUND cap (_cap_loaded_block keeps the top-N variance PCs); on the DERIVED gaborS+mengS it is the
# joint-PCA output size. Set it on the visual/pix blocks + the derived composites' source blocks so the
# whole visual budget is n_feature_pcs. agg/self_mot are raw (no PCA) and untouched.
if n_feature_pcs is not None:
    for _bn in ('pix', 'gaborS', 'mengS', 'mengS2', 'mengS3',
                'gaborS+mengS', 'gaborS+mengS2', 'gaborS+mengS3'):
        if _bn in feature_plan:
            feature_plan[_bn]['n_pca'] = int(n_feature_pcs)
build_cfg    = f_default_build_cfg(side='both', conv_backend=conv_backend, conv_device=conv_device)

# REUSE the pca_feat caches (no reconvolution) — mirrors VR_ca_cebra.py. Set feat_pca_variant to the
# generated variant so gaborS/mengS LOAD from cache; feat_combine composes gaborS+mengS from those
# loaded blocks (concat scores + one PCA, balance=True = the old input_norm='pca' balancing). Blocks
# NOT in the cache (agg/pix/self_mot) still build. feat_pca_variant=None → build gabor/meng from scratch.
feat_pca_variant = 'pcacomb'      # None | 'pcacomb' | 'pcasep' | 'pcaraw'
feat_whiten      = False          # whiten loaded PCA blocks (÷√eigenvalue)
feat_combine     = {}             # gaborS+mengS is now a DERIVED block in f_default_feature_plan
                                  # (from_block: gaborS+mengS) — just list it in `blocks`; no reconv.

# config stamped into the store (for later verification of what produced the scores)
scan_cfg = {'sources': sources, 'blocks': blocks, 'models': models, 'model_kw': model_kw,
            'enc_cfg': enc_cfg, 'enc_tag_cfg': enc.f_enc_config_tag(enc_cfg), 'sigma_frames': sigma_frames,
            'feat_tag': f_default_source()['conv_tag'],
            'num_samp': build_cfg.get('lat_samp'), 'conv_backend': conv_backend, 'samp_ds': samp_ds,
            'feat_pca_variant': feat_pca_variant, 'feat_whiten': feat_whiten, 'n_feature_pcs': n_feature_pcs,
            'feat_combine': {k: dict(v) for k, v in feat_combine.items()},
            'ens_cfg': ens_cfg, 'ens_tag': enspipe.f_ens_config_tag(ens_cfg), 'ens_which': ens_which,
            'enc_tag': ENC_TAG}


#%% SECTION 3 · Build the store path + load-or-resume + planned target grid
store_path = es.f_encode_scan_path(feat_root, ENC_TAG)
os.makedirs(os.path.dirname(store_path), exist_ok=True)

# planned (mouse, dset_name, block, model) grid — needs each mouse's do_proc dset_name order
dsets_of = {}
for m in mice:
    _vr2 = pd.read_excel(vr_data_xlsx)
    _vr2 = _vr2[(_vr2.mouse_id == m) & (_vr2.do_proc == 1)].reset_index(drop=True)
    _names = list(_vr2.dset_name)
    dsets_of[m] = [(nd, _names[nd]) for nd in f_resolve_dsets(DSETS, _vr2, m)]
target = [(m, nm, s, b, mdl) for m in mice for (nd, nm) in dsets_of[m]
          for s in sources for b in blocks for mdl in models]

store = es.f_encode_scan_load(store_path)
if store is None:
    store = es.f_new_store(scan_cfg, target)
else:
    store['target'] = [tuple(t) for t in target]          # refresh planned grid (widen freely)
    store.setdefault('config', scan_cfg)
prog0 = es.f_encode_scan_progress(store_path)
print(f'store: {store_path}')
print(f'planned {len(target)} combos across {sum(len(v) for v in dsets_of.values())} datasets; '
      f'already done: {prog0.get("n_done", 0)}')


#%% SECTION 4 · Run the sweep — loop mice × datasets, save after every combo
# Skips whole datasets whose combos are all done (no load/build); otherwise builds features ONCE per
# dataset, extracts ensembles ONCE (if any pending combo needs them), then fits each remaining
# (source, block, model) and saves atomically. Bad datasets are logged + skipped.
_n_combo = len(sources) * len(blocks) * len(models)
for m in mice:
    src_m = f_default_source(mouse_id=m)
    if feat_pca_variant:                                   # load gaborS/mengS from the pca_feat cache
        src_m = {**src_m, 'cache_tag': f"{src_m['cache_tag']}_{feat_pca_variant}"}
    for nd, dset_name in dsets_of[m]:
        combos = [(s, b, mdl) for s in sources for b in blocks for mdl in models
                  if es._enc_key(m, dset_name, s, b, mdl) not in store['results']]
        if not combos:
            print(f'  {m} d{nd} {dset_name[:22]:22s} — all {_n_combo} combos done, skip')
            continue
        try:
            session = f_load_session(m, nd, data_dir_of(m), data_dir_bh, vr_data_xlsx, verbose=False)
            prep = es.f_encode_prepare(session, blocks, sigma_frames=sigma_frames,
                                       plan=feature_plan, build_cfg=build_cfg, src=src_m,
                                       whiten=feat_whiten, combine=feat_combine, verbose=True)
            # per-source (T, n_units) targets + unit labels; ensembles extracted once if needed
            targets = {'cells': (prep['Y_enc'], prep['cell_ids'])}
            if any(s == 'ensembles' for s, _, _ in combos):
                ens_res = enspipe.f_ensembles_cached(session['est1'], ens_cfg, feat_root + m, m, dset_name,
                                                     fs=prep['fs'], dset_tag=f'{m} d{nd}', verbose=False)
                ens_table = ens_res['ens_table']
                n_ens = len(ens_res['ens_out']['cells']['ens_list'])
                _tg = enspipe.f_ens_targets(ens_res)
                _key = 'ens_all' if ens_which == 'all' else 'ens_sig'
                if _key in _tg:                        # trim to the encoder's clock
                    Y_ens, _lab = _tg[_key]
                    targets['ensembles'] = (Y_ens[:prep['rec']['T']], np.asarray(_lab))
                prep['rec']['ens'] = es.f_ensemble_record(ens_table, n_ens)
                prep['rec']['ens_params'] = {'tag': ens_res['tag'], 'cfg': ens_res['cfg'], 'which': ens_which}
                print(f'  {m} d{nd} {dset_name[:22]:22s} n_act={prep["n_active"]:>4} T={prep["rec"]["T"]:>6} '
                      f'ens={n_ens} (sig {int(prep["rec"]["ens"]["sig"].sum())}) — {len(combos)} combos')
            else:
                print(f'  {m} d{nd} {dset_name[:22]:22s} n_act={prep["n_active"]:>4} T={prep["rec"]["T"]:>6} '
                      f'— {len(combos)} combos')
            es.f_store_set_dataset(store, prep['rec'])
            for s, b, mdl in combos:
                Y_src, labels = targets[s]
                res = es.f_encode_one_cfg(prep['built_blocks'], b, Y_src, enc_cfg, prep['frame_t'],
                                          model=mdl, model_kw=model_kw.get(mdl))
                es.f_store_put(store, m, nd, dset_name, s, b, mdl, res, labels)
                es._save_atomic(store_path, store)          # crash-safe: persist after every combo
                print(f'      {s:9s} {b:14s} {mdl:6s}  median R²={np.nanmedian(res["r2"]):+.3f}  '
                      f'({int(np.isfinite(res["r2"]).sum())}/{len(labels)} units)')
            del session, prep
        except Exception as e:
            import traceback
            print(f'  {m} d{nd} {dset_name[:22]:22s} SKIPPED — {type(e).__name__}: {e}')
            traceback.print_exc()                    # full traceback so the real cause is visible


#%% SECTION 5 · Progress + quick per-source/block/model summary
prog = es.f_encode_scan_progress(store)          # in-memory store (reflects this run)
print(f'\n{ENC_TAG}: {prog["n_done"]}/{prog["n_target"]} combos done, {prog["n_remaining"]} remaining')
print(f'  datasets with results: {len(prog["datasets"])}')
print(f'  by source: {prog.get("by_source", {})}')
print(f'  by block: {prog["by_block"]}   by model: {prog["by_model"]}')

# tidy per-unit table (median R² per source × block × model as a sanity glance)
tbl = es.f_encode_scan_table(store)
if not tbl.empty:
    with pd.option_context('display.width', 200):
        print('\nmedian per-unit R² by source × block × model (all units so far):')
        print(tbl.groupby(['source', 'block', 'model'])['r2'].median().unstack().round(3).to_string())
        # ensembles: median R² over the SIGNIFICANT ensembles only
        _e = tbl[(tbl['source'] == 'ensembles') & (tbl['sig'] == True)]      # noqa: E712
        if not _e.empty:
            print(f'\nSIGNIFICANT ensembles ({_e[["dset_name","unit"]].drop_duplicates().shape[0]} '
                  f'ens across datasets) — median R² by block × model:')
            print(_e.groupby(['block', 'model'])['r2'].median().unstack().round(3).to_string())
