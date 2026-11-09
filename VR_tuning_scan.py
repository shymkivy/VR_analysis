# -*- coding: utf-8 -*-
"""
VR_tuning_scan.py — SCRIPT 1: cross-dataset feature-TUNING sweep (compute + save).

The tuning analogue of VR_encode_scan.py. Loop mice × datasets and, for EACH cell AND ensemble,
reverse-correlate (STA) its activity against the cached gabor/meng features (per orientation × temporal
freq) and the other blocks (agg / self_mot / pix) → a per-unit tuning response. Features are LOADED from
the pca_feat caches (feat_pca_variant='pcasep' → per-orientation×tf PCA groups) with NO reconvolution.
Results accumulate into ONE tag-versioned RESUMABLE joblib store (f_tuning_scan), rewritten atomically
after EVERY (dataset, source, feat_set) combo, so stop/resume never loses progress.

Compute in functions/f_tuning_scan.py (reuses f_encode_prepare + f_tuning reverse-corr/readout + the
f_encode_scan store primitives). Spyder cell-based (#%% markers). Created 2026-07-17.
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

try:                                 # UTF-8 stdout so emoji in library prints can't crash on cp1252
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
from f_encoding import f_lags_from_ms
import f_ensembles as ens
import f_ensemble_pipeline as enspipe
import f_tuning_scan as ts


#%% =========================================================================
# SECTION 1 · SETUP — mice / datasets / paths / tag
# ==========================================================================
mice         = ['L']                 # mouse ids to sweep
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
data_dir_of  = lambda m: 'F:/VR/data_proc/' + m
feat_root    = 'F:/VR/features/'      # shared root; the tuning_scan/ store folder sits here

TUN_TAG      = 'tun_11_6_26'          # versions the store file (bump to start a fresh sweep)
                                     # tun_7_17_26 = old per-script ensembles (not the shared config)
DSETS        = None                   # None/'all' = all do_proc | list of n_dset | {mouse: [...]}


#%% SECTION 2 · CONFIG — sources, feature sets, STA + ensemble knobs, feature plan
sources      = ('cells', 'ensembles')          # unit sources to tune: raw cells AND/OR ensembles
# feat_sets: name → cache blocks that make its filter grid. meng spans the 3 per-tf blocks (→ 4o×3tf).
feat_sets    = dict(ts.DEFAULT_FEAT_SETS)      # {'gabor':['gaborS'], 'meng':['mengS','mengS2','mengS3'],
                                               #  'pix':['pix'], 'agg':['agg'], 'self_mot':['self_mot']}
tune_method  = 'sta'                            # 'sta' (reverse-correlation) | 'ridge'
tune_alpha   = 1.0                              # ridge only
lags_ms      = [0, 100, 200, 300, 400, 500]     # causal FIR window (peak lag picked per unit)
sigma_frames = 6                                # neural smoothing (~100 ms @ 60 Hz)

# feature acquisition — LOAD the pca_feat caches (no render, no convolution). pcasep has the
# per-orientation×tf PCA groups the tuning matrix needs.
feat_pca_variant = 'pcasep'                     # 'pcasep' (per-filter groups) | 'pcacomb' (per-input)
feat_whiten      = False                        # keep UNWHITENED (the magnitude regressor assumes it)
# blocks to load = the union of every feat_set's source blocks
blocks       = sorted({b for bl in feat_sets.values() for b in bl})
feature_plan = f_default_feature_plan()
build_cfg    = f_default_build_cfg(side='both')

# ENSEMBLES (only if 'ensembles' in sources): the SHARED ensemble config (f_ensemble_pipeline; defaults
# = VR_ca_cebra, every parameter documented in _ens_defaults()). Per dataset they LOAD from
# <feat_root><mouse>/ens_cache/ (built by VR_ensembles.py / cebra) or are fit + saved there.
# Override by name, e.g. f_default_ens_config(num_comp=12, membership='cv_consensus').
ens_cfg   = enspipe.f_default_ens_config()
ens_which = 'all'                         # 'all' = tune every ensemble (sig flag stored) | 'sig' = significant only
print(f'ensemble config {enspipe.f_ens_config_tag(ens_cfg)}')

# config stamped into the store (provenance)
scan_cfg = {'sources': sources, 'feat_sets': feat_sets, 'tune_method': tune_method,
            'tune_alpha': tune_alpha, 'lags_ms': lags_ms, 'sigma_frames': sigma_frames,
            'feat_pca_variant': feat_pca_variant, 'feat_whiten': feat_whiten,
            'feat_tag': f_default_source()['cache_tag'], 'ens_cfg': ens_cfg,
            'ens_tag': enspipe.f_ens_config_tag(ens_cfg), 'ens_which': ens_which, 'tun_tag': TUN_TAG}


#%% SECTION 3 · Build the store path + load-or-resume + planned target grid
store_path = ts.f_tuning_scan_path(feat_root, TUN_TAG)
os.makedirs(os.path.dirname(store_path), exist_ok=True)

dsets_of = {}
for m in mice:
    _vr2 = pd.read_excel(vr_data_xlsx)
    _vr2 = _vr2[(_vr2.mouse_id == m) & (_vr2.do_proc == 1)].reset_index(drop=True)
    _names = list(_vr2.dset_name)
    dsets_of[m] = [(nd, _names[nd]) for nd in f_resolve_dsets(DSETS, _vr2, m)]
target = [(m, nm, s, fs) for m in mice for (nd, nm) in dsets_of[m]
          for s in sources for fs in feat_sets]

store = ts.f_tuning_scan_load(store_path)
if store is None:
    store = ts.f_new_store(scan_cfg, target)
else:
    store['target'] = [tuple(t) for t in target]      # refresh planned grid (widen freely)
    store.setdefault('config', scan_cfg)
prog0 = ts.f_tuning_scan_progress(store_path)
print(f'store: {store_path}')
print(f'planned {len(target)} combos across {sum(len(v) for v in dsets_of.values())} datasets; '
      f'already done: {prog0.get("n_done", 0)}')


#%% SECTION 4 · Run the sweep — loop mice × datasets, save after every combo
# Skips whole datasets whose combos are all done; else loads features ONCE (cache, no render),
# extracts ensembles ONCE if needed, then tunes each (source, feat_set) and saves atomically.
_n_combo = len(sources) * len(feat_sets)
for m in mice:
    src_m = f_default_source(mouse_id=m)
    if feat_pca_variant:                              # load gaborS/mengS* / pix from the pca_feat cache
        src_m = {**src_m, 'cache_tag': f"{src_m['cache_tag']}_{feat_pca_variant}"}
    for nd, dset_name in dsets_of[m]:
        combos = [(s, fs) for s in sources for fs in feat_sets
                  if ts._tun_key(m, dset_name, s, fs) not in store['results']]
        if not combos:
            print(f'  {m} d{nd} {dset_name[:22]:22s} — all {_n_combo} combos done, skip')
            continue
        try:
            session = f_load_session(m, nd, data_dir_of(m), data_dir_bh, vr_data_xlsx, verbose=False)
            prep = ts.f_encode_prepare(session, blocks, sigma_frames=sigma_frames,
                                       plan=feature_plan, build_cfg=build_cfg, src=src_m,
                                       whiten=feat_whiten, combine={}, verbose=True)
            lags = f_lags_from_ms(list(lags_ms), prep['fs'])
            # per-source (Y_units, labels): cells → true est1 ids; ensembles → ensemble index
            targets = {'cells': (prep['Y_enc'], prep['cell_ids'])}
            if any(s == 'ensembles' for s, _ in combos):
                ens_res = enspipe.f_ensembles_cached(session['est1'], ens_cfg, feat_root + m, m, dset_name,
                                                     fs=prep['fs'], dset_tag=f'{m} d{nd}', verbose=False)
                ens_table = ens_res['ens_table']
                n_ens = len(ens_res['ens_out']['cells']['ens_list'])
                _tg = enspipe.f_ens_targets(ens_res)
                _key = 'ens_all' if ens_which == 'all' else 'ens_sig'
                if _key in _tg:                        # trim to the tuning clock
                    Y_ens, _lab = _tg[_key]
                    targets['ensembles'] = (np.asarray(Y_ens[:prep['rec']['T']], float), np.asarray(_lab))
                prep['rec']['ens'] = ts.f_ensemble_record(ens_table, n_ens)
                prep['rec']['ens_params'] = {'tag': ens_res['tag'], 'cfg': ens_res['cfg'], 'which': ens_which}
                print(f'  {m} d{nd} {dset_name[:22]:22s} n_act={prep["n_active"]:>4} '
                      f'T={prep["rec"]["T"]:>6} ens={n_ens} '
                      f'(sig {int(prep["rec"]["ens"]["sig"].sum())}) — {len(combos)} combos')
            else:
                print(f'  {m} d{nd} {dset_name[:22]:22s} n_act={prep["n_active"]:>4} '
                      f'T={prep["rec"]["T"]:>6} — {len(combos)} combos')
            # per-dataset monitor-average maps for the visual feat_sets (context; stored once)
            prep['rec']['monitor_maps'] = {
                fs: ts.f_block_monitor_maps(prep['built_blocks'], fs, feat_sets[fs])
                for fs in feat_sets if fs in ts._VISUAL_FEAT}
            ts.f_tuning_store_set_dataset(store, prep['rec'])
            for s, fs in combos:
                Y_src, labels = targets[s]
                res = ts.f_tuning_one(prep['built_blocks'], fs, feat_sets[fs], Y_src, labels, lags,
                                      method=tune_method, fs=prep['fs'], alpha=tune_alpha)
                ts.f_tuning_store_put(store, m, nd, dset_name, s, fs, res)
                ts._save_atomic(store_path, store)        # crash-safe: persist after every combo
                _n_pref = int(np.isfinite(res['peak_abs']).sum())
                print(f'      {s:9s} {fs:9s}  {res["resp"].shape[0]:>3} filters  '
                      f'{_n_pref:>4}/{len(labels)} units  med|peak|={np.nanmedian(res["peak_abs"]):.3g}')
            del session, prep
        except Exception as e:
            import traceback
            print(f'  {m} d{nd} {dset_name[:22]:22s} SKIPPED — {type(e).__name__}: {e}')
            traceback.print_exc()


#%% SECTION 5 · Progress + quick per-source/feat_set summary
prog = ts.f_tuning_scan_progress(store)
print(f'\n{TUN_TAG}: {prog["n_done"]}/{prog["n_target"]} combos done, {prog["n_remaining"]} remaining')
print(f'  datasets with results: {len(prog["datasets"])}')
print(f'  by source: {prog.get("by_source", {})}')
print(f'  by feat_set: {prog["by_feat"]}')

tbl = ts.f_tuning_scan_table(store)
if not tbl.empty:
    with pd.option_context('display.width', 200):
        print('\nmedian |peak| tuning by source × feat_set:')
        print(tbl.groupby(['source', 'feat_set'])['peak_abs'].median().unstack().round(3).to_string())
        print('\nmedian preferred orientation (visual feat_sets):')
        _v = tbl[tbl['feat_set'].isin(list(ts._VISUAL_FEAT))]
        if not _v.empty:
            print(_v.groupby(['source', 'feat_set'])['pref_odeg'].median().unstack().round(1).to_string())
