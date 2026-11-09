# -*- coding: utf-8 -*-
"""
VR_dataset_scan.py — ACROSS-DATASET analyses (TODO #66).

Fast triage: loop mice × do_proc datasets, load calcium + cheap behavior
vectorization (NO render, NO feature cache), and score each session's
encode/decode quality against the cheap ground-truth stimulus (agg + self_mot).
Rank the table to pick which datasets are worth the full feature build + sweeps.

Compute in functions/f_dataset_scan.py. Spyder cells (#%%). This is the shared
home for future cross-dataset encode/decode comparisons (mirror of the per-dataset
scripts, but loads/loops only).
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import os
_thread_cap = 6                     # cap BLAS threads before numpy import
if _thread_cap:
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS'):
        os.environ[_v] = str(_thread_cap)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn
patch_sklearn()
warnings.filterwarnings('ignore', message=r'Some metric_kwargs have been passed',
                        category=UserWarning)

from f_pipeline import f_load_session
from f_dataset_scan import f_scan_dataset, f_count_dsets, f_plot_scan, f_scan_dataset_params
from f_feature_source import f_default_feature_plan, f_default_build_cfg, f_default_source
from f_decoding import f_default_decoder_params, f_plot_decoder_sweep
from f_encoding import f_default_encoder_params

#%% Config — mice, paths, scan knobs
mice        = ['L']                 # one or more mouse ids to scan
data_dir_bh = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
data_dir_of = lambda m: 'F:/VR/data_proc/' + m       # calcium proc dir per mouse

# which datasets per mouse: None = all do_proc, or an explicit list of n_dset indices
dsets_of    = {}                    # e.g. {'L': [0, 2, 4]}; missing mouse → all

scan_kw = dict(
    sigma_frames = 6,               # ~100 ms neural smoothing (60 Hz)
    n_pcs        = 30,              # neural PCs for the decode
    lags_ms      = (0, 100, 200),  # encode FIR window
    embargo_sec  = 0.5,
    n_folds      = 5,
    dec_targets  = ('self_mot', 'agg'),   # neural → these (decode)
    enc_targets  = ('agg', 'self_mot'),   # these → neural (encode)
    r_sig        = 0.1,            # encode "significant" per-cell r threshold
    self_motion_decomp = True,     # fwd/lat/yaw (P2a-9)
    obj_collapse = 'salience_mean',
    obj_size     = {'x': 5, 'y': 20, 'z': 5, 'height': 2},   # for salience_mean agg
)

#%% Run the scan — loop mice × datasets (skip + log any that fail)
rows = []
for m in mice:
    n_avail = f_count_dsets(m, vr_data_xlsx)
    dsets = dsets_of.get(m) or list(range(n_avail))
    print(f'\n=== mouse {m}: {n_avail} do_proc datasets, scanning {list(dsets)} ===')
    for nd in dsets:
        try:
            session = f_load_session(m, nd, data_dir_of(m), data_dir_bh, vr_data_xlsx,
                                     verbose=False)
            row = f_scan_dataset(session, **scan_kw)
            rows.append(row)
            print(f"  d{nd:>2} {row['dset_name'][:28]:28s}  n={row['n_cells']:>4} "
                  f"dec_selfmot={row['dec_self_mot_r2']:+.3f} "
                  f"dec_agg={row['dec_agg_r2']:+.3f} enc_r={row['enc_r_med']:+.3f}")
            del session                                 # free calcium before next
        except Exception as e:
            print(f'  d{nd:>2} SKIPPED — {type(e).__name__}: {e}')

scan_df = pd.DataFrame(rows)
print('\nscan complete:', scan_df.shape[0], 'datasets')

#%% Ranked table + bar plots
if len(scan_df):
    sort_by = 'dec_self_mot_r2'     # 'dec_agg_r2' | 'enc_r_med' | 'part_ratio' | ...
    cols = ['mouse', 'n_dset', 'dset_name', 'n_cells', 'T', 'part_ratio',
            'agg_coverage', 'dec_self_mot_r2', 'dec_agg_r2', 'dec_agg_validfrac',
            'enc_r_med', 'enc_frac_sig']
    cols = [c for c in cols if c in scan_df.columns]
    ranked = scan_df[cols].sort_values(sort_by, ascending=False).reset_index(drop=True)
    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', lambda v: f'{v:+.3f}'):
        print(ranked)
    f_plot_scan(scan_df, sort_by=sort_by)

#%% (optional) save the scan table
if 0:
    out_csv = 'F:/VR/scan/dataset_scan.csv'
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(out_csv, index=False)
    print('saved', out_csv)


#%% FULL-PARAM cross-dataset scan — the SAME dec/enc structs as VR_ca_cebra, dataset = axis
# =============================================================================
# SECTION 2 · FULL-PARAM SCAN — run the exact f_run_decoder / f_run_encoder used in the
#             per-dataset script across datasets, on the real feature blocks (loaded from
#             cache / conv_repo per dataset). Dataset is the sweep axis.
# =============================================================================
# Edit dec_params / enc_params to MATCH the cebra script (same feature_block, decoder, n_pcs,
# lags, fold_scheme, scores). Features are LOADED (not built) per dataset via f_resolve_features
# under the plan; a dataset with no cache/repo for the needed block is skipped + logged.
dec_params = f_default_decoder_params(
    input_source = 'neural',       # keep 'neural' — no CEBRA models are fit in the scan
    n_input_pcs  = 50,
    target_block = 'mengS',        # the feature under test (must exist in each dataset's cache/repo)
    decoder      = 'ridge',
    fold_scheme  = 'interleaved',
    scores       = ['r2', 'r2_vw', 'rv', 'procrustes', 'rsa'],
)
enc_params = f_default_encoder_params(
    feature_block    = 'mengS',
    covariate_blocks = [],         # e.g. ['self_mot'] to regress in locomotion
    decoder          = 'ridge',
    fold_scheme      = 'interleaved',
    scores           = ['r_med', 'frac_sig', 'r2_vw', 'rv', 'procrustes', 'rsa'],
    # pinned: this scan's settings before the defaults became cebra's (2026-11-06)
    n_feature_pcs    = None, feature_pca = 'per_block', ridge_alpha = 1.0, l1_ratio = 0.5,
    lags_ms          = [0, 100, 200, 300],
)
# Load-only: force the scanned visual blocks to the conv_repo so a dataset missing it RAISES →
# the loop skips + logs it (instead of silently rebuilding/rendering). Set _load_origin='cache'
# if you generated joblib caches instead of repos; 'auto' to allow build-on-miss.
_load_origin = 'repo'
_repo_kinds  = {'gaborS', 'mengS', 'gabor', 'meng', 'grid', 'flow'}
_need = {dec_params['target_block'], enc_params['feature_block'], *enc_params['covariate_blocks']}
feature_plan  = f_default_feature_plan(**{b: {'origin': _load_origin}
                                          for b in _need if b in _repo_kinds})
build_cfg     = f_default_build_cfg(side='R')            # only used if a block must be built
scan_sigma_frames = 6                                    # matches the cebra Y_neu smoothing
run_decode, run_encode = True, True

#%% Run the full-param scan across datasets
rows_full = []
for m in mice:
    src_m = f_default_source(mouse_id=m)                 # per-mouse features_dir / tags
    n_avail = f_count_dsets(m, vr_data_xlsx)
    dsets = dsets_of.get(m) or list(range(n_avail))
    print(f'\n=== [full] mouse {m}: scanning {list(dsets)} (dec={dec_params["target_block"]!r}, '
          f'enc={enc_params["feature_block"]!r}) ===')
    for nd in dsets:
        try:
            session = f_load_session(m, nd, data_dir_of(m), data_dir_bh, vr_data_xlsx,
                                     verbose=False)
            r = f_scan_dataset_params(session, dec_params, enc_params, plan=feature_plan,
                                      build_cfg=build_cfg, source=src_m,
                                      sigma_frames=scan_sigma_frames,
                                      decode=run_decode, encode=run_encode)
            rows_full.append(r)
            _d = '  '.join(f'{k}={v:+.2f}' for k, v in {**r['dec_scores'], **r['enc_scores']}.items())
            print(f"  d{nd:>2} {r['dset_name'][:24]:24s} n={r['n_cells']:>4}  {_d}")
            del session
        except Exception as e:
            print(f'  d{nd:>2} SKIPPED — {type(e).__name__}: {e}')
print(f'\nfull-param scan complete: {len(rows_full)} datasets')

#%% Plot the full-param scan — one panel per score, a bar per dataset (decode + encode figs)
if len(rows_full):
    _tag = f'{"+".join(mice)} scan'
    dec_rows = [{'x': r['dset_name'], **r['dec_scores']} for r in rows_full if r['dec_scores']]
    if dec_rows:
        f_plot_decoder_sweep(dec_rows, 'dataset', dec_params['scores'], dset_tag=_tag,
                             config=dec_params)
    enc_rows = [{'x': r['dset_name'], **r['enc_scores']} for r in rows_full if r['enc_scores']]
    if enc_rows:
        f_plot_decoder_sweep(enc_rows, 'dataset', enc_params['scores'], dset_tag=_tag,
                             config=enc_params)
    # tidy table (flatten dec_/enc_ score dicts)
    full_df = pd.DataFrame([{'mouse': r['mouse'], 'n_dset': r['n_dset'],
                             'dset_name': r['dset_name'], 'n_cells': r['n_cells'], 'T': r['T'],
                             **{f'dec_{k}': v for k, v in r['dec_scores'].items()},
                             **{f'enc_{k}': v for k, v in r['enc_scores'].items()}}
                            for r in rows_full])
    with pd.option_context('display.max_rows', None, 'display.width', 220,
                           'display.float_format', lambda v: f'{v:+.3f}'):
        print(full_df)
