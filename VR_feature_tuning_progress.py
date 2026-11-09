# -*- coding: utf-8 -*-
"""
VR_feature_tuning_progress.py — watch / analyze a running (or finished) feature-tuning SCAN file.

Point `scan_file` at the same path VR_feature_tuning.py is writing. The scan file is rewritten
ATOMICALLY after every batch, so this script can read a consistent snapshot WHILE the scan runs
(a different Spyder / process). It reloads the embedded data bundle (dataset identity + cell traces
+ ensemble outputs — no session reload needed), so it can also SCORE the partial scan (tuning so
far) and plot. Compute in f_feature_scan / f_tuning; plots in f_encoding_plots.

Spyder cell-based. Created 2026-07-16.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import f_feature_scan as fscan
import f_feature_source as fsrc
import f_tuning as tun
import f_encoding_plots as encp


#%% Point at the scan file — SAME tag + resolution as VR_feature_tuning.py (builds the same path)
mouse_id         = 'L'
dset_name        = '11_2_25'    # EDIT: the dataset (est1['dset_name']) being scanned
scan_tag         = 'gen_7_16_26'    # must match the scan script's scan_tag
render_lat_samp  = 150
render_vert_samp = 101
scan_sample_ds   = None
src = fsrc.f_default_source(mouse_id=mouse_id)
scan_file = fscan.f_scan_path(src['features_dir'], dset_name, scan_tag,
                              render_lat_samp, render_vert_samp, scan_sample_ds)
print('scan file:', scan_file)


#%% PROGRESS — how many filters done + coverage grid (neural-free; safe mid-scan)
prog = fscan.f_scan_progress(scan_file)
if not prog['exists']:
    print(f'no scan file yet at {scan_file}')
else:
    _rem = (f", {prog['n_remaining']}/{prog['n_target']} pending" if 'n_target' in prog else '')
    print(f"scan {prog.get('dset_name')} [{prog.get('gen_tag')}]: {prog['n_done']} filters done{_rem} "
          f"(gabor {prog['by_kind']['gabor']}, meng {prog['by_kind']['meng']}); "
          f"T={prog['T']} render={prog.get('render')}; "
          f"neural={'yes' if prog['has_neural'] else 'no'} "
          f"ensembles={'yes' if prog['has_ensembles'] else 'no'}")
    # coverage: green=done, blank=pending vs the requested grid recorded in the scan file. (A scan
    # started before target-recording shows done-only/all-green — re-run the SCAN cell once to record it.)
    fscan.f_plot_scan_coverage(scan_file, dset_tag=prog.get('dset_name', ''))


#%% DATASET STATS — how the gabor/meng bank responds (neural-free; guides filter placement)
# magnitude across the param grid (SF / orientation / TF), magnitude vs frame (conv_repo-style),
# and time-mean spatial maps (where activity sits). All read from the scan file.
if 1:
    _tag = prog.get('dset_name', '')
    # per-filter magnitude table
    scan_stats = fscan.f_scan_stats(scan_file)
    with pd.option_context('display.max_rows', 40, 'display.width', 200):
        _c = [c for c in ['kind', 'odeg', 'sf_cyc_deg', 'tf_hz', 'mean', 'std', 'max', 'p5', 'p50', 'p95']
              if c in scan_stats.columns]
        print(scan_stats[_c].to_string(index=False))
    # magnitude vs SF / orientation / TF
    fscan.f_plot_scan_magnitude(scan_file, dset_tag=_tag)
    # magnitude vs frame, grouped by SF (gabor) and by TF (meng)
    fscan.f_plot_scan_traces(scan_file, by='sf_cyc_deg', kind='gabor', dset_tag=_tag)
    fscan.f_plot_scan_traces(scan_file, by='tf_hz', kind='meng', dset_tag=_tag)
    # time-mean spatial activity (where it sits), overall + per orientation (needs spatial_mean=True)
    try:
        fscan.f_plot_scan_spatial(scan_file, group_by='odeg', kind='gabor', dset_tag=_tag)
        fscan.f_plot_scan_spatial(scan_file, group_by='tf_hz', kind='meng', dset_tag=_tag)
    except ValueError as _e:
        print(f'  (spatial maps: {_e})')


#%% SCORE the partial scan (tuning so far) using the EMBEDDED neural + ensembles
# Uses the data bundle stored in the file — no session reload. Re-run anytime to refresh as the scan
# grows. (Requires the scan was written with store_neural=True.)
if 1:
    tuning_sources = ('cells', 'ensembles')
    tuning_methods = ('sta',)
    tuning_lags_ms = [0, 100, 200, 300, 400]
    tuning_ens_which = 'sig'

    X_stim, filt_df = fscan.f_scan_load(scan_file, standardize=True)
    data = fscan.f_scan_load_data(scan_file)
    if 'Y_neu_raw' not in data:
        raise RuntimeError('scan file has no embedded neural (store_neural was False) — '
                           'regenerate the neural/ensembles with the stored dimred_params + seed')
    tun_scan = fscan.f_score_scan(
        X_stim, filt_df, data['Y_neu_raw'], scorer='tuning', sources=tuning_sources,
        ens_out=data.get('ens_out'), ens_table=data.get('ens_table'),
        dimred_params=data.get('dimred_params'), ens_which=tuning_ens_which,
        fs=data.get('fs'), methods=tuning_methods, lags_ms=tuning_lags_ms)
    _tag = data.get('dset_tag', prog.get('dset_name', ''))
    for _src, _r in tun_scan.items():
        _rd = _r['readouts'][tuning_methods[-1]]
        _cols = [c for c in ['unit', 'pref_odeg', 'pref_sf_cyc_deg', 'pref_tf_hz', 'pref_wavelength',
                             'pref_tfreq', 'peak_abs', 'peak_lag_ms', 'osi'] if c in _rd['table'].columns]
        with pd.option_context('display.max_rows', 40, 'display.width', 200):
            print(f'\n{_tag} — {_src} ({_rd["table"].shape[0]} units, {X_stim.shape[1]} filters so far):')
            print(_rd['table'].sort_values('peak_abs', ascending=False)[_cols].to_string(index=False))
        encp.f_plot_tuning_population(_rd['table'], dset_tag=f'{_tag} [{_src}]', method=_rd.get('method', ''))
        encp.f_plot_tuning_marginals(_rd, u=None, dset_tag=f'{_tag} [{_src}]')
        if _r.get('source') == 'ensembles' or _rd['table'].shape[0] <= 20:  # per-unit overlay (one line/ensemble)
            for _nrm in (True, False):                               # peak-normalized shape + raw |tuning|
                encp.f_plot_tuning_marginals_all(_rd, dset_tag=f'{_tag} [{_src}]', normalize=_nrm)


#%% RELOAD saved tuning results (the fitted ridge/STA W response + params) and re-plot — NO recompute
# VR_feature_tuning.py saves <..>_tuning.joblib alongside the scan file. Reload it here to re-plot the
# stored response frames and inspect the params that produced them, without rerunning the scan/scorer.
if 1:
    # plt.close('all')
    tuning_results_file = scan_file.replace('_scan.joblib', '_tuning.joblib')
    tun_saved, saved_params = fscan.f_load_scan_results(tuning_results_file)
    print('saved params:', saved_params)
    _method = saved_params.get('methods', ['sta'])[-1]
    for _src, _r in tun_saved.items():
        _rd = _r['readouts'][_method]                 # full readout: W (n_filter,n_lags,n_units), tables, marginals
        _tag = f"{saved_params.get('dset_tag', '')} [{_src}]"
        encp.f_plot_tuning_population(_rd['table'], dset_tag=_tag, method=_method)
        encp.f_plot_tuning_marginals(_rd, u=None, dset_tag=_tag)
        encp.f_plot_tuning_lags(_rd, dset_tag=_tag)
        _top = _rd['table'].sort_values('peak_abs', ascending=False).index.to_numpy()[:3]
        _has_tf = _rd['by_tfreq'] is not None and np.size(_rd['by_tfreq'])
        for _u in _top:
            encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=_tag, zscore=True, axis='sf')
            if _has_tf:                                   # meng: also orientation × temporal-freq
                encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=_tag, zscore=True, axis='tf')
