# -*- coding: utf-8 -*-
"""
VR_tuning_scan_plots.py — SCRIPT 2: read + plot the cross-dataset feature-TUNING store.

Loads the resumable store written by VR_tuning_scan.py (safe mid-run — atomic snapshots) and plots:
  - population orientation × temporal-freq response matrix per visual feat_set (gabor / meng),
  - per-unit / per-ensemble tuning marginals + example response heatmaps,
  - monitor-average frames (the per-filter time-mean maps stored per dataset),
  - a cell ↔ ensemble tuning comparison,
  - (opt-in) significant cells (re-run the circshift null), and a widefield projection colored by
    preferred orientation / peak tuning (reuse the encode-scan registration path).

Reuses f_tuning_scan (store + f_result_to_readout / f_population_matrix) and f_encoding_plots
(f_plot_tuning_heatmap / f_plot_tuning_marginals_all / f_plot_tuning_monitor / f_plot_rf).
Spyder cell-based. Created 2026-07-17.
"""

import sys
import os
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import f_tuning_scan as ts
from f_encoding_plots import (f_plot_tuning_heatmap, f_plot_tuning_marginals_all, f_plot_tuning_monitor,
                              f_plot_rf)


#%% =========================================================================
# SECTION 1 · CONFIG + LOAD store + tidy table
# ==========================================================================
feat_root = 'F:/VR/features/'
TUN_TAG   = 'tun_7_17_26'

store_path = ts.f_tuning_scan_path(feat_root, TUN_TAG)
store = ts.f_tuning_scan_load(store_path)
if store is None:
    raise FileNotFoundError(f'no tuning-scan store at {store_path} — run VR_tuning_scan.py first')
prog = ts.f_tuning_scan_progress(store)
print(f'{TUN_TAG}: {prog["n_done"]}/{prog["n_target"]} combos, '
      f'{len(prog["datasets"])} datasets; by feat_set {prog["by_feat"]}; by source {prog["by_source"]}')

tbl = ts.f_tuning_scan_table(store)
print(f'table: {len(tbl)} unit rows')
if not tbl.empty:
    print(tbl.groupby(['source', 'feat_set'])['peak_abs'].median().unstack().round(3).to_string())

# results are keyed by ts._tun_key(mouse, dset, source, feat_set); helper to fetch one
def get_result(mouse, dset_name, source, feat_set):
    return store['results'].get(ts._tun_key(mouse, dset_name, source, feat_set))


#%% SECTION 2 · POPULATION orientation × temporal-freq matrix per visual feat_set
# Mean |peak tuning| per (orientation, tf) cell, pooled over all units of a source across datasets.
if 1:
    src_show   = 'cells'                       # 'cells' | 'ensembles'
    feat_show  = 'meng'                         # 'meng' (orientation×tf) | 'gabor' (orientation only)
    res_list = [r for r in store['results'].values()
                if r['source'] == src_show and r['feat_set'] == feat_show]
    if not res_list:
        print(f'no results for {src_show}/{feat_show}')
    else:
        # accumulate the per-filter population magnitude across datasets on a shared (odeg, tf) grid
        Ms, odegs, tfs = [], None, None
        for r in res_list:
            M, od, tf = ts.f_population_matrix(r)
            Ms.append(M); odegs, tfs = od, tf
        Mpop = np.nanmean(np.stack(Ms, axis=0), axis=0)      # (n_odeg, n_tf)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(Mpop, cmap='magma', aspect='auto', origin='lower')
        ax.set_xticks(range(len(tfs))); ax.set_xticklabels([f'{t:g}' for t in tfs])
        ax.set_yticks(range(len(odegs))); ax.set_yticklabels([f'{o:g}' for o in odegs])
        ax.set_xlabel('temporal freq (Hz)'); ax.set_ylabel('orientation (deg)')
        ax.set_title(f'{feat_show} population tuning  ({src_show}, {len(res_list)} datasets)')
        fig.colorbar(im, ax=ax, label='mean |peak| response'); fig.tight_layout()


#%% SECTION 3 · Per-unit / per-ensemble tuning marginals (one line per unit + pop mean)
if 1:
    mouse, dset_name = store['results'][next(iter(store['results']))]['mouse'], None
    src_m, feat_m = 'cells', 'meng'
    # pick the first dataset that has this combo
    for r in store['results'].values():
        if r['source'] == src_m and r['feat_set'] == feat_m:
            dset_name = r['dset_name']; mouse = r['mouse']; break
    r = get_result(mouse, dset_name, src_m, feat_m)
    if r is not None:
        readout = ts.f_result_to_readout(r)
        f_plot_tuning_marginals_all(readout, dset_tag=f'{mouse} {dset_name} {src_m}/{feat_m}',
                                    normalize=True)
        f_plot_tuning_marginals_all(readout, dset_tag=f'{mouse} {dset_name} {src_m}/{feat_m} (raw)',
                                    normalize=False)


#%% SECTION 4 · Example unit response heatmap (orientation × tf) for a few strong units
if 1:
    src_h, feat_h = 'cells', 'meng'
    r = None
    for _r in store['results'].values():
        if _r['source'] == src_h and _r['feat_set'] == feat_h:
            r = _r; break
    if r is not None:
        readout = ts.f_result_to_readout(r)
        order = np.argsort(-np.asarray(r['peak_abs']))       # strongest-tuned first
        for u in order[:4]:
            f_plot_tuning_heatmap(readout, int(u),
                                  dset_tag=f'{r["mouse"]} {r["dset_name"]} unit {r["labels"][u]}',
                                  axis='tf')                 # orientation × temporal-freq


#%% SECTION 5 · Monitor-average frames (per-filter time-mean maps stored per dataset)
if 1:
    feat_mon = 'meng'
    # find a dataset record that has monitor maps for this feat_set
    for dk, rec in store['datasets'].items():
        mm = (rec.get('monitor_maps') or {}).get(feat_mon)
        if mm:
            keys = sorted(mm)
            print(f'{rec["dset_name"]}: {len(keys)} {feat_mon} filter maps')
            for k in keys[:3]:                               # a few example filters (L‖R wide map)
                f_plot_tuning_monitor(mm[k], dset_tag=rec['dset_name'], title=k, n_monitors=2)
            break


#%% SECTION 6 · Cell ↔ ensemble tuning comparison (median |peak| + preferred orientation)
if 1 and not tbl.empty:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, met, lab in zip(axes, ['peak_abs', 'osi'], ['median |peak| response', 'orientation OSI']):
        piv = tbl[tbl['feat_set'].isin(list(ts._VISUAL_FEAT))].groupby(
            ['feat_set', 'source'])[met].median().unstack()
        piv.plot.bar(ax=ax); ax.set_ylabel(lab); ax.set_title(lab); ax.set_xlabel('feat_set')
    fig.suptitle(f'{TUN_TAG} — cell vs ensemble tuning'); fig.tight_layout()


#%% SECTION 7 · (opt-in) SIGNIFICANT cells — re-run the circshift null for one dataset
# Heavier: reloads the session + features (cache, NO render) and re-fits to run f_tuning_significance,
# then re-plots the significant subset. Set the target dataset + feat_set.
if 0:
    from f_pipeline import f_load_session
    from f_feature_source import f_default_feature_plan, f_default_build_cfg, f_default_source
    from f_encoding import f_lags_from_ms
    from f_tuning import f_tuning_significance

    mouse, n_dset = 'L', 0
    feat_sig, src_sig = 'meng', 'cells'
    lags_ms = store['config'].get('lags_ms', [0, 100, 200, 300, 400, 500])
    n_shuffle, alpha = 200, 0.05
    variant = store['config'].get('feat_pca_variant', 'pcasep')

    sess = f_load_session(mouse, n_dset, 'F:/VR/data_proc/' + mouse, 'F:/VR/Bh_data/mice_gcamp/',
                          'F:/VR/data_proc/VR_data.xlsx', verbose=False)
    src_m = f_default_source(mouse_id=mouse)
    src_m = {**src_m, 'cache_tag': f"{src_m['cache_tag']}_{variant}"}
    blocks = ts.DEFAULT_FEAT_SETS[feat_sig]
    prep = ts.f_encode_prepare(sess, blocks, sigma_frames=store['config'].get('sigma_frames', 6),
                               plan=f_default_feature_plan(), build_cfg=f_default_build_cfg(side='both'),
                               src=src_m, whiten=False, combine={}, verbose=True)
    lags = f_lags_from_ms(list(lags_ms), prep['fs'])
    X_stim, filt_df = ts.f_block_regressors(prep['built_blocks'], feat_sig, blocks)
    sig = f_tuning_significance(X_stim[:prep['rec']['T']], filt_df, prep['Y_enc'], lags,
                                method='sta', n_shuffle=n_shuffle, alpha=alpha,
                                labels=list(prep['cell_ids']))
    print(f'{feat_sig}: {int(sig["sig"].sum())}/{len(sig)} cells significant (FDR<{alpha})')


#%% SECTION 8 · (opt-in) widefield projection colored by preferred orientation / peak tuning
# Reuse the encode-scan registration path: reload est1 → footprint centroids → widefield coords, then
# color CELLS by pref_odeg or peak_abs for a chosen feat_set. Ensembles have no footprint (skipped).
if 0:
    from f_pipeline import f_load_session
    vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
    color_by, feat_fov = 'pref_odeg', 'meng'
    sub = tbl[(tbl['source'] == 'cells') & (tbl['feat_set'] == feat_fov)]
    for dk, rec in store['datasets'].items():
        mouse, dset_name = rec['mouse'], rec['dset_name']
        g = sub[(sub['mouse'] == mouse) & (sub['dset_name'] == dset_name)]
        if g.empty:
            continue
        sess = f_load_session(mouse, rec['n_dset'], 'F:/VR/data_proc/' + mouse,
                              'F:/VR/Bh_data/mice_gcamp/', vr_data_xlsx, behavior=False, verbose=False)
        est1 = sess['est1']
        if int(est1['S'].shape[0]) != rec['n_cells_total'] or est1['dset_name'] != dset_name:
            print(f'  {dset_name}: identity mismatch — skip'); continue
        cell_ids = np.asarray(rec['cell_ids'], int)
        try:
            lm = ts.f_load_fov_landmarks(vr_data_xlsx, mouse, dset_name)
            wf, cent, area = ts.f_neuron_widefield_coords(est1, cell_ids, lm['tp_locs'], lm['wf_locs'])
        except Exception as e:
            print(f'  {dset_name}: no widefield ({e})'); continue
        score = g.set_index('unit').reindex(cell_ids)[color_by].to_numpy()
        ts.f_plot_neuron_scores_fov(wf, score, dset_tag=f'{mouse} {dset_name}',
                                    title=f'{feat_fov} {color_by}', cmap='twilight')
        break
