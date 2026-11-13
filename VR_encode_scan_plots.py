# -*- coding: utf-8 -*-
"""
VR_encode_scan_plots.py — SCRIPT 2: open the encoding-sweep store + plot neuron scores vs LOCATION.

Loads the tag-versioned store VR_encode_scan.py wrote (per-neuron R² per feature block × model, keyed
by dataset + neuron id — NO locations). For each dataset it RELOADS est1 (calcium only, fast), VERIFIES
identity against the stored record, computes footprint centroids, and maps them onto the mouse's common
WIDEFIELD map via the VR_data.xlsx 4-corner landmarks (tp_fov_locs→wf_locs). Then plots neuron R² as a
function of 2D location — per (block, model), overlaid across a mouse's datasets — plus R² distributions
and block/model comparisons. Compute in functions/f_encode_scan.py.

Safe to run against a store that is still being written (atomic snapshots). Spyder cell-based. 2026-07-17.
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

from f_pipeline import f_load_session
import f_encode_scan as es

try:
    import tifffile as tiff
except Exception:
    tiff = None


#%% =========================================================================
# SECTION 1 · SETUP — point at the store (same ENC_TAG + feat_root as script 1)
# ==========================================================================
feat_root    = 'F:/VR/features/'
ENC_TAG      = 'enc_11_12_26'        # old: 'enc_7_18_26' / 'enc_11_6_26' (gaborS+mengS blocks)
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
data_dir_of  = lambda m: 'F:/VR/data_proc/' + m
wf_dir       = 'F:/VR/data_proc/2p_fov_reg/'      # widefield images referenced by VR_data.xlsx wf_im

store_path = es.f_encode_scan_path(feat_root, ENC_TAG)
print('store:', store_path)


#%% SECTION 2 · Load store → tidy per-neuron score table
store = es.f_encode_scan_load(store_path)
if store is None:
    raise FileNotFoundError(f'no store at {store_path} — run VR_encode_scan.py first')
prog = es.f_encode_scan_progress(store_path)
print(f'{ENC_TAG}: {prog["n_done"]}/{prog["n_target"]} combos, datasets {prog["datasets"]}')
print(f'  by block {prog["by_block"]}  |  by model {prog["by_model"]}')

score_tbl = es.f_encode_scan_table(store)                 # mouse/dset/source/unit/block/model/r2/r/keep/sig
blocks_present = sorted(score_tbl['block'].unique())
models_present = sorted(score_tbl['model'].unique())
sources_present = sorted(score_tbl['source'].unique())
print('sources:', sources_present, '| blocks:', blocks_present, '| models:', models_present)


#%% SECTION 3 · Attach LOCATIONS — reload est1 per dataset, verify, register to widefield
# For each (mouse, dset) with results: reload calcium (fast), CHECK n_cells_total + dset_name match the
# stored record, compute footprint centroids for the stored cell_ids, and map to widefield coords via
# the 4-corner landmarks. Builds coords_df keyed by (mouse, dset_name, source='cells', unit=cell id).
# Locations apply to CELLS only (ensembles have no footprint). Datasets missing landmarks keep
# session-pixel coords only (wf_x/wf_y NaN).
loc_rows = []
wf_images = {}                                            # (mouse, wf_im) -> image array
for dkey, rec in store['datasets'].items():
    m, dset_name, nd = rec['mouse'], rec['dset_name'], rec['n_dset']
    cell_ids = np.asarray(rec['cell_ids'], int)
    try:
        session = f_load_session(m, nd, data_dir_of(m), data_dir_bh, vr_data_xlsx,
                                 behavior=False, verbose=False)
        est1 = session['est1']
        n_tot = int(np.asarray(est1['S']).shape[0])
        if n_tot != rec['n_cells_total'] or est1['dset_name'] != dset_name:
            print(f'  {m}/{dset_name}: IDENTITY MISMATCH (n_cells {n_tot} vs {rec["n_cells_total"]}, '
                  f'name {est1["dset_name"]} vs {dset_name}) — skipping locations')
            del session
            continue
        cent, area = es.f_footprint_centroids(est1['A'], est1['dims'], cell_ids)
        wf_xy = np.full_like(cent, np.nan)
        try:                                              # register to widefield (if landmarks exist)
            lm = es.f_load_fov_landmarks(vr_data_xlsx, m, dset_name)
            wf_xy, _, _ = es.f_neuron_widefield_coords(est1, cell_ids, lm['tp_locs'], lm['wf_locs'])
            if tiff is not None and lm['wf_im'] and (m, lm['wf_im']) not in wf_images:
                _p = os.path.join(wf_dir, lm['wf_im'])
                wf_images[(m, lm['wf_im'])] = tiff.imread(_p) if os.path.exists(_p) else None
            rec['wf_im'] = lm['wf_im']
        except Exception as _e:
            print(f'  {m}/{dset_name}: no widefield registration ({type(_e).__name__}: {_e})')
        for i, cid in enumerate(cell_ids):
            loc_rows.append({'mouse': m, 'dset_name': dset_name, 'source': 'cells', 'unit': int(cid),
                             'x': cent[i, 0], 'y': cent[i, 1], 'area': area[i],
                             'wf_x': wf_xy[i, 0], 'wf_y': wf_xy[i, 1]})
        del session
        print(f'  {m}/{dset_name}: {len(cell_ids)} neurons located'
              + ('  (+widefield)' if np.isfinite(wf_xy).any() else '  (px only)'))
    except Exception as e:
        print(f'  {m}/{dset_name}: SKIPPED — {type(e).__name__}: {e}')

coords_df = pd.DataFrame(loc_rows)
full = score_tbl.merge(coords_df, on=['mouse', 'dset_name', 'source', 'unit'], how='left')
print(f'\njoined table: {full.shape[0]} rows  (cell rows with widefield coords: '
      f'{int(full["wf_x"].notna().sum())})')


#%% SECTION 4 · Spatial map — CELL R² on the widefield map, per (block, model), across datasets
# Pick a mouse + block + model; overlay all that mouse's datasets on its common widefield frame.
# (Locations are cells only — ensembles have no footprint; use SECTION 7 to compare ensemble R².)
if 1:
    plot_mouse = 'L'
    plot_block = 'pix'          # any of blocks_present
    plot_model = 'ridge'                 # any of models_present
    use_widefield = True                 # False → session pixel coords (not cross-dataset aligned)
    r2_clip = (None, None)                # color scale (vmin, vmax); None = auto
    dot_size = 6                         # scatter marker size (was 18)

    sub = full[(full['source'] == 'cells') & (full['mouse'] == plot_mouse)
               & (full['block'] == plot_block) & (full['model'] == plot_model)].copy()
    xcol, ycol = ('wf_x', 'wf_y') if use_widefield else ('x', 'y')
    coords = sub[[xcol, ycol]].to_numpy()
    scores = sub['r2'].to_numpy()
    # widefield background (first available image for the mouse)
    _wf = next((im for (mm, _fn), im in wf_images.items() if mm == plot_mouse and im is not None), None)
    es.f_plot_neuron_scores_fov(coords, scores, wf_im=(_wf if use_widefield else None),
                                dset_tag=f'{plot_mouse}', vmin=r2_clip[0], vmax=r2_clip[1], s=dot_size,
                                title=f'{plot_block}·{plot_model} R² (all datasets)')


#%% SECTION 4b · Spatial FIELD — smooth topographic R² map (Gaussian-convolved) on the widefield
# Same selection as Section 4, but instead of discrete dots the per-cell R² is convolved into a
# continuous weighted-mean field (Nadaraya-Watson) over widefield space, with topo contours. Reads
# like a gradient/activity map rather than a scatter. sparsely-covered regions are masked out.
if 1:
    plot_mouse = 'L'
    plot_block = 'pix'          # any of blocks_present
    plot_model = 'ridge'                 # any of models_present
    use_widefield = True                 # False → session pixel coords (not cross-dataset aligned)
    r2_clip = (None, None)               # color scale (vmin, vmax); None = auto
    field_sigma = 25.0                   # smoothing bandwidth in widefield pixels (bigger = smoother)
    density_floor = 0.05                 # mask cells with < this fraction of peak coverage
    field_alpha = 0.85                   # field opacity over the background
    n_contours = 6                       # topo contour lines (0 = none)
    show_dots = False                    # overlay the raw cell dots on top of the field

    sub = full[(full['source'] == 'cells') & (full['mouse'] == plot_mouse)
               & (full['block'] == plot_block) & (full['model'] == plot_model)].copy()
    xcol, ycol = ('wf_x', 'wf_y') if use_widefield else ('x', 'y')
    coords = sub[[xcol, ycol]].to_numpy()
    scores = sub['r2'].to_numpy()
    _wf = next((im for (mm, _fn), im in wf_images.items() if mm == plot_mouse and im is not None), None)
    es.f_plot_neuron_field_fov(coords, scores, wf_im=(_wf if use_widefield else None),
                               dset_tag=f'{plot_mouse}', vmin=r2_clip[0], vmax=r2_clip[1],
                               sigma=field_sigma, density_floor=density_floor, field_alpha=field_alpha,
                               contours=n_contours, show_dots=show_dots,
                               title=f'{plot_block}·{plot_model} R² field (all datasets)')


#%% SECTION 5 · Per-dataset small multiples — one widefield panel per dataset (chosen block/model)
if 1:
    plot_mouse = 'L'; plot_block = 'gabmag+mengopp'; plot_model = 'ridge'
    dot_size = 6                         # scatter marker size (was 10)
    dsets = sorted(full.loc[full['mouse'] == plot_mouse, 'dset_name'].unique())
    n = len(dsets)
    if n:
        ncol = int(np.ceil(np.sqrt(n))); nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
        _all = full[(full['source'] == 'cells') & (full['mouse'] == plot_mouse)
                    & (full['block'] == plot_block) & (full['model'] == plot_model)]
        _vmax = float(np.nanpercentile(_all['r2'], 98)) if _all['r2'].notna().any() else 1.0
        for ax, ds in zip(axs.ravel(), dsets):
            g = _all[_all['dset_name'] == ds]
            ok = g['wf_x'].notna() & g['r2'].notna()
            sc = ax.scatter(g.loc[ok, 'wf_x'], g.loc[ok, 'wf_y'], c=g.loc[ok, 'r2'], s=dot_size,
                            cmap='viridis', vmin=0, vmax=_vmax)
            ax.set_title(f'{ds[:18]}', fontsize=8); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axs.ravel()[n:]:
            ax.axis('off')
        fig.colorbar(sc, ax=axs, fraction=0.02, label='R²')
        fig.suptitle(f'{plot_mouse} — {plot_block}·{plot_model} R² per dataset (widefield)')


#%% SECTION 6 · CELL distributions + comparisons (block-vs-block, ridge-vs-cca)
if 1:
    cells = full[(full['mouse'] == 'L') & (full['source'] == 'cells')]
    es.f_plot_score_hist(cells, dset_tag='L cells')
    es.f_plot_block_model_compare(cells, x=('block', 'agg'),
                                  y=('block', 'gabmag+mengopp'), model='ridge', dset_tag='L cells')
    _b = 'gabmag+mengopp'
    _d = cells[cells['block'] == _b]
    _m = _d[_d['model'] == 'ridge'].merge(_d[_d['model'] == 'cca'],
                                          on=['mouse', 'dset_name', 'source', 'unit'],
                                          suffixes=('_ridge', '_cca'))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(_m['r2_ridge'], _m['r2_cca'], s=8, alpha=0.4)
    _lim = [min(0, np.nanmin(_m[['r2_ridge', 'r2_cca']].to_numpy())),
            np.nanmax(_m[['r2_ridge', 'r2_cca']].to_numpy())]
    ax.plot(_lim, _lim, 'k--', lw=0.6)
    ax.set_xlabel('ridge R²'); ax.set_ylabel('cca R²'); ax.set_title(f'L — {_b} cells: ridge vs cca', fontsize=9)
    fig.tight_layout()


#%% SECTION 7 · ENSEMBLES — significant-ensemble R² compared ACROSS datasets
# Ensembles have no location; compare their encoding R² across datasets. For a chosen (block, model),
# plot the per-ensemble R² grouped by dataset, SIGNIFICANT ensembles highlighted, + the sig-only median.
if 1:
    plot_mouse = 'L'; plot_block = 'gabmag+mengopp'; plot_model = 'ridge'
    ens = full[(full['source'] == 'ensembles') & (full['mouse'] == plot_mouse)
               & (full['block'] == plot_block) & (full['model'] == plot_model)].copy()
    if not ens.empty:
        dsets = sorted(ens['dset_name'].unique())
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(dsets)), 4))
        for xi, ds in enumerate(dsets):
            g = ens[ens['dset_name'] == ds]
            sig = g['sig'] == True                       # noqa: E712
            jit = (np.random.RandomState(xi).rand(len(g)) - 0.5) * 0.3
            ax.scatter(xi + jit[~sig.values], g.loc[~sig, 'r2'], s=18, c='0.7', label='ns' if xi == 0 else None)
            ax.scatter(xi + jit[sig.values], g.loc[sig, 'r2'], s=26, c='crimson',
                       label='sig' if xi == 0 else None)
            if sig.any():
                ax.plot([xi - 0.25, xi + 0.25], [g.loc[sig, 'r2'].median()] * 2, 'k-', lw=2)
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xticks(range(len(dsets))); ax.set_xticklabels([d[:14] for d in dsets], rotation=45, ha='right')
        ax.set_ylabel('ensemble R²'); ax.legend(fontsize=8)
        ax.set_title(f'{plot_mouse} — {plot_block}·{plot_model}: ensemble R² across datasets '
                     f'(sig={int((ens["sig"] == True).sum())})', fontsize=9)   # noqa: E712
        fig.tight_layout()
        # sig-ensemble median R² per dataset, block × model matrix
        _sig = ens_full = full[(full['source'] == 'ensembles') & (full['sig'] == True)]  # noqa: E712
        with pd.option_context('display.width', 200):
            print(f'{plot_mouse} — SIGNIFICANT-ensemble median R² by dataset × block (model={plot_model}):')
            print(_sig[_sig['model'] == plot_model].groupby(['dset_name', 'block'])['r2']
                  .median().unstack().round(3).to_string())
