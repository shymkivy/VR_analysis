# -*- coding: utf-8 -*-
"""
VR_decode_analysis.py — compare decoding results ACROSS datasets (TODO #57 follow-up).

LOADS the per-dataset decode caches written by VR_ca_cebra.py's save cell
(f_decode_cache; one <mouse>_<dset>_<tag>_decode.joblib per dataset) and compares the
per-feature blocked-CV R² (neural→behavior) across datasets — which feature blocks
decode well, how consistently, and above the shuffle null. Computes nothing itself:
all decoding is done once per dataset in cebra via the SHARED f_run_decode_sweep, so
single-dataset and cross-dataset analyses use identical numbers.

Spyder cell-based: run the load cell, then any comparison cell.
Created 2026-06-29.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from f_decode_cache import f_list_decode_files, f_load_decode

#%% Load all decode caches for the mouse/mice + tag
MICE       = ['L']
decode_tag = None          # filter to one DECODE_TAG (substring); None = all
base_dir   = 'F:/VR/decode/'      # per-mouse subdir: <base_dir><mouse>

bundles = []
for m in MICE:
    for p in f_list_decode_files(base_dir + m, decode_tag or ''):
        b = f_load_decode(p)
        b['meta'].setdefault('mouse_id', m)
        bundles.append(b)
print(f'loaded {len(bundles)} decode caches:',
      [f"{b['meta'].get('mouse_id')}/{b['meta'].get('dset_name')}" for b in bundles])

#%% Tidy table — one row per (dataset, block, feature, n_pcs)
# r2 = real blocked-CV R²; r2_null = mean over shuffles; delta = r2 − r2_null.
rows = []
for b in bundles:
    meta = b['meta']
    r2   = np.asarray(b['r2_mat'])
    null = b.get('r2_mat_null')
    null_mean = null.mean(axis=0) if null is not None else np.full_like(r2, np.nan)
    npcs = b['n_pcs_sweep']
    for fi, (blk, name) in enumerate(b['targets_all']):
        for ki, k in enumerate(npcs):
            rows.append({
                'mouse': meta.get('mouse_id'), 'dset': meta.get('dset_name'),
                'region': meta.get('region'), 'n_cells': meta.get('n_cells'),
                'block': blk, 'feature': name, 'n_pcs': k,
                'r2': r2[fi, ki], 'r2_null': null_mean[fi, ki],
                'delta': r2[fi, ki] - null_mean[fi, ki]})
df = pd.DataFrame(rows)
print(df.shape, '| blocks:', sorted(df.block.unique()),
      '| datasets:', df.dset.nunique())

# Per-feature ceiling: max over the n_pcs sweep (real R² + Δ at the curve's plateau).
plat = (df.groupby(['mouse', 'dset', 'region', 'block', 'feature'], dropna=False)
          [['r2', 'delta']].max().reset_index())

#%% Per-block decoding across datasets — plateau R² (one point per dataset)
# For each block: mean plateau R² across its features, one dot per dataset, with the
# cross-dataset mean as a bar. Shows which representations decode + how consistent.
metric = 'r2'                       # 'r2' (vs 0) | 'delta' (vs shuffle null)
perblock = (plat.groupby(['dset', 'block'])[metric].mean().reset_index())
blocks = sorted(perblock.block.unique())

fig, ax = plt.subplots(figsize=(1.1 * len(blocks) + 2, 4))
rng = np.random.default_rng(0)
for i, blk in enumerate(blocks):
    v = perblock.loc[perblock.block == blk, metric].values
    ax.scatter(i + 0.12 * rng.standard_normal(len(v)), v, s=28, alpha=0.7, zorder=3)
    ax.hlines(np.nanmean(v), i - 0.25, i + 0.25, color='k', lw=2, zorder=4)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks, rotation=45, ha='right')
ax.set_ylabel(f'plateau {metric}  (mean over block features)')
ax.set_title(f'decoding across {perblock.dset.nunique()} datasets — per block ({metric})')
fig.tight_layout()

#%% Feature × dataset heatmap for ONE block (plateau R²)
hm_block  = 'gabor'                 # which block to expand feature-by-feature
hm_metric = 'r2'                    # 'r2' | 'delta'
sub = plat[plat.block == hm_block].pivot_table(
    index='feature', columns='dset', values=hm_metric)
if sub.size:
    fig, ax = plt.subplots(figsize=(1.0 * sub.shape[1] + 2, 0.25 * sub.shape[0] + 2))
    im = ax.imshow(sub.values, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                   interpolation='none')
    ax.set_xticks(range(sub.shape[1])); ax.set_xticklabels(sub.columns, rotation=45, ha='right')
    ax.set_yticks(range(sub.shape[0])); ax.set_yticklabels(sub.index, fontsize=7)
    ax.set_title(f'{hm_block}: plateau {hm_metric} (feature × dataset)')
    fig.colorbar(im, ax=ax, label=hm_metric)
    fig.tight_layout()
else:
    print(f'block {hm_block!r} not present; available: {sorted(plat.block.unique())}')

#%% Real vs null — per-block Δ (above-chance), one point per dataset
# Same as the per-block cell but Δ R² = real − shuffle-null mean. Points above 0 decode
# above chance after removing blocked-CV / autocorr bias that survives the shuffle.
perblock_d = (plat.groupby(['dset', 'block'])['delta'].mean().reset_index())
blocks = sorted(perblock_d.block.unique())
fig, ax = plt.subplots(figsize=(1.1 * len(blocks) + 2, 4))
for i, blk in enumerate(blocks):
    v = perblock_d.loc[perblock_d.block == blk, 'delta'].values
    ax.scatter(i + 0.12 * rng.standard_normal(len(v)), v, s=28, alpha=0.7, zorder=3)
    ax.hlines(np.nanmean(v), i - 0.25, i + 0.25, color='k', lw=2, zorder=4)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks, rotation=45, ha='right')
ax.set_ylabel('plateau Δ R²  (real − null mean)')
ax.set_title(f'above-chance decoding across {perblock_d.dset.nunique()} datasets')
fig.tight_layout()

#%% (optional) group per-block R² by region (if region is in the cache meta)
if plat.region.notna().any():
    metric = 'r2'
    g = (plat.groupby(['region', 'block'])[metric].mean().unstack('block'))
    fig, ax = plt.subplots(figsize=(1.1 * g.shape[1] + 2, 0.5 * g.shape[0] + 2))
    im = ax.imshow(g.values, aspect='auto', cmap='viridis', interpolation='none')
    ax.set_xticks(range(g.shape[1])); ax.set_xticklabels(g.columns, rotation=45, ha='right')
    ax.set_yticks(range(g.shape[0])); ax.set_yticklabels(g.index)
    ax.set_title(f'mean plateau {metric} by region × block')
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
else:
    print('no region in cache meta — set region in cebra save cell to enable this view.')
