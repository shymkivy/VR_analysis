# -*- coding: utf-8 -*-
"""
VR_inspect_cache.py — load a VR_feature_gen.py feature cache (.joblib) and inspect
its contents. Spyder cell-based: set the path knobs in the first cell, then run the
later cells to print the bundle structure, the built_blocks registry, and a few
blocks in detail.

Caches are written by VR_feature_gen.py via f_save_features (see f_feature_cache.py).
Created 2026-06-25.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import numpy as np

from f_feature_cache import f_features_path, f_load_features

#%% Config — point at one cache file. Two ways: build the path from parts, or set it.
mouse_id     = 'L'
gen_tag      = 'v1'                              # must match VR_feature_gen.py GEN_TAG
dset_name    = '2024_01_15'                      # est1['dset_name'] of the dataset
features_dir = 'F:/VR/features/' + mouse_id

path = f_features_path(features_dir, mouse_id, dset_name, gen_tag)
# path = Path('F:/VR/features/L/L_2024_01_15_v1_features.joblib')   # or set directly

print(f'cache path: {path}')
print(f'exists:     {path.exists()}')

#%% Load the bundle (no staleness check — pass gen_config=cfg to compare)
bundle = f_load_features(path)

built_blocks = bundle['built_blocks']
frame_t      = np.asarray(bundle['frame_t']).ravel()
side_tags    = bundle['side_tags']
meta         = bundle['meta']
gen_config   = bundle['gen_config']
movie        = bundle['movie']

#%% Top-level bundle overview
print('=== bundle keys ===')
for k, v in bundle.items():
    if k == 'built_blocks':
        info = f'dict with {len(v)} blocks: {list(v)}'
    elif hasattr(v, 'shape'):
        info = f'ndarray {v.shape} {v.dtype}'
    elif isinstance(v, dict):
        info = f'dict ({len(v)} keys)'
    elif isinstance(v, (list, tuple)):
        info = f'{type(v).__name__} len={len(v)}: {v}'
    else:
        info = repr(v)
    print(f'  {k:16s} {info}')

print(f'\nframe_t: T={frame_t.size}  '
      f'span={frame_t[-1] - frame_t[0]:.1f}s  '
      f'fs~{1.0 / np.median(np.diff(frame_t)):.2f} Hz')
print(f'side_tags: {side_tags}')
print(f'movie cached: {movie is not None}')

#%% meta + generation config
print('=== meta ===')
for k, v in meta.items():
    print(f'  {k:16s} {v}')

print('\n=== gen_config ===')
import json
print(json.dumps(gen_config, indent=2, default=str)[:3000])

#%% built_blocks registry — per-block shape + names
print('=== built_blocks ===')
for name, blk in built_blocks.items():
    if not isinstance(blk, dict):
        print(f'  {name!r}: {type(blk).__name__}')
        continue
    X = blk.get('X')
    shp = X.shape if hasattr(X, 'shape') else None
    names = blk.get('names')
    n_names = len(names) if names is not None else '?'
    extra = [k for k in blk if k not in ('X', 'names')]
    print(f'  {name:10s} X={shp}  n_names={n_names}  other_keys={extra}')

#%% Inspect one block in detail — edit the key
blk_name = 'agg'
blk = built_blocks[blk_name]
X = np.asarray(blk['X'])
names = blk.get('names')

print(f'=== block {blk_name!r} ===')
print(f'X shape:  {X.shape}  ({X.dtype})')
print(f'NaNs:     {np.isnan(X).sum()} / {X.size}')
print(f'col mean: {np.nanmean(X, axis=0)}')
print(f'col std:  {np.nanstd(X, axis=0)}')
if names is not None:
    print(f'names ({len(names)}):')
    for i, nm in enumerate(names):
        print(f'  [{i:3d}] {nm}')

#%% (optional) peek at first rows of a block as a DataFrame
import pandas as pd
df = pd.DataFrame(np.asarray(built_blocks['agg']['X']),
                  columns=built_blocks['agg'].get('names'))
df.insert(0, 't', frame_t)
print(df.head(20))
