# -*- coding: utf-8 -*-
"""
Decode-results cache — save / load the per-dataset decoding sweep (f_run_decode_sweep
+ f_run_decode_null output) so a cross-dataset analyzer (VR_decode_analysis.py) can
LOAD results instead of recomputing. Mirrors f_feature_cache.py.

Stores the R² / AUC matrices (real + shuffle null) + their feature labels + the decode
config + dataset meta. Deliberately drops the per-target y arrays, the neural matrix and
the full-data PCA — only the small results survive. Each file is named
<mouse>_<dset>_<DECODE_TAG>_decode.joblib.

Created 2026-06-29.
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import joblib

DECODE_FORMAT_VERSION = 1


def _safe(name):
    """Filesystem-safe stem from a dataset / mouse / tag name."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))


def f_decode_path(decode_dir, mouse_id, dset_name, tag=None):
    """Cache path: <dir>/<mouse>_<dset>[_<tag>]_decode.joblib."""
    stem = f'{_safe(mouse_id)}_{_safe(dset_name)}'
    if tag:
        stem += f'_{_safe(tag)}'
    return Path(decode_dir) / f'{stem}_decode.joblib'


def f_decode_exist(decode_dir, mouse_id, dset_name, tag=None):
    """True if the decode cache for this dataset (+ tag) exists."""
    return f_decode_path(decode_dir, mouse_id, dset_name, tag).exists()


def f_list_decode_files(decode_dir, dset_name=None):
    """Sorted list of *_decode.joblib Paths in `decode_dir`, optionally filtered to
    those whose filename contains `dset_name`. [] if the dir is missing."""
    decode_dir = Path(decode_dir)
    if not decode_dir.is_dir():
        return []
    files = sorted(decode_dir.glob('*_decode.joblib'))
    if dset_name:
        files = [p for p in files if _safe(dset_name) in p.name]
    return files


def f_resolve_decode_file(decode_dir, dset_name, name=None, tag=None, index=0):
    """Pick one decode cache for `dset_name`: `name` = exact/unique-substring (wins);
    `tag` = FILTER to files containing it (e.g. DECODE_TAG, shared by all datasets),
    THEN pick by `index` within the matches; `index` selects in the filtered sorted
    list. Narrow with `dset_name` to load ONE dataset. Raises ValueError listing the
    candidates on any miss / bad index."""
    files = f_list_decode_files(decode_dir, dset_name)
    if not files:
        raise ValueError(f'no *_decode.joblib for {dset_name!r} in {decode_dir}')
    if name is not None:
        exact = [p for p in files if p.name == name]
        if exact:
            return exact[0]
        sub = [p for p in files if name in p.name]
        if len(sub) == 1:
            return sub[0]
        raise ValueError(f'filename {name!r} matched {len(sub)} files; '
                         f'have {[p.name for p in files]}')
    if tag is not None:                          # tag FILTERS; index selects within it
        files = [p for p in files if tag in p.name]
        if not files:
            raise ValueError(f'tag {tag!r} matched no files in {decode_dir}')
    have = [p.name for p in files]
    if not 0 <= index < len(files):
        raise ValueError(f'index {index} out of range for {len(files)} files '
                         f'(dset={dset_name!r}, tag={tag!r}); have {have}')
    return files[index]


def _names(targets):
    """(block, name) label pairs from a [(grp, name, y), ...] target list — drops y."""
    return [(g, n) for g, n, _ in targets]


def f_save_decode(path, sweep, null=None, meta=None, gen_config=None, compress=3):
    """Save one dataset's decode results to `path` (joblib).

    Parameters
    ----------
    sweep : dict           f_run_decode_sweep output (r2_mat / auc_mat / targets / ...).
    null : dict | None     f_run_decode_null output (r2_mat_null / auc_mat_null / cfg_null).
    meta : dict            dataset metadata (mouse_id, n_dset, dset_name, region,
                           n_cells, fs, n_frames).
    gen_config : dict      the decode config (sweep['cfg']) — stored for staleness.

    The y arrays / neural / PCA are NOT stored — only the result matrices + labels.
    Returns the written Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        'format_version': DECODE_FORMAT_VERSION,
        'saved_unix':     time.time(),
        'r2_mat':         np.asarray(sweep['r2_mat']),
        'auc_mat':        np.asarray(sweep['auc_mat']),
        'targets_all':    _names(sweep['targets_all']),
        'targets_bin':    _names(sweep['targets_bin']),
        'group_rows':     sweep['group_rows'],
        'groups_present': list(sweep['groups_present']),
        'n_pcs_sweep':    list(sweep['n_pcs_sweep']),
        'r2_mat_null':    (np.asarray(null['r2_mat_null']) if null else None),
        'auc_mat_null':   (np.asarray(null['auc_mat_null']) if null else None),
        'cfg_null':       (null['cfg_null'] if null else None),
        'gen_config':     gen_config or sweep.get('cfg', {}),
        'meta':           meta or {},
    }
    joblib.dump(bundle, path, compress=compress)
    return path


def f_load_decode(path, gen_config=None, on_stale='warn'):
    """Load a decode cache. If `gen_config` is given, warn/raise on a mismatching
    decode config (cache built with different decode knobs). on_stale: 'warn'|'raise'|
    'ignore'. Returns the bundle dict."""
    bundle = joblib.load(Path(path))
    if gen_config is not None and on_stale != 'ignore':
        have = bundle.get('gen_config', {})
        diff = [k for k in (set(gen_config) | set(have))
                if str(gen_config.get(k)) != str(have.get(k))]
        if diff:
            msg = (f'decode cache {Path(path).name} is STALE — decode config differs '
                   f'on {diff}.')
            if on_stale == 'raise':
                raise ValueError(msg)
            warnings.warn(msg)
    return bundle
