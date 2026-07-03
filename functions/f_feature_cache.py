# -*- coding: utf-8 -*-
"""
Feature cache — save / load precomputed built_blocks per dataset so the expensive
render + feature extraction runs ONCE (VR_feature_gen.py over all datasets) and the
analysis scripts (encoding / decoding) just load. Caches FEATURES only: neural
(est1['S']) stays in data_ca; raw monitor movies are skipped by default (the
pix/grid/flow blocks store PCA scores, not pixels).

Each cache file (joblib) stores the GENERATION CONFIG it was built with, so a stale
cache — produced by a different feature method while you were optimizing on a single
dataset — is flagged on load instead of silently used.

Created 2026-06-13.
"""

import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import joblib

CACHE_FORMAT_VERSION = 1


def _safe(name):
    """Filesystem-safe stem from a dataset / mouse / tag name."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))


def f_features_path(features_dir, mouse_id, dset_name, tag=None):
    """Cache path for one dataset:
    <dir>/<mouse>_<dset>[_<tag>]_features.joblib. `tag` = the feature-generation
    version (GEN_TAG) so different configs coexist without overwrite."""
    stem = f'{_safe(mouse_id)}_{_safe(dset_name)}'
    if tag:
        stem += f'_{_safe(tag)}'
    return Path(features_dir) / f'{stem}_features.joblib'


def f_features_exist(features_dir, mouse_id, dset_name, tag=None):
    """True if the cache file for this dataset (+ tag) exists."""
    return f_features_path(features_dir, mouse_id, dset_name, tag).exists()


def f_list_feature_files(features_dir, dset_name=None):
    """Sorted list of *_features.joblib cache Paths in `features_dir`. If
    `dset_name` is given, keep only those whose filename contains it. Returns []
    when the directory is missing."""
    features_dir = Path(features_dir)
    if not features_dir.is_dir():
        return []
    files = sorted(features_dir.glob('*_features.joblib'))
    if dset_name:
        files = [p for p in files if _safe(dset_name) in p.name]
    return files


def f_resolve_feature_file(features_dir, dset_name, name=None, tag=None, index=0):
    """Pick one cache file for `dset_name` from `features_dir`:
        name  — exact filename, else UNIQUE substring match (wins outright).
        tag   — FILTER to files whose name contains the tag (e.g. GEN_TAG, shared by
                all datasets), THEN pick by `index` within the matches.
        index — position in the (dset_name + tag) filtered, sorted list (default 0).
    To load ONE dataset, narrow with `dset_name` (and/or a unique `name`); `tag`+`index`
    browses the datasets sharing a feature config. Raises ValueError listing the
    candidates on any miss / bad index."""
    files = f_list_feature_files(features_dir, dset_name)
    if not files:
        raise ValueError(f'no *_features.joblib for {dset_name!r} in {features_dir}')

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
            raise ValueError(f'tag {tag!r} matched no files in {features_dir}')

    have = [p.name for p in files]
    if not 0 <= index < len(files):
        raise ValueError(f'index {index} out of range for {len(files)} files '
                         f'(dset={dset_name!r}, tag={tag!r}); have {have}')
    return files[index]


def _jsonable(o):
    """Recursively coerce a config to JSON-safe types (tuples→lists, Path→str)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)            # numpy scalars / unknowns → repr


def f_write_config_json(features_dir, tag, config):
    """Write a human-readable JSON provenance sidecar <dir>/<tag>_config.json with
    the full generation config (tuples→lists), so you can see what's in these caches
    without loading any joblib. Returns the written Path."""
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    path = features_dir / f'{_safe(tag) if tag else "default"}_config.json'
    with open(path, 'w') as fh:
        json.dump(_jsonable(config), fh, indent=2, sort_keys=True)
    return path


def f_pick_frames(movie, n=8):
    """`n` evenly-spaced example frames from a movie (T,H,W[,3]) + their indices, for
    stashing in the cache so viz can show the original input without re-rendering.
    Returns (frames, idx) — both None if `movie` is None."""
    if movie is None:
        return None, None
    movie = np.asarray(movie)
    T = movie.shape[0]
    idx = [int(i) for i in np.unique(np.linspace(0, T - 1, min(n, T)).round().astype(int))]
    return movie[idx], idx


def f_save_features(path, built_blocks, frame_t, side_tags,
                    meta=None, gen_config=None, movie=None,
                    example_frames=None, example_frame_idx=None, compress=3):
    """Save one dataset's features to `path` (joblib).

    Parameters
    ----------
    built_blocks : dict     the feature registry (build_feature_blocks output).
    frame_t : ndarray       imaging frame times (the blocks' clock).
    side_tags : list[str]   monitor sides used.
    meta : dict             free-form metadata (n_dset, dset_name, region, n_cells,
                            fs, mean_snr, running_frac, ...).
    gen_config : dict       feature-GENERATION config (block flags, n_pix_pca,
                            agg_mode, obj_collapse, nan_absent, visual_specs, side,
                            render params) — compared on load for staleness. Keep it
                            JSON-able (scalars / strings / lists / nested dicts).
    movie : ndarray | None  optional raw two_mon_frames (large; default skip).
    example_frames : ndarray | None   a few original rendered frames (n,H,W[,3]) —
                            small; saved even when `movie` is skipped, so viz can show
                            the original input next to a feature reconstruction.
    example_frame_idx : list | None   imaging-frame indices of `example_frames`.
    compress : int          joblib compression level.

    Returns the written Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {'format_version':    CACHE_FORMAT_VERSION,
              'saved_unix':        time.time(),
              'built_blocks':      built_blocks,
              'frame_t':           frame_t,
              'side_tags':         list(side_tags),
              'meta':              meta or {},
              'gen_config':        gen_config or {},
              'movie':             movie,
              'example_frames':    example_frames,
              'example_frame_idx': example_frame_idx}
    joblib.dump(bundle, path, compress=compress)
    return path


def _config_diff(want, have):
    """Keys whose values differ between the requested vs cached config dicts.
    Both are JSON-normalized first so tuple-vs-list (Python vs JSON config form)
    isn't a false mismatch."""
    want, have = _jsonable(want), _jsonable(have)
    keys = set(want) | set(have)
    return [k for k in sorted(keys) if want.get(k) != have.get(k)]


def f_load_features(path, gen_config=None, on_stale='warn'):
    """Load a feature cache; optionally check it against a requested gen_config.

    Parameters
    ----------
    path : str / Path
    gen_config : dict | None    if given, compared to the cached config.
    on_stale : {'warn', 'raise', 'ignore'}
        Behavior when the cached generation config differs (cache built with a
        different feature method). 'warn' (default) prints the mismatching keys;
        'raise' errors; 'ignore' skips the check.

    Returns
    -------
    dict bundle: built_blocks, frame_t, side_tags, meta, gen_config, movie,
    saved_unix, format_version.
    """
    bundle = joblib.load(Path(path))
    if gen_config is not None and on_stale != 'ignore':
        diff = _config_diff(gen_config, bundle.get('gen_config', {}))
        if diff:
            msg = (f'feature cache {Path(path).name} is STALE — generation config '
                   f'differs on {diff}. Regenerate via VR_feature_gen.py.')
            if on_stale == 'raise':
                raise ValueError(msg)
            warnings.warn(msg)
    return bundle
