# -*- coding: utf-8 -*-
"""
f_conv_autosave.py — SSD autosave for CONVOLVED+POOLED feature blocks (VR_feature_search).

Persist the RAW (n_pca=None) pooled gabor/meng block that `f_feature_search._raw_block` builds, keyed
by the extraction recipe + the pixel realization (render geometry / fs / sigma_scale / gamma). A later
config that shares the same bank but differs only in proc/marginalize (e.g. the pyr_sets mag/mag2pca/
mag1norm variants) RELOADS the block instead of re-convolving — the in-run raw_cache frees each conv
per unit to bound RAM, so this SSD store is what gives cross-config / cross-run reuse.

Save point is BEFORE any layer1/layer2 proc and before marginalize (the raw pooled block), so ONE saved
block serves every proc/marginalize variant. Storage mirrors f_conv_repo (uint16 quant + manifest) but
at the pooled-block granularity, PER-COLUMN quantized (a pooled block mixes filters/orientations/stats
of different scale, so a single per-array min/max would lose precision).

Layout:  <features_dir>/conv_autosave/<mouse>_<dset>_<geom_tag>/
    <human-tokens>_<recipe_hash>.npz   — one block: Xq (uint16 T×C) + per-col vmin/vmax + names + meta
    frame_t.npy                        — the imaging clock (written once)
    manifest.json                      — provenance (geom + per-block inputs/shape)

Gabor/meng ONLY (pix has no conv and its raw pixel block is huge). Opt-in via
build_cfg['conv_autosave']=True (+ optional 'conv_autosave_tag'). Created 2026-07-24.
"""
import json
import hashlib
from pathlib import Path

import numpy as np

from f_feature_cache import _safe, _jsonable
from f_feature_helpers import f_num_tag
from f_conv_repo import f_deg_per_px


# ── per-column uint16 quantization (the pooled-block analogue of f_conv_repo._quant_u16) ──

def _quant_u16_cols(X):
    """Per-COLUMN linear quant of a (T,C) float array to uint16 over each column's own [min,max].
    Returns (Xq uint16 (T,C), vmin (C,) f32, vmax (C,) f32). Degenerate/NaN columns → zeros (dequant
    to vmin)."""
    X = np.asarray(X, np.float32)
    vmin = np.nanmin(X, axis=0)
    vmax = np.nanmax(X, axis=0)
    span = vmax - vmin
    bad = ~(np.isfinite(vmin) & np.isfinite(vmax)) | (span <= 0)
    vmin = np.where(bad, 0.0, vmin).astype(np.float32)
    span_safe = np.where(bad, 1.0, span).astype(np.float32)
    Xf = np.nan_to_num(X, nan=0.0)
    Xq = np.round((Xf - vmin) / span_safe * 65535.0)
    Xq[:, bad] = 0
    return np.clip(Xq, 0, 65535).astype(np.uint16), vmin, (vmin + span_safe).astype(np.float32)


def _dequant_u16_cols(Xq, vmin, vmax):
    """Inverse of _quant_u16_cols → float32 (T,C)."""
    span = (np.asarray(vmax, np.float32) - np.asarray(vmin, np.float32))
    return (np.asarray(Xq, np.float32) / 65535.0 * span + np.asarray(vmin, np.float32)).astype(np.float32)


# ── keys: the extraction recipe (physical) + the pixel realization (geometry) ──

def _hash8(s):
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def _recipe_str(inputs):
    """Canonical JSON of the extraction recipe. Drops the non-extraction 'proc'/'input_proc' keys
    (applied AFTER conv, so proc-independent → shared across proc variants)."""
    clean = [{k: v for k, v in inp.items() if k not in ('proc', 'input_proc')} for inp in inputs]
    return json.dumps(_jsonable(clean), sort_keys=True)


def _geom_meta(build_cfg, session):
    """The pixel realization that (with the physical recipe) fully determines the conv output: render
    resolution, deg/px, fs, spatial-envelope params, side. Physical `inputs` are geometry-independent,
    so this MUST bind the key (else a map convolved at a different resolution would be reused)."""
    lat = int(build_cfg.get('lat_samp') or build_cfg.get('num_samp', 101))
    vert = int(build_cfg.get('vert_samp') or build_cfg.get('num_samp', 101))
    cam = session.get('cam_params') or build_cfg.get('cam_params')
    dpp = f_deg_per_px(cam, lat, vert) if cam is not None else None
    ft = session.get('frame_t')
    fs = float(1.0 / np.median(np.diff(ft))) if ft is not None and len(np.asarray(ft).ravel()) > 1 else None
    return {'lat_samp': lat, 'vert_samp': vert,
            'deg_per_px': (round(dpp, 8) if dpp is not None else None), 'fs': (round(fs, 6) if fs else None),
            'sigma_scale': float(build_cfg.get('sigma_scale', 0.5)),
            'gamma': float(build_cfg.get('gamma', 1.0)), 'side': build_cfg.get('side', 'both')}


def _geom_tag(build_cfg, session):
    gm = _geom_meta(build_cfg, session)
    base = _hash8(json.dumps(gm, sort_keys=True))
    utag = build_cfg.get('conv_autosave_tag')
    return f'{_safe(utag)}_{base}' if utag else base


def _block_fname(inputs):
    """Human-readable physical-token filename + recipe hash (uniqueness). One-input extraction."""
    inp = inputs[0]
    src = str(inp.get('source', '?'))
    od = inp.get('odegs')
    tok = ['o' + '-'.join(f_num_tag(a, 3) for a in np.atleast_1d(od))] if od is not None \
        else [f"n{inp.get('n_orient', '')}"]
    parts = [src] + tok
    if inp.get('sf_cyc_deg') is not None:
        parts.append('sf' + f_num_tag(np.atleast_1d(inp['sf_cyc_deg'])[0]))
    if inp.get('tf_hz') is not None:
        parts.append('tf' + f_num_tag(np.atleast_1d(inp['tf_hz'])[0]))
    parts.append(f"{inp.get('reduce', 'sample')}{inp.get('ds', '')}")
    parts.append('-'.join(inp.get('stats', ('magnitude',))) if src == 'gabor'
                 else str(inp.get('output', 'opponent')))
    return _safe('_'.join(str(p) for p in parts)) + f'_{_hash8(_recipe_str(inputs))}.npz'


def f_autosave_dir(features_dir, mouse_id, dset_name, geom_tag):
    return Path(features_dir) / 'conv_autosave' / f'{_safe(mouse_id)}_{_safe(dset_name)}_{geom_tag}'


def f_autosave_locate(build_cfg, session, inputs):
    """→ (dir, block_path, geom_meta). Root from build_cfg['conv_autosave_root'] (injected by
    f_feature_search from base_source). Returns None if the root is unset.
    LEGACY input-spec-keyed path — the ACTIVE hook is now spec-level inside build_visual_blocks
    (f_autosave_ctx + f_spec_fname), which every build path shares. Kept for direct callers."""
    root = build_cfg.get('conv_autosave_root')
    if not root:
        return None
    d = f_autosave_dir(root, session.get('mouse_id', '?'), session.get('dset_name', '?'),
                       _geom_tag(build_cfg, session))
    return d, d / _block_fname(inputs), _geom_meta(build_cfg, session)


# ── spec-level key (the ACTIVE path: hooked inside build_visual_blocks) ──
# Keyed on the RESOLVED extractor params rather than the physical input spec. Those params already
# bake in the geometry (sf_cyc_deg → `wavelengths` in px via deg_per_px, ds='nyquist' → `sample_ds`),
# so they are the true determinant of the conv output — no separate geom hash needed beyond the
# movie shape / clock / sides. Because it sits at the ONE point every build path funnels through
# (f_build_features → build_visual_blocks), VR_feature_search and VR_ca_cebra share one store.

AUTOSAVE_KINDS = ('gabor', 'meng', 'fft', 'wavelet')   # conv-heavy; pix/grid/flow/fftglobal cheap

_KEY_SKIP = ('backend', 'device', 'chunk_t', 'verbose')   # perf knobs — do NOT change the result


def f_autosave_ctx(build_cfg, session, features_dir=None):
    """Build the autosave CONTEXT for build_visual_blocks → {'dir', 'verbose'} or None when disabled /
    rootless. Opt in with build_cfg['conv_autosave']=True; root from build_cfg['conv_autosave_root']
    or `features_dir`; build_cfg['conv_autosave_tag'] namespaces the store."""
    if not build_cfg.get('conv_autosave'):
        return None
    root = build_cfg.get('conv_autosave_root') or features_dir
    if not root:
        return None
    tag = _safe(build_cfg.get('conv_autosave_tag')) if build_cfg.get('conv_autosave_tag') else 'spec'
    d = f_autosave_dir(root, session.get('mouse_id', '?'), session.get('dset_name', '?'), tag)
    return {'dir': d, 'verbose': bool(build_cfg.get('conv_autosave_verbose', True))}


def f_spec_fname(fname, ftype, params, side_tags, movie_shapes, n_img):
    """Filename + key metadata for ONE resolved visual spec. Human-readable physical tokens plus a
    hash over (type, result-determining params, sides, movie shape, #imaging frames)."""
    key = {'type': ftype, 'sides': list(side_tags), 'movies': [list(map(int, s)) for s in movie_shapes],
           'n_img': int(n_img),
           'params': {k: v for k, v in sorted(params.items()) if k not in _KEY_SKIP}}
    ks = json.dumps(_jsonable(key), sort_keys=True)
    parts = [ftype]
    wl = params.get('wavelengths')
    if wl is not None:
        parts.append('wl' + '-'.join(f_num_tag(w, 3) for w in np.atleast_1d(wl)))
    od = params.get('odegs')
    parts.append('o' + '-'.join(f_num_tag(a, 3) for a in np.atleast_1d(od)) if od is not None
                 else f"n{params.get('n_orient', '')}")
    tf = params.get('temporal_freqs')
    if tf is not None:
        parts.append('tf' + '-'.join(f_num_tag(t, 3) for t in np.atleast_1d(tf)))
    parts.append(f"{params.get('pool', 'grid')}{params.get('sample_ds', '')}")
    parts.append('-'.join(params.get('stats', ())) if ftype in ('gabor', 'fft', 'fftglobal')
                 else str(params.get('output', 'opponent')))
    return _safe('_'.join(str(p) for p in parts)) + f'_{_hash8(ks)}.npz', key


# ── save / load ──

def f_autosave_load(d, path):
    """Load a saved block → (block_dict, frame_t) shaped exactly like _raw_block's return, or None on
    miss / unreadable (→ caller rebuilds)."""
    try:
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=False)
        X = _dequant_u16_cols(z['Xq'], z['vmin'], z['vmax'])
        names = [str(s) for s in z['names']]
        raw_names = [str(s) for s in z['raw_names']]
        frame_t = np.load(d / 'frame_t.npy')
    except Exception:
        return None
    block = {'X': np.asarray(X, np.float32), 'names': names, 'raw_names': raw_names,
             'raw_dim': int(z['raw_dim']), 'pca': None, 'pca_groups': None, 'kind': str(z['kind'])}
    return block, np.asarray(frame_t).ravel()


def _update_manifest(d, fname, inputs, geom_meta, shape):
    mp = d / 'manifest.json'
    man = {}
    if mp.exists():
        try:
            man = json.loads(mp.read_text())
        except Exception:
            man = {}
    man['geom'] = geom_meta
    man.setdefault('blocks', {})[fname] = {'inputs': _jsonable(inputs), 'shape': [int(s) for s in shape]}
    tmp = d / 'manifest.json.tmp'
    tmp.write_text(json.dumps(man, indent=1))
    tmp.replace(mp)                                    # atomic


def f_autosave_put(d, path, block, frame_t, inputs, geom_meta, verbose=False):
    """Write one pooled block (per-column uint16) + shared frame_t.npy (once) + manifest entry."""
    d.mkdir(parents=True, exist_ok=True)
    X = np.asarray(block['X'], np.float32)
    Xq, vmin, vmax = _quant_u16_cols(X)
    names = list(block.get('names') or [])
    raw_names = list(block.get('raw_names') or names)
    tmp = path.with_name(path.stem + '.tmp.npz')       # ends in .npz → np.savez won't re-append
    np.savez(tmp, Xq=Xq, vmin=vmin, vmax=vmax,
             names=np.asarray(names, dtype='U'), raw_names=np.asarray(raw_names, dtype='U'),
             raw_dim=np.int64(block.get('raw_dim', X.shape[1])),
             kind=np.asarray(str(block.get('kind', '')), dtype='U'))
    tmp.replace(path)                                  # atomic
    ftp = d / 'frame_t.npy'
    if not ftp.exists():
        np.save(ftp, np.asarray(frame_t).ravel())
    _update_manifest(d, path.name, inputs, geom_meta, X.shape)
    if verbose:
        print(f'    autosave WROTE {path.name} ({X.shape[1]} col)', flush=True)
