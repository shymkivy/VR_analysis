# f_tuning_scan.py — cross-dataset feature-TUNING sweep (reverse-correlation / STA; created 2026-07-17).
#
# The tuning analogue of f_encode_scan. Loop many mice × datasets and, for EACH cell AND ensemble,
# reverse-correlate its activity against the cached gabor/meng features (per orientation × temporal-freq)
# and the other blocks (agg / self_mot / pix) → a per-unit tuning response. Features are LOADED from the
# pca_feat caches (pcasep — per-orientation×tf PCA groups) and turned into per-filter whole-monitor
# magnitude regressors WITHOUT reconvolving (cheap PCA-inverse column-mean). Results accumulate in ONE
# tag-versioned, RESUMABLE joblib store (rewritten atomically after every combo; skip-done keyed by
# (mouse, dset, source, feat_set)). Store keeps compact per-unit responses + dataset identity (NO
# locations) — VR_tuning_scan_plots.py joins locations separately (shared with the encode scan).
#
# Compute reuses f_encode_scan.f_encode_prepare (Y + features + ensembles ONCE per dataset),
# f_tuning.f_reverse_corr_tuning / f_tuning_readout (STA fit + orientation readout), and the
# f_encode_scan store primitives. A "feat_set" can span several cache blocks (meng = mengS+mengS2+mengS3
# → 4 orient × 3 tf matrix).

import os
import re

import numpy as np
import pandas as pd
import joblib

from f_encode_scan import (_safe, _save_atomic, _dkey, f_encode_prepare, f_ensemble_record,  # noqa: F401
                          f_footprint_centroids, f_fov_homography, f_apply_homography,
                          f_load_fov_landmarks, f_neuron_widefield_coords, f_plot_neuron_scores_fov)
from f_tuning import f_reverse_corr_tuning, f_tuning_readout, f_select_tuning_units, f_tuning_significance
from f_feature_viz import _recon_raw, _grid_from_names


# default feat_set → cache-block list (one feat_set can span several blocks, e.g. meng across tf)
DEFAULT_FEAT_SETS = {
    'gabor':    ['gaborS'],
    'meng':     ['mengS', 'mengS2', 'mengS3'],
    'pix':      ['pix'],
    'agg':      ['agg'],
    'self_mot': ['self_mot'],
}
_VISUAL_FEAT = ('gabor', 'meng')          # feat_sets that get the orientation×tf readout + monitor maps
_GRP_RE = re.compile(r'_o(\d+)sf(\d+)(?:tf(\d+))?$')   # group prefix → (io, isf, itf) indices


# ─────────────────────────────────────────────────────────────────────────────
# Store — resumable, atomic, tag-versioned (mirror of f_encode_scan)
# ─────────────────────────────────────────────────────────────────────────────

def f_tuning_scan_path(root, tag):
    """Canonical store path — <root>/tuning_scan/<tag>_tuning_scan.joblib (parallel to encode_scan/)."""
    return os.path.join(str(root), 'tuning_scan', f'{_safe(tag)}_tuning_scan.joblib')


def _tun_key(mouse, dset_name, source, feat_set):
    """Combo identity string (skip-done key). No model dim (unlike the encode scan)."""
    return f'{_safe(mouse)}|{_safe(dset_name)}|{source}|{feat_set}'


def f_new_store(config, target=()):
    """Fresh store dict. `target` = full planned list of (mouse, dset_name, source, feat_set) tuples."""
    return {'config': dict(config), 'results': {}, 'datasets': {}, 'target': [tuple(t) for t in target]}


def f_tuning_scan_load(path):
    """Load the store (single atomic snapshot — safe mid-run) or None if it doesn't exist yet."""
    return joblib.load(path) if os.path.exists(path) else None


def f_tuning_store_put(store, mouse, n_dset, dset_name, source, feat_set, result):
    """Insert one combo's compact tuning result into the store (does NOT save)."""
    r = dict(result)
    r.update({'mouse': mouse, 'n_dset': int(n_dset), 'dset_name': dset_name,
              'source': source, 'feat_set': feat_set})
    store['results'][_tun_key(mouse, dset_name, source, feat_set)] = r


def f_tuning_store_set_dataset(store, rec):
    """Record a dataset's identity + neuron index (cell_ids) + per-filter monitor maps for script 2."""
    store['datasets'][_dkey(rec['mouse'], rec['dset_name'])] = dict(rec)


def f_tuning_scan_progress(path_or_store):
    """Progress snapshot (safe mid-run). n_done / n_target / n_remaining + per-source / per-feat_set
    counts + datasets with any results. Accepts a store FILE PATH or an in-memory store dict."""
    st = path_or_store if isinstance(path_or_store, dict) else f_tuning_scan_load(path_or_store)
    if st is None:
        return {'exists': False, 'n_done': 0, 'n_target': 0, 'n_remaining': 0,
                'by_feat': {}, 'by_source': {}, 'datasets': [], 'keys': []}
    res = st['results']
    tkeys = {_tun_key(*t) for t in st.get('target', [])}
    by_feat, by_source = {}, {}
    for r in res.values():
        by_feat[r['feat_set']] = by_feat.get(r['feat_set'], 0) + 1
        by_source[r['source']] = by_source.get(r['source'], 0) + 1
    return {'exists': True, 'n_done': len(res), 'n_target': len(tkeys),
            'n_remaining': len(tkeys - set(res)), 'by_feat': by_feat, 'by_source': by_source,
            'datasets': sorted({(r['mouse'], r['dset_name']) for r in res.values()}), 'keys': sorted(res)}


def f_tuning_scan_table(store):
    """Tidy per-unit table: one row per (dataset, source, unit, feat_set). `unit` = stored label
    (cells → est1 id; ensembles → ensemble index). `sig` = ensemble significance (NaN for cells)."""
    rows = []
    for r in store['results'].values():
        rec = store['datasets'].get(_dkey(r['mouse'], r['dset_name'])) or {}
        ens = rec.get('ens') if r['source'] == 'ensembles' else None
        labels = r['labels']
        for i, lab in enumerate(labels):
            _sig = (bool(ens['sig'][int(lab)]) if ens is not None and int(lab) < ens['n_ens'] else np.nan)
            rows.append({'mouse': r['mouse'], 'n_dset': r['n_dset'], 'dset_name': r['dset_name'],
                         'source': r['source'], 'unit': int(lab) if np.isscalar(lab) or
                         isinstance(lab, (int, np.integer)) else lab, 'feat_set': r['feat_set'],
                         'pref_odeg': float(r['pref_odeg'][i]), 'pref_tf_hz': float(r['pref_tf_hz'][i]),
                         'peak_abs': float(r['peak_abs'][i]), 'peak_z': float(r['peak_z'][i]),
                         'osi': float(r['osi'][i]), 'peak_lag_ms': float(r['peak_lag_ms'][i]), 'sig': _sig})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Regressors from cached blocks (per-filter whole-monitor magnitude; NO reconvolution)
# ─────────────────────────────────────────────────────────────────────────────

def _group_magnitude(X, xslice, pca_g):
    """Whole-monitor (both sides) mean magnitude per frame for one PCA group, from its scores —
    = column-mean of the rank-k PCA reconstruction, computed cheaply as
    X[:,slice] @ components_.mean(1) + mean_.mean(). pca_g=None → the scores ARE the raw cells → mean."""
    s0, s1 = xslice
    Xg = np.asarray(X[:, s0:s1], float)
    if pca_g is None:
        return Xg.mean(axis=1)
    k = min(Xg.shape[1], pca_g.components_.shape[0])
    return Xg[:, :k] @ pca_g.components_[:k].mean(axis=1) + float(pca_g.mean_.mean())


def _block_orient_count(block):
    """Number of distinct orientations in a grouped gabor/meng block (from its group prefixes)."""
    ios = set()
    for pre in (block.get('pca_groups') or {}):
        m = _GRP_RE.search(pre)
        if m:
            ios.add(int(m.group(1)))
    return max(len(ios), 1)


def _phys(inp, key, idx):
    """idx-th value of a block input's physical tuple (sf_cyc_deg / tf_hz), or NaN."""
    v = (inp or {}).get(key)
    if v is None:
        return np.nan
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return float(v[idx]) if 0 <= idx < len(v) else float(v[0])


def f_block_regressors(built_blocks, feat_set, block_names):
    """Build (X_stim (T, n_filter), filt_df) for a feat_set from ALREADY-LOADED cache blocks.

    gabor/meng: one whole-monitor magnitude regressor per per-orientation×tf PCA group (spanning all
    the feat_set's source blocks → the orientation×tf grid). filt_df carries kind/odeg/sf_cyc_deg/tf_hz
    (+ legacy wavelength/tfreq). Physical odeg = io·180/n_orient; sf_cyc_deg/tf_hz from the block inputs.
    pix/agg/self_mot: each block column (PC / channel) is a regressor; filt_df kind=feat_set, no odeg."""
    cols, rows = [], []
    if feat_set in _VISUAL_FEAT:
        for bn in block_names:
            b = built_blocks.get(bn)
            if b is None:
                continue
            pg = b.get('pca_groups') or {}
            pca = b.get('pca') if isinstance(b.get('pca'), dict) else {}
            X = np.asarray(b['X'], float)
            n_or = _block_orient_count(b)
            inp = (b.get('inputs') or [{}])[0]
            for pre, info in pg.items():
                m = _GRP_RE.search(pre)
                if m is None:
                    continue
                io, isf, itf = int(m.group(1)), int(m.group(2)), (int(m.group(3)) if m.group(3) else 0)
                cols.append(_group_magnitude(X, info['xslice'], pca.get(pre)))
                rows.append({'kind': feat_set, 'block': bn,
                             'odeg': io * 180.0 / n_or, 'wavelength': float(isf),
                             'tfreq': (float(itf) if feat_set == 'meng' else np.nan),
                             'sf_cyc_deg': _phys(inp, 'sf_cyc_deg', isf),
                             'tf_hz': (_phys(inp, 'tf_hz', itf) if feat_set == 'meng' else np.nan)})
    else:                                                 # pix / agg / self_mot — columns as regressors
        for bn in block_names:
            b = built_blocks.get(bn)
            if b is None:
                continue
            X = np.asarray(b['X'], float)
            names = list(b.get('names') or [f'{bn}{i}' for i in range(X.shape[1])])
            for j in range(X.shape[1]):
                cols.append(X[:, j])
                rows.append({'kind': feat_set, 'block': bn, 'channel': names[j],
                             'odeg': np.nan, 'wavelength': float(j), 'tfreq': np.nan,
                             'sf_cyc_deg': np.nan, 'tf_hz': np.nan})
    if not cols:
        raise ValueError(f'feat_set {feat_set!r}: no regressors from blocks {block_names} '
                         f'(present: {list(built_blocks)})')
    X_stim = np.column_stack(cols).astype(np.float32)
    return X_stim, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# One tuning fit → compact per-unit result
# ─────────────────────────────────────────────────────────────────────────────

def _peak_profile(W, lags, fs):
    """Per-unit peak-lag tuning profile from W (n_filter, n_lags, n_units). Returns
    (resp (n_filter,n_u) signed profile at each unit's peak lag, peak_lag_idx, peak_abs, peak_z,
    peak_lag_ms)."""
    W = np.asarray(W, float)
    n_f, n_lag, n_u = W.shape
    peak_lag_idx = np.argmax(np.abs(W).max(axis=0), axis=0)     # (n_u,)
    resp = np.stack([W[:, int(peak_lag_idx[u]), u] for u in range(n_u)], axis=1)   # (n_f, n_u)
    at = np.abs(resp)
    peak_abs = at.max(axis=0)
    sd = at.std(axis=0)
    peak_z = np.where(sd > 0, (peak_abs - at.mean(axis=0)) / np.where(sd > 0, sd, 1.0), np.nan)
    fs = float(fs) if fs else 1.0
    peak_lag_ms = np.array([lags[int(pl)] * 1000.0 / fs for pl in peak_lag_idx])
    return resp.astype(np.float32), peak_lag_idx, peak_abs.astype(np.float32), \
        peak_z.astype(np.float32), peak_lag_ms.astype(np.float32)


def f_tuning_one(built_blocks, feat_set, block_names, Y_units, labels, lags, *,
                 method='sta', fs=None, alpha=1.0):
    """Reverse-correlation tuning of `Y_units` (T, n_units) against one feat_set's regressors → a
    COMPACT result dict: peak-lag tuning profile per unit (`resp` (n_filter, n_kept)), the filter grid
    (`filt_df`), and per-unit summaries. Visual feat_sets (gabor/meng) also carry orientation readout
    fields (pref_odeg/pref_tf_hz/osi). `labels` = full-length unit labels (subset by keep_cells)."""
    X_stim, filt_df = f_block_regressors(built_blocks, feat_set, block_names)
    T = min(int(X_stim.shape[0]), int(np.asarray(Y_units).shape[0]))
    fit = f_reverse_corr_tuning(X_stim[:T], filt_df, np.asarray(Y_units)[:T], lags,
                                method=method, alpha=alpha, standardize=True)
    keep = np.asarray(fit['keep_cells'], bool)
    lab_keep = list(np.asarray(labels, dtype=object)[keep]) if len(labels) == len(keep) else list(labels)
    resp, _pli, peak_abs, peak_z, peak_lag_ms = _peak_profile(fit['W'], list(fit['lags']), fs)
    n_u = resp.shape[1]
    out = {'feat_set': feat_set, 'method': method, 'labels': np.asarray(lab_keep, dtype=object),
           'filt_df': filt_df, 'resp': resp, 'lags': list(fit['lags']),
           'peak_abs': peak_abs, 'peak_z': peak_z, 'peak_lag_ms': peak_lag_ms,
           'pref_odeg': np.full(n_u, np.nan, np.float32), 'pref_tf_hz': np.full(n_u, np.nan, np.float32),
           'osi': np.full(n_u, np.nan, np.float32)}
    if feat_set in _VISUAL_FEAT:                          # orientation / tf readout
        readout = f_tuning_readout(fit, fs=fs, labels=labels)
        tab = readout['table']
        out['table'] = tab                                # full per-unit readout table (for the viewers)
        out['pref_odeg'] = tab['pref_odeg'].to_numpy(np.float32)
        if 'pref_tf_hz' in tab.columns:
            out['pref_tf_hz'] = tab['pref_tf_hz'].to_numpy(np.float32)
        out['osi'] = tab['osi'].to_numpy(np.float32)
        out['by_odeg'] = np.asarray(readout['by_odeg'], np.float32)
        out['by_wavelength'] = np.asarray(readout['by_wavelength'], np.float32)
        out['by_tfreq'] = (np.asarray(readout['by_tfreq'], np.float32)
                           if readout['by_tfreq'] is not None else None)
        out['uniq_odeg'] = np.asarray(readout['uniq_odeg'], float)
        out['uniq_wavelength'] = np.asarray(readout['uniq_wavelength'], float)
        out['uniq_tf_hz'] = np.asarray(readout['uniq_tfreq'], float)
        out['sf_label'] = readout.get('sf_label', ''); out['tf_label'] = readout.get('tf_label', '')
    return out


def f_result_to_readout(result):
    """Reconstruct a f_tuning_readout-style dict from a stored COMPACT result so the existing
    f_encoding_plots viewers (f_plot_tuning_heatmap / f_plot_tuning_marginals_all) work unchanged. The
    stored `resp` (n_filter, n_unit) is the peak-lag tuning profile, so W is faked as a single-lag cube
    (W[:,0,u] = resp[:,u]) and every unit's peak lag is 0."""
    resp = np.asarray(result['resp'], float)
    n_f, n_u = resp.shape
    tab = result.get('table')                             # full readout table stored for visual feat_sets
    if tab is None:                                       # non-visual: minimal table
        tab = pd.DataFrame({'unit': list(result['labels']),
                            'pref_odeg': result['pref_odeg'], 'pref_wavelength': np.nan,
                            'osi': result['osi'], 'peak_abs': result['peak_abs'],
                            'peak_lag_ms': result['peak_lag_ms']})
    return {'W': resp[:, None, :], 'filt_df': result['filt_df'], 'lags': [0],
            'peak_lag_idx': np.zeros(n_u, int), 'table': tab,
            'by_odeg': result.get('by_odeg'), 'by_wavelength': result.get('by_wavelength'),
            'by_tfreq': result.get('by_tfreq'), 'by_lag': np.abs(resp).max(0)[:, None],
            'uniq_odeg': result.get('uniq_odeg'), 'uniq_wavelength': result.get('uniq_wavelength'),
            'uniq_tfreq': result.get('uniq_tf_hz'), 'lags_ms': [0.0],
            'sf_label': result.get('sf_label', ''), 'tf_label': result.get('tf_label', ''),
            'keep_cells': None, 'method': result.get('method')}


def f_population_matrix(result):
    """Aggregate orientation × temporal-freq matrix over units for a visual feat_set:
    mean |resp| per (odeg, tf_hz) cell. Returns (M (n_odeg, n_tf), odegs, tfs). For gabor (no tf) the
    tf axis is a single NaN column."""
    fdf = result['filt_df']; resp = np.abs(np.asarray(result['resp'], float))
    od = fdf['odeg'].to_numpy(float)
    tf = (fdf['tf_hz'].to_numpy(float) if 'tf_hz' in fdf.columns else np.full(len(fdf), np.nan))
    odegs = np.unique(od[~np.isnan(od)])
    tfs = np.unique(tf[~np.isnan(tf)]) if np.any(~np.isnan(tf)) else np.array([np.nan])
    M = np.full((len(odegs), len(tfs)), np.nan)
    pop = resp.mean(axis=1)                              # mean over units → per-filter magnitude
    for i, o in enumerate(odegs):
        for j, t in enumerate(tfs):
            mask = (od == o) & (np.isnan(tf) if np.isnan(t) else (tf == t))
            if mask.any():
                M[i, j] = pop[mask].mean()
    return M, odegs, tfs


def f_block_monitor_maps(built_blocks, feat_set, block_names):
    """Per-filter TIME-MEAN monitor-average maps for a visual feat_set (context, stored once/dataset).
    Reconstructs each block's raw sampled cells (inverse-PCA), time-means, and reshapes to the L‖R wide
    monitor lattice via `_grid_from_names`. Returns {label: (ny, nx·n_sides)} or {} for non-visual."""
    if feat_set not in _VISUAL_FEAT:
        return {}
    maps = {}
    for bn in block_names:
        b = built_blocks.get(bn)
        if b is None or not b.get('raw_names'):
            continue
        raw = _recon_raw(b, None)                          # (T, raw_dim) reconstructed cells
        raw_names = list(b['raw_names'])
        grid = _grid_from_names([n.split('_', 1)[1] if '_' in n else n for n in raw_names])
        if grid is None:
            continue
        ny, nx = grid
        # split sides (side tag prefix), average over time, paint the lattice per side, hstack L‖R
        sides = sorted({n.split('_', 1)[0] for n in raw_names})
        mean_t = raw.mean(axis=0)                          # (raw_dim,)
        # group columns by (side, prefix-without-side) → cell (iy,ix)
        pat = re.compile(r'_r(\d+)c(\d+)$')
        # one map per orientation-group prefix (side-stripped), L‖R side-by-side
        by_pref = {}
        for c, nm in enumerate(raw_names):
            side = nm.split('_', 1)[0]
            stripped = nm.split('_', 1)[1]
            m = pat.search(stripped)
            if m is None:
                continue
            pre = stripped[:m.start()].rstrip('_')
            iy, ix = int(m.group(1)), int(m.group(2))
            by_pref.setdefault(pre, {})[(side, iy, ix)] = mean_t[c]
        for pre, cells in by_pref.items():
            img = np.full((ny, nx * len(sides)), np.nan, np.float32)
            for (side, iy, ix), v in cells.items():
                sx = sides.index(side)
                img[iy, sx * nx + ix] = v
            maps[f'{bn}:{pre}'] = img
    return maps
