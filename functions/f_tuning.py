# f_tuning.py — feature tuning of cells / ensembles by reverse-correlation on the conv-repo
# Gabor / motion-energy per-filter responses (created 2026-07-14).
#
# The idea (spike-triggered-average / reverse-correlation, on the CONTINUOUS calcium rate — no spike
# detection): weight each dense per-filter conv response (whole-monitor magnitude, over orientation ×
# wavelength [× temporal-freq]) by a unit's activity and read where it peaks → the unit's preferred
# tuning. Compute reuses the encoding arm's estimators (feature-agnostic): f_encode_sta (literal
# response-weighted average) and f_encode_fit_weights (whitened / ridge — sharper for correlated
# filters). The tuning READOUT (peak over the filter axis → preferred odeg/wavelength/tfreq + OSI) is
# the only new logic; it mirrors f_encoding.f_encode_latency_stats' argmax/selectivity pattern over the
# LAG axis, applied here over the FILTER axis. Regressors come from f_conv_repo.f_conv_activity; both
# the conv responses and the neural activity are on the same imaging clock (no resampling). Driven by
# the FEATURE TUNING section of VR_ca_cebra.py.

import numpy as np
import pandas as pd


def _reg_transform(X, kind):
    """Per-regressor value map for tuning: 'sqrt'/'signed_sqrt' (ENERGY→amplitude, e.g. meng) |
    'log1p'/'signed_log1p' | None."""
    if not kind:
        return X
    X = np.asarray(X, float)
    if kind == 'sqrt':
        return np.sqrt(np.clip(X, 0, None))
    if kind == 'signed_sqrt':
        return np.sign(X) * np.sqrt(np.abs(X))
    if kind == 'log1p':
        return np.log1p(np.clip(X, 0, None))
    if kind == 'signed_log1p':
        return np.sign(X) * np.log1p(np.abs(X))
    raise ValueError(f"unknown transform {kind!r} (use 'sqrt'|'signed_sqrt'|'log1p'|'signed_log1p'|None)")


def f_conv_regressors(repo_dir, kind='gabor', wavelengths=None, odegs=None, tfreqs=None,
                      combine_sides='mean', reduce='absmean', frame_ds=1,
                      transform=None, standardize=False):
    """Assemble the per-filter regressor matrix for ONE conv kind from a conv_repo. Calls
    f_conv_repo.f_conv_activity with a `select` predicate for the requested kind/param subset, then
    collapses the L/R rows per filter identity `(odeg, wavelength, tfreq)`. Returns:
      X_stim  : (T, n_filter) float32 — whole-monitor response magnitude per distinct filter, columns
                sorted by (odeg, wavelength, tfreq).
      filt_df : DataFrame, one row per column — kind, odeg, wavelength, tfreq.
    `wavelengths/odegs/tfreqs` (iterables) limit which filters load (None = all in the repo). One
    `kind` at a time so the tuning grid stays clean (gabor: odeg×wavelength; meng adds tfreq).
    `transform` maps each regressor before use — 'sqrt' brings meng ENERGY to amplitude (comparable
    to gabor magnitude). `standardize` z-scores each filter column (unit variance) so filters compare
    regardless of magnitude — recommended for method='sta' (ridge already standardizes internally)."""
    from f_conv_repo import f_conv_activity
    _wl = set(wavelengths) if wavelengths is not None else None
    _od = set(odegs) if odegs is not None else None
    _tf = set(tfreqs) if tfreqs is not None else None

    def _sel(e):
        if e.get('kind') != kind:
            return False
        if _wl is not None and e.get('wavelength') not in _wl:
            return False
        if _od is not None and e.get('odeg') not in _od:
            return False
        if _tf is not None and e.get('tfreq') not in _tf:
            return False
        return True

    df, ts = f_conv_activity(repo_dir, select=_sel, reduce=reduce, frame_ds=frame_ds)
    ts = np.asarray(ts, float)
    if df is None or len(df) == 0 or ts.size == 0:
        raise ValueError(f"f_conv_regressors: no {kind!r} filters matched in {repo_dir}")
    df = df.reset_index(drop=True)
    cols, rows = [], []
    for key, idx in df.groupby(['odeg', 'wavelength', 'tfreq'], dropna=False).groups.items():
        ii = np.asarray(list(idx), int)                       # rows (per side) for this filter
        v = ts[ii].mean(0) if combine_sides == 'mean' else ts[ii[0]]
        cols.append(v)
        od, wl, tf = key
        rows.append({'kind': kind, 'odeg': od, 'wavelength': wl, 'tfreq': tf})
    X = np.vstack(cols).T.astype(np.float32)                  # (T, n_filter)
    filt_df = pd.DataFrame(rows)
    order = filt_df.sort_values(['odeg', 'wavelength', 'tfreq'],
                                na_position='last').index.to_numpy()
    X = X[:, order]
    if transform:                                             # e.g. meng energy → amplitude (sqrt)
        X = _reg_transform(X, transform)
    if standardize:                                           # z-score each filter (comparable magnitudes)
        mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
    return X.astype(np.float32), filt_df.iloc[order].reset_index(drop=True)


def f_reverse_corr_tuning(X_stim, filt_df, Y_units, lags, method='ridge', alpha=1.0, standardize=True):
    """Reverse-correlate `Y_units` (T, n_units) against the per-filter regressors `X_stim` (T,
    n_filter) at integer frame `lags`. `method='sta'` → f_encode_sta (response-weighted average, the
    literal STA); `method='ridge'` → f_encode_fit_weights (whitened / ridge, de-correlates the
    filters). Returns the encoder fit dict (W (n_filter, n_lags, n_kept), keep_cells, lags, …) with
    `filt_df`/`method` attached. Both estimators drop constant/NaN units (keep_cells)."""
    from f_encoding import f_encode_sta, f_encode_fit_weights
    X = np.asarray(X_stim, float); Y = np.asarray(Y_units, float)
    if method == 'sta':
        fit = f_encode_sta(X, Y, lags)
    elif method == 'ridge':
        fit = f_encode_fit_weights(X, Y, lags, alpha=alpha, standardize=standardize)
    else:
        raise ValueError(f"method must be 'sta' or 'ridge', got {method!r}")
    fit['filt_df'] = filt_df.reset_index(drop=True)
    fit['method'] = method
    return fit


def f_tuning_readout(fit, fs=None, labels=None):
    """Per-unit tuning readout from a reverse-correlation fit (f_reverse_corr_tuning). For each unit:
    pick the PEAK LAG = argmax over lags of max_filter |W|; the tuning profile is `W[:, peak_lag, u]`
    (signed) at that lag. Map to the filter grid via `fit['filt_df']` and report preferred
    orientation / wavelength / tfreq, peak amplitude, peak-lag (ms if `fs`), orientation OSI (axial
    resultant over odeg), and simple SF/TF selectivity. Units are in KEPT order (aligned to W); the
    table is NOT sorted (sort a copy for display). Returns dict with `table` (one row per unit) +
    marginal curves `by_odeg/by_wavelength/by_tfreq` (n_units × n_level) and grid axes."""
    W = np.asarray(fit['W'], float)                           # (n_filter, n_lags, n_units)
    filt_df = fit['filt_df']; lags = list(fit['lags'])
    n_f, n_lag, n_u = W.shape
    fs = float(fs) if fs else 1.0
    od = filt_df['odeg'].to_numpy(float)
    wl = filt_df['wavelength'].to_numpy(float)
    tf = filt_df['tfreq'].to_numpy(float)
    sfd = filt_df['sf_cyc_deg'].to_numpy(float) if 'sf_cyc_deg' in filt_df.columns else None
    tfh = filt_df['tf_hz'].to_numpy(float) if 'tf_hz' in filt_df.columns else None
    # SF / TF marginal axes in PHYSICAL units when the repo/scan is physical (cyc/deg + Hz), else the
    # legacy pixel wavelength / cyc-frame. sf/wl are 1:1, so grouping on either gives the same curve.
    sfx = sfd if sfd is not None else wl
    tfx = tfh if tfh is not None else tf
    sf_label = 'spatial freq (cyc/deg)' if sfd is not None else 'wavelength (px)'
    tf_label = 'temporal freq (Hz)' if tfh is not None else 'tfreq (cyc/frame)'
    uniq_od = np.unique(od[~np.isnan(od)])
    uniq_wl = np.unique(sfx[~np.isnan(sfx)])                   # SF-axis values (cyc/deg if physical)
    uniq_tf = np.unique(tfx[~np.isnan(tfx)]) if np.any(~np.isnan(tfx)) else np.array([])

    keep = fit.get('keep_cells')
    if labels is not None and keep is not None and len(labels) == len(keep):
        labels = list(np.asarray(labels, dtype=object)[np.asarray(keep, bool)])
    peak_lag_idx = np.argmax(np.abs(W).max(axis=0), axis=0)   # (n_units,)

    rows, by_od, by_wl, by_tf = [], [], [], []
    _th = np.radians(uniq_od)
    for u in range(n_u):
        pl = int(peak_lag_idx[u])
        tun = W[:, pl, u]; at = np.abs(tun)
        pf = int(np.argmax(at))                               # preferred filter
        m_od = np.array([at[od == o].mean() for o in uniq_od]) if uniq_od.size else np.array([])
        m_wl = np.array([at[sfx == w].mean() for w in uniq_wl]) if uniq_wl.size else np.array([])
        m_tf = np.array([at[tfx == t].mean() for t in uniq_tf]) if uniq_tf.size else np.array([])
        if m_od.size and m_od.sum() > 0:                      # axial (orientation) resultant → OSI
            vec = np.sum(m_od * np.exp(2j * _th))
            osi = float(np.abs(vec) / m_od.sum())
            pref_circ = float(np.degrees(0.5 * np.angle(vec)) % 180)
        else:
            osi, pref_circ = np.nan, np.nan
        sf_sel = (float((m_wl.max() - m_wl.min()) / (m_wl.max() + m_wl.min() + 1e-12))
                  if m_wl.size > 1 else np.nan)
        tf_sel = (float((m_tf.max() - m_tf.min()) / (m_tf.max() + m_tf.min() + 1e-12))
                  if m_tf.size > 1 else np.nan)
        _asd = float(at.std())                                # peak vs the filter distribution (scale-free)
        peak_z = float((at[pf] - at.mean()) / _asd) if _asd > 0 else np.nan
        _row = {'unit': (labels[u] if labels is not None else u),
                'pref_odeg': float(od[pf]), 'pref_odeg_circ': pref_circ,
                'pref_wavelength': float(wl[pf]),
                'pref_tfreq': (float(tf[pf]) if not np.isnan(tf[pf]) else np.nan),
                'peak': float(tun[pf]), 'peak_abs': float(at[pf]), 'peak_z': peak_z,
                'peak_lag_ms': float(lags[pl] * 1000.0 / fs),
                'osi': osi, 'sf_sel': sf_sel, 'tf_sel': tf_sel}
        if sfd is not None:                                   # physical units (cyc/deg + Hz) if present
            _row['pref_sf_cyc_deg'] = float(sfd[pf])
        if tfh is not None:
            _row['pref_tf_hz'] = float(tfh[pf]) if not np.isnan(tfh[pf]) else np.nan
        rows.append(_row)
        by_od.append(m_od); by_wl.append(m_wl); by_tf.append(m_tf)
    by_lag = np.abs(W).max(axis=0).T                          # (n_units, n_lags) tuning strength per lag
    lags_ms = [float(l * 1000.0 / fs) for l in lags]
    return {'table': pd.DataFrame(rows),
            'by_odeg': np.array(by_od), 'by_wavelength': np.array(by_wl),
            'by_tfreq': (np.array(by_tf) if uniq_tf.size else None),
            'by_lag': by_lag, 'lags_ms': lags_ms,
            'uniq_odeg': uniq_od, 'uniq_wavelength': uniq_wl, 'uniq_tfreq': uniq_tf,
            'sf_label': sf_label, 'tf_label': tf_label,
            'W': W, 'filt_df': filt_df, 'lags': lags, 'peak_lag_idx': peak_lag_idx,
            'keep_cells': keep, 'method': fit.get('method')}


def f_select_tuning_units(source, Y_neu_raw, ens_out=None, ens_table=None, dimred_params=None,
                          ens_src_which='sig', fs=None):
    """Build the (T, n_units) activity matrix for tuning from a source (mirrors neural_source):
      'cells'          → Y_neu_raw (all active cells)
      'ensembles'      → ens_out['scores'][sel].T (sel = sig | all via ens_src_which)
      'ens_cells'      → raw cells that are members of the selected ensembles (union)
      'reliable_cells' → f_reduce_neural(mode='reliable_cells') cells
    Returns (Y_units, labels, kind_str). `Y_neu_raw` is (T, n_active) in the extraction cell order."""
    Y = np.asarray(Y_neu_raw, float)
    if source == 'cells':
        return Y, list(range(Y.shape[1])), 'cells'
    if source in ('ensembles', 'ens_cells'):
        if ens_out is None:
            raise ValueError(f"source={source!r} needs ens_out (run the ensembles cell)")
        if ens_src_which == 'all':
            sel = list(range(len(ens_out['cells']['ens_list'])))
        else:
            sel = ens_table.loc[ens_table['sig'].astype(bool), 'ensemble'].astype(int).tolist()
        if source == 'ensembles':
            if not sel:
                raise ValueError('no ensembles selected for source=ensembles')
            return (np.asarray(ens_out['scores'][np.asarray(sel, int), :].T, float),
                    [f'ens{i}' for i in sel], 'ensembles')
        act = np.flatnonzero(ens_out['active_cells_mask'])    # ens_cells
        sel = [i for i in sel if len(ens_out['cells']['ens_list'][i]) > 0]
        if not sel:
            raise ValueError('no non-empty ensembles selected for source=ens_cells')
        memb = sorted(set().union(*[set(np.asarray(ens_out['cells']['ens_list'][i], int).tolist())
                                    for i in sel]))
        cols = act[np.asarray(memb, int)]
        return Y[:, cols], list(int(c) for c in cols), 'ens_cells'
    if source == 'reliable_cells':
        from f_ensembles import f_reduce_neural
        out = f_reduce_neural(dimred_params, Y.T, mode='reliable_cells', fs=fs)
        Yr = np.asarray(out['low_rank'], float)
        labels = (list(int(c) for c in np.flatnonzero(out['keep_mask']))
                  if 'keep_mask' in out else list(range(Yr.shape[1])))
        return Yr, labels, 'reliable_cells'
    raise ValueError(f"unknown tuning source {source!r}")


def f_tuning_significance(X_stim, filt_df, Y_units, lags, method='sta', n_shuffle=200,
                          alpha=0.05, seed=0, min_shift=None, labels=None):
    """Per-unit tuning SIGNIFICANCE via a circular-shift null (the shuffle test, not a noise ceiling).
    Statistic = peak |tuning| = max over (filter, lag) of |W| for each unit. For `n_shuffle` draws each
    unit's activity is independently circular-shifted (decouples it from the stimulus, preserves its
    autocorrelation) and the tuning refit; the null peak distribution gives per-unit z = (obs −
    null_mean)/null_std and one-sided p = (1 + #{null ≥ obs})/(n_shuffle + 1), Benjamini-Hochberg
    FDR-corrected at `alpha`. Returns a DataFrame aligned to the KEPT units of the observed fit:
      col      : column index into Y_units (the kept unit's position)
      unit     : label (from `labels`, else col)
      peak_abs, null_mean, null_std, z, p, p_fdr, sig
    Reusable for any source (cells / ensembles / …). method 'sta' | 'ridge' matches f_reverse_corr_tuning."""
    from f_encoding import f_encode_sta, f_encode_fit_weights
    X = np.asarray(X_stim, float); Y = np.asarray(Y_units, float)
    T = Y.shape[0]

    def _stat(Ym):
        fit = (f_encode_sta(X, Ym, lags) if method == 'sta'
               else f_encode_fit_weights(X, Ym, lags, standardize=True))
        W = np.abs(np.asarray(fit['W'], float))               # (n_filter, n_lags, n_kept)
        return W.max(axis=(0, 1)), np.asarray(fit['keep_cells'], bool)

    obs, keep = _stat(Y)
    cols = np.flatnonzero(keep)                               # Y_units columns that survived
    Yk = Y[:, cols]
    n_u = Yk.shape[1]
    rng = np.random.default_rng(seed)
    lo = int(min_shift) if min_shift is not None else max(1, T // 100)
    null = np.empty((n_shuffle, n_u), float)
    for s in range(n_shuffle):
        shifts = rng.integers(lo, T - lo, size=n_u)
        Ysh = np.column_stack([np.roll(Yk[:, j], int(shifts[j])) for j in range(n_u)])
        null[s], _ = _stat(Ysh)
    nmean = null.mean(0); nstd = null.std(0)
    z = np.where(nstd > 0, (obs - nmean) / np.where(nstd > 0, nstd, 1.0), np.nan)
    p = (1.0 + (null >= obs[None, :]).sum(0)) / (n_shuffle + 1.0)
    order = np.argsort(p)                                     # Benjamini-Hochberg FDR
    p_sorted = p[order] * n_u / np.arange(1, n_u + 1)
    p_bh = np.minimum.accumulate(p_sorted[::-1])[::-1]
    p_fdr = np.empty(n_u); p_fdr[order] = np.clip(p_bh, 0, 1)
    lab = [labels[c] for c in cols] if labels is not None else list(cols)
    return pd.DataFrame({'col': cols, 'unit': lab, 'peak_abs': obs, 'null_mean': nmean,
                         'null_std': nstd, 'z': z, 'p': p, 'p_fdr': p_fdr, 'sig': p_fdr <= alpha})


# ─────────────────────────────────────────────────────────────────────────────
# Verification views — re-read the conv-repo dense maps for ONE unit's PREFERRED filter to show
# WHERE the tuning sits (location map) and WHAT the selected feature looks like in monitor space
# (activity-weighted STA). Both take a `filt_row` (a row of readout['filt_df']: kind/odeg/wavelength/
# tfreq) + the unit's activity trace (T,).
# ─────────────────────────────────────────────────────────────────────────────

def f_preferred_filt_row(readout, u):
    """The preferred-filter row (kind/odeg/wavelength/tfreq) for kept-unit `u`: argmax |W| at its peak
    lag, indexed into readout['filt_df']. Returns a pandas Series."""
    W = readout['W']; pl = int(readout['peak_lag_idx'][u])
    pf = int(np.argmax(np.abs(W[:, pl, u])))
    return readout['filt_df'].iloc[pf]


def _filt_match(filt_row):
    """Predicate matching a conv-repo entry to a filt_row (kind/odeg/wavelength/tfreq)."""
    kind = str(filt_row['kind']); od = float(filt_row['odeg']); wl = float(filt_row['wavelength'])
    tf = filt_row.get('tfreq', np.nan)
    def _m(e):
        if e.get('kind') != kind or abs(e.get('odeg', 1e9) - od) > 1e-6 or abs(e.get('wavelength', 1e9) - wl) > 1e-6:
            return False
        if not (isinstance(tf, float) and np.isnan(tf)):
            return e.get('tfreq') is not None and abs(e['tfreq'] - tf) < 1e-9
        return True
    return _m


def _combine_monitors(sidemap_fn, sides, side):
    """Combine per-monitor 2D maps: side='both' → hstack (L on left, R on right → wide); 'L'/'R' →
    that monitor alone. `sidemap_fn(s)` returns the 2D map for monitor `s`; `sides` = list of the
    monitor tag per column (to know which are present)."""
    order = [s for s in ('L', 'R') if s in sides] or list(dict.fromkeys(sides))
    if side == 'both':
        return np.hstack([sidemap_fn(s) for s in order])
    if side in order:
        return sidemap_fn(side)
    raise ValueError(f"side must be 'both'/'L'/'R'; repo has {sorted(set(sides))}")


def f_tuning_location_map(repo_dir, filt_row, activity, ds=8, metric='corr', side='both'):
    """Reverse-correlation LOCATION map for one filter: point-sample the filter's dense response at
    stride `ds` (repo, RAM-safe) and correlate EACH monitor location's response with `activity` →
    a map of WHERE on the screen the tuning sits (peak = RF center). `side`: 'both' = L and R monitors
    placed SIDE BY SIDE (L left, R right) → wide (ny, 2·nx) | 'L' | 'R' (single monitor, ny×nx).
    metric 'corr' (Pearson, signed) | 'sta' (rectified-activity-weighted mean). Returns the map or None."""
    import re
    from f_conv_repo import f_sample_conv_to_block
    X, names = f_sample_conv_to_block(repo_dir, int(ds), select=_filt_match(filt_row), verbose=False)
    X = np.asarray(X, float)
    if X.size == 0:
        return None
    sides, iy, ix = [], [], []
    for nm in names:
        m = re.search(r'r(\d+)c(\d+)$', nm); iy.append(int(m.group(1))); ix.append(int(m.group(2)))
        sides.append(nm.split('_', 1)[0])
    ny, nx = max(iy) + 1, max(ix) + 1
    a = np.asarray(activity, float); T = min(len(a), X.shape[0]); a = a[:T]; X = X[:T]
    if metric == 'sta':
        w = np.clip(a - np.median(a), 0, None)
        val = (X * w[:, None]).sum(0) / (w.sum() + 1e-12)             # per-column weighted mean
    else:
        am = a - a.mean(); Xc = X - X.mean(0)
        val = (Xc * am[:, None]).sum(0) / ((np.sqrt((Xc ** 2).sum(0)) + 1e-12) * (np.sqrt((am ** 2).sum()) + 1e-12))

    def _sidemap(s):                                                  # (ny,nx) map for one monitor
        M = np.full((ny, nx), np.nan)
        for j, (sj, y, x) in enumerate(zip(sides, iy, ix)):
            if sj == s:
                M[y, x] = val[j]
        return M
    return _combine_monitors(_sidemap, sides, side)


def f_tuning_feature_sta(repo_dir, filt_row, activity, frame_ds=20, side='both'):
    """Activity-weighted STA of the PREFERRED filter's FULL-res dense map → the feature in monitor
    space = 'what was on screen when this unit was active'. Reads every `frame_ds`-th frame (mmap +
    subsample-then-dequant, RAM-safe), averages the map weighted by rectified `activity`, PER MONITOR.
    `side`: 'both' = L and R side by side (L left, R right) → wide (H, 2·W) | 'L' | 'R'. Returns the
    image or None."""
    from pathlib import Path
    from f_conv_repo import f_list_conv_filters, _dequant_u16
    ents = [e for e in f_list_conv_filters(repo_dir) if _filt_match(filt_row)(e)]
    if not ents:
        return None
    a = np.asarray(activity, float)
    per = {}                                                                  # side → [acc (H,W), wsum]
    for e in ents:
        q = np.load(Path(repo_dir) / e['file'], mmap_mode='r')                # uint16 (T,H,W)
        idx = np.arange(0, min(q.shape[0], len(a)), int(frame_ds))
        sub = _dequant_u16(np.asarray(q[idx]), e['vmin'], e['vmax'])          # (n,H,W) float32
        w = np.clip(a[idx] - np.median(a), 0, None)
        s = (sub * w[:, None, None]).sum(0); sd = e['side']
        if sd in per:
            per[sd][0] += s; per[sd][1] += float(w.sum())
        else:
            per[sd] = [s, float(w.sum())]
    maps = {sd: acc / (ws + 1e-12) for sd, (acc, ws) in per.items()}
    return _combine_monitors(lambda s: maps[s], list(per.keys()), side)
