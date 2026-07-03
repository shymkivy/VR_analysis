# -*- coding: utf-8 -*-
"""
f_feature_optim.py — feature-PARAMETER vs decoding-PERFORMANCE sweep (TODO #47/#66).

Compute-only. Answers "how few features, at what extraction settings, still decode?"
for the gabor bank: grid density × #orientations × spatial-frequency subset × post-PCA
rank (+ decoder lags/type), swept marginally or jointly.

TWO scoring paths share the same extraction/subset/PCA machinery:
  - RECONSTRUCTION (feature → MOVIE, Chain-1): f_sweep_recon / f_sweep_marginal /
    f_plot_sweep_marginal — a FEATURE-AGNOSTIC engine (gabor + meng reuse it) scoring via
    f_feature_recon.f_recon_score (pooled R²). Driven by VR_feature_optim.py. [primary]
  - NEURAL (block decoded FROM neural PCs): f_sweep_gabor_params / f_score_block [parked].

Efficiency — the gabor convolution (the slow part, per orientation×SF) is
GRID-INDEPENDENT, so we convolve a SUPERSET bank ONCE and:
  - pool it to EVERY requested grid density (cheap `_pool`),
  - subset orientations / spatial frequencies by COLUMN (free),
  - sweep post-PCA rank by nested truncation (free).
Only the master bank extraction pays the conv cost — everything downstream is
column selection + PCA + CV. Run on a FRAME SUBSET (extraction is minutes/dataset).

Driven by VR_feature_optim.py. Reuses the production extractor internals
(`_gabor_pair`, `_spatial_fftconv_chunked`, `_pool` math, `_pca_fit`) so the swept
features are byte-identical to what f_visual_features.build_visual_blocks would build.
"""

import time
import numpy as np

from f_visual_features import (f_grid_bounds, _gabor_pair, _spatial_fftconv_chunked,
                               _temporal_quad, _pca_fit)
from f_feature_helpers import f_resample_to_imaging
from f_decoding import f_pca_prefix, f_shuffle_neural, f_lags_from_ms
from f_cebra_helpers import f_blocked_cv_r2
from f_feature_recon import f_recon_score      # reconstruction scorer (feature → movie)


# ─────────────────────────────────────────────────────────────────────────────
# Master-bank extraction: convolve once, pool to every grid
# ─────────────────────────────────────────────────────────────────────────────

def f_extract_gabor_multigrid(movie, grids, n_orient=8, wavelengths=(4, 6, 9, 13, 19),
                              stats=('magnitude',), sigma_scale=0.5, gamma=1.0,
                              chunk_t=2000, verbose=True):
    """One monitor side → pooled gabor channels at SEVERAL grid densities, sharing the
    convolution. For each (orientation × wavelength) the quadrature pair is convolved
    ONCE (the expensive step); the requested `stats` maps are then mean-pooled into
    every grid in `grids`.

    Parameters mirror f_visual_features.f_gabor_features; `grids` is a list of (ny,nx).
    Returns {grid: (X (T, d), names)} on the movie's (behavior) clock, channel order
    cell-major, names 'gab{tok}_o{io}sf{isf}_r{iy}c{ix}' — identical to the production
    extractor so a subset here == a smaller-bank build there.
    """
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    wavelengths = list(np.atleast_1d(wavelengths))
    n_sf = len(wavelengths)
    thetas = np.arange(n_orient) * np.pi / n_orient

    # (token, fn(re, ro, mag) -> (T,H,W)) per requested stat — same as f_gabor_features.
    chan = []
    if 'magnitude' in stats:
        chan.append(('M', lambda re, ro, mag: mag))
    if 'phase' in stats:
        chan.append(('Pc', lambda re, ro, mag: np.where(mag > 1e-6, re / np.maximum(mag, 1e-6), 0.0)))
        chan.append(('Ps', lambda re, ro, mag: np.where(mag > 1e-6, ro / np.maximum(mag, 1e-6), 0.0)))
    if 'even' in stats:
        chan.append(('Ev', lambda re, ro, mag: np.abs(re)))
    if not chan:
        raise ValueError(f"no valid stats in {stats!r} (magnitude/phase/even)")

    # Per-grid pooling operator (bounds + cell areas), built once.
    grid_ops = {}
    for g in grids:
        ny, nx = g
        ys, xs = f_grid_bounds(H, W, g)
        area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)
        grid_ops[g] = (ys, xs, area)

    def _pool(F, g):
        ys, xs, area = grid_ops[g]
        return (np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=1), xs[:-1], axis=2)
                / area[None])

    # pooled[grid][(io, isf, tok)] -> (T, ny, nx)
    pooled = {g: {} for g in grids}
    _t0 = time.perf_counter()
    for io, th in enumerate(thetas):
        for isf, lam in enumerate(wavelengths):
            ev, od = _gabor_pair(th, lam, sigma_scale * lam, gamma)
            re = _spatial_fftconv_chunked(movie, ev, chunk_t)
            ro = _spatial_fftconv_chunked(movie, od, chunk_t)
            mag = np.sqrt(re * re + ro * ro)
            for tok, fn in chan:
                m = fn(re, ro, mag)
                for g in grids:
                    pooled[g][(io, isf, tok)] = _pool(m, g).astype(np.float32)
    if verbose:
        print(f'    ⏱ gabor multigrid conv (o={n_orient} sf={n_sf} '
              f'grids={grids}): {time.perf_counter() - _t0:.2f}s')

    out = {}
    for g in grids:
        ny, nx = g
        feats, names = [], []
        for iy in range(ny):
            for ix in range(nx):
                for io in range(n_orient):
                    for isf in range(n_sf):
                        for tok, _ in chan:
                            feats.append(pooled[g][(io, isf, tok)][:, iy, ix])
                            names.append(f'gab{tok}_o{io}sf{isf}_r{iy}c{ix}')
        out[g] = (np.column_stack(feats).astype(np.float32), names)
    return out


def f_extract_meng_multigrid(movie, grids, n_orient=4, wavelengths=(8,),
                             temporal_freqs=(0.15,), temporal_support=11, temporal_sigma=None,
                             sigma_scale=0.5, gamma=1.0, output='opponent', chunk_t=1000,
                             verbose=True):
    """One monitor side → pooled Adelson–Bergen MOTION-ENERGY channels at SEVERAL grid densities,
    sharing the spatial+temporal convolution. Mirrors f_visual_features.f_motion_energy_features
    (complex spatial Gabor × complex temporal Gabor → opponent |·|² energy, halo-chunked) but
    pools each energy map to EVERY grid (conv/temporal-filter ONCE). Names
    'meng{dt}_o{io}sf{isf}tf{itf}_r{iy}c{ix}' (dt='opp' | 'd0'/'d1' for output='both').
    Returns {grid: (X (T,d), names)}, byte-compatible with the production extractor per grid."""
    from scipy.signal import fftconvolve
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    wavelengths = list(np.atleast_1d(wavelengths))
    temporal_freqs = list(np.atleast_1d(temporal_freqs))
    n_sf, n_tf = len(wavelengths), len(temporal_freqs)
    thetas = np.arange(n_orient) * np.pi / n_orient
    halo = temporal_support // 2
    tsig = temporal_sigma if temporal_sigma else temporal_support / 4.0

    grid_ops = {}
    for g in grids:
        ys, xs = f_grid_bounds(H, W, g)
        area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)
        grid_ops[g] = (ys, xs, area)

    def _pool(F, g):
        ys, xs, area = grid_ops[g]
        return (np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=1), xs[:-1], axis=2)
                / area[None])

    sgab = {(io, isf): _gabor_pair(thetas[io], wavelengths[isf], sigma_scale * wavelengths[isf], gamma)
            for io in range(n_orient) for isf in range(n_sf)}
    tk = {itf: _temporal_quad(tf, temporal_support, tsig) for itf, tf in enumerate(temporal_freqs)}
    n_out = 1 if output == 'opponent' else 2
    store = {g: {(io, isf, itf, d): np.empty((T, g[0], g[1]), np.float32)
                 for io in range(n_orient) for isf in range(n_sf)
                 for itf in range(n_tf) for d in range(n_out)} for g in grids}

    _t0 = time.perf_counter()
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        lo, hi = max(0, s - halo), min(T, e + halo)
        sub = movie[lo:hi]
        vs, ve = s - lo, s - lo + (e - s)
        for (io, isf), (ev, od) in sgab.items():
            re = fftconvolve(sub, ev[None], mode='same', axes=(1, 2))
            ro = fftconvolve(sub, od[None], mode='same', axes=(1, 2))
            Cs = (re + 1j * ro).astype(np.complex64)
            for itf in range(n_tf):
                te, to = tk[itf]
                R = fftconvolve(Cs, (te + 1j * to).reshape(-1, 1, 1), mode='same', axes=0)
                L = fftconvolve(Cs, (te - 1j * to).reshape(-1, 1, 1), mode='same', axes=0)
                eR = (R.real ** 2 + R.imag ** 2).astype(np.float32)[vs:ve]
                eL = (L.real ** 2 + L.imag ** 2).astype(np.float32)[vs:ve]
                for g in grids:
                    if output == 'opponent':
                        store[g][(io, isf, itf, 0)][s:e] = _pool(eR - eL, g)
                    else:
                        store[g][(io, isf, itf, 0)][s:e] = _pool(eR, g)
                        store[g][(io, isf, itf, 1)][s:e] = _pool(eL, g)
    if verbose:
        print(f'    ⏱ meng multigrid (o={n_orient} sf={n_sf} tf={n_tf} '
              f'grids={grids}): {time.perf_counter() - _t0:.2f}s')

    dtoks = ['opp'] if output == 'opponent' else ['d0', 'd1']
    out = {}
    for g in grids:
        ny, nx = g
        feats, names = [], []
        for iy in range(ny):
            for ix in range(nx):
                for io in range(n_orient):
                    for isf in range(n_sf):
                        for itf in range(n_tf):
                            for d, dt in enumerate(dtoks):
                                feats.append(store[g][(io, isf, itf, d)][:, iy, ix])
                                names.append(f'meng{dt}_o{io}sf{isf}tf{itf}_r{iy}c{ix}')
        out[g] = (np.column_stack(feats).astype(np.float32), names)
    return out


def f_build_banks(extract_fn, movies, side_tags, beh_t, frame_t, pulse_delay,
                  resample_kind='linear', **extract_kw):
    """Both monitor sides → per-grid RAW banks on the IMAGING clock, for ANY multigrid
    extractor. `extract_fn(movie, **extract_kw) -> {grid: (X, names)}` per side (e.g.
    f_extract_gabor_multigrid, or a meng one later). Concatenates sides (tag-prefixed names,
    as build_visual_blocks does) and resamples to the imaging clock. NO PCA — subset + PCA
    happen per sweep point. Feature-agnostic → gabor and meng share it.
    Returns {grid: {'X': (T_img, d) raw, 'names': [...], 'grid': grid}}."""
    per_side = [extract_fn(m, **extract_kw) for m in movies]
    banks = {}
    for g in per_side[0]:
        Xs, names = [], []
        for ps, tag in zip(per_side, side_tags):
            Xg, base = ps[g]
            Xs.append(Xg)
            names.extend(f'{tag}_{b}' for b in base)
        X_beh = np.concatenate(Xs, axis=1)
        X_img = f_resample_to_imaging(beh_t, X_beh, frame_t, pulse_delay, kind=resample_kind)
        banks[g] = {'X': X_img, 'names': names, 'grid': g}
    return banks


def f_build_gabor_banks(movies, side_tags, beh_t, frame_t, pulse_delay, grids,
                        n_orient=8, wavelengths=(4, 6, 9, 13, 19), stats=('magnitude',),
                        sigma_scale=0.5, gamma=1.0, chunk_t=2000, resample_kind='linear',
                        verbose=True):
    """Gabor banks — thin wrapper over f_build_banks with f_extract_gabor_multigrid; also
    tags each bank with n_orient/wavelengths/stats meta (used by the config generator)."""
    banks = f_build_banks(f_extract_gabor_multigrid, movies, side_tags, beh_t, frame_t,
                          pulse_delay, resample_kind=resample_kind, grids=grids,
                          n_orient=n_orient, wavelengths=wavelengths, stats=stats,
                          sigma_scale=sigma_scale, gamma=gamma, chunk_t=chunk_t, verbose=verbose)
    for g in banks:
        banks[g].update(n_orient=n_orient,
                        wavelengths=list(np.atleast_1d(wavelengths)), stats=tuple(stats))
    return banks


def f_build_meng_banks(movies, side_tags, beh_t, frame_t, pulse_delay, grids,
                       n_orient=4, wavelengths=(8,), temporal_freqs=(0.15,),
                       temporal_support=11, temporal_sigma=None, sigma_scale=0.5, gamma=1.0,
                       output='opponent', chunk_t=1000, resample_kind='linear', verbose=True):
    """Meng banks — thin wrapper over f_build_banks with f_extract_meng_multigrid; tags each
    bank with n_orient/wavelengths/temporal_freqs/output meta (used by f_meng_configs)."""
    banks = f_build_banks(f_extract_meng_multigrid, movies, side_tags, beh_t, frame_t,
                          pulse_delay, resample_kind=resample_kind, grids=grids, n_orient=n_orient,
                          wavelengths=wavelengths, temporal_freqs=temporal_freqs,
                          temporal_support=temporal_support, temporal_sigma=temporal_sigma,
                          sigma_scale=sigma_scale, gamma=gamma, output=output, chunk_t=chunk_t,
                          verbose=verbose)
    for g in banks:
        banks[g].update(n_orient=n_orient, wavelengths=list(np.atleast_1d(wavelengths)),
                        temporal_freqs=list(np.atleast_1d(temporal_freqs)), output=output)
    return banks


# ─────────────────────────────────────────────────────────────────────────────
# Column subsetting (orientation / spatial frequency / stat) — free
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gabor_name(name):
    """'L_gabM_o2sf1_r0c3' -> dict(side='L', tok='M', io=2, isf=1, iy=0, ix=3)."""
    side, rest = (name.split('_', 1) if '_' in name else (None, name))
    tok = rest[3:rest.index('_o')]                 # after 'gab', before '_o'
    io = int(rest[rest.index('_o') + 2:rest.index('sf')])
    isf = int(rest[rest.index('sf') + 2:rest.index('_r')])
    iy = int(rest[rest.index('_r') + 2:rest.index('c')])
    ix = int(rest[rest.index('c') + 1:])
    return {'side': side, 'tok': tok, 'io': io, 'isf': isf, 'iy': iy, 'ix': ix}


def _parse_meng_name(name):
    """'L_mengopp_o2sf0tf1_r0c3' -> dict(side, tok='opp', io=2, isf=0, itf=1, iy=0, ix=3)."""
    side, rest = (name.split('_', 1) if '_' in name else (None, name))
    tok = rest[4:rest.index('_o')]                 # after 'meng', before '_o'
    io = int(rest[rest.index('_o') + 2:rest.index('sf')])
    isf = int(rest[rest.index('sf') + 2:rest.index('tf')])
    itf = int(rest[rest.index('tf') + 2:rest.index('_r')])
    iy = int(rest[rest.index('_r') + 2:rest.index('c')])
    ix = int(rest[rest.index('c') + 1:])
    return {'side': side, 'tok': tok, 'io': io, 'isf': isf, 'itf': itf, 'iy': iy, 'ix': ix}


def _subset_cols(names, parse_fn, keep):
    """Column indices whose parsed fields all match `keep` (dict field→iterable|None; None =
    keep all on that field). Generic over any channel naming (gabor/meng) via `parse_fn`."""
    keep = {k: (None if v is None else set(v)) for k, v in keep.items()}
    return [i for i, nm in enumerate(names)
            if all(keep[f] is None or parse_fn(nm)[f] in keep[f] for f in keep)]


def f_orient_indices(n_master, k):
    """Evenly-spaced orientation indices for a k-orientation bank drawn from an
    n_master bank. Requires k to divide n_master (so the kept angles equal a true
    k-orientation bank: 0, π/k, …). e.g. n_master=8, k=4 -> [0,2,4,6]."""
    if n_master % k:
        raise ValueError(f'n_orient={k} must divide the master n_orient={n_master} '
                         f'(so kept angles match a real {k}-orientation bank)')
    return list(range(0, n_master, n_master // k))


def f_gabor_subset(bank, io_keep=None, isf_keep=None, stats_keep=None):
    """Column-select a raw gabor bank to kept orientations (io_keep), spatial-frequency
    indices (isf_keep), and stat tokens (stats_keep, e.g. ('M',)). None = keep all on
    that axis. Returns {'X', 'names'} (still raw — PCA is applied by the scorer)."""
    cols = _subset_cols(bank['names'], _parse_gabor_name,
                        {'io': io_keep, 'isf': isf_keep, 'tok': stats_keep})
    return {'X': bank['X'][:, cols], 'names': [bank['names'][i] for i in cols]}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — decode the feature block FROM neural (blocked CV, multi-output R²)
# ─────────────────────────────────────────────────────────────────────────────

def _cv_kwargs(cfg):
    """CV kwargs shared with the production decode sweep (f_run_decode_sweep)."""
    return dict(n_folds=cfg.get('n_folds', 5),
                embargo=int(round(cfg.get('embargo_sec', 0.0) * cfg['fs'])),
                standardize=cfg.get('standardize_emb', True),
                decoder=cfg.get('decoder', 'ridge'),
                ridge_alpha=cfg.get('ridge_alpha', 1.0),
                k_nn=cfg.get('k_nn', 15),
                ridgecv_alphas=cfg.get('ridgecv_alphas', None),
                multioutput=cfg.get('multioutput', 'variance_weighted'))


def f_score_block(block_X, pca_neu, cfg, n_pca=None):
    """Blocked-CV multi-output R² decoding block_X (optionally truncated to its first
    n_pca columns — nested PCA) from the neural PCs pca_neu. Scalar (variance-weighted
    across the block's output dims, so high-variance channels dominate)."""
    Y = block_X if n_pca is None else block_X[:, :n_pca]
    if Y.shape[1] == 0:
        return np.nan
    return float(f_blocked_cv_r2(pca_neu, Y, **_cv_kwargs(cfg)))


def f_pca_block_scores(sub_X, n_pca_max):
    """PCA-fit a subset's raw channels once at n_pca_max; return the score matrix so
    smaller ranks are exact prefixes (nested). Returns (scores (T, k), pca)."""
    k = min(n_pca_max, sub_X.shape[1])
    scores, pca = _pca_fit(sub_X, k)
    return scores, pca


# ─────────────────────────────────────────────────────────────────────────────
# Sweep orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def f_sweep_gabor_params(banks, Y_neu, cfg, n_orient_list=None, wl_subsets=None,
                         stats_keep=('M',), n_pca_list=(5, 10, 20, 40),
                         n_pcs_neu=30, add_null=False, null_seed=0, verbose=True):
    """Sweep gabor extraction parameters and score each by neural decodability.

    Outer axes (re-PCA per point, cheap): grid (keys of `banks`) × n_orient (subset of
    the master) × wavelength-subset. Inner axis (nested, free): n_pca. For each point the
    kept channels are PCA'd once at max(n_pca_list) and scored at every n_pca via prefix.

    banks : f_build_gabor_banks output {grid: {'X','names','n_orient','wavelengths',...}}.
    Y_neu : (T_img, N) neural (SAME frames as the banks' movie subset).
    cfg   : must include 'fs' (imaging Hz); optional CV knobs (see _cv_kwargs).
    n_orient_list : orientation counts to test (each must divide the master n_orient);
                    None → just the master count.
    wl_subsets    : list of wavelength-value lists to test (each a subset of the master
                    wavelengths); None → the full master set only.
    stats_keep    : stat tokens to keep, e.g. ('M',) magnitude, ('M','Pc','Ps') mag+phase.
    n_pca_list    : post-PCA ranks to score.
    n_pcs_neu     : #neural PCs used as the decoder input (held fixed across the sweep).
    add_null      : also score against a per-cell circshift neural null (chance ref).

    Returns a list of row dicts (grid, n_orient, n_sf, wavelengths, stat, n_pca, dim,
    r2[, r2_null]) — build a DataFrame in the driver.
    """
    pca_neu = f_pca_prefix(Y_neu, n_pcs_neu)
    pca_neu_null = None
    if add_null:
        pca_neu_null = f_pca_prefix(f_shuffle_neural(Y_neu, seed=null_seed), n_pcs_neu)

    n_pca_max = max(n_pca_list)
    rows = []
    for g, bank in banks.items():
        master_no = bank['n_orient']
        master_wl = list(bank['wavelengths'])
        no_list = n_orient_list or [master_no]
        wl_list = wl_subsets or [master_wl]
        for no in no_list:
            io_keep = f_orient_indices(master_no, no)
            for wls in wl_list:
                isf_keep = [master_wl.index(w) for w in wls]
                sub = f_gabor_subset(bank, io_keep=io_keep, isf_keep=isf_keep,
                                     stats_keep=stats_keep)
                if sub['X'].shape[1] == 0:
                    continue
                scores, _ = f_pca_block_scores(sub['X'], n_pca_max)
                stat_lbl = '+'.join(stats_keep)
                for npca in n_pca_list:
                    k = min(npca, scores.shape[1])
                    row = {'grid': f'{g[0]}x{g[1]}', 'grid_cells': g[0] * g[1],
                           'n_orient': no, 'n_sf': len(wls),
                           'wavelengths': tuple(wls), 'stat': stat_lbl,
                           'n_pca': k, 'raw_dim': sub['X'].shape[1],
                           'r2': f_score_block(scores, pca_neu, cfg, n_pca=k)}
                    if add_null:
                        row['r2_null'] = f_score_block(scores, pca_neu_null, cfg, n_pca=k)
                    rows.append(row)
                    if verbose:
                        _n = f"  null={row['r2_null']:+.3f}" if add_null else ''
                        print(f"  grid={row['grid']:>5s} o={no} sf={len(wls)} "
                              f"npca={k:>2d} dim={sub['X'].shape[1]:>4d}  "
                              f"r2={row['r2']:+.3f}{_n}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plots — decoding score vs each swept parameter (return None; Spyder-echo safe)
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_param_curves(rows, x, hue=None, y='r2', title='', ax=None,
                        show_null=True, agg='max'):
    """Line plot of decoding score vs one parameter axis, one line per `hue` level.
    `rows` = f_sweep_gabor_params output (list of dicts). For each (x, hue) the scores
    over the OTHER swept axes are aggregated by `agg` ('max' = best achievable, 'mean').
    Draws the null band (if present) as a dashed reference. Returns None."""
    import matplotlib.pyplot as plt
    import numpy as _np
    if not rows:
        print('no rows to plot'); return None
    made = ax is None
    if made:
        _, ax = plt.subplots(figsize=(6, 4))
    aggf = _np.nanmax if agg == 'max' else _np.nanmean

    def _key(r):
        v = r[hue] if hue else '_'
        return tuple(v) if isinstance(v, tuple) else v

    levels = sorted({_key(r) for r in rows}, key=lambda z: (str(type(z)), z))
    for lev in levels:
        sel = [r for r in rows if _key(r) == lev]
        xs = sorted({r[x] if not isinstance(r[x], tuple) else len(r[x]) for r in sel})
        ys = [aggf([r[y] for r in sel
                    if (r[x] if not isinstance(r[x], tuple) else len(r[x])) == xv]) for xv in xs]
        ax.plot(xs, ys, '-o', label=(f'{hue}={lev}' if hue else None))
        if show_null and any('r2_null' in r for r in sel):
            yn = [aggf([r.get('r2_null', _np.nan) for r in sel
                        if (r[x] if not isinstance(r[x], tuple) else len(r[x])) == xv]) for xv in xs]
            ax.plot(xs, yn, '--', color=ax.lines[-1].get_color(), alpha=0.4)
    ax.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax.set_xlabel(x); ax.set_ylabel(f'{agg} blocked-CV R²  (block ← neural)')
    ax.set_title(title or f'decoding vs {x}')
    if hue:
        ax.legend(fontsize=8, title=hue)
    if made:
        ax.figure.tight_layout()
    return None


def f_plot_optim_overview(rows, suptitle=''):
    """2×2 overview: decoding score vs n_pca (per grid), vs grid cells (per n_orient),
    vs n_orient (per grid), vs n_sf (per grid). Best-over-other-axes (agg='max').
    Returns None."""
    import matplotlib.pyplot as plt
    if not rows:
        print('no rows to plot'); return None
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    f_plot_param_curves(rows, x='n_pca', hue='grid', ax=ax[0][0], title='vs post-PCA rank')
    f_plot_param_curves(rows, x='grid_cells', hue='n_orient', ax=ax[0][1], title='vs grid density')
    f_plot_param_curves(rows, x='n_orient', hue='grid', ax=ax[1][0], title='vs #orientations')
    f_plot_param_curves(rows, x='n_sf', hue='grid', ax=ax[1][1], title='vs #spatial-freqs')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION-scoring sweep — feature-agnostic engine + thin feature adapters
# ─────────────────────────────────────────────────────────────────────────────
# The score/truncation/lags/decoder/marginal/plot logic is identical for gabor and meng,
# so it lives ONCE (f_sweep_recon / f_sweep_marginal / f_plot_sweep_marginal); only the
# extractor (f_extract_*_multigrid) + config generator (f_*_configs) are feature-specific.

def _lags_lbl(ms):
    ms = list(ms)
    return f'{ms[0]}..{ms[-1]}ms' if len(ms) > 1 else f'{ms[0]}ms'


def _dec_lbl(dec, alpha):
    return f'ridge@{alpha:g}' if (dec == 'ridge' and alpha is not None) else dec


def f_gabor_configs(banks, grids=None, n_orient_list=None, wl_subsets=None, stats_keep=('M',)):
    """Gabor ADAPTER: yield (meta, raw_X) per (grid × #orient × SF-subset) by column-subsetting
    the raw banks (free — no re-extraction). `grids` (list of (ny,nx)) filters bank keys; None=all.
    n_orient_list values must divide the bank's master n_orient. wl_subsets = lists of wavelengths
    (subsets of the master). meta = {grid, grid_cells, n_orient, n_sf, wavelengths, stat}."""
    keys = list(banks) if grids is None else [g for g in banks if g in set(map(tuple, grids))]
    for g in keys:
        bank = banks[g]
        master_no, master_wl = bank['n_orient'], list(bank['wavelengths'])
        for no in (n_orient_list or [master_no]):
            io_keep = f_orient_indices(master_no, no)
            for wls in (wl_subsets or [tuple(master_wl)]):
                isf_keep = [master_wl.index(w) for w in wls]
                sub = f_gabor_subset(bank, io_keep=io_keep, isf_keep=isf_keep, stats_keep=stats_keep)
                if sub['X'].shape[1] == 0:
                    continue
                meta = {'grid': f'{g[0]}x{g[1]}', 'grid_cells': g[0] * g[1], 'n_orient': no,
                        'n_sf': len(wls), 'wavelengths': tuple(wls), 'stat': '+'.join(stats_keep)}
                yield meta, sub['X']


def f_meng_configs(banks, grids=None, n_orient_list=None, wl_subsets=None, tf_subsets=None,
                   stats_keep=None):
    """Meng ADAPTER: yield (meta, raw_X) per (grid × #orient(directions) × SF-subset × TF-subset)
    by column-subsetting the raw meng banks (free). n_orient_list values must divide the master;
    wl_subsets / tf_subsets = value lists (subsets of the masters). stats_keep = tok subset (None
    = all; 'opp' opponent, 'd0'/'d1' for output='both'). meta = {grid, grid_cells, n_orient,
    n_sf, n_tf, wavelengths, temporal_freqs, stat}. Reuses the generic engine downstream."""
    keys = list(banks) if grids is None else [g for g in banks if g in set(map(tuple, grids))]
    for g in keys:
        bank = banks[g]
        master_no, master_wl = bank['n_orient'], list(bank['wavelengths'])
        master_tf = list(bank['temporal_freqs'])
        for no in (n_orient_list or [master_no]):
            io_keep = f_orient_indices(master_no, no)
            for wls in (wl_subsets or [tuple(master_wl)]):
                isf_keep = [master_wl.index(w) for w in wls]
                for tfs in (tf_subsets or [tuple(master_tf)]):
                    itf_keep = [master_tf.index(t) for t in tfs]
                    cols = _subset_cols(bank['names'], _parse_meng_name,
                                        {'io': io_keep, 'isf': isf_keep, 'itf': itf_keep,
                                         'tok': stats_keep})
                    if not cols:
                        continue
                    meta = {'grid': f'{g[0]}x{g[1]}', 'grid_cells': g[0] * g[1], 'n_orient': no,
                            'n_sf': len(wls), 'n_tf': len(tfs), 'wavelengths': tuple(wls),
                            'temporal_freqs': tuple(tfs), 'stat': 'meng'}
                    yield meta, bank['X'][:, cols]


def f_sweep_recon(configs, target, cv_kw, fs, n_pca_list=(20,), lags_list=([0, 100, 200],),
                  decoder_list=(('ridge', 1.0),), motion=None, verbose=True):
    """Feature-AGNOSTIC sweep engine. `configs` = iterable of (meta_dict, raw_X). For each:
    PCA at max(n_pca_list) once, then score reconstruction R² over n_pca (nested truncation) ×
    lags × decoder via f_recon_score (the ONLY CV/score path — no duplication). `decoder_list`
    entries = (decoder_name, ridge_alpha|None). Row = meta + {n_pca, dim, lags_ms, lags_lbl,
    decoder, alpha, decoder_lbl, r2 (=pooled), r2_perfold, r2_hi, r2_lo}. Works for gabor/meng."""
    n_pca_max = max(n_pca_list)
    rows = []
    for meta, raw_X in configs:
        scores, _ = f_pca_block_scores(raw_X, n_pca_max)
        for npca in n_pca_list:
            k = min(npca, scores.shape[1])
            feat = scores[:, :k]
            for lags_ms in lags_list:
                lags = f_lags_from_ms(lags_ms, fs)
                for dec, alpha in decoder_list:
                    cvk = dict(cv_kw); cvk['decoder'] = dec
                    if alpha is not None:
                        cvk['ridge_alpha'] = alpha
                    r = f_recon_score(feat, target['scores'], cvk, lags=lags, motion=motion)
                    row = dict(meta)
                    row.update(n_pca=k, dim=raw_X.shape[1],
                               lags_ms=tuple(lags_ms), lags_lbl=_lags_lbl(lags_ms),
                               decoder=dec, alpha=alpha, decoder_lbl=_dec_lbl(dec, alpha),
                               r2=r['r2_pooled'], r2_perfold=r['r2'],
                               r2_hi=r.get('r2_hi', np.nan), r2_lo=r.get('r2_lo', np.nan))
                    # compact labels for the categorical wl / tf axes (the actual SF/TF set)
                    if 'wavelengths' in meta:
                        row['wl_lbl'] = '/'.join(str(w) for w in meta['wavelengths'])
                    if 'temporal_freqs' in meta:
                        row['tf_lbl'] = '/'.join(str(t) for t in meta['temporal_freqs'])
                    rows.append(row)
                    if verbose:
                        print(f"  {row['grid']:>5s} o{row['n_orient']} sf{row['n_sf']} "
                              f"npca={k:>2d} {row['lags_lbl']:>10s} {row['decoder_lbl']:>10s} "
                              f"dim={row['dim']:>4d}  r2={row['r2']:+.3f}")
    return rows


# extraction axes consumed by the config generator (gabor: grid/n_orient/wl; meng: +tf) vs
# score axes consumed by f_sweep_recon. `grid` baseline is a list of (ny,nx); the others hold a
# single value wrapped into a singleton list.
_EXTRACT_AXES = {'grid': 'grids', 'n_orient': 'n_orient_list', 'wl': 'wl_subsets',
                 'tf': 'tf_subsets'}
_SCORE_AXES = {'n_pca': 'n_pca_list', 'lags': 'lags_list', 'decoder': 'decoder_list'}


def f_sweep_marginal(config_fn, target, cv_kw, fs, baseline, axes, stats_keep=('M',),
                     motion=None, verbose=True):
    """MARGINAL sweep: for each axis in `axes` (name→list of values), vary it while holding all
    other axes at `baseline`, tag rows with `axis`, concat. `config_fn(stats_keep, **ext) ->
    configs` (e.g. f_gabor_configs / f_meng_configs bound to `banks`). `baseline` supplies the
    fixed value for every axis (extraction: grid [list], n_orient, wl, [tf]; score: n_pca, lags,
    decoder). Only extraction axes PRESENT in `baseline` are forwarded to config_fn, so gabor
    (no 'tf') and meng ('tf') both work. JOINT sweep = call f_sweep_recon directly."""
    rows = []
    for a, vals in axes.items():
        if a not in _EXTRACT_AXES and a not in _SCORE_AXES:
            raise ValueError(f'unknown axis {a!r}; pick from {list(_EXTRACT_AXES)+list(_SCORE_AXES)}')
        # extraction args from baseline (only axes the feature actually has → gabor omits tf)
        ext = {}
        for axn, argn in _EXTRACT_AXES.items():
            if axn in baseline:
                ext[argn] = baseline[axn] if axn == 'grid' else [baseline[axn]]
        if a in _EXTRACT_AXES:
            ext[_EXTRACT_AXES[a]] = vals
        configs = list(config_fn(stats_keep=stats_keep, **ext))
        skw = dict(n_pca_list=[baseline['n_pca']], lags_list=[baseline['lags']],
                   decoder_list=[baseline['decoder']])
        if a in _SCORE_AXES:
            skw[_SCORE_AXES[a]] = vals
        if verbose:
            print(f'— sweeping {a} ({len(vals)} values) —')
        r = f_sweep_recon(configs, target, cv_kw, fs, motion=motion, verbose=verbose, **skw)
        for row in r:
            row['axis'] = a
        rows += r
    return rows


def f_plot_sweep_marginal(rows, y='r2', suptitle=''):
    """One panel per swept `axis`: numeric axis (n_pca/grid_cells/n_orient/n_sf) → line
    (pooled + hi/lo motion), categorical (lags/decoder) → bar (pooled). `rows` from
    f_sweep_marginal. Returns None."""
    import matplotlib.pyplot as plt
    if not rows:
        print('no rows to plot'); return None
    # numeric axes → line (x = ordinal value); categorical → bar (x = the actual set/label).
    # wl / tf are SETS of frequencies, so they're categorical (labeled by the tuple), NOT the
    # count — a line over #SFs would collapse different subsets of the same size.
    num_field = {'n_pca': 'n_pca', 'grid': 'grid_cells', 'n_orient': 'n_orient'}
    cat_field = {'lags': 'lags_lbl', 'decoder': 'decoder_lbl',
                 'wl': 'wl_lbl', 'tf': 'tf_lbl'}
    order = []
    for r in rows:
        if r['axis'] not in order:
            order.append(r['axis'])
    ncol = min(3, len(order)); nrow = int(np.ceil(len(order) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
    axs = axs.ravel()
    for ax, a in zip(axs, order):
        sel = [r for r in rows if r['axis'] == a]
        if a in num_field:
            xf = num_field[a]
            xs = sorted({r[xf] for r in sel})
            best = lambda key, xv: np.nanmax([r[key] for r in sel if r[xf] == xv])
            ax.plot(xs, [best(y, xv) for xv in xs], '-o', color='C4',
                    label=('pooled' if y == 'r2' else y))
            if y == 'r2' and all('r2_hi' in r for r in sel):   # add the motion split for pooled
                ax.plot(xs, [best('r2_hi', xv) for xv in xs], '--o', color='C3', ms=3, label='high mot')
                ax.plot(xs, [best('r2_lo', xv) for xv in xs], '--o', color='C2', ms=3, label='low mot')
            ax.set_xlabel(a); ax.legend(fontsize=7)
        else:
            xf = cat_field.get(a, a)
            labs = []
            for r in sel:
                if r[xf] not in labs:
                    labs.append(r[xf])
            vals = [np.nanmax([r[y] for r in sel if r[xf] == lb]) for lb in labs]
            ax.bar(range(len(labs)), vals, color='C4')
            ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, ha='right', fontsize=7)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylabel('reconstruction R²  (feature → movie)')
        ax.set_title(f'vs {a}', fontsize=10)
    for ax in axs[len(order):]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return None
