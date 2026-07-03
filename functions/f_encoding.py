# -*- coding: utf-8 -*-
"""
Encoding toolkit — predict per-neuron deconvolved firing-rate proxy from
stimulus + self-motion features (the reverse of the decoding pipeline).

v1 = minimal validation pass (TODO #57): a short causal FIR design matrix from
built_blocks → multi-output ridge under the shared blocked-CV harness → per-neuron
out-of-fold correlation, against a per-cell circular-shift null. Reuses
make_cebra_supervision (concat + z-score + NaN-fill), f_blocked_cv_r2 (leak-free
blocked CV, train-only standardize, embargo) and f_shuffle_neural (uniform null).

Temporal model: deconvolved S removes the calcium decay tail, so the
stimulus→activity transform is a SHORT causal delay (~100-300 ms = a few frames
at 60 Hz; rise/peak, not decay). Lag-embed features over that small window and
let ridge learn the per-cell kernel.

Created 2026-06-13.
"""

import numpy as np

from f_cebra_helpers import f_blocked_cv_r2, make_cebra_supervision
# f_lag_embed / f_lags_from_ms moved to f_decoding (shared base) so f_blocked_cv_r2
# can lag the decoder's neural input too; re-exported here for back-compat.
from f_decoding import f_shuffle_neural, f_lag_embed, f_lags_from_ms


def _per_col_pearson(A, B):
    """Per-column Pearson r between two (n, p) arrays."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ac = A - A.mean(axis=0); Bc = B - B.mean(axis=0)
    num = (Ac * Bc).sum(axis=0)
    den = np.sqrt((Ac ** 2).sum(axis=0) * (Bc ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def _per_col_r2(yt, yh):
    """Per-column R² (1 - SS_res/SS_tot, test-column mean baseline)."""
    yt = np.asarray(yt, float); yh = np.asarray(yh, float)
    ss_res = ((yt - yh) ** 2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - ss_res / ss_tot


def f_encode_per_neuron(built_blocks, block_names, Y_enc, lags,
                        embargo=0, n_folds=5, standardize=True,
                        decoder='ridge', ridge_alpha=1.0, k_nn=15,
                        ridgecv_alphas=None, n_components=20):
    """Per-neuron encoding score under blocked CV.

    Build a lagged stimulus design from `block_names` (must include self_mot —
    locomotion is a predictor), fit a multi-output model per fold (one solve for
    all neurons) and return per-neuron out-of-fold Pearson r + R². For pls/cca
    this is a joint reduced-rank fit (neurons share latent factors); n_components
    is auto-clamped per fold.

    Parameters
    ----------
    built_blocks : dict             feature registry (build_feature_blocks).
    block_names : list[str]         stimulus blocks, e.g. ['self_mot','beh','agg'].
    Y_enc : ndarray (T, n_cells)    deconvolved firing-rate proxy (S_smn.T).
    lags : list[int]                causal frame lags (e.g. f_lags_from_ms(...)).
    embargo, n_folds, standardize : CV knobs (see f_blocked_cv_r2).
    decoder : {'ridge','ridgecv','knn','pls','cca'}, plus ridge_alpha / k_nn /
              ridgecv_alphas / n_components — passed to make_decoder.

    Returns
    -------
    dict: r_per_cell, r2_per_cell (over KEPT cells), keep_cells (bool over all
    cells; drops constant/NaN cells that would nuke the any-NaN-row CV), oof_true,
    oof_pred (n_oof, n_kept), design_names, n_design.
    """
    X_stim, stim_names = make_cebra_supervision(block_names, built_blocks)
    X_design, design_names = f_lag_embed(X_stim, lags, names=stim_names)
    ll = [int(l) for l in lags]
    lo = max([0] + [l for l in ll if l > 0])        # leading rows undefined (+lags)
    hi_drop = max([0] + [-l for l in ll if l < 0])  # trailing rows undefined (-lags)
    Y = np.asarray(Y_enc, float)
    T_full = X_design.shape[0]
    X_design = X_design[lo:T_full - hi_drop]
    Y = Y[lo:T_full - hi_drop]

    # Drop degenerate cells (constant / non-finite). f_blocked_cv_r2 drops any
    # row with a NaN in ANY target column, so one silent cell would nuke all rows.
    keep_cells = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep_cells]

    out = f_blocked_cv_r2(X_design, Yk, n_folds=n_folds, decoder=decoder,
                          ridge_alpha=ridge_alpha, k_nn=k_nn,
                          ridgecv_alphas=ridgecv_alphas, n_components=n_components,
                          embargo=embargo, standardize=standardize,
                          return_predictions=True)
    yt, yh = out['y_true'], out['y_pred']            # (n_oof, n_kept)
    return {'r_per_cell':  _per_col_pearson(yt, yh),
            'r2_per_cell': _per_col_r2(yt, yh),
            'keep_cells':  keep_cells,
            'oof_true':    yt, 'oof_pred': yh, 'idx': out['idx'],
            'row_offset':  lo,   # abs frame = row_offset + idx (both-ends trim)
            'design_names': design_names, 'n_design': X_design.shape[1]}


# ─────────────────────────────────────────────────────────────────────────────
# P2a — per-neuron WEIGHT fit (the RF) + reshape + relative ceiling  (sub-plan D1/D5)
# f_encode_per_neuron gives CV SCORES; these give the fitted WEIGHTS = the receptive
# field, plus the reshape back to monitor space and the r_full-relative ceiling.
# ─────────────────────────────────────────────────────────────────────────────

def f_encode_fit_weights(X_stim, Y_enc, lags, alpha=1.0, standardize=True):
    """Fit a per-neuron RIDGE encoder on ALL rows and return the WEIGHTS (the RF), not a
    CV score. `X_stim` (T, d_stim) is the stimulus design in its NATIVE spatial layout —
    raw movie pixels flattened, or raw grid features — so the weights reshape back to space.
    `lags` = frame lags (f_lag_embed; lag-major columns). Weights are in standardized-X
    units (RF SHAPE, not a calibrated gain) and fit on all data (CV scoring stays in
    f_encode_per_neuron). Returns dict:
        W        (d_stim, n_lags, n_kept)   weights reshaped by lag  [the RF]
        W_flat   (d_design, n_kept),  intercept (n_kept,),  keep_cells (n_cells bool),
        lags, d_stim, row_offset, alpha.
    """
    from sklearn.linear_model import Ridge
    ll = [int(l) for l in lags]
    X_design, _ = f_lag_embed(np.asarray(X_stim, float), ll)     # (T, d_stim*n_lags)
    lo = max([0] + [l for l in ll if l > 0])                     # +lags → leading NaN rows
    hi = max([0] + [-l for l in ll if l < 0])                    # -lags → trailing NaN rows
    T = X_design.shape[0]
    X = X_design[lo:T - hi]
    Y = np.asarray(Y_enc, float)[lo:T - hi]
    keep = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep]
    if standardize:
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
    reg = Ridge(alpha=alpha, fit_intercept=True).fit(X, Yk)
    W_flat = np.atleast_2d(reg.coef_).T if Yk.shape[1] == 1 else reg.coef_.T   # (d_design, n_kept)
    d_stim, n_lags = int(np.asarray(X_stim).shape[1]), len(ll)
    W = W_flat.reshape(n_lags, d_stim, -1).transpose(1, 0, 2)    # (d_stim, n_lags, n_kept)
    return {'W': W, 'W_flat': W_flat, 'intercept': np.atleast_1d(reg.intercept_),
            'keep_cells': keep, 'lags': ll, 'd_stim': d_stim, 'row_offset': lo, 'alpha': alpha}


def f_rf_pixel_frames(fit, cell_idx, hw):
    """Spatiotemporal PIXEL RF for one KEPT cell (index into fit['keep_cells']-True cells):
    (n_lags, H, W) from the fitted weights — a short weight-movie (blob motion across lags =
    direction tuning). Weights are signed (±)."""
    H, W = hw
    w = fit['W'][:, :, cell_idx]                     # (d_stim = H*W, n_lags)
    return w.T.reshape(len(fit['lags']), H, W)


def f_rf_grid_location(fit, cell_idx, grid, n_chan):
    """Feature-grid RF LOCATION map for one kept cell: |weights| summed over the `n_chan`
    channels per grid cell → (n_lags, ny, nx). Assumes the raw grid block is ordered
    location-major (grid cell outer, channel inner), as f_gabor_features/f_grid_features emit."""
    ny, nx = grid
    w = fit['W'][:, :, cell_idx].reshape(ny * nx, n_chan, len(fit['lags']))
    return np.abs(w).sum(axis=1).T.reshape(len(fit['lags']), ny, nx)


def f_encode_ceiling_fraction(r_model, r_full, floor=0.05):
    """Per-cell fraction of the full-model relative ceiling a feature/model recovers:
    r_model / r_full (the #57 stance — read a feature by how much of r_full it reaches, not
    as absolute variance). Cells whose ceiling r_full ≤ `floor` (≈chance) → NaN (ratio
    meaningless there)."""
    r_model = np.asarray(r_model, float); r_full = np.asarray(r_full, float)
    out = np.full(r_model.shape, np.nan)
    ok = r_full > floor
    out[ok] = r_model[ok] / r_full[ok]
    return out


def f_encode_null(built_blocks, block_names, Y_enc, lags, rng, **kw):
    """Null encoding scores: per-cell circular-shift of the target (uniform null),
    same design + CV as f_encode_per_neuron. Breaks stimulus↔neural alignment,
    preserves each cell's autocorrelation."""
    Y_shuf = f_shuffle_neural(np.asarray(Y_enc, float), rng)
    return f_encode_per_neuron(built_blocks, block_names, Y_shuf, lags, **kw)


def _bh_fdr(p):
    """Benjamini-Hochberg FDR q-values for a 1-D array of p-values (NaN-safe)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    qv = np.empty(m)
    qv[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = qv
    return q


def _null_one_shuffle(seed, X_design, Yk, kw):
    """One circshift-null encoding fit for f_encode_null_dist (module-level so loky
    can pickle it). Shuffles the (fixed) design's target, refits, returns per-cell r."""
    Ysh = f_shuffle_neural(Yk, np.random.default_rng(seed))
    out = f_blocked_cv_r2(X_design, Ysh, return_predictions=True, **kw)
    return _per_col_pearson(out['y_true'], out['y_pred'])


def f_encode_null_dist(built_blocks, block_names, Y_enc, lags, rng,
                       n_shuffles=200, n_jobs=-1, **kw):
    """Per-cell NULL distribution of encoding r over N circular-shift shuffles.

    The lag-embedded design is FIXED; only the target is circshifted per cell
    (autocorr-preserving null). Build the design ONCE, refit per shuffle. Used by
    f_encode_signif for a per-cell significance test (replaces a noise ceiling).

    Speed: design built once, float32, shuffles run in parallel over the loky
    PROCESS backend (sklearnex has no threading backend). n_jobs=1 = serial.
    keep_cells matches f_encode_per_neuron (same trim), so it aligns with `real`.

    Returns dict: null_r (n_shuffles, n_kept), keep_cells.
    """
    X_stim, _ = make_cebra_supervision(block_names, built_blocks)
    X_design, _ = f_lag_embed(X_stim, lags)
    ll = [int(l) for l in lags]
    lo = max([0] + [l for l in ll if l > 0])
    hi = max([0] + [-l for l in ll if l < 0])
    T = X_design.shape[0]
    X_design = np.asarray(X_design[lo:T - hi], np.float32)
    Y = np.asarray(Y_enc, np.float32)[lo:T - hi]
    keep = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep]
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_shuffles)]

    if n_jobs == 1:
        rows = [_null_one_shuffle(s, X_design, Yk, kw) for s in seeds]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(                # loky (processes)
            delayed(_null_one_shuffle)(s, X_design, Yk, kw) for s in seeds)
    return {'null_r': np.vstack(rows), 'keep_cells': keep}


def f_encode_signif(real, null_dist, alpha=0.05):
    """Per-cell significance of encoding r vs the circshift null distribution.

    A statistical test in place of a noise ceiling. Per kept cell:
      p   = (1 + #{null_r >= real_r}) / (N + 1)   one-sided, never 0
      z   = (real_r - mean_null) / std_null       null-referenced effect size
            (ceiling-free normalized magnitude; NaN when std_null == 0)
      q   = Benjamini-Hochberg FDR over cells;  sig = q < alpha
    real['keep_cells'] must match null_dist['keep_cells'] (same Y_enc + lags).

    Returns dict: r, p, q, z, sig, null_mean, null_std, n_sig, n_shuffles.
    """
    if not np.array_equal(real['keep_cells'], null_dist['keep_cells']):
        raise ValueError('real and null keep_cells differ — same Y_enc/lags needed')
    nd = np.asarray(null_dist['null_r'], float)          # (N, n_kept)
    r = np.asarray(real['r_per_cell'], float)            # (n_kept,)
    N = nd.shape[0]
    ge = (nd >= r[None, :]).sum(axis=0)
    p = (1.0 + ge) / (N + 1.0)
    mu = nd.mean(axis=0)
    sd = nd.std(axis=0)
    z = (r - mu) / np.where(sd == 0, np.nan, sd)
    q = _bh_fdr(p)
    sig = q < alpha
    return {'r': r, 'p': p, 'q': q, 'z': z, 'sig': sig,
            'null_mean': mu, 'null_std': sd,
            'n_sig': int(np.nansum(sig)), 'n_shuffles': N}


def f_encode_block_importance(built_blocks, block_names, Y_enc, lags,
                              embargo=0, n_folds=5, standardize=True,
                              decoder='ridge', ridge_alpha=1.0, k_nn=15,
                              ridgecv_alphas=None, n_components=20):
    """Per-cell, per-block feature importance via out-of-fold correlation.

    For each block: MARGINAL = r with that block ALONE; UNIQUE = r(full) −
    r(full minus that block) (drop-one Δr). Unique handles redundancy between
    blocks (e.g. motion ↔ self_mot); marginal is the standalone predictive power.
    Same kept cells across runs (determined by Y_enc only). Model knobs
    (decoder / ridge_alpha / k_nn / ridgecv_alphas / n_components) as
    f_encode_per_neuron.

    Returns dict: block_names, r_full (n_kept,), marginal / unique (n_kept, n_blocks),
    keep_cells (bool over all cells).
    """
    kw = dict(embargo=embargo, n_folds=n_folds, standardize=standardize,
              decoder=decoder, ridge_alpha=ridge_alpha, k_nn=k_nn,
              ridgecv_alphas=ridgecv_alphas, n_components=n_components)
    full = f_encode_per_neuron(built_blocks, block_names, Y_enc, lags, **kw)
    r_full, keep = full['r_per_cell'], full['keep_cells']
    n_b = len(block_names)
    marginal = np.full((r_full.size, n_b), np.nan)
    unique   = np.full((r_full.size, n_b), np.nan)
    for bi, b in enumerate(block_names):
        marginal[:, bi] = f_encode_per_neuron(built_blocks, [b], Y_enc, lags,
                                              **kw)['r_per_cell']
        rest = [x for x in block_names if x != b]
        if rest:
            r_drop = f_encode_per_neuron(built_blocks, rest, Y_enc, lags,
                                         **kw)['r_per_cell']
            unique[:, bi] = r_full - r_drop
        else:
            unique[:, bi] = r_full
    return {'block_names': list(block_names), 'r_full': r_full,
            'marginal': marginal, 'unique': unique, 'keep_cells': keep}


def _scan_one_lag(L, X_stim, Yk, kw):
    """One single-lag CV fit for f_encode_latency_scan (module-level so loky can
    pickle it). Shifts X_stim by L, trims undefined rows, returns per-cell OOF r."""
    L = int(L)
    T = X_stim.shape[0]
    lo = L if L > 0 else 0
    hi = -L if L < 0 else 0
    Xd = np.empty_like(X_stim)
    if L == 0:
        Xd[:] = X_stim
    elif L > 0:
        Xd[L:] = X_stim[:T - L]                # rows 0:L undefined → trimmed by lo
    else:
        Xd[:T - hi] = X_stim[hi:]              # rows T-hi: undefined → trimmed by hi
    out = f_blocked_cv_r2(Xd[lo:T - hi], Yk[lo:T - hi],
                          return_predictions=True, **kw)
    return _per_col_pearson(out['y_true'], out['y_pred'])


def f_encode_latency_scan(built_blocks, block_names, Y_enc, lags_frames,
                          n_jobs=-1, **kw):
    """Encoding performance vs latency — fit with ONE lag at a time.

    For each frame lag in `lags_frames`, fit a single-lag model and collect the
    per-cell OOF r. The lag that best predicts a cell is its preferred encoding
    latency; the population curve (median r vs lag) shows where the drive
    concentrates (expected peak ~100–200 ms). `kw` passes through (decoder /
    ridge_alpha / n_components / embargo / n_folds).

    Speed: the supervision matrix and kept-cell mask are built ONCE (invariant
    across lags); each lag only shifts + trims. float32 design + targets (~2x
    faster matmul / half the memory bandwidth; sklearnex float32 path). Lags run
    in parallel over the loky PROCESS backend (sklearnex doesn't support joblib's
    threading backend), each worker an independent single-threaded sklearn — so no
    BLAS oversubscription. n_jobs=1 = serial (no worker pool). keep_cells is
    computed once over the full session (constant across lags), so every r_by_lag
    row aligns to one cell set.

    Returns dict: lags (frames), r_by_lag (n_lags, n_kept), keep_cells.
    """
    X_stim, _ = make_cebra_supervision(block_names, built_blocks)   # built ONCE
    X_stim = np.asarray(X_stim, np.float32)
    Y = np.asarray(Y_enc, np.float32)
    keep = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)        # ONCE, global
    Yk = Y[:, keep]

    if n_jobs == 1:
        rows = [_scan_one_lag(L, X_stim, Yk, kw) for L in lags_frames]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(                # loky (processes) by default
            delayed(_scan_one_lag)(L, X_stim, Yk, kw) for L in lags_frames)
    return {'lags': [int(L) for L in lags_frames],
            'r_by_lag': np.vstack(rows), 'keep_cells': keep}


def f_encode_latency_stats(scan):
    """Per-cell latency-tuning stats from a f_encode_latency_scan result.

    Quantifies how strongly each cell prefers a lag (the argmax histogram alone
    can't — a flat curve still has an argmax). Per (kept) cell:
      r_pref     max r over lags;  lag_idx argmax;  lag_pref = lags[lag_idx] (frames)
      prominence r_pref − median_lag r   (peak above a typical lag)
      benefit    r_pref − r(lag 0)        (gain over instantaneous; NaN if no lag 0)
      width      # lags with r ≥ r_pref/2 (half-max sharpness; NaN if r_pref ≤ 0)
    width is a lag-count (convert to ms via the lag spacing at the call site).
    Returns dict of (n_kept,) arrays + lags, keep_cells.
    """
    R = np.asarray(scan['r_by_lag'], float)            # (n_lags, n_kept)
    lags = np.asarray(scan['lags'])
    Rf = np.where(np.isfinite(R), R, -np.inf)
    lag_idx = Rf.argmax(axis=0)
    r_pref = Rf.max(axis=0)
    prominence = r_pref - np.nanmedian(R, axis=0)
    zero = np.where(lags == 0)[0]
    benefit = (r_pref - R[zero[0]]) if zero.size else np.full(r_pref.size, np.nan)
    half = np.where(r_pref > 0, r_pref / 2.0, np.inf)  # width undefined for r_pref≤0
    width = (R >= half[None, :]).sum(axis=0).astype(float)
    width[r_pref <= 0] = np.nan
    return {'lags': lags, 'keep_cells': scan['keep_cells'],
            'r_pref': r_pref, 'lag_idx': lag_idx, 'lag_pref': lags[lag_idx],
            'prominence': prominence, 'benefit': benefit, 'width': width}


def f_encode_summary(real, null, pct=95):
    """Compare real vs null per-cell r: null threshold, fraction of real cells
    above it, and median r for each. Both dicts from f_encode_per_neuron/_null."""
    r, rn = real['r_per_cell'], null['r_per_cell']
    thr = float(np.nanpercentile(rn, pct))
    return {'null_pct': pct, 'null_thresh': thr,
            'frac_above_null': float(np.nanmean(r > thr)),
            'median_r_real': float(np.nanmedian(r)),
            'median_r_null': float(np.nanmedian(rn)),
            'n_cells': int(np.isfinite(r).sum())}
