# -*- coding: utf-8 -*-
"""
f_dataset_scan.py — FAST per-dataset encode/decode quality triage (across-dataset).

Cheap screening so we can rank sessions and focus the expensive feature build +
sweeps on the best ones. Scores against the CHEAP ground-truth stimulus (object
location `agg` + `self_mot`) built straight from the behavior vectorization — NO
rendering, NO feature cache — so a whole mouse's datasets scan in minutes.

Per dataset (one blocked-CV fit each, all leak-free):
  - decode R²  neural PCs → self_mot   (always defined; robust neural→behavior read)
  - decode R²  neural PCs → agg         (object location; stimulus read; NaN rows drop)
  - encode r   agg+self_mot → neural    (median per-cell OOF r; frac significant)
plus near-free covariates (n_cells, T, neural participation ratio, object coverage)
so a low score isn't misread as a bad recording when it's really low N / low coverage
(the P0 matched-N caveat). These use only agg+self_mot, so they rank by stimulus-
location / self-motion coupling — a TRIAGE proxy, not the full gabor/meng result.

Compute only; the driver loop + plots live in VR_dataset_scan.py.
Created 2026-07-11 (TODO #66 across-dataset).
"""

import numpy as np

from f_utils import f_gauss_smooth
from f_feature_helpers import build_feature_blocks, f_zscore, f_proc
from f_cebra_helpers import f_blocked_cv_r2
from f_decoding import f_imaging_fs, f_lags_from_ms
from f_encoding import f_encode_per_neuron


def f_participation_ratio(X):
    """Effective dimensionality PR = (Σλ)² / Σλ² of the covariance eigenvalues of
    X (T, d) — high when variance is spread over many dims, ~1 when one PC dominates.
    A cheap population-reliability proxy (computed via the SVD singular values)."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(axis=0)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = s ** 2
    tot = lam.sum()
    return float(tot ** 2 / (lam ** 2).sum()) if tot > 0 else np.nan


def f_build_cheap_blocks(session, agg_mode='angular', obj_collapse='salience_mean',
                         nan_absent=True, self_motion_decomp=True, obj_size=None):
    """agg + self_mot(+decomp) + beh from the cheap vectorization (NO render). Returns
    the built_blocks dict. side_tags fixed to ['L','R'] (both monitors). `obj_size` is
    required by obj_collapse='salience_mean' (object angular size for the salience weight)."""
    side_tags = ['L', 'R']
    vec_use = [session['vec_data_l'], session['vec_data_r']]
    return build_feature_blocks(
        vec_use, side_tags, session['beh_t'], session['frame_t'], session['pulse_delay'],
        build_pix=False,                                   # no movie in the fast scan
        agg_mode=agg_mode, obj_size=obj_size, obj_collapse=obj_collapse, nan_absent=nan_absent,
        build_self_motion=True, mov_data=session['mov_data'],
        self_motion_decomp=self_motion_decomp,
        build_lick_reward=True, lr_data=session['lr_data'])


def _decode_r2(Yneu, target_X, n_pcs, embargo, n_folds, standardize=True, pred_proc=None):
    """Leak-free blocked-CV decode of `target_X` from the within-fold neural predictor. NaN target rows
    drop. Default path = within-fold PCA(n_pcs) + `standardize`. `pred_proc` (f_proc step-list) OVERRIDES
    that with a unified within-fold pipeline on the neural predictor (then pass Yneu un-z-scored so the
    proc owns it). Returns a dict of decode metrics over the target columns:
      r2         VARIANCE-WEIGHTED R² (per-fold, averaged) — big-variance cols dominate (the default).
      r2_uniform UNIFORM-average R² over cols (pooled OOF) — every col counts equally.
      r2_med     MEDIAN per-col R² (pooled OOF) — robust to a few bad cols.
      r2_min/p5/p25/p75/p95/max   distribution of the per-col R² (whiskers around the median).
      validfrac  fraction of finite (usable) time rows."""
    from sklearn.metrics import r2_score
    out = f_blocked_cv_r2(None, np.asarray(target_X, float), Y_raw=Yneu, n_pcs=n_pcs,
                          embargo=embargo, n_folds=n_folds, standardize=standardize, pred_proc=pred_proc,
                          multioutput='variance_weighted', return_predictions=True)
    yt, yh = np.asarray(out['y_true']), np.asarray(out['y_pred'])          # pooled OOF (time-ordered)
    if yt.size and yt.ndim == 2 and yt.shape[1] > 1:                       # per-column R² from pooled OOF
        rc = r2_score(yt, yh, multioutput='raw_values')
    elif yt.size:
        rc = np.array([r2_score(yt.ravel(), yh.ravel())])
    else:
        rc = np.array([np.nan])
    return {'r2': float(out['r2']), 'r2_uniform': float(np.mean(rc)), 'r2_med': float(np.median(rc)),
            **{f'r2_{k}': v for k, v in _dist_stats(rc).items()},
            'validfrac': float(out['n_valid'] / max(1, out['T']))}


# Measured peak of _decode_r2 / (T · n_col · 8 bytes): the target is cast to float64, then the per-fold
# OOF lists, their concatenation and the time-order reindex all coexist. Linear in n_col (3.70 at
# 500/1000/2000/4000 cols), so it extrapolates — the knob that decides if a RAW block is runnable.
DECODE_PEAK_FACTOR = 3.7


def _pca_var_kept(pca):
    """Fraction of input variance a fitted PCA's KEPT components explain (NaN if there is no PCA)."""
    evr = getattr(pca, 'explained_variance_ratio_', None)
    return float(np.sum(evr)) if evr is not None else np.nan


def _block_size(block, n_cells=None):
    """Size/shape record for one built block → the dict merged into each f_score_representations row.
    `n_dim` is the width actually scored, `raw_dim` the width BEFORE any layer/PCA proc (equal when no
    reduction ran), `decode_peak_gb` the projected _decode_r2 peak (see DECODE_PEAK_FACTOR).
    VARIANCE KEPT by the components taken: `var_kept` = the block/layer2 PCA's explained-variance sum;
    `l1_var_kept` = the per-filter layer1 PCAs pooled (width-weighted, set by f_iter_grid_cells);
    `var_kept_total` = their product = the fraction of the RAW block variance still present. Each is
    NaN where that layer ran no PCA (nothing dropped is reported as 1.0 by the producer, not here)."""
    X = np.asarray(block['X'])
    T, d = int(X.shape[0]), int(X.shape[1])
    raw = int(block.get('raw_dim') or d)
    v2 = _pca_var_kept(block.get('pca'))
    v1 = block.get('l1_var_kept', np.nan)
    v1 = float(v1) if v1 is not None else np.nan
    tot = np.nanprod([x for x in (v1, v2) if x == x]) if (v1 == v1 or v2 == v2) else np.nan
    l1_in = block.get('l1_dim_in')                       # width before layer1 (grid blocks only)
    return {'n_dim': d, 'raw_dim': raw, 'l1_dim_in': float(l1_in) if l1_in else np.nan,
            'n_frames': T, 'n_cells_kept': n_cells,
            'block_mb': float(X.nbytes / 1e6),
            'decode_peak_gb': float(DECODE_PEAK_FACTOR * T * d * 8 / 1e9),
            'l1_var_kept': v1, 'var_kept': v2, 'var_kept_total': float(tot) if tot == tot else np.nan}


def _dist_stats(a):
    """min / p5 / p25 / p75 / p95 / max of a 1-D array (NaN-aware); all NaN if empty."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: np.nan for k in ('min', 'p5', 'p25', 'p75', 'p95', 'max')}
    p5, p25, p75, p95 = np.percentile(a, [5, 25, 75, 95])
    return {'min': float(a.min()), 'p5': float(p5), 'p25': float(p25),
            'p75': float(p75), 'p95': float(p95), 'max': float(a.max())}


_POP_KEYS = ('pop_r2', 'pop_r2_uniform', 'pop_r', 'pop_n_pc',
             *(f'pop_{k}' for k in ('min', 'p5', 'p25', 'p75', 'p95', 'max')))


def f_nan_metrics():
    """Full NaN metric-column dict (every column a scored row carries, minus method/status) — used for
    missing/failed rows so downstream column selects never KeyError even when nothing scored."""
    dist = ('min', 'p5', 'p25', 'p75', 'p95', 'max')
    d = {'decode_r2': np.nan, 'decode_r2_uniform': np.nan, 'decode_r2_med': np.nan,
         'decode_validfrac': np.nan, 'encode_r_med': np.nan, 'encode_r_mean': np.nan,
         'encode_r2_med': np.nan, 'encode_r2_mean': np.nan, 'encode_frac_sig': np.nan,
         'n_dim': 0, 'raw_dim': 0, 'l1_dim_in': np.nan, 'n_frames': 0, 'n_cells_kept': np.nan,
         'block_mb': np.nan, 'decode_peak_gb': np.nan,
         'l1_var_kept': np.nan, 'var_kept': np.nan, 'var_kept_total': np.nan}
    d.update({f'decode_r2_{k}': np.nan for k in dist})
    d.update({f'encode_r_{k}': np.nan for k in dist})
    d.update({f'encode_r2_{k}': np.nan for k in dist})
    d.update({f'encode_{k}': np.nan for k in _POP_KEYS})
    return d


def _pop_encode(yt, yh, steps):
    """Population encoding score: reduce the per-cell OOF true/pred (yt, yh; T×cells) to population
    dimensions with the f_proc step-list `steps` (e.g. [('norm','zscore'),('pca',20)]) — FIT on yt
    (unsupervised; the regression was already cross-validated), TRANSFORM yh with the SAME fitted params
    — then score how well the features reconstruct those dimensions. Returns R²/Pearson r over the output
    dims, VARIANCE-WEIGHTED by each dim's variance in the true data, + the per-dim R² distribution. All
    NaN if too few cells/rows."""
    from sklearn.metrics import r2_score
    yt, yh = np.asarray(yt, float), np.asarray(yh, float)
    if yt.ndim != 2 or yt.shape[1] < 2 or yt.shape[0] < 3:
        return {k: np.nan for k in _POP_KEYS}
    Zt, fitted = f_proc(yt, steps)                           # fit the reduction on OOF true
    Zh, _ = f_proc(yh, steps, fit=fitted)                    # SAME transform on OOF pred
    Zt, Zh = np.atleast_2d(Zt), np.atleast_2d(Zh)
    r2pc = r2_score(Zt, Zh, multioutput='raw_values')        # per-dim R²
    zt_c, zh_c = Zt - Zt.mean(0), Zh - Zh.mean(0)            # per-dim Pearson r
    denom = np.sqrt((zt_c ** 2).sum(0) * (zh_c ** 2).sum(0))
    rpc = np.divide((zt_c * zh_c).sum(0), denom, out=np.zeros_like(denom), where=denom > 0)
    v = Zt.var(axis=0)                                       # weight each dim by its true-data variance
    w = v / v.sum() if v.sum() > 0 else np.full(len(v), 1.0 / len(v))
    return {'pop_r2': float(np.sum(w * r2pc)), 'pop_r2_uniform': float(np.mean(r2pc)),
            'pop_r': float(np.sum(w * rpc)), 'pop_n_pc': int(Zt.shape[1]),
            **{f'pop_{k2}': v2 for k2, v2 in _dist_stats(r2pc).items()}}


def f_score_representations(built_blocks, Y_neu, method_names=None, *, n_pcs=30,
                            lags_ms=(0, 100, 200), embargo=0, n_folds=5, r_sig=0.1, fs=60.0,
                            standardize=True, neural_zscore=True, feature_proc=None, neural_proc=None,
                            encode_neural_proc=None, proc_before_cv=False, scores=('decode', 'encode'),
                            verbose=False, timing=False):
    """Compact per-METHOD encode+decode score (one row per feature representation) — the method
    analogue of f_scan_dataset (same metric definitions/format). `method_names=None` scores EVERY
    block in built_blocks (order preserved); pass a list to restrict. For each block name in
    `method_names` (e.g. gabor-mag / meng at various sampling resolutions + 'pix'):
      decode_r2      neural PCs → the feature block (leak-free within-fold PCA, variance-weighted);
                     'how well the neurons carry this representation'.
      encode_r_med / encode_frac_sig   the feature block → neural (f_encode_per_neuron, median OOF
                     r + fraction of cells with r > r_sig).
      n_dim          block width.
    `Y_neu` = (T, n_cells) neural rate proxy (already smoothed; dead/constant cells dropped here).
    `embargo` in FRAMES. A missing block scores NaN (never raises).
    `standardize` = within-fold z-score of the model PREDICTOR (encode: feature columns; decode: neural
    PCs), leak-free. `neural_zscore` = per-cell z-score the neural before the within-fold PCA → sets
    which directions that PCA picks (decode-only; encode's per-cell r/R² are scale-invariant). Both
    default True (prior behavior). `feature_proc`/`neural_proc` = f_proc step-lists that REPLACE the
    within-fold encode/decode predictor normalization with the unified pipeline (leak-free); when
    `neural_proc` is given the decode uses the RAW neural (Yk) so the proc owns its z-score+PCA.
    `encode_neural_proc` (f_proc step-list, e.g. [('norm','zscore'),('pca',20)] | None) adds a
    POPULATION encode score: reduce the per-cell OOF true/pred to population dimensions with that proc
    (fit on the true OOF, applied to pred) and report the variance-weighted (by each dim's true variance)
    R² and Pearson r over the dims (encode_pop_r2 / encode_pop_r + the per-dim R² distribution
    encode_pop_{...}). An int/float is coerced to [('pca', n)]. None (default) → those columns are NaN.
    `proc_before_cv` (default False → the leak-free within-fold path above): when True, ALL procs are
    applied GLOBALLY before the CV split and the folds add NO further normalization — the decode neural
    predictor is `f_proc(Yk, neural_proc)` computed ONCE, and both the decode and encode predictors enter
    the fold as-is (empty within-fold pred_proc). In this mode `standardize`/`neural_zscore`/`n_pcs` are
    ignored (superseded by the global procs); feature procs must be applied upstream (they are — the search
    applies layer/pix procs globally). `scores` = which fits to run ('decode', 'encode'); a skipped one
    leaves its columns NaN. Returns a tidy DataFrame."""
    import pandas as pd
    Y = np.asarray(Y_neu, float)
    keep = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep]
    n_cells = int(keep.sum())
    if proc_before_cv:                            # procs global, CV only splits — no within-fold normalization
        Y_dec = f_proc(np.asarray(Yk, float), neural_proc or [])[0]    # global decode-neural proc
        _dec_pp, _enc_pp, _std = [], [], False                        # identity within-fold; use as-is
    else:                                         # leak-free within-fold path (prior behavior)
        Yz = f_zscore(Yk) if neural_zscore else Yk   # sets the decode neural-PCA directions (decode-only)
        Y_dec = np.asarray(Yk, float) if neural_proc is not None else Yz   # neural_proc owns z-score → raw Yk
        _dec_pp, _enc_pp, _std = neural_proc, feature_proc, standardize
    npc = int(min(n_pcs, max(1, n_cells - 1)))
    lags = f_lags_from_ms(lags_ms, fs)
    if method_names is None:
        method_names = list(built_blocks)
    import time
    rows, M = [], len(method_names)
    for j, m in enumerate(method_names, 1):
        if m not in built_blocks:
            rows.append({'method': m, **f_nan_metrics()})
            if verbose:
                print(f'      [{j:>2}/{M}] {m:<26} (missing)')
            continue
        _t = time.perf_counter()
        X = built_blocks[m]['X']
        if timing:
            print(f'        [{time.strftime("%H:%M:%S")}] → decode {m} ({np.asarray(X).shape[1]} feat)',
                  flush=True)
        _td = time.perf_counter()
        _nan = f_nan_metrics()
        if 'decode' in scores:
            dec = _decode_r2(Y_dec, X, npc, embargo, n_folds, standardize=_std, pred_proc=_dec_pp)
            dec_cols = {'decode_r2': dec['r2'], 'decode_r2_uniform': dec['r2_uniform'],
                        'decode_r2_med': dec['r2_med'], 'decode_validfrac': dec['validfrac'],
                        **{f'decode_r2_{k}': dec[f'r2_{k}']
                           for k in ('min', 'p5', 'p25', 'p75', 'p95', 'max')}}
        else:                                                # decode skipped → NaN columns
            dec_cols = {k: v for k, v in _nan.items() if k.startswith('decode_r2')
                        or k == 'decode_validfrac'}
        if timing:
            print(f'        [{time.strftime("%H:%M:%S")}] ✓ decode {time.perf_counter() - _td:.1f}s'
                  f'   → encode ({Yk.shape[1]} cells)', flush=True)
        _te = time.perf_counter()
        if 'encode' in scores:
            enc = f_encode_per_neuron(built_blocks, [m], Yk, lags, embargo=embargo, n_folds=n_folds,
                                      standardize=_std, pred_proc=_enc_pp)
            r = np.asarray(enc['r_per_cell'], float)
            r2c = np.asarray(enc['r2_per_cell'], float)
            _enp = ([('pca', encode_neural_proc)] if isinstance(encode_neural_proc, (int, float))
                    else encode_neural_proc)                 # int/float → [('pca', n)]; else a step-list
            pop = (_pop_encode(enc['oof_true'], enc['oof_pred'], _enp)
                   if _enp else {k: np.nan for k in _POP_KEYS})
            enc_cols = {'encode_r_med': float(np.nanmedian(r)), 'encode_r_mean': float(np.nanmean(r)),
                        'encode_r2_med': float(np.nanmedian(r2c)),
                        'encode_r2_mean': float(np.nanmean(r2c)),
                        'encode_frac_sig': float(np.nanmean(r > r_sig)),
                        **{f'encode_r_{k}': v for k, v in _dist_stats(r).items()},
                        **{f'encode_r2_{k}': v for k, v in _dist_stats(r2c).items()},
                        **{f'encode_{k}': pop[k] for k in _POP_KEYS}}
        else:                                                # encode skipped → NaN columns
            enc_cols = {k: v for k, v in _nan.items() if k.startswith('encode_')}
        if timing:
            print(f'        [{time.strftime("%H:%M:%S")}] ✓ encode {time.perf_counter() - _te:.1f}s',
                  flush=True)
        r_med = enc_cols['encode_r_med']
        rows.append({'method': m, **dec_cols, **enc_cols,
                     **_block_size(built_blocks[m], n_cells)})
        if verbose:
            _s = rows[-1]
            _v = ''.join(f' {lbl}{100 * _s[k]:.0f}%' for lbl, k in
                         (('L1var', 'l1_var_kept'), ('var', 'var_kept')) if _s[k] == _s[k])
            print(f'      [{j:>2}/{M}] {m:<26} dec={dec_cols["decode_r2"]:+.3f}  enc={r_med:+.3f}   '
                  f'[{_s["n_frames"]}x{_s["n_dim"]}'
                  + (f' of {_s["raw_dim"]}' if _s['raw_dim'] != _s['n_dim'] else '')
                  + _v
                  + f', {_s["block_mb"]:.0f}MB, decode~{_s["decode_peak_gb"]:.1f}GB]   '
                  + f'{time.perf_counter() - _t:.1f}s')
    return pd.DataFrame(rows)


def f_scan_dataset(session, *, sigma_frames=6, n_pcs=30, lags_ms=(0, 100, 200),
                   embargo_sec=0.5, n_folds=5, dec_targets=('self_mot', 'agg'),
                   enc_targets=('agg', 'self_mot'), r_sig=0.1,
                   agg_mode='angular', obj_collapse='salience_mean', nan_absent=True,
                   self_motion_decomp=True, obj_size=None):
    """Fast encode+decode quality metrics for ONE loaded session (f_load_session output).
    Returns a flat dict (one row of the scan table). Never raises on a degenerate block —
    missing/empty targets score NaN. See module docstring for the metric definitions."""
    est1 = session['est1']
    fs = f_imaging_fs(session['frame_t'])
    embargo = int(round(embargo_sec * fs))
    built = f_build_cheap_blocks(session, agg_mode=agg_mode, obj_collapse=obj_collapse,
                                 nan_absent=nan_absent, self_motion_decomp=self_motion_decomp,
                                 obj_size=obj_size)

    # neural rate proxy → z-scored (drop dead/constant cells)
    Y = f_gauss_smooth(np.asarray(est1['S'], float), sigma_frames=sigma_frames).T   # (T,n)
    keep = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep]
    n_cells, T = int(keep.sum()), Y.shape[0]
    Yz = f_zscore(Yk)
    npc = int(min(n_pcs, max(1, n_cells - 1)))

    row = {'mouse': session['mouse_id'], 'n_dset': session['n_dset'],
           'dset_name': session['dset_name'], 'n_cells': n_cells, 'T': T,
           'part_ratio': f_participation_ratio(Yz), 'n_pcs': npc}

    # object coverage: fraction of frames with an object present (agg row not all-NaN)
    if 'agg' in built:
        agg_X = built['agg']['X']
        row['agg_coverage'] = float(np.isfinite(agg_X).any(axis=1).mean())
    else:
        row['agg_coverage'] = np.nan

    # DECODE: neural PCs → each cheap target
    for tgt in dec_targets:
        if tgt in built:
            dec = _decode_r2(Yz, built[tgt]['X'], npc, embargo, n_folds)
            r2, vf = dec['r2'], dec['validfrac']
        else:
            r2, vf = np.nan, np.nan
        row[f'dec_{tgt}_r2'] = r2
        row[f'dec_{tgt}_validfrac'] = vf

    # ENCODE: agg+self_mot → neural (median OOF r + frac significant)
    enc_blocks = [b for b in enc_targets if b in built]
    if enc_blocks:
        enc = f_encode_per_neuron(built, enc_blocks, Yk, f_lags_from_ms(lags_ms, fs),
                                  embargo=embargo, n_folds=n_folds)
        r = enc['r_per_cell']
        row['enc_r_med'] = float(np.nanmedian(r))
        row['enc_frac_sig'] = float(np.nanmean(r > r_sig))
    else:
        row['enc_r_med'] = np.nan
        row['enc_frac_sig'] = np.nan
    return row


def f_scan_dataset_params(session, dec_params=None, enc_params=None, *, plan=None,
                          build_cfg=None, source=None, sigma_frames=6, decode=True,
                          encode=True, verbose=False):
    """FULL-param cross-dataset scan for ONE session: acquire the feature block(s) the params
    reference (from cache / conv_repo / build via f_resolve_features), build the neural rate proxy
    the SAME way VR_ca_cebra.py does, then run the EXACT f_run_decoder(dec_params) and
    f_run_encoder(enc_params) — so a config tuned on one session is applied unchanged across
    datasets (dataset = the sweep axis). Returns a row dict:
        {mouse, n_dset, dset_name, n_cells, T, dec_scores{...}, enc_scores{...}}.
    `source` = f_default_source kwargs for this mouse (features_dir / cache_tag / conv_tag / …);
    only the referenced blocks are loaded. dec_params should use input_source='neural' (no CEBRA
    models are fit here). A missing cache/repo for a needed block raises from f_resolve_features —
    the caller loop catches + logs it (so one un-generated dataset doesn't abort the scan)."""
    from f_feature_source import f_resolve_features
    from f_decoding import f_run_decoder
    from f_encoding import f_run_encoder
    est1 = session['est1']
    need = set()
    if decode and dec_params is not None:
        need.add(dec_params['target_block'])
    if encode and enc_params is not None:
        fb = enc_params['feature_block']
        need |= {fb} if isinstance(fb, str) else set(fb)
        need |= set(enc_params.get('covariate_blocks', []))
    built, frame_t, _side, _mov = f_resolve_features(
        session, plan, build_cfg, dset_name=session['dset_name'],
        only=sorted(need), verbose=verbose, **(source or {}))

    # neural rate proxy — peak-normalized smoothed S (matches VR_ca_cebra Y_neu); cols = cells.
    S_sm = f_gauss_smooth(np.asarray(est1['S'], float), sigma_frames=sigma_frames)
    S_smn = S_sm / (np.max(S_sm, axis=1)[:, None] + 1e-12)
    Y_neu = S_smn.T.astype(np.float32)                    # (T, n_cells)
    keep = np.isfinite(Y_neu).all(axis=0) & (Y_neu.std(axis=0) > 0)

    row = {'mouse': session['mouse_id'], 'n_dset': session['n_dset'],
           'dset_name': session['dset_name'], 'n_cells': int(keep.sum()),
           'T': int(Y_neu.shape[0]), 'dec_scores': {}, 'enc_scores': {}}
    if decode and dec_params is not None:
        row['dec_scores'] = f_run_decoder(dec_params, built, Y_neu, frame_t=frame_t)['scores']
    if encode and enc_params is not None:
        row['enc_scores'] = f_run_encoder(enc_params, built, Y_neu, frame_t=frame_t)['scores']
    return row


def f_count_dsets(mouse_id, vr_data_xlsx):
    """# of do_proc==1 datasets for a mouse (the valid n_dset range, 0..N-1)."""
    import pandas as pd
    vr = pd.read_excel(vr_data_xlsx)
    return int(((vr.mouse_id == mouse_id) & (vr.do_proc == 1)).sum())


def f_plot_scan(df, sort_by='dec_self_mot_r2', metrics=None, figsize=(11, 6)):
    """Horizontal bars of the scan metrics, one panel per metric, datasets sorted by
    `sort_by`. `df` = the scan DataFrame; `metrics` = columns to plot (default the 3
    headline scores). Returns None (no Spyder Figure echo)."""
    import matplotlib.pyplot as plt
    if metrics is None:
        metrics = ['dec_self_mot_r2', 'dec_agg_r2', 'enc_r_med']
    metrics = [m for m in metrics if m in df.columns]
    d = df.sort_values(sort_by).reset_index(drop=True)
    labels = [f"{r.mouse}d{r.n_dset}" for r in d.itertuples()]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True, squeeze=False)
    y = np.arange(len(d))
    for ax, m in zip(axes[0], metrics):
        ax.barh(y, d[m].values, color='steelblue')
        ax.axvline(0, color='gray', lw=0.6)
        ax.set_xlabel(m, fontsize=9)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    axes[0][0].set_ylabel('dataset (sorted by ' + sort_by + ')')
    fig.suptitle('Across-dataset encode/decode quality triage')
    fig.tight_layout()
    return None
