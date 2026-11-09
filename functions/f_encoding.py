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
from f_decoding import (f_shuffle_neural, f_lag_embed, f_lags_from_ms,
                        f_imaging_fs, f_structure_scores, f_plot_decoder_sweep)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER CONFIG DEFAULTS — edit here to change them for EVERY script (restart the kernel after).
# Values = VR_ca_cebra's enc_params on 2026-11-06 (PLAN_encoding_config.md). Read by
# f_default_encoder_params → f_run_encoder / f_encoder_sweep.
#
# Override from a script instead (one run / one script; these defaults stay as they are):
#   import f_encoding as enc
#   enc_cfg = enc.f_default_encoder_params()                              # all defaults
#   enc_cfg = enc.f_default_encoder_params(ridge_alpha=10)                # one key
#   enc_cfg = enc.f_default_encoder_params(feature_block=['gabmag', 'mengopp'], n_feature_pcs=None,
#                                          lags_ms=[0, 100, 200], covariate_blocks=['self_mot'])
#   enc_cfg['ridge_alpha']                                                # read a value
# Unknown keys raise.
# ─────────────────────────────────────────────────────────────────────────────

def _enc_defaults():
    return {
        # ── neural target ──────────────────────────────────────────────────────────────────────
        'target_sigma_frames': 6,       # Gaussian σ on deconvolved spikes (imaging frames) | 0 = none
                                        #   (used by f_enc_target; cebra's Y_neu uses 6)
        'target_peak_norm':    True,    # True = divide each cell by its max | False = raw scale
        'target_smooth_ms':    0,       # extra Gaussian σ (ms) on the target inside the fit | 0 = off
                                        #   (raises r trivially — hold fixed when comparing)
        'n_neural_pcs':        None,    # None = per-neuron target | int = predict the top-k neural PCs
                                        #   (reduced rank), then back-project → per-neuron r
        # ── features ───────────────────────────────────────────────────────────────────────────
        'feature_block':    ['gabmag'],  # block / bank name, or a LIST (concatenated). A vis_sets bank
                                         #   expands to all its '<bank>:<cell>' blocks; legacy names
                                         #   (pix | gaborS | mengS | meng | agg | self_mot …) work too
        'n_feature_pcs':    150,        # None = all columns | int = top-k feature PCs
                                        #   (set None when the feature set already reduces itself,
                                        #   e.g. feat_sets layer1/layer2 procs — avoids double reduction)
        'feature_pca':      'joint',    # 'per_block' = top-k of EACH block, then lags |
                                        #   'joint' = concat blocks → ONE PCA(k) → then lags |
                                        #   'joint_postlag' = concat → lag-embed → PCA(k)
                                        #   (whole-session PCA — mild leak, pipeline-wide stance)
        'covariate_blocks': [],         # always-in predictors, e.g. ['self_mot'] | [] = pure stimulus
        'standardize':      True,       # z-score the feature design (global + per fold) |
                                        #   False = keep PC-score variance (ridge favours top PCs);
                                        #   pls / cca ignore it
        # ── model ──────────────────────────────────────────────────────────────────────────────
        'decoder':          'ridge',    # 'ridge' (fixed α) | 'ridgecv' (α per fold) | 'elasticnet' |
                                        #   'lasso' | 'huber' (robust loss) | 'lad' (abs-error, SLOW) |
                                        #   'knn' | 'pls' | 'cca'
        'ridge_alpha':      1e3,        # ridge α (larger = more shrinkage); also elasticnet / lasso
                                        #   strength (sklearn scale differs — re-tune much smaller)
        'l1_ratio':         1e-4,       # elasticnet only: 0 = pure L2 … 1 = pure L1
        'ridgecv_alphas':   None,       # ridgecv only: α grid | None = logspace(-2, 4, 13)
        'k_nn':             15,         # knn only: neighbours
        'n_components':     20,         # pls / cca only: components (clamped per fold)
        'roughness_lambda': 0,          # temporal output-roughness penalty λ·‖D(Xw)‖² | 0 = off
                                        #   (>0 forces contiguous folds)
        'roughness_order':  2,          # 1 = penalize jumps | 2 = penalize spikes / curvature
        # ── cross-validation ───────────────────────────────────────────────────────────────────
        'lags_ms':          [0, 100, 200, 300, 400, 500],   # CAUSAL lags (stimulus precedes response)
        'n_folds':          5,
        'fold_scheme':      'interleaved',  # 'contiguous' | 'interleaved' (small blocks round-robin)
        'n_blocks':         None,       # interleaved only: blocks | None = n_folds × 10
        'embargo_sec':      0.0,        # gap between train and test folds (s)
        # ── scores ─────────────────────────────────────────────────────────────────────────────
        'scores':           ['r_med', 'r2_med', 'frac_sig', 'r2_vw', 'rv', 'procrustes', 'rsa'],
                                        # per-unit: 'r_med' (+ r_p05/p25/p75/p95/r_max) | 'r2_med'
                                        #   (+ spread) | 'frac_sig'; population: 'r2' | 'r2_vw' | 'rv' |
                                        #   'procrustes' | 'rsa'
        'r_sig':            0.1,        # frac_sig = fraction of units with r > r_sig
        'rsa_subsample':    2000,       # frames sampled for 'rsa'
        'seed':             42,
    }


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
                        decoder='ridge', ridge_alpha=1.0, k_nn=15, l1_ratio=0.5,
                        ridgecv_alphas=None, n_components=20,
                        fold_scheme='contiguous', n_blocks=None, row_mask=None, pred_proc=None):
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
    # zscore here only when NOT using pred_proc (else the within-fold pred_proc owns normalization).
    X_stim, stim_names = make_cebra_supervision(block_names, built_blocks,
                                                zscore=(standardize and pred_proc is None))
    X_design, design_names = f_lag_embed(X_stim, lags, names=stim_names)
    ll = [int(l) for l in lags]
    lo = max([0] + [l for l in ll if l > 0])        # leading rows undefined (+lags)
    hi_drop = max([0] + [-l for l in ll if l < 0])  # trailing rows undefined (-lags)
    Y = np.asarray(Y_enc, float)
    T_full = X_design.shape[0]
    X_design = X_design[lo:T_full - hi_drop]
    Y = Y[lo:T_full - hi_drop]

    # Optional row_mask (e.g. movement-only / object-on-screen frames), applied AFTER lag embedding
    # so lags still reference true adjacent frames. Boolean over the ORIGINAL T; trimmed to match.
    # NB: with a mask the OOF frames are a subset — idx/row_offset no longer map to absolute frames.
    if row_mask is not None:
        rm = np.asarray(row_mask, bool)
        if rm.shape[0] != T_full:
            raise ValueError(f"row_mask len {rm.shape[0]} != n frames {T_full}")
        rm = rm[lo:T_full - hi_drop]
        X_design, Y = X_design[rm], Y[rm]

    # Drop degenerate cells (constant / non-finite). f_blocked_cv_r2 drops any
    # row with a NaN in ANY target column, so one silent cell would nuke all rows.
    keep_cells = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep_cells]

    out = f_blocked_cv_r2(X_design, Yk, n_folds=n_folds, decoder=decoder,
                          ridge_alpha=ridge_alpha, k_nn=k_nn, l1_ratio=l1_ratio,
                          ridgecv_alphas=ridgecv_alphas, n_components=n_components,
                          embargo=embargo, standardize=standardize,
                          fold_scheme=fold_scheme, n_blocks=n_blocks,
                          pred_proc=pred_proc, return_predictions=True)
    yt, yh = out['y_true'], out['y_pred']            # (n_oof, n_kept)
    return {'r_per_cell':  _per_col_pearson(yt, yh),
            'r2_per_cell': _per_col_r2(yt, yh),
            'keep_cells':  keep_cells,
            'oof_true':    yt, 'oof_pred': yh, 'idx': out['idx'],
            'row_offset':  lo,   # abs frame = row_offset + idx (both-ends trim)
            'design_names': design_names, 'n_design': X_design.shape[1]}


def f_encode_roughness_cv(built_blocks, block_names, Y_enc, lags, lam_rough=1.0, order=2,
                          lam_ridge=1.0, n_folds=5, embargo=0, standardize=True, row_mask=None):
    """Per-neuron encoding CV with a TEMPORAL OUTPUT-ROUGHNESS penalty (drop-in twin of
    f_encode_per_neuron): min ‖Y−Xw‖² + lam_rough·‖D(Xw)‖² + lam_ridge·‖w‖², where D is the `order`-th
    temporal difference of the PREDICTION (order 1 = penalize jumps, 2 = penalize spikes/curvature).
    Penalizes spiky PREDICTED output — distinct from ridge (weight magnitude) and target-smoothing
    (low-passes Y). D is GAP-AWARE (no penalty across fold/embargo gaps), so folds are CONTIGUOUS
    only. Closed form via ‖D(Xw)‖²=wᵀ(DX)ᵀ(DX)w (row-differences of X — no dense D matrix). Same
    return contract as f_encode_per_neuron."""
    X_stim, stim_names = make_cebra_supervision(block_names, built_blocks, zscore=standardize)
    X_design, design_names = f_lag_embed(X_stim, lags, names=stim_names)
    ll = [int(l) for l in lags]
    lo = max([0] + [l for l in ll if l > 0]); hi = max([0] + [-l for l in ll if l < 0])
    Y = np.asarray(Y_enc, float); T_full = X_design.shape[0]
    X = X_design[lo:T_full - hi]; Y = Y[lo:T_full - hi]
    if row_mask is not None:
        rm = np.asarray(row_mask, bool)
        if rm.shape[0] != T_full:
            raise ValueError(f"row_mask len {rm.shape[0]} != n frames {T_full}")
        rm = rm[lo:T_full - hi]; X, Y = X[rm], Y[rm]
    keep_cells = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep_cells]
    T, d = X.shape
    fold = T // n_folds
    eyeR = lam_ridge * np.eye(d)
    oof = np.full(Yk.shape, np.nan)
    for f in range(n_folds):
        te_lo = f * fold; te_hi = (f + 1) * fold if f < n_folds - 1 else T
        te = np.zeros(T, bool); te[te_lo:te_hi] = True
        ban = np.zeros(T, bool)
        if embargo > 0:
            ban[max(0, te_lo - embargo):min(T, te_hi + embargo)] = True
        idx_tr = np.where((~te) & (~ban))[0]
        Xtr, Ytr, Xte = X[idx_tr], Yk[idx_tr], X[te]
        if standardize:
            mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        # fit an INTERCEPT (center X & Y on train, add Y-mean back on predict) — matches ridge's
        # fit_intercept=True. Without it R² collapses on a nonzero-mean (calcium) target (r is
        # offset-invariant so it looks fine, but r2 tanks). Ridge penalty not applied to the mean.
        mx = Xtr.mean(axis=0); my = Ytr.mean(axis=0)
        Xc = Xtr - mx
        dd = np.diff(idx_tr) == 1                       # adjacent-in-time train pairs (gap-aware)
        if order == 1:
            a = np.where(dd)[0]; DX = Xc[a + 1] - Xc[a]
        else:                                            # 2nd difference over consecutive triples
            tri = np.where(dd[:-1] & dd[1:])[0]
            DX = Xc[tri + 2] - 2 * Xc[tri + 1] + Xc[tri]
        P = lam_rough * (DX.T @ DX) + eyeR
        W = np.linalg.solve(Xc.T @ Xc + P, Xc.T @ (Ytr - my))
        oof[te] = (Xte - mx) @ W + my
    return {'r_per_cell': _per_col_pearson(Yk, oof), 'r2_per_cell': _per_col_r2(Yk, oof),
            'keep_cells': keep_cells, 'oof_true': Yk, 'oof_pred': oof,
            'idx': np.arange(T), 'row_offset': lo,
            'design_names': design_names, 'n_design': d}


# ─────────────────────────────────────────────────────────────────────────────
# Encoding scan — params struct + generic single-axis sweep (the feature→neural mirror
# of f_decoding.f_default_decoder_params / f_run_decoder / f_decoder_sweep). Reports the
# per-neuron encoding readout (r_med / frac_sig / r2_med) AND the population-structure
# scores (r2_vw / rv / procrustes / rsa) from ONE f_encode_per_neuron fit.
# ─────────────────────────────────────────────────────────────────────────────

def f_default_encoder_params(**overrides):
    """Encoder config (feature → neural): _enc_defaults() (top of this module; every key + its
    options; defaults = VR_ca_cebra's enc_params) updated with `overrides`. Unknown keys raise.
    Consumed by f_run_encoder / f_encoder_sweep / f_encode_over_inputs."""
    import copy
    cfg = _enc_defaults()
    bad = [k for k in overrides if k not in cfg]
    if bad:
        raise ValueError(f'f_default_encoder_params: unknown key(s) {bad}; have {sorted(cfg)}')
    cfg.update(copy.deepcopy(overrides))
    return cfg


def f_enc_target(est1, cfg, verbose=True):
    """Per-cell encoding target from the encoder config: f_utils.f_neural_rate with
    cfg['target_sigma_frames'] / cfg['target_peak_norm'] (cells in est1 order; no sort — per-unit
    scores and the population scores are order-invariant). Returns {'Y' (T, n_active) float32,
    'cell_ids' (est1 row per column), 'active' bool}. Ensemble targets: f_ensemble_pipeline.f_ens_targets."""
    from f_utils import f_neural_rate
    R, active = f_neural_rate(est1['S'], sigma_frames=cfg['target_sigma_frames'],
                              peak_norm=cfg['target_peak_norm'])
    if verbose:
        print(f"encoding target: {int(active.sum())}/{active.size} active cells x {R.shape[1]} frames "
              f"(sigma={cfg['target_sigma_frames']}, peak_norm={cfg['target_peak_norm']})")
    return {'Y': R.T.astype(np.float32), 'cell_ids': np.flatnonzero(active), 'active': active}


def f_enc_config_tag(cfg):
    """Short stable hash of an encoder config, for store keys / labels."""
    import hashlib
    import json
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _joint_feature_block(built_blocks, fbs, covs, lags, n_pcs, mode, seed=42):
    """Build a JOINT-PCA feature design: concatenate the feature blocks `fbs` and fit ONE PCA →
    top-`n_pcs` components (a shared basis + a TOTAL dimensionality cap, vs per-block top-k).
      mode='joint'         → PCA the concatenated features, THEN lag-embed (design = (k+cov)·n_lags).
      mode='joint_postlag' → lag-embed the concatenated features FIRST, then PCA (design = k + cov·
                             n_lags): the PCA caps the feature×lag dimensionality; covariates are
                             lag-embedded and baked in. Global PCA (mild leak, matching the pipeline's
                             convention). Returns (bb2, feat_block_names, lags2, covs2) to pass on."""
    from sklearn.decomposition import PCA
    Xf, _ = make_cebra_supervision(fbs, built_blocks, zscore=False)      # (T, Σ feat cols), NaN-free
    if mode == 'joint':
        k = int(min(n_pcs, *Xf.shape))
        scores = PCA(n_components=k, random_state=seed).fit_transform(Xf)
        bb2 = {**built_blocks, '_FEAT_JOINT': {'X': scores.astype(np.float32),
               'names': [f'jpc{i}' for i in range(k)], 'pca': None}}
        return bb2, ['_FEAT_JOINT'], lags, list(covs)
    # joint_postlag: lag-embed features, PCA on the lag-embedded design, bake covariates in
    Xfl, _ = f_lag_embed(Xf, lags)
    m = np.isfinite(Xfl).all(axis=1)
    k = int(min(n_pcs, m.sum(), Xfl.shape[1]))
    pca = PCA(n_components=k, random_state=seed).fit(Xfl[m])
    scores = np.full((Xfl.shape[0], k), np.nan)
    scores[m] = pca.transform(Xfl[m])
    parts, names = [scores], [f'jpc{i}' for i in range(k)]
    if covs:                                                             # lag-embed covariates, concat
        Xc, cn = make_cebra_supervision(covs, built_blocks, zscore=False)
        Xcl, cnl = f_lag_embed(Xc, lags, names=cn)
        parts.append(Xcl); names += list(cnl)
    final = np.concatenate(parts, axis=1).astype(np.float32)
    bb2 = {**built_blocks, '_FEAT_JOINT_PL': {'X': final, 'names': names, 'pca': None}}
    return bb2, ['_FEAT_JOINT_PL'], [0], []                              # lags baked in → no re-embed


def f_run_encoder(config, built_blocks, Y_neu, frame_t=None, frame_mask=None, verbose=False):
    """ONE encoding fit under `config` (see f_default_encoder_params): assemble covariate_blocks +
    feature_block (optionally truncated to top-k PCs) → per-neuron blocked-CV encode via
    f_encode_per_neuron → the per-neuron readout (r_med / frac_sig / r2_med) AND population-
    structure scores (f_structure_scores on the OOF neural matrix), from ONE fit. `frame_mask`
    (boolean over T) restricts the ENCODING to a frame subset (e.g. movement / object-on-screen),
    applied AFTER lag embedding — the neural target PCA (n_neural_pcs) is still fit on all frames.
    Returns {'scores': {name: val}, 'n_valid': int, 'res': <encode dict>}."""
    fs = f_imaging_fs(frame_t) if frame_t is not None else 60.0
    embargo = int(round(config.get('embargo_sec', 0.0) * fs))
    lags = f_lags_from_ms(config.get('lags_ms', [0]), fs)

    # A grid BANK name expands to all its `<bank>:<cell>` blocks; exact names pass through unchanged.
    from f_feature_source import f_expand_blocks
    fbs = f_expand_blocks(built_blocks, config['feature_block'], what='feature_block')
    bb = built_blocks
    ntp = config.get('n_feature_pcs')
    fpca = config.get('feature_pca', 'per_block')         # 'per_block' | 'joint' | 'joint_postlag'
    covs = f_expand_blocks(built_blocks, config.get('covariate_blocks', []), what='covariate_block')
    if ntp is not None and fpca in ('joint', 'joint_postlag'):
        # concat feature blocks → ONE PCA (total k), optionally after lag-embedding (joint_postlag)
        bb, fbs, lags, covs = _joint_feature_block(built_blocks, fbs, covs, lags, int(ntp), fpca,
                                                    config.get('seed', 42))
        block_names = covs + fbs
    elif ntp is not None:                                 # per-block: truncate EACH block to top-k PCs
        bb = dict(built_blocks)
        for b in fbs:
            binfo = built_blocks[b]
            if isinstance(binfo.get('pca'), dict) or binfo.get('pca_groups'):
                raise ValueError(f"feature_block {b!r} uses grouped PCA — top-k truncation invalid; "
                                 f"set n_feature_pcs=None, feature_pca='joint', or single-PCA blocks")
            bb[b] = {**binfo, 'X': np.asarray(binfo['X'], float)[:, :int(ntp)],
                     'names': list(binfo['names'])[:int(ntp)]}
        block_names = covs + fbs
    else:
        block_names = covs + fbs

    # Optional neural-target PCA reduction (reduced-rank encoding): predict the top-k neural PCs,
    # then inverse-transform predictions back to cells so per-neuron r is preserved (scored against
    # the rank-k projection of the true neural). PCA is global (mild target-basis leak, like the
    # decode side's global prefix). None → per-neuron target (the default deliverable).
    # optional TARGET smoothing (Gaussian, ms → frames): predict the smoothed rate instead of the
    # raw spiky deconvolved trace (bias-for-variance; not the same as ridge). Same smoothing in train
    # & test (it's a target preprocessing step). NB: more smoothing trivially raises r — hold it fixed
    # when comparing models.
    Y_in = np.asarray(Y_neu, float)
    _tsm = config.get('target_smooth_ms', 0)
    if _tsm:
        from f_utils import f_gauss_smooth
        _sb = max(0, int(round(_tsm * fs / 1000.0)))
        if _sb:
            Y_in = f_gauss_smooth(Y_in.T, sigma_frames=_sb).T

    n_npc = config.get('n_neural_pcs')
    pca_neu = None
    Y_target = Y_in
    if n_npc is not None:
        from sklearn.decomposition import PCA
        Yf = np.asarray(Y_in, float)
        keepc = np.isfinite(Yf).all(axis=0) & (Yf.std(axis=0) > 0)
        Yk = Yf[:, keepc]
        k = int(min(int(n_npc), max(1, Yk.shape[1] - 1)))
        pca_neu = PCA(n_components=k, random_state=config.get('seed', 42)).fit(Yk)
        Y_target = pca_neu.transform(Yk)              # (T, k) PC scores as the encoding target

    _rl = config.get('roughness_lambda', 0)
    if _rl and _rl > 0:                                   # temporal output-roughness penalty path
        if config.get('fold_scheme', 'contiguous') != 'contiguous':
            print('  roughness_lambda>0: forcing contiguous folds (roughness needs time contiguity)')
        enc = f_encode_roughness_cv(
            bb, block_names, Y_target, lags, lam_rough=_rl,
            order=config.get('roughness_order', 2), lam_ridge=config.get('ridge_alpha', 1.0),
            n_folds=config.get('n_folds', 5), embargo=embargo,
            standardize=config.get('standardize', True), row_mask=frame_mask)
    else:
        enc = f_encode_per_neuron(
            bb, block_names, Y_target, lags, embargo=embargo,
            n_folds=config.get('n_folds', 5), standardize=config.get('standardize', True),
            decoder=config.get('decoder', 'ridge'), ridge_alpha=config.get('ridge_alpha', 1.0),
            l1_ratio=config.get('l1_ratio', 0.5),
            k_nn=config.get('k_nn', 15), ridgecv_alphas=config.get('ridgecv_alphas'),
            n_components=config.get('n_components', 20),
            fold_scheme=config.get('fold_scheme', 'contiguous'), n_blocks=config.get('n_blocks'),
            row_mask=frame_mask)

    # Per-neuron readout: raw target → straight from f_encode_per_neuron; PCA target → reconstruct
    # predictions + rank-k-projected truth to cell space, then per-cell r. Structure scores use the
    # PC-space OOF when reduced (orthonormal-invariant), else the full neural OOF.
    if pca_neu is None:
        true_cells = np.asarray(enc['oof_true'], float)   # (n_oof, n_kept) per-neuron OOF
        pred_cells = np.asarray(enc['oof_pred'], float)
        r = np.asarray(enc['r_per_cell'], float)
        r2c = np.asarray(enc['r2_per_cell'], float)
    else:
        W = pca_neu.components_                       # (k, n_kept)
        true_cells = np.asarray(enc['oof_true']) @ W  # rank-k projection of the true neural
        pred_cells = np.asarray(enc['oof_pred']) @ W
        r = _per_col_pearson(true_cells, pred_cells)  # per-neuron (kept cells)
        r2c = _per_col_r2(true_cells, pred_cells)
    struct_true, struct_pred = enc['oof_true'], enc['oof_pred']
    which = tuple(config.get('scores', ('r_med', 'frac_sig')))
    scores = {}
    if 'r_med' in which:
        scores['r_med'] = float(np.nanmedian(r))
        if np.isfinite(r).any():                     # per-neuron r spread → sweep band (5/25/75/95)
            for _p, _k in ((5, 'r_p05'), (25, 'r_p25'), (75, 'r_p75'), (95, 'r_p95')):
                scores[_k] = float(np.nanpercentile(r, _p))
            scores['r_max'] = float(np.nanmax(r))    # best cell / ensemble
    if 'frac_sig' in which:
        scores['frac_sig'] = float(np.nanmean(r > config.get('r_sig', 0.1)))
    if 'r2_med' in which:
        scores['r2_med'] = float(np.nanmedian(r2c))
        if np.isfinite(r2c).any():                   # per-neuron R² spread → sweep band (5/25/75/95)
            for _p, _k in ((5, 'r2_p05'), (25, 'r2_p25'), (75, 'r2_p75'), (95, 'r2_p95')):
                scores[_k] = float(np.nanpercentile(r2c, _p))
            scores['r2_max'] = float(np.nanmax(r2c))  # best cell / ensemble
    struct = [s for s in which if s in ('r2', 'r2_vw', 'rv', 'procrustes', 'rsa')]
    if struct:
        scores.update(f_structure_scores(
            struct_true, struct_pred, which=tuple(struct),
            rsa_subsample=config.get('rsa_subsample', 2000),
            rng=np.random.default_rng(config.get('seed', 42))))
    if verbose:
        print(f"  [{'+'.join(block_names)} → neural]  n_feat_pcs={ntp}  "
              + '  '.join(f'{k}={v:+.3f}' for k, v in scores.items()))
    return {'scores': scores, 'n_valid': int(enc['oof_true'].shape[0]), 'res': enc,
            'r_per_cell': r, 'r2_per_cell': r2c,
            'oof_true_cells': true_cells, 'oof_pred_cells': pred_cells,
            'keep_cells': enc['keep_cells'], 'idx': enc['idx'],
            'row_offset': enc['row_offset']}


def f_encoder_sweep(config, sweep_var, sweep_values, built_blocks, Y_neu,
                    frame_t=None, frame_mask=None, verbose=True):
    """Vary ONE key of `config` across `sweep_values`, run f_run_encoder at each, collect scores —
    the encoding mirror of f_decoder_sweep. `sweep_var` = a config key ('n_feature_pcs',
    'feature_block', 'decoder', 'fold_scheme', ...) OR the virtual axis 'lag_ms' (single-lag
    tuning: each scalar sets lags_ms=[v]). `frame_mask` (boolean over T) restricts encoding to a
    frame subset (movement / object-on-screen). Bad combos (grouped block, etc.) skip with a printed
    reason. Returns tidy rows [{'x': value, <score>: val, ..., 'n_valid': int}].
    Plot with f_decoding.f_plot_decoder_sweep (direction-agnostic)."""
    if sweep_var not in config and sweep_var != 'lag_ms':
        raise ValueError(f"sweep_var {sweep_var!r} not an encoder-params key "
                         f"({list(config)}) or the virtual axis 'lag_ms'")
    rows = []
    for v in sweep_values:
        cfg = ({**config, 'lags_ms': [v]} if sweep_var == 'lag_ms'
               else {**config, sweep_var: v})
        try:
            out = f_run_encoder(cfg, built_blocks, Y_neu, frame_t, frame_mask=frame_mask)
        except (ValueError, KeyError) as e:
            print(f"  skip {sweep_var}={v!r}: {e}")
            continue
        rows.append({'x': v, **out['scores'], 'n_valid': out['n_valid']})
        if verbose:
            print(f"  {sweep_var}={str(v):>12s}  "
                  + '  '.join(f'{k}={rows[-1][k]:+.3f}' for k in out['scores']))
    return rows


def f_encode_sweep(enc_cfg, axes, targets, built_blocks, frame_t=None, frame_mask=None,
                   keep_units=False, verbose=True):
    """Encoding sweep over one or more axes → tidy table (one row per axis combination × target).
    `enc_cfg` from f_default_encoder_params. `axes` = ordered {axis: values}: an encoder key
    ('ridge_alpha', 'lags_ms', 'decoder', 'n_feature_pcs', 'fold_scheme', 'target_smooth_ms', …), the
    virtual 'lag_ms' (each value v → lags_ms=[v]) or 'target'. Every combination is run (1 axis = 1-D,
    2 = 2-D). `targets` = {name: (T, n) array or (Y, labels)}, e.g. {'cells': f_enc_target(...)['Y'],
    **f_ens_targets(ens)}; if 'target' is not an axis, every target is run. Features are built once
    (`built_blocks`); feature-set axes come in a later step. `keep_units` → per-unit r / R² per row.
    Bad combinations are skipped with the reason printed.
    Returns (df, units): df columns = axis names + 'target' + every score + 'n_valid' + 'enc_tag';
    units = {row index: {'r', 'r2', 'labels'}} (empty unless keep_units). Plot: f_encoding_plots.f_plot_encode_sweep."""
    import itertools
    import pandas as pd
    tg = {k: (v if isinstance(v, tuple) else (v, None)) for k, v in targets.items()}
    axes = dict(axes)
    for a in axes:
        if a not in enc_cfg and a not in ('lag_ms', 'target'):
            raise ValueError(f"axis {a!r} is not an encoder key ({sorted(enc_cfg)}), 'lag_ms' or 'target'")
    if 'target' in axes:
        bad = [t for t in axes['target'] if t not in tg]
        if bad:
            raise ValueError(f'target(s) {bad} not in targets {list(tg)}')
    names = list(axes) + ([] if 'target' in axes else ['target'])
    values = [list(axes[a]) for a in axes] + ([] if 'target' in axes else [list(tg)])
    rows, units = [], {}
    for combo in itertools.product(*values):
        point = dict(zip(names, combo))
        cfg = dict(enc_cfg)
        for a, v in point.items():
            if a == 'lag_ms':
                cfg['lags_ms'] = [v]
            elif a != 'target':
                cfg[a] = v
        Y, lab = tg[point['target']]
        try:
            out = f_run_encoder(cfg, built_blocks, Y, frame_t, frame_mask=frame_mask)
        except (ValueError, KeyError) as e:
            print(f'  skip {point}: {e}')
            continue
        rows.append({**point, **out['scores'], 'n_valid': out['n_valid'], 'enc_tag': f_enc_config_tag(cfg)})
        if keep_units:
            res = out['res']
            units[len(rows) - 1] = {'r': np.asarray(res.get('r_per_cell'), float),
                                    'r2': np.asarray(res.get('r2_per_cell'), float), 'labels': lab}
        if verbose:
            _s = '  '.join(f'{k}={out["scores"][k]:+.3f}' for k in ('r_med', 'r2_med', 'frac_sig')
                           if k in out['scores'])
            print('  ' + '  '.join(f'{a}={v}' for a, v in point.items()) + f'   {_s}')
    return pd.DataFrame(rows), units


def f_encode_over_inputs(config, built_blocks, targets, frame_t=None, frame_mask=None, verbose=True):
    """Run the per-neuron encoder for SEVERAL neural TARGET sets and collect each set's per-unit
    scores — a sweep over the INPUT (what's being encoded) rather than a config key. `targets` =
    dict {label: Y (T, n_units)} (build with f_tuning.f_select_tuning_units — e.g. 'cells',
    'reliable_cells', 'ensembles', 'ens_cells'). Feature side + config are identical across sets; only
    the neural target changes. Returns {label: f_run_encoder output} (each has r_per_cell /
    r2_per_cell). Plot the overlay with f_plot_encode_ecdf_joint."""
    out = {}
    for lab, Y in targets.items():
        Y = np.asarray(Y, float)
        try:
            res = f_run_encoder(config, built_blocks, Y, frame_t, frame_mask=frame_mask)
        except (ValueError, KeyError) as e:
            print(f"  skip input {lab!r}: {e}")
            continue
        out[lab] = res
        if verbose:
            r = np.asarray(res.get('r_per_cell', []), float); r = r[np.isfinite(r)]
            r2 = np.asarray(res.get('r2_per_cell', []), float); r2 = r2[np.isfinite(r2)]
            print(f"  input {lab:16s} n_units={Y.shape[1]:4d}  "
                  f"r_med={np.median(r) if r.size else np.nan:+.3f}  "
                  f"r2_med={np.median(r2) if r2.size else np.nan:+.3f}")
    return out


def f_lno_vs_encode_groups(groups, config, built_blocks, Y_neu_raw, frame_t=None, frame_mask=None,
                           ens_out=None, ens_table=None, dimred_params=None, fs=None,
                           ens_lno_metric='cv_r2', which='sig', verbose=True):
    """Per-unit (LNO R², encoding score) for several GROUPS → overlay-scatter data. For each group
    (via f_tuning.f_select_tuning_units): x = LNO R² — cells: population leave-one-neuron-out
    (f_ensembles.f_run_dimred per_cell_r2); ensembles: ens_table[ens_lno_metric] (within-ensemble
    member LNO). y = encoding score (feature→unit, per-unit, f_run_encoder). Aligned on the encoder's
    kept units. `which` = ensembles 'sig'|'all'. Returns {group: {'lno', 'enc_r', 'enc_r2', 'labels'}}."""
    from f_tuning import f_select_tuning_units
    from f_ensembles import f_run_dimred
    cell_lno = None                                     # population per-cell LNO (computed once, lazy)
    out = {}
    for g in groups:
        try:
            Yg, labels, kind = f_select_tuning_units(g, Y_neu_raw, ens_out=ens_out, ens_table=ens_table,
                                                     dimred_params=dimred_params, ens_src_which=which, fs=fs)
        except Exception as e:
            print(f"  skip group {g!r}: {e}"); continue
        if kind == 'ensembles':                         # per-ensemble LNO = ens_table metric
            if ens_table is None:
                print(f"  skip {g!r}: no ens_table"); continue
            _by = ens_table.set_index('ensemble')[ens_lno_metric]
            lno_units = np.array([float(_by.get(int(str(l)[3:]), np.nan)) for l in labels])
        else:                                           # cell-type: population LNO per active cell
            if cell_lno is None:
                cell_lno = np.asarray(f_run_dimred(dimred_params, np.asarray(Y_neu_raw, float).T,
                                                   fs=fs)['per_cell_r2'], float)
            lno_units = cell_lno[np.asarray(labels, int)]
        try:
            res = f_run_encoder(config, built_blocks, np.asarray(Yg, float), frame_t, frame_mask=frame_mask)
        except (ValueError, KeyError) as e:
            print(f"  skip group {g!r} (encode): {e}"); continue
        keep = np.asarray(res['res'].get('keep_cells', np.ones(np.asarray(Yg).shape[1], bool)), bool)
        lno_k = lno_units[keep] if lno_units.shape[0] == keep.shape[0] else lno_units
        out[g] = {'lno': lno_k, 'enc_r2': np.asarray(res['r2_per_cell'], float),
                  'enc_r': np.asarray(res['r_per_cell'], float), 'labels': labels}
        if verbose:
            print(f"  {g:16s} n={out[g]['enc_r2'].size:4d}  LNO med {np.nanmedian(lno_k):+.3f}  "
                  f"enc_r2 med {np.nanmedian(out[g]['enc_r2']):+.3f}")
    return out


def f_plot_lno_vs_encode(data, metric='r2', dset_tag='', title=None):
    """Scatter of LNO R² (x, reliability) vs encoding score (y, feature→unit) per unit — one color
    per group (from f_lno_vs_encode_groups). `metric`='r2'|'r' picks the y-axis (encoding). Identity
    + 0 lines drawn. Returns None."""
    import matplotlib.pyplot as plt
    key = {'r2': 'enc_r2', 'r': 'enc_r'}[metric]
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.2))
    for i, (g, d) in enumerate(data.items()):
        x = np.asarray(d['lno'], float); y = np.asarray(d[key], float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=16, alpha=0.6, color=f'C{i}', label=f'{g} (n={int(m.sum())})')
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6, alpha=0.6)     # identity
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('LNO R²  (reliability)'); ax.set_ylabel(f'encoding {metric}  (feature→unit)')
    ax.legend(fontsize=8, loc='best')
    ax.set_title(title or f'{dset_tag} — LNO R² vs encoding {metric} by group')
    fig.tight_layout()
    return None


def f_plot_encode_rasters(out, Y_neu=None, frame_t=None, dset_tag='', vmin=0.0, vmax=0.5,
                          cmap='viridis', sort=None, title=None):
    """TRUE vs feature-ENCODED neural rasters in the VR_ca_analysis style: viridis, vmin/vmax on
    the S_smn scale, cells × imaging time, stacked and sharing the x-axis. `out` = f_run_encoder
    result. If Y_neu (T×n_cells, hclust-ordered like the script's Y_neu) and frame_t are given the
    TRUE panel is the full continuous raster and PRED is the OOF predictions placed at their
    absolute frames (dropped/embargo frames = NaN, shown blank) with a seconds x-axis via extent —
    matching the ca_analysis rasters. Without them both panels are the OOF matrices vs OOF-frame
    index. `sort`: None keeps the given (hclust) cell order; 'r' re-sorts by encoding r. Returns
    None (no Spyder Figure echo)."""
    import matplotlib.pyplot as plt
    keep = np.asarray(out['keep_cells'], bool)
    r = np.asarray(out.get('r_per_cell', np.full(int(keep.sum()), np.nan)), float)
    order = np.argsort(-r) if sort == 'r' else np.arange(int(keep.sum()))

    if Y_neu is not None and frame_t is not None:            # full-time, ca_analysis style
        true = np.asarray(Y_neu, float)[:, keep].T           # (n_kept, T) full true, hclust order
        pred = np.full_like(true, np.nan)
        abs_fr = np.asarray(out['row_offset']) + np.asarray(out['idx'], int)
        pred[:, abs_fr] = np.asarray(out['oof_pred_cells'], float).T
        ft = np.asarray(frame_t).ravel()
        extent = [float(ft[0]), float(ft[-1]), true.shape[0], 0]
        xlabel = 'imaging time (s)'
    else:                                                    # OOF-only fallback
        true = np.asarray(out['oof_true_cells'], float).T
        pred = np.asarray(out['oof_pred_cells'], float).T
        extent, xlabel = None, 'out-of-fold frame'

    true, pred, r = true[order], pred[order], r[order]
    kw = dict(aspect='auto', vmin=vmin, vmax=vmax, interpolation='none', cmap=cmap, extent=extent)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].imshow(true, **kw); ax[0].set_title('TRUE neural')
    im = ax[1].imshow(pred, **kw); ax[1].set_title('ENCODED from features (OOF)')
    ax[1].set_xlabel(xlabel)
    for a in ax:
        a.set_ylabel('cell')
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(title or f'{dset_tag} — encoding raster: true vs feature-reconstructed  '
                 f'(median r={np.nanmedian(r):+.3f}, n_cells={true.shape[0]})')
    return None


def f_plot_encode_ecdf(out, dset_tag='', metrics=('r2', 'r'), title=None):
    """ECDF of per-neuron encoding scores from an f_run_encoder result — one step curve per metric
    ('r2' = per-cell OOF R², 'r' = per-cell OOF Pearson r). Shows how the WHOLE population's fits
    are distributed (median marked; the right tail = the well-encoded cells). Returns None."""
    import matplotlib.pyplot as plt
    key = {'r2': 'r2_per_cell', 'r': 'r_per_cell'}
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.2))
    for mtr in metrics:
        v = np.asarray(out.get(key[mtr], []), float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        xs = np.sort(v)
        ys = np.arange(1, xs.size + 1) / xs.size
        line, = ax.step(xs, ys, where='post',
                        label=f'{mtr}  (median {np.median(v):+.3f}, n={v.size})')
        ax.axvline(np.median(v), color=line.get_color(), lw=0.8, ls=':')
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel('per-neuron score'); ax.set_ylabel('cumulative fraction of cells')
    ax.set_ylim(0, 1); ax.legend(fontsize=8, loc='lower right')
    ax.set_title(title or f'{dset_tag} — per-neuron encoding score ECDF')
    fig.tight_layout()
    return None


def f_plot_encode_ecdf_joint(results, metric='r2', dset_tag='', title=None):
    """Overlay per-unit encoding-score ECDFs for SEVERAL input sets (from f_encode_over_inputs) —
    one step curve per label (e.g. cells vs responsive cells vs ensembles), median marked. `metric`
    = 'r2' (per-unit OOF R²) | 'r' (per-unit OOF Pearson r). `results` = {label: f_run_encoder out}.
    Returns None."""
    import matplotlib.pyplot as plt
    key = {'r2': 'r2_per_cell', 'r': 'r_per_cell'}[metric]
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.4))
    for lab, out in results.items():
        v = np.asarray(out.get(key, []), float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        xs = np.sort(v); ys = np.arange(1, xs.size + 1) / xs.size
        line, = ax.step(xs, ys, where='post',
                        label=f'{lab}  (median {np.median(v):+.3f}, n={v.size})')
        ax.axvline(np.median(v), color=line.get_color(), lw=0.8, ls=':')
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel(f'per-unit {metric}'); ax.set_ylabel('cumulative fraction of units')
    ax.set_ylim(0, 1); ax.legend(fontsize=8, loc='lower right')
    ax.set_title(title or f'{dset_tag} — encoding {metric} ECDF by input set')
    fig.tight_layout()
    return None


def f_plot_encode_traces(out, n_show=25, sort_by_r=True, offset=4.0, frame_t=None,
                         dset_tag='', title=None):
    """Overlaid TRUE vs feature-ENCODED per-unit OOF traces, stacked in rows going DOWN with a
    vertical `offset` per unit (best-encoded on top). Each unit is z-scored by its OWN true stats so
    amplitudes compare; true = black, encoded = red. `out` = f_run_encoder result (uses
    oof_true_cells / oof_pred_cells, per-unit `r`). `n_show` = # units (top by r if `sort_by_r`).
    x-axis = OOF frame index (imaging-seconds via `frame_t` only when its length matches the OOF
    length — i.e. no CV reordering / frame_mask). Returns None."""
    import matplotlib.pyplot as plt
    yt = np.asarray(out['oof_true_cells'], float)          # (n_oof, n_units)
    yh = np.asarray(out['oof_pred_cells'], float)
    r = np.asarray(out['r_per_cell'], float)
    n_units = yt.shape[1]
    order = np.argsort(-np.nan_to_num(r, nan=-1.0)) if sort_by_r else np.arange(n_units)
    order = order[:min(int(n_show), n_units)]
    use_t = frame_t is not None and len(frame_t) == yt.shape[0]
    t = np.asarray(frame_t, float) if use_t else np.arange(yt.shape[0])
    fig, ax = plt.subplots(figsize=(12, max(3, 0.42 * len(order))))
    for k, u in enumerate(order):
        mu = np.nanmean(yt[:, u]); sd = np.nanstd(yt[:, u]) or 1.0
        base = -k * offset                                 # k=0 (best) on top, rows go down
        ax.plot(t, (yt[:, u] - mu) / sd + base, color='k', lw=0.6, label=('true' if k == 0 else None))
        ax.plot(t, (yh[:, u] - mu) / sd + base, color='r', lw=0.6, alpha=0.85,
                label=('encoded' if k == 0 else None))
        ax.text(t[0], base + offset * 0.28, f'r={r[u]:+.2f}', fontsize=6, color='dimgray',
                ha='left', va='bottom')
    ax.set_yticks([]); ax.set_xlabel('imaging time (s)' if use_t else 'OOF frame')
    ax.set_ylabel('unit (best on top)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(title or f'{dset_tag} — true vs encoded traces (top {len(order)} by r)')
    fig.tight_layout()
    return None


def f_encode_cell_scores(out, labels=None):
    """Per-cell encoding scores from an f_run_encoder result → tidy DataFrame, one row per KEPT unit,
    sorted by R² (desc). Columns: unit, r (Pearson), r2 (R²; offset/scale-sensitive), r_spearman
    (rank corr — robust to outliers / monotone nonlinearity), nrmse (RMSE ÷ std(true); lower better).
    Different metrics rank cells differently (e.g. a cell can have high r but negative R² = good shape,
    wrong scale/offset), so this exposes WHICH units encode well under WHICH criterion. `labels` =
    optional per-unit names (length = n_cells before keep-filter, or n_kept)."""
    import pandas as pd
    from scipy.stats import rankdata
    yt = np.asarray(out['oof_true_cells'], float); yh = np.asarray(out['oof_pred_cells'], float)
    r = np.asarray(out['r_per_cell'], float); r2 = np.asarray(out['r2_per_cell'], float)
    n = yt.shape[1]
    sp = np.full(n, np.nan); nrmse = np.full(n, np.nan)
    for i in range(n):
        a, b = yt[:, i], yh[:, i]; m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 2 and a[m].std() > 0:
            sp[i] = np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1]
            nrmse[i] = float(np.sqrt(np.mean((a[m] - b[m]) ** 2)) / (a[m].std() + 1e-12))
    keep = out.get('keep_cells')
    if labels is not None and keep is not None and len(labels) == len(keep):
        labels = list(np.asarray(labels, dtype=object)[np.asarray(keep, bool)])
    lab = list(labels) if labels is not None else list(range(n))
    df = pd.DataFrame({'unit': lab, 'r': r, 'r2': r2, 'r_spearman': sp, 'nrmse': nrmse})
    return df.sort_values('r2', ascending=False, na_position='last').reset_index(drop=True)


def f_plot_encode_score_scatter(out, labels=None, annotate=True, dset_tag='', title=None):
    """Per-cell r vs R² scatter — which units encode well, and where r and R² DISAGREE (points below
    the r²=r² curve = right shape but wrong scale/offset; r>0 with R²<0 = overfit). Each point is
    labeled with its unit number (`labels`, default the kept-unit index — e.g. ensemble #); set
    `annotate=False` to hide. Returns None."""
    import matplotlib.pyplot as plt
    r = np.asarray(out['r_per_cell'], float); r2 = np.asarray(out['r2_per_cell'], float)
    n = r.shape[0]
    keep = out.get('keep_cells')
    if labels is not None and keep is not None and len(labels) == len(keep):
        labels = list(np.asarray(labels, dtype=object)[np.asarray(keep, bool)])
    lab = list(labels) if labels is not None else list(range(n))
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.scatter(r, r2, s=14, alpha=0.6)
    if annotate:
        for xi, yi, li in zip(r, r2, lab):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(str(li), (xi, yi), fontsize=6, color='dimgray',
                            xytext=(2, 2), textcoords='offset points')
    xs = np.linspace(0, np.nanmax(np.r_[r, 0.01]), 50)
    ax.plot(xs, xs ** 2, 'k--', lw=0.7, label='R²=r² (ideal linear)')
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel('Pearson r'); ax.set_ylabel('R²'); ax.legend(fontsize=8, loc='lower right')
    ax.set_title(title or f'{dset_tag} — per-cell r vs R² ({int(np.isfinite(r).sum())} units)')
    fig.tight_layout()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# P2a — per-neuron WEIGHT fit (the RF) + reshape + relative ceiling  (sub-plan D1/D5)
# f_encode_per_neuron gives CV SCORES; these give the fitted WEIGHTS = the receptive
# field, plus the reshape back to monitor space and the r_full-relative ceiling.
# ─────────────────────────────────────────────────────────────────────────────

def _encode_design(X_stim, Y_enc, lags):
    """Shared design-build for the weight/STA/smooth fits: lag-embed X_stim (lag-major), trim
    the lag-undefined end rows, drop constant/NaN cells. Returns (X, Yk, keep, ll, lo)."""
    ll = [int(l) for l in lags]
    X_design, _ = f_lag_embed(np.asarray(X_stim, float), ll)      # (T, d_stim*n_lags)
    lo = max([0] + [l for l in ll if l > 0])                     # +lags → leading NaN rows
    hi = max([0] + [-l for l in ll if l < 0])                    # -lags → trailing NaN rows
    T = X_design.shape[0]
    X = X_design[lo:T - hi]
    Y = np.asarray(Y_enc, float)[lo:T - hi]
    keep = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    return X, Y[:, keep], keep, ll, lo


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
    X, Yk, keep, ll, lo = _encode_design(X_stim, Y_enc, lags)
    if standardize:
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
    reg = Ridge(alpha=alpha, fit_intercept=True).fit(X, Yk)
    W_flat = np.atleast_2d(reg.coef_).T if Yk.shape[1] == 1 else reg.coef_.T   # (d_design, n_kept)
    d_stim, n_lags = int(np.asarray(X_stim).shape[1]), len(ll)
    W = W_flat.reshape(n_lags, d_stim, -1).transpose(1, 0, 2)    # (d_stim, n_lags, n_kept)
    return {'W': W, 'W_flat': W_flat, 'intercept': np.atleast_1d(reg.intercept_),
            'keep_cells': keep, 'lags': ll, 'd_stim': d_stim, 'row_offset': lo, 'alpha': alpha}


def f_encode_sta(X_stim, Y_enc, lags):
    """Spike-triggered-average RF (P3 method-3 baseline) — the response-weighted mean stimulus
    per lag (normalized reverse correlation), parameter-free (no ridge α to tune). Returns the
    SAME shape as f_encode_fit_weights (`W` (d_stim, n_lags, n_kept)) so f_rf_pixel_frames /
    f_plot_rf work on it unchanged. NOTE: raw STA is BIASED for correlated / intensity-
    asymmetric natural movies [Theunissen 2001; Meyer 2014] — a baseline to show that, not the
    RF; the whitened STA equals ridge regression (f_encode_fit_weights)."""
    X, Yk, keep, ll, lo = _encode_design(X_stim, Y_enc, lags)
    w = np.clip(Yk, 0, None)                          # deconvolved events as ≥0 weights
    wsum = w.sum(axis=0); wsum[wsum == 0] = 1.0
    Xc = X - X.mean(axis=0)                           # centered stimulus
    sta = (Xc.T @ w) / wsum                           # (d_design, n_kept) response-weighted mean
    d_stim, n_lags = int(np.asarray(X_stim).shape[1]), len(ll)
    W = sta.reshape(n_lags, d_stim, -1).transpose(1, 0, 2)
    return {'W': W, 'W_flat': sta, 'keep_cells': keep, 'lags': ll, 'd_stim': d_stim,
            'row_offset': lo, 'method': 'sta'}


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


# ─────────────────────────────────────────────────────────────────────────────
# P2a-3 / D2 — SMOOTHNESS-regularized RF fit + σ×λ joint CV
# Generalized Tikhonov: minimize ‖y−Xw‖² + wᵀPw, P = a graph-Laplacian penalty coupling
# neighboring SPATIAL units (pixels / grid cells) and adjacent LAGS. The standard data-
# efficient RF prior [ASD/ALD] → a legible RF where plain ridge is salt-and-pepper. Deconv
# σ (target smoothing) and the smoothness λ are tuned in ONE CV (§2d: σ ≈ redundant with lags).
# ─────────────────────────────────────────────────────────────────────────────

def _grid_laplacian(H, W):
    """4-neighbor graph Laplacian (D−A) over an H×W grid, dense (n,n), n=H·W. wᵀLw =
    Σ_edges (w_i−w_j)² = a first-difference smoothness penalty (PSD; null space = constants)."""
    n = H * W
    L = np.zeros((n, n))
    for i in range(H):
        for j in range(W):
            k = i * W + j
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < H and 0 <= jj < W:
                    L[k, ii * W + jj] -= 1.0
                    L[k, k] += 1.0
    return L


def _chain_laplacian(n):
    """1-D chain graph Laplacian over n ordered nodes (adjacent-lag smoothness)."""
    L = np.zeros((n, n))
    for k in range(n):
        if k > 0:
            L[k, k - 1] -= 1.0; L[k, k] += 1.0
        if k < n - 1:
            L[k, k + 1] -= 1.0; L[k, k] += 1.0
    return L


def _smooth_penalty(spatial_shape, n_lags, lam_space, lam_lag, lam_ridge):
    """Tikhonov penalty P for a lag-major weight vector (lag outer, space inner):
    lam_space·(I_lag⊗L_space) [smooth within each lag] + lam_lag·(L_lag⊗I_space) [smooth
    across lags] + lam_ridge·I. spatial_shape = (H,W) pixels or (ny,nx) grid cells."""
    Ls = _grid_laplacian(*spatial_shape)
    d_space = Ls.shape[0]
    Lt = _chain_laplacian(n_lags)
    P = (lam_space * np.kron(np.eye(n_lags), Ls)
         + lam_lag * np.kron(Lt, np.eye(d_space))
         + lam_ridge * np.eye(d_space * n_lags))
    return P


def f_encode_fit_smooth(X_stim, Y_enc, lags, spatial_shape, lam_space=1.0, lam_lag=1.0,
                        lam_ridge=1e-3, standardize=False):
    """Smoothness-regularized RF fit (all rows) — solves w=(XᵀX+P)⁻¹Xᵀy with the Tikhonov
    penalty `P` (_smooth_penalty). Legible RF at low sample counts where plain ridge is ragged.
    `spatial_shape` = (H,W) pixels (or (ny,nx) grid) so d_stim = prod(spatial_shape); the fit
    couples neighboring locations + adjacent lags. Same return shape as f_encode_fit_weights.
    NOTE: d_design = prod(spatial_shape)·n_lags — DOWNSAMPLE to keep the d×d solve tractable."""
    X, Yk, keep, ll, lo = _encode_design(X_stim, Y_enc, lags)
    if standardize:
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
    n_lags = len(ll)
    d_space = int(np.prod(spatial_shape))
    if d_space * n_lags != X.shape[1]:
        raise ValueError(f'spatial_shape {spatial_shape}×{n_lags} lags = {d_space * n_lags} '
                         f'≠ design width {X.shape[1]}')
    P = _smooth_penalty(spatial_shape, n_lags, lam_space, lam_lag, lam_ridge)
    W_flat = np.linalg.solve(X.T @ X + P, X.T @ Yk)              # (d_design, n_kept)
    W = W_flat.reshape(n_lags, d_space, -1).transpose(1, 0, 2)
    return {'W': W, 'W_flat': W_flat, 'keep_cells': keep, 'lags': ll, 'd_stim': d_space,
            'row_offset': lo, 'lam': (lam_space, lam_lag, lam_ridge), 'method': 'smooth'}


def f_encode_smooth_cv(X_stim, Y_enc, lags, spatial_shape, lam_space=1.0, lam_lag=1.0,
                       lam_ridge=1e-3, n_folds=5, embargo=0, standardize=False):
    """Blocked-CV out-of-fold Pearson r for the smoothness-regularized encoder — fit smooth
    weights on the train fold, predict the held-out test fold (contiguous folds + embargo,
    train-only fit; the RF prior mirrors f_blocked_cv_r2 but with the Tikhonov solve). Returns
    {r_per_cell (OOF), keep_cells, median_r}. Used to tune (σ, λ) jointly (§2d)."""
    X, Yk, keep, ll, lo = _encode_design(X_stim, Y_enc, lags)
    n_lags = len(ll)
    P = _smooth_penalty(spatial_shape, n_lags, lam_space, lam_lag, lam_ridge)
    T = X.shape[0]
    fold = T // n_folds
    oof = np.full(Yk.shape, np.nan)
    for f in range(n_folds):
        te_lo = f * fold
        te_hi = (f + 1) * fold if f < n_folds - 1 else T
        te = np.zeros(T, bool); te[te_lo:te_hi] = True
        ban = np.zeros(T, bool)
        if embargo > 0:
            ban[max(0, te_lo - embargo):min(T, te_hi + embargo)] = True
        tr = (~te) & (~ban)
        Xtr, Ytr = X[tr], Yk[tr]
        if standardize:
            mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd
            Xte = (X[te] - mu) / sd
        else:
            Xte = X[te]
        W = np.linalg.solve(Xtr.T @ Xtr + P, Xtr.T @ Ytr)
        oof[te] = Xte @ W
    r = _per_col_pearson(Yk, oof)
    return {'r_per_cell': r, 'keep_cells': keep, 'median_r': float(np.nanmedian(r))}


def f_encode_tune_sigma_lambda(S_raw, X_stim, lags, spatial_shape, sigma_frames_list,
                               lam_list, n_folds=5, embargo=0, report_frac=0.25, verbose=True):
    """JOINT deconv-σ × smoothness-λ CV (§2d, D2). For each σ, RE-SMOOTH the raw deconvolved
    `S_raw` (n_cells, T) → peak-normalized target; for each λ (list of (lam_space, lam_lag[,
    lam_ridge])), score encoding with f_encode_smooth_cv. σ and lags are partly redundant, so
    this is ONE search, not two.

    GUARD (P2 §7, nested split): (σ,λ) are SELECTED by blocked CV on the leading
    (1-report_frac) of time, then the winner's r is REPORTED by a fresh blocked CV on the
    held-out report tail (an `embargo` gap separates them) — so the reported r carries no
    hyperparameter-selection optimism (the P0 low-sample overfitting risk). `sel_r` = the
    selection score every row is ranked on; `best['report_r']` = the honest held-out number.
    report_frac=0 → old single-split behavior (report_r == sel_r; select==report).
    Returns rows [{sigma_frames, lam_space, lam_lag, lam_ridge, sel_r}] + best."""
    from f_utils import f_gauss_smooth
    S_raw = np.asarray(S_raw, float)
    X_stim = np.asarray(X_stim, float)
    T = S_raw.shape[1]
    n_rep = int(round(report_frac * T))
    sel_hi = T - n_rep                                             # select [0,sel_hi); report tail
    rep_lo = min(T, sel_hi + embargo)                             # embargo gap before report split

    def _target(sig):                                             # σ → peak-normalized (T,n_cells)
        S_sm = f_gauss_smooth(S_raw, sigma_frames=sig)
        rowmax = np.max(S_sm, axis=1, keepdims=True)
        return (S_sm / np.where(rowmax == 0, 1.0, rowmax)).T

    rows = []
    for sig in sigma_frames_list:
        Y = _target(sig)
        Xs, Ys = (X_stim[:sel_hi], Y[:sel_hi]) if n_rep > 0 else (X_stim, Y)
        for lam in lam_list:
            ls, ll_, lr = (list(lam) + [1e-3])[:3]
            cv = f_encode_smooth_cv(Xs, Ys, lags, spatial_shape, lam_space=ls, lam_lag=ll_,
                                    lam_ridge=lr, n_folds=n_folds, embargo=embargo)
            rows.append({'sigma_frames': sig, 'lam_space': ls, 'lam_lag': ll_, 'lam_ridge': lr,
                         'sel_r': cv['median_r']})
            if verbose:
                print(f'  σ={sig:>3} frames  λ=({ls:g},{ll_:g})  select OOF r={cv["median_r"]:+.3f}')
    best = max(rows, key=lambda r: r['sel_r'] if np.isfinite(r['sel_r']) else -np.inf)
    # REPORT: winner re-scored on the held-out tail (never seen during selection).
    if n_rep > 0:
        Y = _target(best['sigma_frames'])
        rep = f_encode_smooth_cv(X_stim[rep_lo:], Y[rep_lo:], lags, spatial_shape,
                                 lam_space=best['lam_space'], lam_lag=best['lam_lag'],
                                 lam_ridge=best['lam_ridge'], n_folds=n_folds, embargo=embargo)
        best['report_r'] = rep['median_r']
        best['report_r_per_cell'] = rep['r_per_cell']
    else:
        best['report_r'] = best['sel_r']
    if verbose:
        print(f'  BEST σ={best["sigma_frames"]} λ=({best["lam_space"]:g},{best["lam_lag"]:g})  '
              f'select r={best["sel_r"]:+.3f}  held-out report r={best["report_r"]:+.3f}')
    return rows, best


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


# ─────────────────────────────────────────────────────────────────────────────
# P2a-5 / P2b-1 — BANDED ridge (per-feature-space λ) + variance partition (sub-plan D3)
# One λ per band beats a single global λ for collinear banks of different scale/dimensionality
# [Nunez-Elizalde 2019]; the joint fit yields a per-band unique/shared variance decomposition
# [Dupré la Tour 2022] — the honest replacement for f_encode_block_importance's separate-fit
# drop-one. band_groups = [(band_name, [block,...]), ...]. Banded fit = BandedRidge (f_decoding).
# ─────────────────────────────────────────────────────────────────────────────

def _band_columns(built_blocks, band_groups):
    """(band_name → [blocks]) → (flat block list, per-STIM-column band id, band names). Design
    columns are lag-major (n_lags outer, d_stim inner), so design bands = tile(stim_band,n_lags).
    Widths come from each block's 'names'."""
    flat, stim_band, band_names = [], [], []
    for bi, (name, blocks) in enumerate(band_groups):
        band_names.append(name)
        for b in blocks:
            if b not in built_blocks:
                raise ValueError(f'band {name!r}: block {b!r} not built '
                                 f'(available: {list(built_blocks)})')
            flat.append(b)
            stim_band.extend([bi] * len(built_blocks[b]['names']))
    return flat, np.asarray(stim_band, int), band_names


def f_encode_banded(built_blocks, band_groups, Y_enc, lags, band_alphas,
                    embargo=0, n_folds=5, standardize=True, sel_range=None):
    """Per-neuron BANDED-ridge encoding score: one λ per band. `band_groups` = [(name,[blocks]),
    ...]; `band_alphas` = {band_name: α}. Same design/trim/keep as f_encode_per_neuron but the
    decoder is BandedRidge over the per-column band ids (bands survive standardize — columns
    aren't reordered). `sel_range=(r0,r1)` restricts to a contiguous row slab of the trimmed
    design (for the nested α-selection split). Returns r_per_cell / r2_per_cell / keep_cells /
    oof_true / oof_pred + band_names / bands_design / d_stim / row_offset."""
    flat, stim_band, band_names = _band_columns(built_blocks, band_groups)
    X_stim, stim_names = make_cebra_supervision(flat, built_blocks)
    X_design, design_names = f_lag_embed(X_stim, lags, names=stim_names)
    ll = [int(l) for l in lags]
    lo = max([0] + [l for l in ll if l > 0])
    hi = max([0] + [-l for l in ll if l < 0])
    T_full = X_design.shape[0]
    X_design = X_design[lo:T_full - hi]
    Y = np.asarray(Y_enc, float)[lo:T_full - hi]
    if sel_range is not None:
        X_design = X_design[sel_range[0]:sel_range[1]]
        Y = Y[sel_range[0]:sel_range[1]]
    keep_cells = np.isfinite(Y).all(axis=0) & (Y.std(axis=0) > 0)
    Yk = Y[:, keep_cells]
    d_stim, n_lags = int(X_stim.shape[1]), len(ll)
    bands_design = np.tile(stim_band, n_lags)                 # lag-major → tile per lag
    alpha_map = {i: float(band_alphas[band_names[i]]) for i in range(len(band_names))}
    out = f_blocked_cv_r2(X_design, Yk, n_folds=n_folds, decoder='banded',
                          bands=bands_design, band_alphas=alpha_map, embargo=embargo,
                          standardize=standardize, return_predictions=True)
    yt, yh = out['y_true'], out['y_pred']
    return {'r_per_cell':  _per_col_pearson(yt, yh),
            'r2_per_cell': _per_col_r2(yt, yh),
            'keep_cells':  keep_cells, 'oof_true': yt, 'oof_pred': yh,
            'band_names':  band_names, 'bands_design': bands_design,
            'd_stim':      d_stim, 'row_offset': lo}


def f_encode_tune_bands(built_blocks, band_groups, Y_enc, lags, alpha_grid,
                        embargo=0, n_folds=5, report_frac=0.25, verbose=True):
    """Select per-band α on a leading time split, REPORT the winner on a held-out tail (nested
    guard, P2 §7 — mirrors f_encode_tune_sigma_lambda). `alpha_grid` = candidate α applied to
    EVERY band (joint product); B bands × G grid → G**B combos (the researcher's two-band-first
    keeps this ~O(G²)). Combos ranked by median select OOF r. Returns rows [{band_alphas, sel_r}]
    + best (best['band_alphas'] = winning per-band α; best['report_r'] = honest held-out r).
    report_frac=0 → report_r == sel_r (single split)."""
    import itertools
    band_names = [g[0] for g in band_groups]
    ll = [int(l) for l in lags]
    lo = max([0] + [l for l in ll if l > 0]); hi = max([0] + [-l for l in ll if l < 0])
    T_tr = int(np.asarray(Y_enc).shape[0]) - lo - hi                  # trimmed row count
    n_rep = int(round(report_frac * T_tr)); sel_hi = T_tr - n_rep
    rep_lo = min(T_tr, sel_hi + embargo)
    rows = []
    for combo in itertools.product(alpha_grid, repeat=len(band_names)):
        ba = {band_names[i]: combo[i] for i in range(len(band_names))}
        sr = (0, sel_hi) if n_rep > 0 else None
        res = f_encode_banded(built_blocks, band_groups, Y_enc, lags, ba,
                              embargo=embargo, n_folds=n_folds, sel_range=sr)
        m = float(np.nanmedian(res['r_per_cell']))
        rows.append({'band_alphas': ba, 'sel_r': m})
        if verbose:
            print('  α=%s  select median r=%+.3f' % (ba, m))
    best = max(rows, key=lambda r: r['sel_r'] if np.isfinite(r['sel_r']) else -np.inf)
    if n_rep > 0:
        rep = f_encode_banded(built_blocks, band_groups, Y_enc, lags, best['band_alphas'],
                              embargo=embargo, n_folds=n_folds, sel_range=(rep_lo, T_tr))
        best['report_r'] = float(np.nanmedian(rep['r_per_cell']))
    else:
        best['report_r'] = best['sel_r']
    if verbose:
        print('  BEST α=%s  select r=%+.3f  held-out report r=%+.3f'
              % (best['band_alphas'], best['sel_r'], best['report_r']))
    return rows, best


def f_encode_variance_partition(built_blocks, band_groups, Y_enc, lags, band_alphas,
                                embargo=0, n_folds=5, standardize=True):
    """Per-cell variance decomposition from the banded joint fit (P2b-1 / P2b-5). For each band:
    UNIQUE = R²_full − R²_(full minus that band), drop-one on the JOINT banded model with the
    other bands keeping their selected α. SHARED = R²_full − Σ unique (aggregate variance carried
    jointly by ≥2 bands — e.g. the closed-loop visual↔self_mot overlap). OOF R² per cell; same
    kept cells across full/drop (Y_enc rows unchanged). To read visual-unique vs behavior-unique
    (P2b-5), group blocks into a 'visual' band and a 'behavior' band. Returns band_names, r2_full
    (n_kept,), unique (n_kept,n_bands), shared (n_kept,), keep_cells."""
    kw = dict(embargo=embargo, n_folds=n_folds, standardize=standardize)
    full = f_encode_banded(built_blocks, band_groups, Y_enc, lags, band_alphas, **kw)
    r2_full, keep, names = full['r2_per_cell'], full['keep_cells'], full['band_names']
    nb = len(names)
    unique = np.full((r2_full.size, nb), np.nan)
    for bi in range(nb):
        rest = [band_groups[j] for j in range(nb) if j != bi]
        if rest:
            ba_rest = {g[0]: band_alphas[g[0]] for g in rest}
            r2_drop = f_encode_banded(built_blocks, rest, Y_enc, lags, ba_rest,
                                      **kw)['r2_per_cell']
            unique[:, bi] = r2_full - r2_drop
        else:
            unique[:, bi] = r2_full
    shared = r2_full - np.nansum(unique, axis=1)
    return {'band_names': names, 'r2_full': r2_full, 'unique': unique,
            'shared': shared, 'keep_cells': keep}


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
