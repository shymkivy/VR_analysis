# -*- coding: utf-8 -*-
"""
CEBRA-specific helpers for VR_ca_cebra.py.

Holds functions tightly coupled to the CEBRA training pipeline. Generic
feature engineering (clock alignment, monitor feature aggregation, motion
features, z-scoring) lives in `f_feature_helpers.py`.

Functions:
    make_cebra_supervision — Build a CEBRA-ready supervision matrix from
                             registry blocks. Concatenates via combine_blocks,
                             fills NaN→0 (CEBRA can't handle NaN), z-scores.
    f_run_cebra            — Thin wrapper around cebra.CEBRA().fit / transform
                             that returns a dict with model + embedding + metadata.

Conventions for both functions:
    - 'cosine' distance + L2-normalized output → embedding lies on a unit
      sphere; standard for behavior-supervised CEBRA.
    - 'time_delta' conditional → positives within a learned time window
      weighted by behavior similarity. The right pick for continuous-
      supervised CEBRA on calcium imaging.
    - device='cuda_if_available' → uses GPU if present.

Moved out of inline definitions in VR_ca_cebra.py on 2026-05-21.
"""

import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score

from f_feature_helpers import f_zscore
from f_decoding import make_decoder


def f_blocked_cv_r2(emb, target, n_folds=5, embargo=0, standardize=True,
                    decoder='ridge', ridge_alpha=1.0, k_nn=15,
                    ridgecv_alphas=None, multioutput='uniform_average',
                    Y_raw=None, n_pcs=None, random_state=42, return_std=False,
                    return_predictions=False, task='regression'):
    """Unified blocked-CV decoding score (single- or multi-output), NaN-aware.

    One implementation behind the three former inline CV helpers in
    VR_ca_cebra.py (f_blocked_cv_r2 [KNN A/B bar], f_cv_r2_single [per-feature
    PCA sweep], _cv_r2_multi [Ridge multi-output decoding]). Unified 2026-06-03
    so embargo / train-only standardization / within-fold PCA / RidgeCV apply
    everywhere and the helpers can't drift apart.

    Contiguous time-blocked folds (not random split) avoid the temporal-
    autocorrelation leakage that inflates random-split CV on imaging data.

    Parameters
    ----------
    emb : ndarray (T, d) or None
        Precomputed embedding used as decoder input. Ignored if Y_raw is given.
    target : ndarray (T,) or (T, p)
        Behavior target(s). Rows where ANY passed column is NaN are dropped from
        train and test each fold. For a single column this is per-column masking;
        for multi-output it is the unavoidable any-column row drop (the decoder
        predicts all columns jointly and can't fit NaN). Pass columns separately
        if you want to keep the maximal frame set per target.
    n_folds : int
        Number of contiguous CV folds.
    embargo : int
        # frames purged on EACH side of the test block (removed from train) to
        cut autocorrelation leakage. 0 = train right up to the test boundary.
    standardize : bool
        z-score embedding columns using TRAIN-fold stats only (stops a high-
        variance PC from dominating KNN distance / the Ridge penalty).
    decoder : {'ridge', 'ridgecv', 'knn'}
        'ridgecv' tunes alpha by efficient leave-one-out CV inside each train
        fold over ridgecv_alphas.
    ridge_alpha, k_nn, ridgecv_alphas
        Decoder hyperparameters. ridgecv_alphas defaults to logspace(-2, 4, 13).
    multioutput : str or array
        Passed to sklearn r2_score for multi-output targets
        ('uniform_average' | 'variance_weighted').
    Y_raw : ndarray (T, N) or None
        If given with n_pcs, fit PCA on TRAIN rows only inside each fold
        (leak-free embedding axes); emb is then ignored.
    n_pcs : int
        # PCs for the within-fold PCA path.
    return_std : bool
        If True return (mean, std) across folds; else just the mean.
    return_predictions : bool
        If True, return a dict with the concatenated, time-ordered out-of-fold
        predictions (for trace / scatter diagnostics) instead of a scalar — see
        Returns. Takes precedence over return_std.
    task : {'regression', 'classification', 'auto'}
        'regression'     — fit the regressor (decoder) and score per-fold R².
        'classification' — BINARY target only: mirror the analog decoder with
                           its classification twin (ridge→LogisticRegression,
                           ridgecv→LogisticRegressionCV, knn→KNeighborsClassifier),
                           predict P(class=1), and score per-fold ROC-AUC. The
                           returned scalar / dict['r2'] then holds AUC (chance =
                           0.5), NOT R². Folds whose train OR test set is single-
                           class score NaN. Requires a single target column.
        'auto'           — classification if the (single-column) target takes
                           only values in {0, 1}; else regression. Lets the
                           caller stay metric-agnostic; the metric used is
                           reported in dict['metric'] (predictions mode).

    Returns
    -------
    If return_predictions is False:
        float, or (mean, std) if return_std. The value is R² (regression) or
        ROC-AUC (classification). NaN (or (NaN, NaN)) if no fold scored.
    If return_predictions is True, a dict with keys
        'r2'       : mean R² across folds (matches the scalar return).
        'r2_std'   : std of per-fold R².
        'r2_folds' : list of per-fold (test) R² (np.nan for skipped folds).
        'r2_train' : mean per-fold TRAIN (in-sample) score.
        'r2_folds_train' : list of per-fold train scores. High train + low/neg
                     test ⇒ overfitting / nonstationarity; low train too ⇒ the
                     feature isn't (linearly) in the embedding at all.
        'idx'      : (n_pts,) original frame indices of the OOF test points,
                     sorted ascending (time order). Excludes NaN-target rows
                     and embargoed frames are never in any test fold anyway.
        'y_true'   : (n_pts,) or (n_pts, p) true target on those frames.
        'y_pred'   : same shape — out-of-fold prediction on those frames.
        'fold'     : (n_pts,) fold id each point was held out in.
        'n_valid'  : # non-NaN target rows. 'T' : total rows.
        'metric'   : 'r2' or 'roc_auc' — which score the values represent.
    """
    target = np.asarray(target)
    was_1d = target.ndim == 1
    if target.ndim == 1:
        target = target[:, None]
    single = target.shape[1] == 1   # fit/score 1-D for one column (avoids the
                                    # sklearnex column-vector-y DataConversionWarning)
    T = (Y_raw.shape[0] if Y_raw is not None else emb.shape[0])
    fold_size = T // n_folds
    valid = ~np.isnan(target).any(axis=1)
    if ridgecv_alphas is None:
        ridgecv_alphas = np.logspace(-2, 4, 13)

    # Resolve 'auto' → regression / classification from the target's support.
    if task == 'auto':
        u = np.unique(target[valid]) if target.shape[1] == 1 else np.array([])
        task = ('classification' if (target.shape[1] == 1 and u.size <= 2
                                     and np.all(np.isin(u, (0.0, 1.0))))
                else 'regression')
    is_clf = (task == 'classification')
    if is_clf and target.shape[1] != 1:
        raise ValueError('classification task requires a single target column')
    metric = 'roc_auc' if is_clf else 'r2'

    def _make():
        # Estimator construction consolidated into f_decoding.make_decoder so the
        # decoder choice can't drift between here and the diagnostic cells.
        return make_decoder(decoder, ridge_alpha=ridge_alpha, k_nn=k_nn,
                            ridgecv_alphas=ridgecv_alphas, task=task)

    r2s = []
    r2s_train = []                                   # per-fold TRAIN score (in-sample);
                                                    # only filled when return_predictions
    p_idx, p_true, p_hat, p_fold = [], [], [], []   # OOF collectors
    for f in range(n_folds):
        te_lo = f * fold_size
        te_hi = (f + 1) * fold_size if f < n_folds - 1 else T
        te_idx = np.zeros(T, dtype=bool); te_idx[te_lo:te_hi] = True
        ban = np.zeros(T, dtype=bool)
        if embargo > 0:
            ban[max(0, te_lo - embargo):min(T, te_hi + embargo)] = True
        tr = (~te_idx) & (~ban) & valid
        te = te_idx & valid
        if tr.sum() < 2 or te.sum() < 2:
            r2s.append(np.nan); r2s_train.append(np.nan); continue
        if Y_raw is not None:
            _p  = PCA(n_components=n_pcs, random_state=random_state)
            Xtr = _p.fit_transform(Y_raw[tr])
            Xte = _p.transform(Y_raw[te])
        else:
            Xtr, Xte = emb[tr], emb[te]
        if standardize:
            mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        reg = _make()
        if is_clf:
            ytr = target[tr].ravel(); yte = target[te].ravel()
            if np.unique(ytr).size < 2:        # can't fit a 1-class train fold
                r2s.append(np.nan); r2s_train.append(np.nan); continue
            reg.fit(Xtr, ytr)
            pos = int(np.argmax(reg.classes_))         # P(class == larger label)
            pred = reg.predict_proba(Xte)[:, pos]      # (n_te,) prob of class 1
            # AUC undefined if the test fold is single-class → NaN score, but
            # still keep the predictions for the trace/scatter diagnostics.
            r2s.append(roc_auc_score(yte, pred) if np.unique(yte).size >= 2
                       else np.nan)
            if return_predictions:                     # in-sample (train) AUC
                ptr = reg.predict_proba(Xtr)[:, pos]
                r2s_train.append(roc_auc_score(ytr, ptr))
            else:
                r2s_train.append(np.nan)
            pred = pred[:, None]
        else:
            yt_fit = target[tr][:, 0] if single else target[tr]   # 1-D if 1 col
            reg.fit(Xtr, yt_fit)
            pred = reg.predict(Xte)
            yte_true = target[te][:, 0] if single else target[te]
            r2s.append(r2_score(yte_true, pred, multioutput=multioutput))
            if return_predictions:                     # in-sample (train) R²
                r2s_train.append(r2_score(yt_fit, reg.predict(Xtr),
                                          multioutput=multioutput))
            else:
                r2s_train.append(np.nan)
        if return_predictions:
            te_where = np.where(te)[0]
            p_idx.append(te_where)
            p_true.append(target[te])
            p_hat.append(pred if np.ndim(pred) == 2 else np.asarray(pred)[:, None])
            p_fold.append(np.full(te_where.size, f, dtype=int))

    mean = np.nan if all(np.isnan(r2s)) else float(np.nanmean(r2s))
    std  = np.nan if all(np.isnan(r2s)) else float(np.nanstd(r2s))

    if return_predictions:
        if p_idx:
            idx = np.concatenate(p_idx)
            yt  = np.concatenate(p_true, axis=0)
            yh  = np.concatenate(p_hat,  axis=0)
            fld = np.concatenate(p_fold)
            order = np.argsort(idx, kind='stable')   # restore time order
            idx, yt, yh, fld = idx[order], yt[order], yh[order], fld[order]
            if was_1d:
                yt = yt.ravel(); yh = yh.ravel()
        else:
            idx = np.array([], dtype=int)
            yt  = np.empty(0) if was_1d else np.empty((0, target.shape[1]))
            yh  = np.empty_like(yt)
            fld = np.array([], dtype=int)
        r2_train = (np.nan if all(np.isnan(r2s_train))
                    else float(np.nanmean(r2s_train)))
        return {'r2': mean, 'r2_std': std, 'r2_folds': r2s,
                'r2_train': r2_train, 'r2_folds_train': r2s_train,
                'idx': idx, 'y_true': yt, 'y_pred': yh, 'fold': fld,
                'n_valid': int(valid.sum()), 'T': int(T), 'metric': metric}

    if np.isnan(mean):
        return (np.nan, np.nan) if return_std else np.nan
    return (mean, std) if return_std else mean


def make_cebra_supervision(group_names, built_blocks, blocks=None):
    """Build a CEBRA-ready supervision matrix from registry blocks.

    Concatenates selected blocks, fills NaN with 0 (CEBRA can't handle NaN —
    empty-frame masking would crash training), and z-scores column-wise.
    Returns the (T, d_total) matrix + column names; ready to pass to
    CEBRA's model.fit(neural, behavior).

    Parameters
    ----------
    group_names : str | list of str
        Block name(s) from `built_blocks`, e.g. ['agg', 'motion'] or 'agg'.
    built_blocks : dict
        The feature registry (from f_feature_helpers.build_feature_blocks).
    blocks : dict, optional
        Override for `built_blocks`. Defaults to the passed `built_blocks`.

    Returns
    -------
    X_sup : ndarray (T, d_total)
        z-scored, NaN-free supervision matrix.
    names : list of str
        Column names matching the concatenation order.
    """
    if blocks is None:
        blocks = built_blocks
    if isinstance(group_names, str):
        group_names = [group_names]
    missing = [g for g in group_names if g not in blocks]
    if missing:
        raise ValueError(f'block(s) {missing!r} not built; available: {list(blocks)}')
    Xs, ns = [], []
    for g in group_names:
        Xs.append(blocks[g]['X'])
        ns.extend(blocks[g]['names'])
    X = np.concatenate(Xs, axis=1)
    X_filled = np.where(np.isnan(X), 0.0, X)
    return f_zscore(X_filled), ns


def f_run_cebra(neural, behavior, out_dim=3, max_iter=5000,
                batch_size=512, lr=3e-4, temperature=1.0,
                arch='offset10-model', random_state=42, verbose=True):
    """Train a CEBRA model and return {model, embedding, metadata} dict.

    Parameters
    ----------
    neural : ndarray (T, N)
        Neural activity, one row per imaging frame.
    behavior : ndarray (T, d)
        Supervision signal (typically z-scored). Must be NaN-free.
    out_dim : int
        Embedding dimensionality (CEBRA `output_dimension`).
    max_iter, batch_size, lr, temperature, arch, random_state, verbose
        Standard CEBRA hyperparameters.

    Returns
    -------
    out : dict with keys
        'model'     : trained CEBRA instance
        'embedding' : (T, out_dim) embedding from model.transform(neural)
        'out_dim', 'max_iter', 'arch' : echoed hyperparameters
        'duration'  : fit wall time (s)

    Notes
    -----
    distance='cosine' → embedding L2-normalized to unit sphere.
    conditional='time_delta' → continuous-supervised mode.
    device='cuda_if_available' → uses GPU when present.
    """
    from cebra import CEBRA
    import torch
    torch.manual_seed(random_state)
    model = CEBRA(model_architecture=arch,
                  batch_size=batch_size,
                  learning_rate=lr,
                  temperature=temperature,
                  output_dimension=out_dim,
                  max_iterations=max_iter,
                  distance='cosine',
                  conditional='time_delta',
                  device='cuda_if_available',
                  verbose=verbose)
    t0 = time.perf_counter()
    model.fit(neural, behavior)
    emb = model.transform(neural)
    return {'model':       model,
            'embedding':   emb,
            'out_dim':     out_dim,
            'max_iter':    max_iter,
            'duration':    time.perf_counter() - t0,
            'arch':        arch}
