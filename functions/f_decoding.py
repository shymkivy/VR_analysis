# -*- coding: utf-8 -*-
"""
Decoding toolkit for VR_ca_cebra.py — compute + plotting helpers.

Consolidates the repeated decoding machinery that had drifted across the cells
of VR_ca_cebra.py (imaging-rate recompute, decoder-by-name factory, per-cell
shuffle null, PCA-prefix embedding, per-block detrend, target building,
embedding-source resolution) plus the shared decoding figures.

Compute and plots live together here per project preference (2026-06-04);
the blocked-CV scorer itself stays in f_cebra_helpers.f_blocked_cv_r2.

Dependencies (one-way): f_feature_helpers (detrend), f_analysis (circshift,
lazy-imported so this module loads even without the RNN_scripts path set).

Created 2026-06-04 during the VR_ca_cebra.py modularization pass (group 1:
low-risk compute utils).
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score

from f_feature_helpers import f_detrend_col, f_resolve_detrend_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — low-risk compute utils
# ─────────────────────────────────────────────────────────────────────────────

def f_imaging_fs(frame_t, default=60.0):
    """Imaging sample rate (Hz) = 1 / median(diff(frame_t)).

    Returns `default` if frame_t is unusable (too short / non-numeric).
    Replaces the try/except median-diff snippet repeated across the decoding
    cells (the bar, sweep, and decoding-vs-n_pcs cells each had their own copy).
    """
    try:
        return 1.0 / float(np.median(np.diff(np.asarray(frame_t).ravel())))
    except Exception:
        return default


def make_decoder(name, ridge_alpha=1.0, k_nn=15, ridgecv_alphas=None,
                 task='regression'):
    """Build a decoder estimator by name — single source of truth.

    Mirrors the analog decoder with its classification twin when
    task='classification' (ridge→LogisticRegression with C=1/α, ridgecv→
    LogisticRegressionCV, knn→KNeighborsClassifier). Used by f_blocked_cv_r2
    and by the diagnostic cells that fit folds directly, so the choice of
    estimator can't drift between them.

    Parameters
    ----------
    name : {'ridge', 'ridgecv', 'knn'}
    ridge_alpha : float       Ridge penalty (α). For classification, C = 1/α.
    k_nn : int                neighbors for the knn decoder.
    ridgecv_alphas : array    α grid for ridgecv; None → logspace(-2, 4, 13).
    task : {'regression', 'classification'}

    Returns
    -------
    An unfitted sklearn estimator.
    """
    if ridgecv_alphas is None:
        ridgecv_alphas = np.logspace(-2, 4, 13)
    if task == 'classification':
        if name == 'ridge':
            return LogisticRegression(C=1.0 / ridge_alpha, max_iter=1000)
        if name == 'ridgecv':
            return LogisticRegressionCV(Cs=np.sort(1.0 / ridgecv_alphas),
                                        max_iter=1000)
        if name == 'knn':
            return KNeighborsClassifier(n_neighbors=k_nn, weights='distance')
        raise ValueError(f'unknown decoder: {name!r}')
    if name == 'ridge':
        return Ridge(alpha=ridge_alpha)
    if name == 'ridgecv':
        return RidgeCV(alphas=ridgecv_alphas)
    if name == 'knn':
        return KNeighborsRegressor(n_neighbors=k_nn, weights='distance')
    raise ValueError(f'unknown decoder: {name!r}')


def f_shuffle_neural(Y, rng, min_shift=0):
    """Per-cell circular shift of neural data — the standard decoding null.

    Breaks cross-cell timing (and any neural↔behavior alignment) while keeping
    each cell's rate, autocorrelation, and burstiness. So whatever a decoder
    extracts from the shuffled data is the "no real co-firing structure"
    baseline.

    Parameters
    ----------
    Y : ndarray (T, N)        neural, frames × cells (the script's Y_neu layout).
    rng : np.random.Generator seeded generator so shuffles are reproducible.
    min_shift : int           minimum |shift| in frames. 0 = uniform null (no
                              cell stays near-unshifted (the old global-RNG
                              randint(0,T) could leave cells effectively aligned).

    Returns
    -------
    ndarray (T, N) : shuffled neural. f_circshift_rates works on (cells, T),
    so this transposes in and out. Pair with f_pca_prefix for the embedding.
    """
    from f_analysis import f_circshift_rates   # lazy: needs RNN_scripts path
    return f_circshift_rates(Y.T, min_shift=min_shift, rng=rng).T


def f_insample_predict(emb, y, decoder='ridge', ridge_alpha=1.0, k_nn=15,
                       ridgecv_alphas=None, standardize=True, task='regression'):
    """Fit the decoder on ALL non-NaN rows and predict in-sample (no CV).

    The "reconstruction ceiling" — what the linear (or KNN) map can capture with
    no held-out generalization. Compare its R²/AUC and trace to the out-of-fold
    (test) result: a small in-sample score means the feature is only weakly /
    non-linearly in the embedding at all; a large in-sample-minus-OOF gap means
    it fits but doesn't generalize (drift / #8).

    Parameters mirror f_blocked_cv_r2's decoder knobs. `task='auto'` detects a
    0/1 target → classification (ROC-AUC) else regression (R²).

    Returns
    -------
    dict with 'idx' (valid-row indices), 'y_true', 'y_pred' (on valid rows),
    'r2' (in-sample R² or AUC), 'metric'.
    """
    y = np.asarray(y)
    valid = ~np.isnan(y)
    yv = y[valid]
    if task == 'auto':
        u = np.unique(yv)
        task = ('classification' if (u.size <= 2 and np.all(np.isin(u, (0.0, 1.0))))
                else 'regression')
    is_clf = (task == 'classification')
    X = emb[valid]
    if standardize:
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
    reg = make_decoder(decoder, ridge_alpha=ridge_alpha, k_nn=k_nn,
                       ridgecv_alphas=ridgecv_alphas,
                       task='classification' if is_clf else 'regression')
    reg.fit(X, yv)
    if is_clf:
        pos = int(np.argmax(reg.classes_))
        pred = reg.predict_proba(X)[:, pos]
        score = roc_auc_score(yv, pred) if np.unique(yv).size >= 2 else np.nan
        metric = 'roc_auc'
    else:
        pred = reg.predict(X)
        score = r2_score(yv, pred)
        metric = 'r2'
    return {'idx': np.where(valid)[0], 'y_true': yv, 'y_pred': pred,
            'r2': float(score), 'metric': metric}


def f_pca_prefix(Y, n_pcs, random_state=42):
    """Top-`n_pcs` PCA scores of Y (T, N) — fit_transform at exactly n_pcs.

    The 'fit PCA, use the leading components as the embedding' idiom repeated
    across the sweep / decoding / recon cells. For a sweep over n_pcs, fit once
    at the max and slice prefixes; for a single k, call this directly.
    """
    return PCA(n_components=n_pcs, random_state=random_state).fit_transform(Y)


def f_apply_block_detrend(X, block_name, detrend_blocks, detrend_sigma_sec, fs):
    """High-pass every column of one feature block per the detrend config.

    Resolves the σ for `block_name` via f_resolve_detrend_sigma (block-selective:
    a dict {block: σ_sec}, a list of blocks at the global σ, or None = global),
    then NaN-aware high-passes each column with f_detrend_col. Returns X
    unchanged when this block isn't detrended. Consolidates the per-block
    detrend loop duplicated in the bar / sweep / decoding cells.

    Parameters
    ----------
    X : ndarray (T, d)        one block's feature columns.
    block_name : str          registry block name ('agg', 'pix', ...).
    detrend_blocks : None | dict | list   block-selective config.
    detrend_sigma_sec : float | None      global σ in seconds.
    fs : float                imaging sample rate (Hz).

    Returns
    -------
    ndarray (T, d) : detrended (or the original X if σ resolves to None).
    """
    sig_fr = f_resolve_detrend_sigma(block_name, detrend_blocks,
                                     detrend_sigma_sec, fs)
    if not sig_fr:
        return X
    return np.column_stack([f_detrend_col(X[:, c], sig_fr)
                            for c in range(X.shape[1])])


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — target building from the feature registry
# ─────────────────────────────────────────────────────────────────────────────

def f_is_binary_col(y):
    """True if y's non-NaN support is ⊆ {0, 1} (≤2 distinct values).

    Used to route presence-style channels to ROC-AUC scoring instead of R²
    (R² is misleading for a base-rate-skewed binary target).
    """
    u = np.unique(y[~np.isnan(y)])
    return u.size <= 2 and np.all(np.isin(u, (0.0, 1.0)))


def build_target_columns(built_blocks, block_names, detrend_blocks=None,
                         detrend_sigma_sec=None, fs=None, split_binary=False):
    """Per-column decoding targets from the registry.

    One entry per (block, column): (grp, name, column_vector). The shape the
    per-feature sweep, the detrend grid, and the per-fold diagnostics all want.

    Parameters
    ----------
    built_blocks : dict          feature registry.
    block_names : list of str    which blocks to pull columns from.
    detrend_blocks, detrend_sigma_sec, fs
        Block-selective high-pass config. If fs is None, no detrend is applied
        (raw columns) — what the detrend grid wants (it high-passes per σ later).
    split_binary : bool
        If True, return (analog, binary) where binary holds the 0/1 columns
        (presence) and analog the rest. Binary columns are NEVER detrended
        (high-passing a 0/1 signal is meaningless). If False, return a single
        list with every column (binary included, detrended like the rest).

    Returns
    -------
    list of (grp, name, y)             if split_binary is False
    (analog_list, binary_list)         if split_binary is True
    """
    analog, binary = [], []
    for grp in block_names:
        if grp not in built_blocks:
            raise ValueError(f'block {grp!r} not built; available: '
                             f'{list(built_blocks)}')
        b = built_blocks[grp]
        sig_fr = (f_resolve_detrend_sigma(grp, detrend_blocks, detrend_sigma_sec, fs)
                  if fs is not None else None)
        for i, name in enumerate(b['names']):
            y = b['X'][:, i]
            if np.unique(y[~np.isnan(y)]).size < 2:
                # Constant / all-NaN column — decodes to silent NaN (single-class
                # AUC or ss_tot=0 R²). Skip with a warning instead.
                print(f'  [build_target_columns] skip {grp}:{name} — degenerate '
                      '(<2 distinct valid values)')
                continue
            if split_binary and f_is_binary_col(y):
                binary.append((grp, name, y))          # AUC path, never detrended
                continue
            if sig_fr:
                y = f_detrend_col(y, sig_fr)
            analog.append((grp, name, y))
    return (analog, binary) if split_binary else analog


def build_target_matrix(built_blocks, block_names, detrend_blocks=None,
                        detrend_sigma_sec=None, fs=None, skip_presence=False):
    """Concatenated multi-output target matrix + names from the registry.

    The shape the multi-output decoders want (decoding-vs-n_pcs, real-vs-
    reconstructed). Columns are concatenated across blocks in order; each block
    is block-selectively high-passed (when fs is given) before concatenation.

    Parameters
    ----------
    built_blocks : dict
    block_names : list of str
    detrend_blocks, detrend_sigma_sec, fs
        Block-selective high-pass config. fs=None → no detrend.
    skip_presence : bool
        Drop 'pres_*' columns (binary; awkward under a regression R² metric).

    Returns
    -------
    X : ndarray (T, sum_d)   (always a fresh array — np.concatenate copies)
    names : list of str
    """
    Xs, names = [], []
    for grp in block_names:
        if grp not in built_blocks:
            raise ValueError(f'block {grp!r} not built; available: '
                             f'{list(built_blocks)}')
        Xb = built_blocks[grp]['X']
        if fs is not None:
            Xb = f_apply_block_detrend(Xb, grp, detrend_blocks,
                                       detrend_sigma_sec, fs)
        Xs.append(Xb)
        names.extend(built_blocks[grp]['names'])
    X = np.concatenate(Xs, axis=1)
    if skip_presence:
        keep = [i for i, nm in enumerate(names) if not nm.startswith('pres')]
        if not keep:
            raise ValueError('skip_presence removed all target columns')
        X = X[:, keep]
        names = [names[i] for i in keep]
    return X, names


_LEGACY_TARGET_BLOCKS = {
    'X_agg':       ['agg'],
    'X_pix':       ['pix'],
    'X_mot':       ['motion'],
    'X_agg+X_mot': ['agg', 'motion'],
    'both':        ['agg', 'pix'],
}


def legacy_target_blocks(choice):
    """Translate a legacy target_choice string to a list of block names.

    Single source for the two duplicated _legacy_map dicts (decoding +
    real-vs-reconstructed cells).
    """
    if choice not in _LEGACY_TARGET_BLOCKS:
        raise ValueError(f'legacy target choice {choice!r} not recognized; '
                         f'available: {list(_LEGACY_TARGET_BLOCKS)}')
    return list(_LEGACY_TARGET_BLOCKS[choice])


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — shared decoding figures
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_feature_heatmap(mat, xticklabels, row_labels, group_sizes=None,
                           title='', cbar_label='', vmin=-0.5, vmax=0.5,
                           xlabel='# neural PCs in embedding'):
    """Feature × x heatmap with optional horizontal group dividers.

    The shared layout behind the per-feature R² heatmap (sweep) and the
    real−null Δ-heatmap (shuffle cell): RdBu_r, clipped, row per target,
    column per x value, black lines between feature-block groups.

    Parameters
    ----------
    mat : ndarray (n_rows, n_cols)
    xticklabels : sequence            column tick labels (e.g. n_pcs_sweep).
    row_labels : sequence of str      one per row (e.g. 'agg:lat_R').
    group_sizes : sequence of int     # rows per group, for dividers (optional).
    title, cbar_label, xlabel : str
    vmin, vmax : float                colour limits.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, max(4, 0.18*len(row_labels))))
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   interpolation='none')
    ax.set_xticks(range(len(xticklabels))); ax.set_xticklabels(xticklabels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel(xlabel)
    if group_sizes:
        cum = 0
        for g in list(group_sizes)[:-1]:
            cum += g
            ax.axhline(cum - 0.5, color='k', lw=0.5)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label=cbar_label)
    fig.tight_layout()


def f_plot_pred_scatter(y_mat, pred_mat, names, suptitle='', max_panels=8,
                        ncols=4, point_size=2, alpha=0.25):
    """Grid of real-vs-predicted scatter panels (one per target column).

    Each panel shows the scatter, a y=x identity line, and the per-feature R²
    in its title. NaN-aware (frames where real or pred is NaN are dropped).

    Parameters
    ----------
    y_mat, pred_mat : ndarray (T, p)   real and predicted targets.
    names : list of str                column names (len ≥ p plotted columns).
    suptitle : str                     figure suptitle (skipped if empty).
    max_panels : int                   cap on # columns plotted.
    ncols, point_size, alpha           layout / scatter style.
    """
    n_plot = min(max_panels, y_mat.shape[1])
    ncols = min(ncols, n_plot)
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.atleast_2d(axes).flatten()
    for ci in range(n_plot):
        ax = axes[ci]
        y = y_mat[:, ci]; p = pred_mat[:, ci]
        m = ~(np.isnan(y) | np.isnan(p))
        if m.sum() < 2:
            ax.set_title(f'{names[ci]}  (no valid)', fontsize=8); ax.axis('off')
            continue
        ax.scatter(y[m], p[m], s=point_size, alpha=alpha)
        lo = float(min(y[m].min(), p[m].min()))
        hi = float(max(y[m].max(), p[m].max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
        ss_res = float(np.sum((y[m] - p[m])**2))
        ss_tot = float(np.sum((y[m] - y[m].mean())**2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        ax.set_xlabel('real', fontsize=8); ax.set_ylabel('predicted', fontsize=8)
        ax.set_title(f'{names[ci]}  R²={r2:+.3f}', fontsize=8)
    for ax in axes[n_plot:]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()


def f_plot_pred_traces(y_mat, pred_mat, names, t, suptitle='', max_panels=8):
    """Stacked real (blue) vs predicted (red) temporal traces.

    One row per target column on a shared time axis.

    Parameters
    ----------
    y_mat, pred_mat : ndarray (T, p)   real and predicted targets.
    names : list of str                column names.
    t : ndarray (T,)                   time axis (imaging seconds).
    suptitle : str                     figure suptitle (skipped if empty).
    max_panels : int                   cap on # columns plotted.
    """
    n_plot = min(max_panels, y_mat.shape[1])
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 1.6*n_plot), sharex=True)
    axes = np.atleast_1d(axes)
    for ci in range(n_plot):
        ax = axes[ci]
        ax.plot(t, y_mat[:, ci],   color='C0', lw=0.8, alpha=0.85, label='real')
        ax.plot(t, pred_mat[:, ci], color='C3', lw=0.8, alpha=0.85, label='predicted')
        ax.set_ylabel(names[ci], fontsize=8)
        if ci == 0:
            ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('imaging time (s)')
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()


def f_plot_decode_bars(feat_names, emb_names, r2_table, r2_std, title='',
                       ylabel='5-fold blocked-CV R²'):
    """Grouped bar plot of decoding R² — feature groups × embeddings.

    Parameters
    ----------
    feat_names : list of str          x-axis groups (target feature blocks).
    emb_names : list of str           bars within each group (embeddings).
    r2_table, r2_std : ndarray (n_feats, n_embs)   mean R² and its across-fold std.
    title, ylabel : str
    """
    fig, ax = plt.subplots(1, 1, figsize=(max(6, 1.6*len(feat_names)+3), 3.5))
    x = np.arange(len(feat_names))
    n_embs = len(emb_names)
    w = 0.8 / n_embs    # leave 20% gap between groups
    offsets = (np.arange(n_embs) - (n_embs - 1) / 2.0) * w
    for j, ename in enumerate(emb_names):
        ax.bar(x + offsets[j], r2_table[:, j], yerr=r2_std[:, j], width=w, label=ename)
    ax.set_xticks(x); ax.set_xticklabels(feat_names)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()


def f_plot_decode_sweep_summary(n_pcs_sweep, r2_mat, group_rows,
                                groups_present, targets_bin, auc_mat,
                                dset_tag='', decoder='ridge'):
    """2-axis decoding-performance-vs-n_pcs summary.

    Left  — analog R² per group: mean (o-), median (s--), top-3 (^:), with a
            25–75th pct band across the group's features.
    Right — binary ROC-AUC per feature (chance 0.5). Only drawn when targets_bin
            is non-empty (else a single-axis figure).

    Parameters
    ----------
    n_pcs_sweep : sequence            x values.
    r2_mat : ndarray (n_feat, n_pcs)  per-(analog-feature) R².
    group_rows : dict                 group name → row indices into r2_mat.
    groups_present : list of str      group order.
    targets_bin : list                binary (grp, name, y) entries (may be []).
    auc_mat : ndarray (n_bin, n_pcs)  per-binary-feature AUC.
    dset_tag, decoder : str           title metadata.
    """
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ncols = 2 if targets_bin else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 4.5), squeeze=False)
    ax = axes[0, 0]
    for gi, grp in enumerate(groups_present):
        group_r2 = r2_mat[group_rows[grp]]
        color = palette[gi % len(palette)]
        mean_   = group_r2.mean(axis=0)
        median_ = np.median(group_r2, axis=0)
        top3 = np.sort(group_r2, axis=0)[-3:, :].mean(axis=0)
        q25 = np.percentile(group_r2, 25, axis=0)
        q75 = np.percentile(group_r2, 75, axis=0)
        ax.fill_between(n_pcs_sweep, q25, q75, color=color, alpha=0.12)
        ax.plot(n_pcs_sweep, mean_,   'o-',  color=color, label=f'{grp} — mean',   lw=1.8)
        ax.plot(n_pcs_sweep, median_, 's--', color=color, label=f'{grp} — median', lw=1.2, alpha=0.8)
        ax.plot(n_pcs_sweep, top3,    '^:',  color=color, label=f'{grp} — top-3 mean', lw=1.2, alpha=0.8)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('# neural PCs in embedding')
    ax.set_ylabel('blocked-CV R²')
    ax.set_title(f'analog targets — R²  ({decoder}; shaded = 25-75th pct)')
    ax.legend(fontsize=8)
    if targets_bin:
        axb = axes[0, 1]
        for fi, (grp, name, _) in enumerate(targets_bin):
            axb.plot(n_pcs_sweep, auc_mat[fi], 'o-', label=f'{grp}:{name}', alpha=0.85)
        axb.axhline(0.5, color='gray', ls=':', alpha=0.7, label='chance (0.5)')
        axb.set_ylim(0.4, 1.0)
        axb.set_xlabel('# neural PCs in embedding')
        axb.set_ylabel('blocked-CV ROC-AUC')
        axb.set_title(f'binary targets — ROC-AUC  (PCA → {decoder} classifier)')
        axb.legend(fontsize=7, ncol=2)
    fig.suptitle(f'{dset_tag}\nembedding decoding performance vs n_neural_PCs',
                 fontsize=10)
    fig.tight_layout()


def f_plot_decode_real_vs_null(n_pcs_sweep, r2_mat, r2_mat_null, group_rows,
                               groups_present, targets_bin, auc_mat, auc_mat_null,
                               dset_tag='', decoder='ridge', shuffle_mode='',
                               n_shuffles=0):
    """2-axis real-vs-shuffled-null overlay.

    Left  — analog R² group means: real (o-), null mean (:), shaded null
            2.5–97.5th pct across shuffles.
    Right — binary ROC-AUC per feature: real vs null band (chance 0.5). Only
            drawn when targets_bin is non-empty.

    Parameters mirror f_plot_decode_sweep_summary plus the *_null arrays:
    r2_mat_null : (n_shuf, n_feat, n_pcs) ; auc_mat_null : (n_shuf, n_bin, n_pcs).
    """
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ncols = 2 if targets_bin else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.5*ncols, 4.5), squeeze=False)
    ax = axes[0, 0]
    for gi, grp in enumerate(groups_present):
        rows    = group_rows[grp]
        real_r2 = r2_mat[rows]
        null_r2 = r2_mat_null[:, rows, :]
        color   = palette[gi % len(palette)]
        label   = f'{grp} targets'
        real_mean          = real_r2.mean(axis=0)
        null_mean_per_shuf = null_r2.mean(axis=1)        # (n_shuf, n_pcs)
        null_mean          = null_mean_per_shuf.mean(axis=0)
        null_lo            = np.percentile(null_mean_per_shuf,  2.5, axis=0)
        null_hi            = np.percentile(null_mean_per_shuf, 97.5, axis=0)
        ax.fill_between(n_pcs_sweep, null_lo, null_hi, color=color, alpha=0.18,
                        label=f'{label} — null 95% band')
        ax.plot(n_pcs_sweep, null_mean, ':',  color=color, lw=1.5, alpha=0.9,
                label=f'{label} — null mean')
        ax.plot(n_pcs_sweep, real_mean, 'o-', color=color, lw=2.0,
                label=f'{label} — real mean')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('# neural PCs in embedding')
    ax.set_ylabel('blocked-CV R²')
    ax.set_title(f'analog targets — R² real vs null  ({shuffle_mode}, n={n_shuffles})')
    ax.legend(fontsize=7, ncol=2)
    if targets_bin:
        axb = axes[0, 1]
        for fi, (grp, name, _) in enumerate(targets_bin):
            color    = palette[fi % len(palette)]
            real_auc = auc_mat[fi]
            null_auc = auc_mat_null[:, fi, :]            # (n_shuf, n_pcs)
            null_lo  = np.nanpercentile(null_auc,  2.5, axis=0)
            null_hi  = np.nanpercentile(null_auc, 97.5, axis=0)
            axb.fill_between(n_pcs_sweep, null_lo, null_hi, color=color, alpha=0.18)
            axb.plot(n_pcs_sweep, np.nanmean(null_auc, axis=0), ':', color=color,
                     lw=1.5, alpha=0.9, label=f'{grp}:{name} — null')
            axb.plot(n_pcs_sweep, real_auc, 'o-', color=color, lw=2.0,
                     label=f'{grp}:{name} — real')
        axb.axhline(0.5, color='gray', ls='--', alpha=0.6, label='chance (0.5)')
        axb.set_ylim(0.35, 1.0)
        axb.set_xlabel('# neural PCs in embedding')
        axb.set_ylabel('blocked-CV ROC-AUC')
        axb.set_title(f'binary targets — AUC real vs null  ({decoder})')
        axb.legend(fontsize=7, ncol=2)
    fig.suptitle(f'{dset_tag}\nreal vs shuffled control  ({shuffle_mode}, '
                 f'n_shuf={n_shuffles}, {decoder})', fontsize=10)
    fig.tight_layout()


def f_plot_sweep_lines(n_pcs_sweep, r2_mat, targets_all, group_rows,
                       groups_present, dset_tag=''):
    """Per-feature R²-vs-n_pcs line plots, one panel per feature group.

    Parameters
    ----------
    n_pcs_sweep : sequence            x values.
    r2_mat : ndarray (n_feat, n_pcs)  per-feature R².
    targets_all : list of (grp, name, y)   feature metadata (name used for labels).
    group_rows : dict                 group → row indices into r2_mat / targets_all.
    groups_present : list of str      panel order.
    """
    n_groups = len(groups_present)
    fig, axes = plt.subplots(1, n_groups, figsize=(5*max(n_groups, 1), 4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for gi, grp in enumerate(groups_present):
        for ri in group_rows[grp]:
            axes[gi].plot(n_pcs_sweep, r2_mat[ri], 'o-',
                          label=targets_all[ri][1], alpha=0.85)
        axes[gi].set_title(f'{grp} features')
        axes[gi].set_xlabel('# neural PCs in embedding')
        axes[gi].axhline(0, color='gray', ls=':', alpha=0.5)
        axes[gi].legend(fontsize=7, ncol=2)
    axes[0].set_ylabel('blocked-CV R²')
    fig.suptitle(dset_tag, fontsize=10)
    fig.tight_layout()


def f_plot_perfeature_null_grid(n_pcs_sweep, r2_mat, r2_mat_null, group_rows,
                                groups_present, targets_all, dset_tag='',
                                decoder='ridge', shuffle_mode='', n_shuffles=0):
    """Per-feature real-vs-null 2×N grid (one column per feature group).

    Top row    — raw R² per feature, with the pooled null 95% band (across all
                 shuffles × features in the group) and null median.
    Bottom row — Δ R² (real − per-feature null mean); zero = chance.

    Parameters
    ----------
    n_pcs_sweep : sequence
    r2_mat : ndarray (n_feat, n_pcs)
    r2_mat_null : ndarray (n_shuf, n_feat, n_pcs)
    group_rows : dict                 group → row indices.
    groups_present : list of str      column order.
    targets_all : list of (grp, name, y)   for feature labels.
    dset_tag, decoder, shuffle_mode, n_shuffles : title metadata.
    """
    n_groups = len(groups_present)
    fig, axes = plt.subplots(2, n_groups, figsize=(6*max(n_groups, 1), 7.5),
                             sharex='col', squeeze=False)
    for col, grp in enumerate(groups_present):
        rows    = group_rows[grp]
        real_r2 = r2_mat[rows]
        null_r2 = r2_mat_null[:, rows, :]
        names   = [targets_all[r][1] for r in rows]

        # pooled null band per n_pcs (all shuffles × features in the group)
        null_pool = null_r2.reshape(-1, null_r2.shape[-1])
        null_lo  = np.percentile(null_pool,  2.5, axis=0)
        null_hi  = np.percentile(null_pool, 97.5, axis=0)
        null_med = np.median(null_pool, axis=0)

        ax_top = axes[0, col]
        ax_top.fill_between(n_pcs_sweep, null_lo, null_hi, color='gray', alpha=0.25,
                            label='null 95% band')
        ax_top.plot(n_pcs_sweep, null_med, '--', color='gray', alpha=0.8,
                    label='null median')
        for fi, name in enumerate(names):
            ax_top.plot(n_pcs_sweep, real_r2[fi], 'o-', alpha=0.8, label=name)
        ax_top.axhline(0, color='gray', ls=':', alpha=0.4)
        ax_top.set_ylabel('blocked-CV R²')
        ax_top.set_title(f'{grp} features')
        ax_top.legend(fontsize=6, ncol=2, loc='best')

        # Δ R² per feature against the per-feature null mean
        delta_grp = real_r2 - null_r2.mean(axis=0)
        ax_bot = axes[1, col]
        ax_bot.axhline(0, color='gray', ls='--', alpha=0.6, label='chance (real = null)')
        for fi, name in enumerate(names):
            ax_bot.plot(n_pcs_sweep, delta_grp[fi], 'o-', alpha=0.8, label=name)
        ax_bot.set_xlabel('# neural PCs in embedding')
        ax_bot.set_ylabel('Δ R²  (real − null mean)')
        ax_bot.set_title('signal above chance')
        ax_bot.legend(fontsize=6, ncol=2, loc='best')

    fig.suptitle(f'{dset_tag}\nper-feature decoding vs shuffled control  '
                 f'({shuffle_mode}, n_shuf={n_shuffles}, {decoder})', fontsize=10)
    fig.tight_layout()


def f_plot_oof_trace_scatter(emb, targets, t, dset_tag='', emb_tag='',
                             block_label='', n_folds=5, embargo=0,
                             standardize=True, decoder='ridge', ridge_alpha=1.0,
                             k_nn=15):
    """Per-feature OOF trace (left) + predicted-vs-actual scatter (right).

    Computes out-of-fold predictions for each target column via
    f_blocked_cv_r2(return_predictions=True) (lazy-imported to avoid a circular
    import), then draws a small-multiples grid — one row per target:
      left  — real (line) vs OOF prediction (dots) over time, fold boundaries.
      right — pred-vs-actual scatter colored by held-out fold, y=x line + global
              OLS fit (slope in the panel title).

    The OLS slope disambiguates negative R²: ~0 → predicts train mean (blocked-CV
    pessimism); tilted/negative → drift-misled (#8); ~1 → genuine decoding.

    Parameters
    ----------
    emb : ndarray (T, d)              decoder input embedding.
    targets : list of (name, y)       prepared target columns (detrend already
                                      applied by the caller, to match the sweep).
    t : ndarray (T,)                  time axis (imaging seconds).
    dset_tag, emb_tag, block_label : str   title metadata.
    n_folds, embargo, standardize, decoder, ridge_alpha, k_nn
        CV knobs forwarded to f_blocked_cv_r2.
    """
    from f_cebra_helpers import f_blocked_cv_r2     # lazy: avoids import cycle
    from scipy.stats import spearmanr
    n_rows = len(targets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.4*n_rows),
                             gridspec_kw={'width_ratios': [2.4, 1]}, squeeze=False)
    for r, (name, y) in enumerate(targets):
        res = f_blocked_cv_r2(emb, y, n_folds=n_folds, embargo=embargo,
                              standardize=standardize, decoder=decoder,
                              ridge_alpha=ridge_alpha, k_nn=k_nn,
                              return_predictions=True)
        idx, yt, yh, fld = res['idx'], res['y_true'], res['y_pred'], res['fold']
        rho = spearmanr(yt, yh).correlation if idx.size > 5 else np.nan
        T_ = res['T']

        axL = axes[r, 0]
        axL.plot(t, y, 'C0', lw=0.6, label='real')
        if idx.size:
            axL.plot(t[idx], yh, 'C3.', ms=2, alpha=0.6, label='OOF pred')
        for f in range(1, n_folds):
            axL.axvline(t[min(f*(T_//n_folds), T_-1)], color='gray', ls=':', alpha=0.4)
        axL.set_ylabel(name, fontsize=8)
        axL.set_title(f'{block_label}:{name}   R²={res["r2"]:+.3f}   ρ={rho:+.2f}',
                      fontsize=9)
        if r == 0:
            axL.legend(fontsize=7, loc='upper right')
        if r == n_rows - 1:
            axL.set_xlabel('imaging time (s)')

        axR = axes[r, 1]
        if idx.size:
            axR.scatter(yt, yh, c=fld, cmap='viridis', s=6, alpha=0.5)
            lo = float(min(yt.min(), yh.min())); hi = float(max(yt.max(), yh.max()))
            axR.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.7)
            if np.std(yt) > 1e-9:
                m_, b_ = np.polyfit(yt, yh, 1)
                xs = np.array([lo, hi])
                axR.plot(xs, m_*xs + b_, 'r-', lw=0.8, alpha=0.7)
                axR.set_title(f'slope={m_:+.2f}', fontsize=8)
        axR.set_xlabel('actual', fontsize=8); axR.set_ylabel('pred', fontsize=8)
        axR.tick_params(labelsize=7)

    fig.suptitle(f'{dset_tag}\nOOF decode diagnostic — {emb_tag} → {block_label} '
                 f'({decoder}, embargo={embargo}fr, std={standardize}, '
                 f'fold color = viridis)', fontsize=10)
    fig.tight_layout()


def f_plot_real_vs_shuffle_line(x, real, shuf_runs, xlabel='# neural PCs',
                                ylabel='', title='', hline=0.0, legend_loc='best'):
    """Real vs per-cell-shuffle line plot with a shuffle min–max band.

    Shared by the unsupervised PCA-reconstruction-R² and the supervised
    decoding-R²-vs-n_pcs cells.

    Parameters
    ----------
    x : sequence                      x axis (# neural PCs).
    real : sequence                   real-data curve (C0 line).
    shuf_runs : ndarray (n_shuf, len(x))
        per-shuffle curves; band = min..max across shuffles, C1 line = mean.
    xlabel, ylabel, title : str
    hline : float | None              horizontal reference line (None = none).
    legend_loc : str
    """
    shuf_runs = np.asarray(shuf_runs)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.fill_between(x, shuf_runs.min(0), shuf_runs.max(0), color='C1', alpha=0.2,
                    label=f'shuffle range (n={shuf_runs.shape[0]})')
    ax.plot(x, shuf_runs.mean(0), 'o-', color='C1', lw=1.5, alpha=0.85,
            label='shuffled — mean (per-cell circshift)')
    ax.plot(x, real, 'o-', color='C0', lw=2.0, label='real')
    if hline is not None:
        ax.axhline(hline, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9, loc=legend_loc)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def f_plot_input_raster(S_sorted, t, agg_X, pix_X, side_tags, n_per_side,
                        dist_name='dist', n_pix_plot=3, dset_tag='', side='',
                        agg_mode=''):
    """5-panel decoder-input diagnostic on a shared imaging-time axis.

    Panels: (1) neural raster, (2) presence, (3) lateral angle, (4) distance
    channel, (5) top pix-PCs. Lets you eyeball behavior↔neural alignment and
    slow nonstationarity in the top pix-PCs (#9). Per-side agg channel layout:
    0=pres, 1=lat, 2=vert, 3=dist within each n_per_side block.

    Parameters
    ----------
    S_sorted : ndarray (n_cells, T)   hclust-sorted neural raster.
    t : ndarray (T,)                  imaging time axis (s).
    agg_X, pix_X : ndarray (T, ·)     agg + pix feature blocks.
    side_tags : list of str           monitor side tags (e.g. ['R'] or ['L','R']).
    n_per_side : int                  agg channels per side.
    dist_name : str                   'dist' or 'ang_size' (panel 4 label).
    n_pix_plot : int                  # pix-PCs in panel 5.
    dset_tag, side, agg_mode : title metadata.
    """
    colors = ['C0', 'C1']
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(12, 9),
                           gridspec_kw={'height_ratios': [6, 1, 1, 1, 2]})
    ax[0].imshow(S_sorted, aspect='auto', vmin=0, vmax=0.5, interpolation='none',
                 extent=[float(t[0]), float(t[-1]), S_sorted.shape[0], 0])
    ax[0].set_ylabel('cells (hclust)')
    ax[0].set_title(f'{dset_tag}: raster + decoder inputs  '
                    f'(side={side!r}, agg_mode={agg_mode!r})')
    for s, tag in enumerate(side_tags):
        base = s * n_per_side
        ax[1].plot(t, agg_X[:, base + 0], color=colors[s % len(colors)],
                   label=f'pres_{tag}', lw=0.8)
    ax[1].set_ylabel('presence'); ax[1].set_ylim(-0.1, 1.2)
    ax[1].legend(fontsize=7, loc='upper right')
    for s, tag in enumerate(side_tags):
        base = s * n_per_side
        ax[2].plot(t, agg_X[:, base + 1], color=colors[s % len(colors)],
                   label=f'lat_{tag}', lw=0.8)
    ax[2].set_ylabel('lat angle (rad)'); ax[2].legend(fontsize=7, loc='upper right')
    for s, tag in enumerate(side_tags):
        base = s * n_per_side
        ax[3].plot(t, agg_X[:, base + 3], color=colors[s % len(colors)],
                   label=f'{dist_name}_{tag}', lw=0.8)
    ax[3].set_ylabel(dist_name); ax[3].legend(fontsize=7, loc='upper right')
    for i in range(n_pix_plot):
        ax[4].plot(t, pix_X[:, i], label=f'pix_PC{i+1}', alpha=0.85, lw=0.8)
    ax[4].axhline(0, color='gray', ls=':', alpha=0.4)
    ax[4].set_ylabel('pix-PCs'); ax[4].set_xlabel('imaging time (s)')
    ax[4].legend(fontsize=7, loc='upper right')
    fig.tight_layout()


def f_plot_block_traces(X, names, t, S_sorted=None, title='', max_lines=14,
                        zscore=True, heatmap=None, cmap='viridis'):
    """Generic per-channel viewer for ANY feature block on a shared time axis.

    Block-agnostic companion to f_plot_input_raster (which is hardwired to
    agg+pix): pass any built_blocks entry's X / names and see its channels over
    imaging time the same way — agg, pix, grid, flow, motion, self_mot, pix_mot,
    beh. Few channels render as line traces; many render as a channel×time
    heatmap. An optional neural raster is stacked on top with a shared x-axis so
    behavior↔neural alignment and slow nonstationarity stay easy to eyeball.

    Parameters
    ----------
    X : ndarray (T, d)              feature values (built_blocks[name]['X']).
    names : list[str]               d channel names (built_blocks[name]['names']).
    t : ndarray (T,)                imaging-time axis (s).
    S_sorted : ndarray (n_cells, T) | None
                                    optional neural raster for a top panel.
    title : str                     figure title.
    max_lines : int                 d <= max_lines → line traces, else heatmap.
    zscore : bool                   per-channel NaN-aware z-score before display
                                    so unlike scales share an axis (off → native).
    heatmap : bool | None           force heatmap (True) / lines (False); None →
                                    auto by max_lines.
    cmap : str                      heatmap colormap.
    """
    X = np.asarray(X, float)
    t = np.asarray(t, float).ravel()
    T, d = X.shape
    names = list(names)
    if len(names) != d:
        names = [f'ch{i}' for i in range(d)]          # fall back if mismatched

    disp = X.copy()
    if zscore:
        mu = np.nanmean(disp, axis=0)
        sd = np.nanstd(disp, axis=0)
        sd[sd == 0] = 1.0
        disp = (disp - mu) / sd
    units = 'z-scored' if zscore else 'native'

    use_heatmap = (d > max_lines) if heatmap is None else heatmap

    n_panels = 1 + (S_sorted is not None)
    ratios = ([4, 5] if use_heatmap else [4, 3])[:n_panels] if n_panels == 2 else [1]
    fig, axes = plt.subplots(n_panels, 1, sharex=True,
                             figsize=(12, 3 + 2.2 * n_panels),
                             gridspec_kw={'height_ratios': ratios},
                             squeeze=False)
    axes = axes[:, 0]
    ai = 0
    if S_sorted is not None:
        axes[ai].imshow(S_sorted, aspect='auto', vmin=0, vmax=0.5,
                        interpolation='none',
                        extent=[float(t[0]), float(t[-1]), S_sorted.shape[0], 0])
        axes[ai].set_ylabel('cells (hclust)')
        ai += 1

    axf = axes[ai]
    if use_heatmap:
        vlim = float(np.nanpercentile(np.abs(disp), 99)) or 1.0
        im = axf.imshow(disp.T, aspect='auto', interpolation='none', cmap=cmap,
                        vmin=-vlim if zscore else None, vmax=vlim if zscore else None,
                        extent=[float(t[0]), float(t[-1]), d - 0.5, -0.5])
        # label every channel if few enough rows, else a thinned subset
        step = max(1, d // 30)
        ticks = list(range(0, d, step))
        axf.set_yticks(ticks)
        axf.set_yticklabels([names[i] for i in ticks], fontsize=6)
        axf.set_ylabel(f'channels ({units})')
        fig.colorbar(im, ax=axf, fraction=0.025, pad=0.01,
                     label=units if zscore else None)
    else:
        for i in range(d):
            axf.plot(t, disp[:, i], lw=0.8, alpha=0.85, label=names[i])
        if zscore:
            axf.axhline(0, color='gray', ls=':', alpha=0.4)
        axf.set_ylabel(f'value ({units})')
        axf.legend(fontsize=7, loc='upper right', ncol=max(1, (d + 7) // 8))
    axf.set_xlabel('imaging time (s)')

    axes[0].set_title(title or f'feature block: {d} channels, T={T}')
    fig.tight_layout()
    return None


def f_plot_grid_delta_heatmap(grid_delta, grid_p, combo_labels, row_labels,
                              dset_tag='', n_pcs_grid=None, decoder='ridge',
                              vmax=0.3):
    """Δ R² (real − null) heatmap over a detrend×embargo grid (#8).

    Blue = real beats chance; red = real below chance (the #8 pathology).
    A '*' marks cells where real > null at p < 0.05.

    Parameters
    ----------
    grid_delta : ndarray (n_feat, n_combos)   real − null mean.
    grid_p : ndarray (n_feat, n_combos)       one-sided empirical p (P(null≥real)).
    combo_labels : list of str                σ|embargo column labels.
    row_labels : list of str                  feature labels.
    dset_tag, n_pcs_grid, decoder : title metadata. vmax : colour limit.
    """
    n_gf, n_combos = grid_delta.shape
    fig, ax = plt.subplots(1, 1, figsize=(max(6, 1.2*n_combos+2), max(4, 0.22*n_gf)))
    im = ax.imshow(grid_delta, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='none')
    ax.set_xticks(range(n_combos))
    ax.set_xticklabels(combo_labels, rotation=40, ha='right', fontsize=7)
    ax.set_yticks(range(n_gf)); ax.set_yticklabels(row_labels, fontsize=6)
    for fi in range(n_gf):
        for cj in range(n_combos):
            if (not np.isnan(grid_p[fi, cj]) and grid_p[fi, cj] < 0.05
                    and grid_delta[fi, cj] > 0):
                ax.text(cj, fi, '*', ha='center', va='center', fontsize=8, color='k')
    ax.set_title(f'{dset_tag}\nΔR² (real − null) over detrend×embargo grid  '
                 f'(n_pcs={n_pcs_grid}, {decoder}); * = real>null @ p<0.05',
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label=f'Δ R²  (clipped ±{vmax})')
    fig.tight_layout()


def f_plot_grid_focus_lines(detrend_grid, embargo_grid, combos, grid_real,
                            focus_null_all, focus_feats, grid_names,
                            dset_tag='', n_pcs_grid=None):
    """Focus-feature real-vs-null lines vs detrend σ (#8 rescue plot).

    One panel per focus feature; one real line + null 95% band per embargo level.
    The #8 rescue reads as the real line rising from below the band (σ off) up to
    / above it as the high-pass strengthens.

    Parameters
    ----------
    detrend_grid : list           σ values (None = off) → x axis (0 for None).
    embargo_grid : list           embargo levels (one real line + band each).
    combos : list of (σ, embargo)  index map into grid_real columns.
    grid_real : ndarray (n_feat, n_combos)   real R² per feature × combo.
    focus_null_all : dict          name → (n_combos, n_shuf) null draws.
    focus_feats : list of str      features to panel.
    grid_names : list of str       feature order (to locate rows).
    dset_tag, n_pcs_grid : title metadata.
    """
    sig_x = [0 if ds is None else ds for ds in detrend_grid]
    fig, axes = plt.subplots(1, len(focus_feats),
                             figsize=(5.5*len(focus_feats), 4.2), squeeze=False)
    for pi, nm in enumerate(focus_feats):
        ax = axes[0, pi]
        if nm not in grid_names:
            ax.set_title(f'{nm} (not in grid_blocks)'); ax.axis('off'); continue
        fi = grid_names.index(nm)
        for eb in embargo_grid:
            cjs = [combos.index((ds, eb)) for ds in detrend_grid]
            line, = ax.plot(sig_x, grid_real[fi, cjs], 'o-', lw=2,
                            label=f'real  emb={eb:g}s')
            null_lo, null_hi, null_md = [], [], []
            for ds in detrend_grid:
                cj = combos.index((ds, eb))
                nd = focus_null_all[nm][cj]; nd = nd[~np.isnan(nd)]
                null_lo.append(np.percentile(nd, 2.5)  if nd.size else np.nan)
                null_hi.append(np.percentile(nd, 97.5) if nd.size else np.nan)
                null_md.append(np.median(nd)           if nd.size else np.nan)
            ax.fill_between(sig_x, null_lo, null_hi, color=line.get_color(),
                            alpha=0.15, label=f'null 95%  emb={eb:g}s')
            ax.plot(sig_x, null_md, ':', color=line.get_color(), alpha=0.7)
        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('detrend σ (s; 0 = off)')
        ax.set_ylabel('blocked-CV R²')
        ax.set_title(f'{nm}  (n_pcs={n_pcs_grid})')
        ax.legend(fontsize=7)
    fig.suptitle(f'{dset_tag}\ndetrend rescue of slow targets — real vs circshift null',
                 fontsize=10)
    fig.tight_layout()
