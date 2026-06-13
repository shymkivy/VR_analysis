# -*- coding: utf-8 -*-
"""
f_ensembles.py — NMF / PCA ensemble extraction pipeline.

Python port of the MATLAB ensemble_analysis_YS pipeline
(C:/Users/ys2605/Desktop/stuff/ensemble_analysis_YS/). Compute layer only;
visuals live in f_ensemble_plots.py (one-way import like
f_render_diagnostics → f_functions).

Four methods (see CLAUDE.md / memory project_ensembles.md):

  A. Cross-validated grid sweep over (smooth_SD × num_comp) with
     leave-neuron-out test error.
        f_cv_estimate_grid, f_cv_estimate_one, f_dred_test_lno
  B. Auto num-components via shuffle PCA eigenvalues.
        f_estimate_dim_corr
  C. Threshold-based cell/trial → ensemble assignment.
        f_ens_get_thresh, f_apply_thresh
  D. Cluster-based cell/trial → ensemble assignment.
        f_filter_cells_by_shuf_corr, f_cluster_cells, f_extract_clust

Top-level entry: f_ensemble_extract.

Behavior-clamped helpers (exploratory):
        f_residualize_on_behavior, f_NMF_constrained

Dim-red wrappers (moved here from VR_ca_dimred.py so the CV / shuffle
loops can call them without circular imports):
        f_NMF, f_PCA, f_sparsePCA, f_mini_batch_sparsePCA,
        f_dred_add_error, f_hoyer_sparsity, f_component_stability

Convention: outside this module, X is (n_cells, n_t) — MATLAB orientation.
Inside the sklearn wrappers, the input is (n_t, n_cells) = (samples,
features), and `components` is (k, n_cells), `scores` is (n_t, k).
The orchestrator transposes at the boundary so downstream `ens_out` has
MATLAB shapes (coeffs = (n_cells, k), scores = (k, n_t)).
"""

import time
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF, PCA, SparsePCA, MiniBatchSparsePCA
from sklearn.linear_model import Ridge
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage as sp_linkage, fcluster

from f_utils import f_gauss_smooth
from f_analysis import f_circshift_rates


# =============================================================================
# Dim-red wrappers (moved verbatim from VR_ca_dimred.py).
# The CV loop (Method A), the shuffle null (Method B), and the threshold
# shuffle (Method C) all refit NMF/PCA — keeping these inline in the script
# would force circular imports. The script keeps an `from f_ensembles import
# f_NMF, ...` line for backward compat.
# =============================================================================

def f_NMF(X, num_comp, max_iter=500, random_state=None, tol=1e-4,
          solver='cd', beta_loss='frobenius', l1_ratio=0.0,
          alpha_W=0, alpha_H='same'):
    """
    sklearn NMF wrapper returning a dict with `components` (k, n_features),
    `scores` (n_samples, k), `min_val`, and metadata. X must be non-negative
    or shifted by min_val (handled internally).
    """
    start_time = time.perf_counter()

    min_val = np.min(X)
    model = NMF(
        n_components=num_comp,
        init='nndsvda',
        solver=solver,                  # cd (frobenius only), mu (KL / IS)
        beta_loss=beta_loss,            # frobenius, kullback-leibler, itakura-saito
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,              # 0=Ridge, 1=Lasso, 0.5=ElasticNet
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )

    W = model.fit_transform(X - min_val)

    if beta_loss == 'kullback-leibler':
        prefix = 'KL_NMF'
    elif beta_loss == 'itakura-saito':
        prefix = 'IS_NMF'
    else:
        prefix = 'NMF'

    if l1_ratio == 0 or l1_ratio is None:
        algo = prefix
    elif l1_ratio == 0.5:
        algo = f'elastic_net_{prefix}'
    elif l1_ratio == 1:
        algo = f'lasso_{prefix}'
    else:
        algo = f'{prefix}_l1={l1_ratio}'

    return {
        'algo':         algo,
        'num_comp':     num_comp,
        'components':   model.components_,
        'scores':       W,
        'min_val':      min_val,
        'model':        model,
        'max_iter':     max_iter,
        'random_state': random_state,
        'beta_loss':    beta_loss,
        'l1_ratio':     l1_ratio,
        'alpha_W':      alpha_W,
        'alpha_H':      alpha_H,
        'tol':          tol,
        'duration':     time.perf_counter() - start_time,
    }


def f_PCA(X, num_comp, svd_solver='randomized', random_state=None, tol=0.0):
    """sklearn PCA wrapper, same return-dict shape as f_NMF."""
    start_time = time.perf_counter()
    model = PCA(n_components=num_comp, svd_solver=svd_solver,
                random_state=random_state, tol=tol)
    X_reduced = model.fit_transform(X)
    return {
        'algo':         'PCA',
        'num_comp':     num_comp,
        'components':   model.components_,
        'scores':       X_reduced,
        'model':        model,
        'svd_solver':   svd_solver,
        'tol':          tol,
        'duration':     time.perf_counter() - start_time,
    }


def f_sparsePCA(X, num_comp, random_state=None, alpha=1.0, ridge_alpha=0.01,
                n_jobs=5, method='lars', tol=1e-08):
    start_time = time.perf_counter()
    model = SparsePCA(
        n_components=num_comp,
        method=method,                  # lars, cd
        alpha=alpha,
        ridge_alpha=ridge_alpha,
        max_iter=500,
        tol=tol,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    X_reduced = model.fit_transform(X)
    return {
        'algo':         'sparsePCA',
        'num_comp':     num_comp,
        'components':   model.components_,
        'scores':       X_reduced,
        'model':        model,
        'tol':          tol,
        'duration':     time.perf_counter() - start_time,
    }


def f_mini_batch_sparsePCA(X, num_comp, random_state=None, batch_size=3,
                           alpha=1.0, ridge_alpha=0.01, n_jobs=5,
                           method='lars', tol=1e-03):
    start_time = time.perf_counter()
    model = MiniBatchSparsePCA(
        n_components=num_comp,
        method=method,
        alpha=alpha,
        ridge_alpha=ridge_alpha,
        max_iter=500,
        batch_size=batch_size,
        tol=tol,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    X_reduced = model.fit_transform(X)
    return {
        'algo':         'mini_batch_sparsePCA',
        'num_comp':     num_comp,
        'components':   model.components_,
        'scores':       X_reduced,
        'model':        model,
        'batch_size':   batch_size,
        'tol':          tol,
        'duration':     time.perf_counter() - start_time,
    }


def f_dred_add_error(X, dred_data):
    """Mutates dred_data with frob/KL/exp_var reconstruction metrics."""
    min_val = dred_data.get('min_val', 0)
    mean = dred_data['model'].mean_ if hasattr(dred_data['model'], 'mean_') else 0

    data_rec = dred_data['scores'] @ dred_data['components'] + min_val + mean

    frob_data = np.linalg.norm(X, 'fro')
    frob_error = np.linalg.norm(X - data_rec, 'fro')
    rel_error = frob_error / frob_data

    ss_res = np.sum((X - data_rec) ** 2)
    ss_tot = np.sum((X - X.mean()) ** 2)
    explained_var = 1 - ss_res / ss_tot

    # KL (generalized Bregman) divergence — see VR_ca_dimred.py comment for
    # rationale (eps-clip handles PCA reconstructions that go negative).
    eps_kl = 1e-10
    X_pos = np.maximum(X, eps_kl)
    X_rec_pos = np.maximum(data_rec, eps_kl)
    kl_loss = np.sum(X_pos * np.log(X_pos / X_rec_pos) - X_pos + X_rec_pos)

    dred_data['frob_data'] = frob_data
    dred_data['frob_rec'] = np.linalg.norm(data_rec, 'fro')
    dred_data['frob_error'] = frob_error
    dred_data['rel_error'] = rel_error
    dred_data['ss_res'] = ss_res
    dred_data['ss_tot'] = ss_tot
    dred_data['explained_var'] = explained_var
    dred_data['kl_loss'] = kl_loss


def f_hoyer_sparsity(x):
    """Hoyer index in [0, 1]: 0 = fully dense, 1 = single non-zero entry."""
    x = np.abs(np.asarray(x, dtype=float))
    n = x.size
    if n <= 1:
        return 0.0
    l1 = x.sum()
    l2 = np.sqrt((x ** 2).sum())
    if l2 == 0:
        return 0.0
    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)


def f_component_stability(comps_list):
    """
    Hungarian-matched |corr| of components across seeds.

    Parameters
    ----------
    comps_list : list of (k, n_features) ndarrays
        One components matrix per random seed. Seed 0 is the reference.

    Returns
    -------
    sim_per_seed : (n_seeds - 1, k) ndarray
        sim_per_seed[s, i] = |corr| between ref component i and its
        Hungarian-matched partner in seed s+1. Stability per ref component
        is `sim_per_seed.mean(axis=0)`.
    """
    ref = comps_list[0]
    k = ref.shape[0]
    n_seeds = len(comps_list)
    sim_per_seed = np.zeros((n_seeds - 1, k))
    for s in range(1, n_seeds):
        other = comps_list[s]
        cross = np.abs(np.corrcoef(np.vstack([ref, other])))[:k, k:]
        row_idx, col_idx = linear_sum_assignment(-cross)
        sim_per_seed[s - 1, :] = cross[row_idx, col_idx]
    return sim_per_seed


# =============================================================================
# Shared helpers — port of MATLAB f_normalize, f_shuffle_data,
# f_make_crossval_groups.
# =============================================================================

def f_normalize_rows(X, mode='norm_mean_std'):
    """
    Per-row (per-cell) normalization. Port of MATLAB f_normalize.

    Parameters
    ----------
    X : (n_cells, n_t) ndarray
    mode : {'norm_mean_std', 'norm_mean', 'norm_std', 'norm_rms', 'none'}
        - 'norm_mean_std' : z-score per cell  (subtract mean, divide std)
        - 'norm_mean'     : center per cell
        - 'norm_std'      : divide by std per cell
        - 'norm_rms'      : divide by RMS per cell (sign-preserving; best for NMF
                            because mean isn't subtracted, so values stay ≥ 0
                            if X was ≥ 0)
        - 'none'          : no-op

    Returns
    -------
    Xn : ndarray, same shape as X. NaNs (e.g. from divide-by-zero rows) → 0.
    """
    X = np.asarray(X, dtype=float)
    if mode == 'none' or mode is None:
        return X.copy()

    if mode == 'norm_mean_std':
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True, ddof=0)
        Xn = (X - mu) / sd
    elif mode == 'norm_mean':
        Xn = X - X.mean(axis=1, keepdims=True)
    elif mode == 'norm_std':
        Xn = X / X.std(axis=1, keepdims=True, ddof=0)
    elif mode == 'norm_rms':
        rms = np.sqrt((X ** 2).mean(axis=1, keepdims=True))
        Xn = X / rms
    else:
        raise ValueError(f"Unknown normalize mode: {mode!r}")

    Xn[~np.isfinite(Xn)] = 0.0
    return Xn


def f_shuffle_data(X, mode='circ_shift', random_state=None, min_shift=0):
    """
    Per-cell shuffling. Port of MATLAB f_shuffle_data.

    Parameters
    ----------
    X : (n_cells, n_t) ndarray
    mode : {'circ_shift', 'scramble'}
        'circ_shift' preserves per-cell autocorr by random circular roll
        (uses f_circshift_rates from RNN_scripts/f_analysis.py).
        'scramble' breaks all temporal structure via np.random.permutation
        on each row.
    random_state : int or np.random.Generator or None
        Honored for BOTH modes now. For circ_shift it seeds f_circshift_rates'
        rng (was previously dropped → non-reproducible, global RNG).
    min_shift : int
        Minimum |circular shift| per cell. 0 = uniform null (unbiased). A
        nonzero value biases the null by excluding near-aligned shuffles.
    """
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    if mode == 'circ_shift':
        return f_circshift_rates(X, min_shift=min_shift, rng=rng)
    elif mode == 'scramble':
        Xs = X.copy()
        for i in range(Xs.shape[0]):
            Xs[i, :] = rng.permutation(Xs[i, :])
        return Xs
    else:
        raise ValueError(f"Unknown shuffle mode: {mode!r}")


def f_shuffle_chunked(X, chunk_size, random_state=None):
    """
    Permute chunks of time bins (preserving local autocorr inside chunks
    while breaking trial-scale structure across chunks). Used by Method A's
    CV when chunked_shuffle=True.
    """
    rng = np.random.default_rng(random_state)
    n_cells, n_t = X.shape
    n_chunks = int(np.ceil(n_t / chunk_size))
    pad = n_chunks * chunk_size - n_t
    if pad > 0:
        Xpad = np.concatenate([X, np.zeros((n_cells, pad))], axis=1)
    else:
        Xpad = X
    # split → permute → concat
    chunks = np.split(Xpad, n_chunks, axis=1)
    perm = rng.permutation(n_chunks)
    Xs = np.concatenate([chunks[i] for i in perm], axis=1)
    return Xs[:, :n_t]


def f_make_cv_groups(n_t, k_folds=5, chunked_shuffle=False,
                     chunk_size=None, random_state=None):
    """
    Contiguous k-fold partition over n_t time bins. Port of MATLAB
    f_make_crossval_groups.

    Parameters
    ----------
    n_t : int
    k_folds : int
    chunked_shuffle : bool
        If True, permute chunks of indices before partitioning. Keeps
        local autocorr while breaking trial-scale structure.
    chunk_size : int or None
        Default ceil(n_t / (k_folds * 100)) — matches MATLAB.

    Returns
    -------
    dict with keys
        'train_idx' : list of length k_folds; each is an ndarray of indices
        'test_idx'  : same
        'n_t'       : int
        'k_folds'   : int
    """
    if chunked_shuffle:
        if chunk_size is None:
            chunk_size = int(np.ceil(n_t / (k_folds * 100)))
        rng = np.random.default_rng(random_state)
        n_chunks = int(np.ceil(n_t / chunk_size))
        # build chunked permutation of indices
        idx = np.arange(n_chunks * chunk_size)
        chunks = np.split(idx, n_chunks)
        perm = rng.permutation(n_chunks)
        idx_perm = np.concatenate([chunks[i] for i in perm])[:n_t]
    else:
        idx_perm = np.arange(n_t)

    fold_size = int(np.ceil(n_t / k_folds))
    train_idx, test_idx = [], []
    for k in range(k_folds):
        a = k * fold_size
        b = min((k + 1) * fold_size, n_t)
        test = idx_perm[a:b]
        train = np.concatenate([idx_perm[:a], idx_perm[b:]])
        train_idx.append(train)
        test_idx.append(test)
    return {'train_idx': train_idx, 'test_idx': test_idx,
            'n_t': n_t, 'k_folds': k_folds}


# =============================================================================
# Helpers to unify factor extraction across methods (PCA vs NMF live in
# different orientations in the dred dict; this hides the difference).
# =============================================================================

def _get_factors(dred, n_cells):
    """
    Return (L, mu) where L is (n_cells, k) factor basis and mu is the
    per-cell offset added back during reconstruction.
        NMF : L = components.T, mu = min_val (subtracted during fit, added back)
        PCA : L = components.T, mu = model.mean_
    """
    L = dred['components'].T              # (n_cells, k)
    if 'min_val' in dred:
        # NMF was fit on (X - min_val); recon Y = W H + min_val.
        mu = np.full(n_cells, dred['min_val'])
    elif hasattr(dred['model'], 'mean_') and dred['model'].mean_ is not None:
        mu = np.asarray(dred['model'].mean_, dtype=float)
    else:
        mu = np.zeros(n_cells)
    return L, mu


# =============================================================================
# Method A — CV grid sweep for (smooth_SD × num_comp) with LNO test error.
# =============================================================================

def f_dred_test_lno(X_test, dred, method='nmf'):
    """
    Leave-neuron-out prediction on held-out test data.

    For each held-out cell i, solve x_i = (L_{-i}' L_{-i})^{-1} L_{-i}'
    (Y_{-i} - μ_{-i})  and predict Y_i = L_i x_i + μ_i. Returns the full
    predicted matrix Ycs (same shape as X_test).

    Parameters
    ----------
    X_test : (n_cells, n_t_test) ndarray
        Test fold, original (un-smoothed) data.
    dred : dict
        Dim-red result from f_NMF / f_PCA (fit on the training fold).
    method : {'nmf', 'pca', ...}
        Used only to pick the orientation convention; current logic
        delegates to _get_factors which handles both.

    Returns
    -------
    Ycs : (n_cells, n_t_test) ndarray
    """
    n_cells, n_t = X_test.shape
    L, mu = _get_factors(dred, n_cells)   # L (n_cells, k), mu (n_cells,)
    k = L.shape[1]

    Ycs = np.zeros_like(X_test, dtype=float)
    for i in range(n_cells):
        idx = np.r_[:i, i + 1:n_cells]
        L_mi = L[idx, :]                   # (n_cells-1, k)
        Y_mi = X_test[idx, :] - mu[idx, None]
        # lstsq solves L_mi @ x = Y_mi → x has shape (k, n_t)
        x_i, *_ = np.linalg.lstsq(L_mi, Y_mi, rcond=None)
        Ycs[i, :] = L[i, :] @ x_i + mu[i]
    return Ycs


def f_cv_estimate_one(X, smooth_sigma_bins, num_comp, method='nmf',
                      k_folds=5, chunked_shuffle=False,
                      normalize='norm_mean_std', dred_kwargs=None,
                      random_state=None):
    """
    Per-(smooth_SD, num_comp) k-fold CV evaluation.

    Each fold:
      1. Build train/test split (contiguous; optional chunked shuffle).
      2. Smooth + normalize both halves.
      3. Fit dim-red on smoothed train (using f_NMF / f_PCA).
      4. Predict held-out cells on raw test via f_dred_test_lno.
      5. Accumulate ||Y_test - Ycs||_F / n_cells.

    Returns
    -------
    dict with `train_err`, `train_err_sm`, `test_err`, `test_err_sm`,
    `test_norm`, `test_norm_sm` (means over folds).
    """
    n_cells, n_t = X.shape
    cv = f_make_cv_groups(n_t, k_folds=k_folds,
                          chunked_shuffle=chunked_shuffle,
                          random_state=random_state)
    dred_kwargs = dict(dred_kwargs or {})
    dred_kwargs.setdefault('random_state', random_state)

    fit_fn = {'nmf': f_NMF, 'pca': f_PCA, 'svd': f_PCA,
              'spca': f_sparsePCA}.get(method.lower())
    if fit_fn is None:
        raise ValueError(f"Unknown method for CV: {method!r}")

    metrics = {k: [] for k in ('train_err', 'train_err_sm', 'test_err',
                                'test_err_sm', 'test_norm', 'test_norm_sm')}
    for k in range(k_folds):
        tr_idx, te_idx = cv['train_idx'][k], cv['test_idx'][k]
        X_train = X[:, tr_idx]
        X_test = X[:, te_idx]

        if smooth_sigma_bins:
            X_train_sm = f_gauss_smooth(X_train, sigma_frames=smooth_sigma_bins)
            X_test_sm = f_gauss_smooth(X_test, sigma_frames=smooth_sigma_bins)
        else:
            X_train_sm, X_test_sm = X_train, X_test

        X_train_sm_n = f_normalize_rows(X_train_sm, normalize)
        X_test_n = f_normalize_rows(X_test, normalize)
        X_test_sm_n = f_normalize_rows(X_test_sm, normalize)

        # fit on smoothed train (samples = time, features = cells)
        dred = fit_fn(X_train_sm_n.T, num_comp, **dred_kwargs)
        # in-sample train error on smoothed train
        train_rec = dred['scores'] @ dred['components']
        if 'min_val' in dred:
            train_rec += dred['min_val']
        elif hasattr(dred['model'], 'mean_'):
            train_rec += dred['model'].mean_
        train_err_sm = np.linalg.norm(X_train_sm_n.T - train_rec) / n_cells

        # LNO prediction on raw test
        Ycs = f_dred_test_lno(X_test_n, dred, method=method)
        Ycs_sm = f_dred_test_lno(X_test_sm_n, dred, method=method)

        metrics['train_err'].append(train_err_sm)        # only smoothed train
        metrics['train_err_sm'].append(train_err_sm)
        metrics['test_err'].append(np.linalg.norm(X_test_n - Ycs) / n_cells)
        metrics['test_err_sm'].append(np.linalg.norm(X_test_sm_n - Ycs_sm) / n_cells)
        metrics['test_norm'].append(np.linalg.norm(X_test_n) / n_cells)
        metrics['test_norm_sm'].append(np.linalg.norm(X_test_sm_n) / n_cells)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def f_cv_estimate_grid(X, smooth_SDs_bins, num_comps, method='nmf',
                       k_folds=5, reps=1, with_shuffle=False,
                       chunked_shuffle=False, normalize='norm_mean_std',
                       dred_kwargs=None, random_state=None, verbose=True):
    """
    Cartesian-product sweep over (smooth_SD, num_comp) × reps. Optional
    shuffled-null pass (`with_shuffle=True`).

    Returns
    -------
    df : pandas.DataFrame
        Columns: smooth_SD (bins), num_comp, rep, is_shuf, train_err,
        test_err, test_err_sm, test_norm.
    """
    rows = []
    rng = np.random.default_rng(random_state)
    total = len(smooth_SDs_bins) * len(num_comps) * reps * (2 if with_shuffle else 1)
    done = 0
    for is_shuf in ([False, True] if with_shuffle else [False]):
        X_use = f_shuffle_data(X, mode='circ_shift') if is_shuf else X
        for s in smooth_SDs_bins:
            for k in num_comps:
                for r in range(reps):
                    rs = int(rng.integers(0, 2**31 - 1))
                    res = f_cv_estimate_one(
                        X_use, smooth_sigma_bins=s, num_comp=k,
                        method=method, k_folds=k_folds,
                        chunked_shuffle=chunked_shuffle, normalize=normalize,
                        dred_kwargs=dred_kwargs, random_state=rs,
                    )
                    rows.append({
                        'smooth_SD': s, 'num_comp': k, 'rep': r,
                        'is_shuf': is_shuf, **res,
                    })
                    done += 1
                    if verbose:
                        print(f"  CV [{done}/{total}] smooth={s} k={k} "
                              f"rep={r} shuf={is_shuf} test_err={res['test_err']:.4f}")
    return pd.DataFrame(rows)


# =============================================================================
# Method B — Auto-num-comp via shuffle PCA eigenvalues.
# =============================================================================

def f_estimate_dim_corr(X, n_shuf=50, n_comp_max=None,
                        normalize='norm_mean_std',
                        shuffle_method='circ_shift', random_state=None,
                        verbose=False):
    """
    Estimate dimensionality by counting PCA components whose eigenvalue
    exceeds the maximum eigenvalue of the shuffled-null distribution.

    Real X gets explained_variance_ratio_ → 'd_explained_real'. For each
    of n_shuf shuffles, PCA fits and reports its explained_variance_ratio_;
    the max across components per shuffle → 'max_lamb_shuf'.

    `dimensionality_corr = mean over shuffles of #(d_explained_real > max_lamb_shuf)`.

    Returns
    -------
    dict with d_explained_real, d_explained_shuf, max_lamb_shuf,
    dimensionality_corr (float; ceil → recommended num_comp), n_shuf.
    """
    n_cells, n_t = X.shape
    if n_comp_max is None:
        n_comp_max = min(n_cells, n_t)

    Xn = f_normalize_rows(X, normalize)
    # sklearn PCA wants (samples × features) = (n_t × n_cells)
    pca = PCA(n_components=n_comp_max, svd_solver='randomized',
              random_state=random_state)
    pca.fit(Xn.T)
    # Raw eigenvalues (not the within-fit-normalized ratio): circular shift
    # preserves total variance, so real vs shuffle eigenvalues are comparable.
    # Ratios would flatter the shuffle spectrum and over-count dimensions.
    d_explained_real = pca.explained_variance_

    rng = np.random.default_rng(random_state)
    d_explained_shuf = np.zeros((n_shuf, n_comp_max))
    max_lamb_shuf = np.zeros(n_shuf)
    for s in range(n_shuf):
        Xs = f_shuffle_data(X, mode=shuffle_method,
                             random_state=int(rng.integers(0, 2**31 - 1)))
        Xs_n = f_normalize_rows(Xs, normalize)
        pca_s = PCA(n_components=n_comp_max, svd_solver='randomized',
                    random_state=int(rng.integers(0, 2**31 - 1)))
        pca_s.fit(Xs_n.T)
        d_explained_shuf[s, :] = pca_s.explained_variance_
        max_lamb_shuf[s] = d_explained_shuf[s, :].max()
        if verbose:
            print(f"  Method B shuffle {s+1}/{n_shuf}")

    # MATLAB: dimensionality_corr = mean(sum(d_explained > max_lamb_shuff'))
    dim_corr = np.mean([np.sum(d_explained_real > ml) for ml in max_lamb_shuf])

    return {
        'd_explained_real':  d_explained_real,
        'd_explained_shuf':  d_explained_shuf,
        'max_lamb_shuf':     max_lamb_shuf,
        'dimensionality_corr': float(dim_corr),
        'n_shuf':            n_shuf,
    }


# =============================================================================
# Method C — Threshold-based ensemble extraction.
# =============================================================================

def _signal_z_thresh(factor, signal_z_thresh):
    """
    Two-sided z-score threshold (MATLAB f_ens_get_thresh signal_z branch).
    Uses median + sample-std (n-1 denominator) for robustness.
    """
    factor = np.asarray(factor, dtype=float).ravel()
    center = np.median(factor)
    spread = np.sqrt(np.sum((factor - center) ** 2) / max(len(factor) - 1, 1))
    pos = center + signal_z_thresh * spread
    neg = center - signal_z_thresh * spread
    has_neg = np.any(factor < 0)
    if has_neg:
        return np.array([neg, pos])           # two-sided
    return np.array([pos])                    # one-sided (NMF-like)


def f_ens_get_thresh(X, coeffs, scores, mode='signal_z',
                     signal_z_thresh=2.5, shuff_percent=95, n_shuf=50,
                     dred_method='nmf', dred_kwargs=None,
                     random_state=None, verbose=False):
    """
    Per-component thresholds on coefficients (cells) and scores (time bins).

    Parameters
    ----------
    X : (n_cells, n_t) ndarray
        Used by `shuff` mode to refit NMF on shuffled copies.
    coeffs : (n_cells, k) ndarray
    scores : (k, n_t) ndarray
    mode : {'signal_z', 'shuff'}
        'signal_z' : center + signal_z_thresh × spread (per component, both
                     sides only if factor has negative values).
        'shuff'    : pool 50 shuffled NMF coefficients/scores → percentile
                     threshold. Retries NMF up to 3× on convergence failure.
    signal_z_thresh : float
        SD multiplier for signal_z mode (MATLAB default 2.5).
    shuff_percent : float
        Percentile for shuff mode (MATLAB default 95).
    dred_method : {'nmf', 'pca'}
        Used by `shuff` mode to refit.
    dred_kwargs : dict or None

    Returns
    -------
    thresh_coeffs : list of length k, each entry an ndarray of length
        1 (one-sided) or 2 (two-sided) — the cutoff(s) for that component.
    thresh_scores : same shape for scores.
    """
    n_cells, k = coeffs.shape
    _, n_t = scores.shape

    if mode == 'signal_z':
        thresh_coeffs = [_signal_z_thresh(coeffs[:, i], signal_z_thresh)
                         for i in range(k)]
        thresh_scores = [_signal_z_thresh(scores[i, :], signal_z_thresh)
                         for i in range(k)]
        return thresh_coeffs, thresh_scores

    if mode != 'shuff':
        raise NotImplementedError(
            f"thresh mode {mode!r} not yet ported (only signal_z and shuff so far)")

    # ---- shuff mode ----
    fit_fn = {'nmf': f_NMF, 'pca': f_PCA}.get(dred_method.lower())
    if fit_fn is None:
        raise ValueError(f"Unsupported dred_method for shuff thresh: {dred_method!r}")
    dred_kwargs = dict(dred_kwargs or {})

    rng = np.random.default_rng(random_state)
    coeffs_pool = []
    scores_pool = []
    for s in range(n_shuf):
        for attempt in range(3):
            try:
                Xs = f_shuffle_data(
                    X, mode='circ_shift',
                    random_state=int(rng.integers(0, 2**31 - 1)))
                dk = dict(dred_kwargs)
                dk['random_state'] = int(rng.integers(0, 2**31 - 1))
                dred_s = fit_fn(f_normalize_rows(Xs, 'norm_mean_std').T, k, **dk)
                coeffs_pool.append(dred_s['components'].T)   # (n_cells, k)
                scores_pool.append(dred_s['scores'].T)        # (k, n_t)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                if verbose:
                    print(f"  Method C shuff {s+1}/{n_shuf} attempt {attempt+1} "
                          f"failed ({e}); retrying.")
        if verbose:
            print(f"  Method C shuff {s+1}/{n_shuf}")

    coeffs_pool = np.concatenate(coeffs_pool, axis=1)     # (n_cells, k*n_shuf)
    scores_pool = np.concatenate(scores_pool, axis=1)     # (k, n_t*n_shuf)

    # NMF factors are non-negative → one-sided percentile
    one_sided = (dred_method.lower() == 'nmf')
    thresh_coeffs, thresh_scores = [], []
    for i in range(k):
        if one_sided:
            tc = np.array([np.percentile(coeffs_pool, shuff_percent)])
            ts = np.array([np.percentile(scores_pool, shuff_percent)])
        else:
            tail = (100.0 - shuff_percent) / 2.0
            tc = np.array([np.percentile(coeffs_pool, tail),
                           np.percentile(coeffs_pool, 100 - tail)])
            ts = np.array([np.percentile(scores_pool, tail),
                           np.percentile(scores_pool, 100 - tail)])
        thresh_coeffs.append(tc)
        thresh_scores.append(ts)
    return thresh_coeffs, thresh_scores


def f_apply_thresh(coeffs, scores, thresh_coeffs, thresh_scores,
                   min_members=2):
    """
    Assign cells (coeffs > thresh) and trials/frames (scores > thresh) to
    ensembles. Port of MATLAB f_ensemble_apply_thresh.

    Parameters
    ----------
    coeffs : (n_cells, k)
    scores : (k, n_t)
    thresh_coeffs : list of k arrays, each length 1 (one-sided) or 2 (two-sided)
    thresh_scores : same
    min_members : int
        Flag (don't drop) ensembles smaller than this.

    Returns
    -------
    ens_out : dict with `cells.{ens_list, clust_ident, non_ens_list, ens_scores}`,
    `trials.{ens_list, clust_ident, non_ens_list}`, `low_count_ens`.
    """
    n_cells, k = coeffs.shape
    _, n_t = scores.shape

    cells_ens_list = []
    trials_ens_list = []
    clust_ident_cells = np.zeros(n_cells, dtype=int)
    clust_ident_trials = np.zeros(n_t, dtype=int)
    low_count = np.zeros(k, dtype=bool)
    ens_scores_proj = np.zeros((k, n_t))

    for i in range(k):
        tc = np.asarray(thresh_coeffs[i]).ravel()
        ts = np.asarray(thresh_scores[i]).ravel()
        # one-sided: positive side only; two-sided: above max OR below min
        cell_pos = coeffs[:, i] > tc.max()
        trial_pos = scores[i, :] > ts.max()
        if tc.size == 2:
            cell_pos = cell_pos | (coeffs[:, i] < tc.min())
            trial_pos = trial_pos | (scores[i, :] < ts.min())

        ci = np.where(cell_pos)[0]
        ti = np.where(trial_pos)[0]
        cells_ens_list.append(ci)
        trials_ens_list.append(ti)
        # last-assignment-wins for overlapping ensembles (MATLAB-equivalent
        # behavior; threshold extraction naturally allows overlap, the
        # clust_ident label just records the most recent ensemble seen).
        clust_ident_cells[ci] = i + 1
        clust_ident_trials[ti] = i + 1
        if ci.size < min_members:
            low_count[i] = True
        # per-ensemble projected score: mean(coeff_in_ensemble) @ scores
        if ci.size > 0:
            ens_scores_proj[i, :] = coeffs[ci, i].mean() * scores[i, :]
        else:
            ens_scores_proj[i, :] = scores[i, :]

    non_cells = np.where(clust_ident_cells == 0)[0]
    non_trials = np.where(clust_ident_trials == 0)[0]

    return {
        'cells': {
            'ens_list':      cells_ens_list,
            'clust_ident':   clust_ident_cells,
            'non_ens_list':  non_cells,
            'ens_scores':    ens_scores_proj,
        },
        'trials': {
            'ens_list':      trials_ens_list,
            'clust_ident':   clust_ident_trials,
            'non_ens_list':  non_trials,
        },
        'low_count_ens': low_count,
    }


# =============================================================================
# Method D — Cluster-based ensemble extraction.
# =============================================================================

def f_filter_cells_by_shuf_corr(X, dist_metric='correlation', n_shuf=100,
                                percent=95, random_state=None, verbose=False):
    """
    Boolean mask of cells whose mean pairwise correlation with all others
    exceeds the shuffle-null at the given percentile (per cell).

    `dist_metric` is the scipy distance name; 'correlation' is 1 - corrcoef.
    'cosine' is also reasonable for non-negative data. MATLAB used cosine
    by default in the filter step.
    """
    n_cells, n_t = X.shape

    def _mean_pwcorr(Y):
        D = squareform(pdist(Y, metric=dist_metric))   # (n_cells, n_cells)
        np.fill_diagonal(D, np.nan)
        sim = 1.0 - D
        return np.nanmean(sim, axis=1)

    real_pwcorr = _mean_pwcorr(X)

    rng = np.random.default_rng(random_state)
    shuf_pwcorr = np.zeros((n_shuf, n_cells))
    for s in range(n_shuf):
        Xs = f_shuffle_data(X, mode='circ_shift',
                             random_state=int(rng.integers(0, 2**31 - 1)))
        shuf_pwcorr[s, :] = _mean_pwcorr(Xs)
        if verbose:
            print(f"  Method D filter shuffle {s+1}/{n_shuf}")
    thresh = np.percentile(shuf_pwcorr, percent, axis=0)
    return real_pwcorr > thresh, real_pwcorr, thresh


def f_cluster_cells(coeffs, n_clust, linkage='ward', metric='euclidean'):
    """
    Hierarchical clustering on coefficients. Returns labels in {1..n_clust}.

    For NMF: metric='euclidean' (magnitude-sensitive — matches MATLAB).
    For PCA/ICA: metric='cosine' is a reasonable alternative.
    'ward' linkage requires Euclidean.
    """
    if linkage == 'ward' and metric != 'euclidean':
        # ward only supports Euclidean; quietly upgrade method
        linkage = 'average'
    Z = sp_linkage(coeffs, method=linkage, metric=metric)
    return fcluster(Z, t=n_clust, criterion='maxclust')


def f_clust_ens_scores(coeffs, scores, clust_labels, sigma_thresh=3.0):
    """
    Per-cluster projected score time-courses and threshold-marked active
    trials. Port of f_ensemble_clust_cell's score-projection logic.

    Returns
    -------
    ens_scores : (n_clust, n_t)
    trial_lists : list of length n_clust; active-trial indices per cluster.
    """
    uniq = np.unique(clust_labels[clust_labels > 0])
    n_t = scores.shape[1]
    ens_scores = np.zeros((len(uniq), n_t))
    trial_lists = []
    for j, c in enumerate(uniq):
        mask = clust_labels == c
        # mean across the cells in cluster c → (k,); project through scores → (n_t,)
        mean_coef = coeffs[mask, :].mean(axis=0)            # (k,)
        proj = mean_coef @ scores                            # (n_t,)
        ens_scores[j, :] = proj
        sigma = proj.std(ddof=0)
        trial_lists.append(np.where(np.abs(proj) > sigma_thresh * sigma)[0])
    return ens_scores, trial_lists


def f_extract_clust(coeffs, scores, X, num_ens, dist_metric='correlation',
                    cluster_metric='euclidean', linkage_method='ward',
                    n_shuf=100, percent=95, sigma_thresh=3.0,
                    random_state=None, verbose=False):
    """
    Cluster-based ensemble extraction. Port of MATLAB f_ensemble_clust_cell.

    1. Filter cells whose mean pwcorr > shuffle null (percent).
    2. Cluster the filtered (correlated) cells into num_ens groups.
    3. Cluster the uncorrelated cells separately (one extra cluster).
    4. Project mean(coeffs_in_cluster) @ scores → per-ensemble time courses.

    Returns ens_out matching Method C's shape.
    """
    n_cells, k = coeffs.shape
    _, n_t = scores.shape

    correlated, _pwcorr, _thr = f_filter_cells_by_shuf_corr(
        X, dist_metric=dist_metric, n_shuf=n_shuf, percent=percent,
        random_state=random_state, verbose=verbose)
    corr_idx = np.where(correlated)[0]
    uncorr_idx = np.where(~correlated)[0]

    clust_ident = np.zeros(n_cells, dtype=int)
    if corr_idx.size >= num_ens:
        labels_corr = f_cluster_cells(coeffs[corr_idx, :], num_ens,
                                       linkage=linkage_method,
                                       metric=cluster_metric)
        clust_ident[corr_idx] = labels_corr
    else:
        # not enough cells for the requested partition — assign all to one ens
        clust_ident[corr_idx] = 1

    # uncorrelated cells get their own bucket labelled n_ens + 1 (or 0 to
    # mark "unassigned"). MATLAB keeps them appended to the dendrogram; here
    # we leave them as 0 so apply_thresh-shaped output stays interpretable.
    # (clust_ident already 0 for these.)

    ens_scores, trial_lists = f_clust_ens_scores(
        coeffs, scores, clust_ident, sigma_thresh=sigma_thresh)

    uniq = np.unique(clust_ident[clust_ident > 0])
    cells_ens_list = [np.where(clust_ident == c)[0] for c in uniq]

    clust_ident_trials = np.zeros(n_t, dtype=int)
    for j, ti in enumerate(trial_lists):
        clust_ident_trials[ti] = uniq[j]

    non_cells = np.where(clust_ident == 0)[0]
    non_trials = np.where(clust_ident_trials == 0)[0]

    return {
        'cells': {
            'ens_list':      cells_ens_list,
            'clust_ident':   clust_ident,
            'non_ens_list':  non_cells,
            'ens_scores':    ens_scores,
        },
        'trials': {
            'ens_list':      trial_lists,
            'clust_ident':   clust_ident_trials,
            'non_ens_list':  non_trials,
        },
        'low_count_ens': np.array([len(ci) < 2 for ci in cells_ens_list]),
    }


# =============================================================================
# Top-level orchestrator — port of MATLAB f_ensemble_analysis_YS_raster.
# =============================================================================

def f_ensemble_extract(X, num_comp=None, smooth_sigma_bins=None,
                       dred_method='nmf', extraction='thresh',
                       normalize='norm_mean_std', thresh_mode='signal_z',
                       signal_z_thresh=2.5, n_shuf=50,
                       dred_kwargs=None, random_state=None,
                       drop_inactive=True, return_intermediates=True,
                       verbose=False):
    """
    End-to-end ensemble extraction pipeline.

    Pipeline:
      1. Drop inactive cells (sum across time == 0).
      2. Smooth (Gaussian, sigma in bins) if `smooth_sigma_bins`.
      3. Normalize per cell.
      4. If `num_comp is None`: auto-pick via Method B.
      5. Fit dim-red (NMF/PCA) — wrapper expects (n_t, n_cells).
      6. Convert to MATLAB orientation: coeffs (n_cells, k), scores (k, n_t).
      7. Extract via `extraction='thresh'` (Method C) or `'clust'` (Method D).
      8. Return merged ens_out dict.

    Parameters
    ----------
    X : (n_cells, n_t) ndarray
        Raw firing-rate-style matrix.
    num_comp : int or None
        If None, runs Method B to auto-pick.
    smooth_sigma_bins : int or None
        Gaussian sigma in bins for f_gauss_smooth. None = skip.
    dred_method : {'nmf', 'pca', 'spca'}
    extraction : {'thresh', 'clust'}
    normalize : passed to f_normalize_rows.
    thresh_mode : {'signal_z', 'shuff'}
    n_shuf : int
        Used by `shuff` thresh mode AND by auto-num-comp if num_comp is None.

    Returns
    -------
    ens_out : dict with
        'coeffs'             : (n_cells, k)
        'scores'             : (k, n_t)
        'dred'               : raw f_NMF / f_PCA output (the full dict)
        'cells'              : as in f_apply_thresh
        'trials'             : as in f_apply_thresh
        'ord_cell'           : ndarray — hclust ordering of cells by coeff
        'num_comps'          : int
        'extraction_method'  : str
        'ensemble_method'    : str
        'active_cells_mask'  : ndarray bool — which original cells survived
        'dim_info'           : optional, included if Method B was used
    """
    X = np.asarray(X, dtype=float)
    n_cells_orig, n_t = X.shape

    # 1. drop inactive
    if drop_inactive:
        active = X.sum(axis=1) > 0
    else:
        active = np.ones(n_cells_orig, dtype=bool)
    X_act = X[active, :]
    n_cells = X_act.shape[0]

    # 2. smooth
    if smooth_sigma_bins:
        X_sm = f_gauss_smooth(X_act, sigma_frames=smooth_sigma_bins)
    else:
        X_sm = X_act

    # 3. normalize
    X_norm = f_normalize_rows(X_sm, normalize)

    # 4. auto num_comp
    dim_info = None
    if num_comp is None:
        dim_info = f_estimate_dim_corr(X_act, n_shuf=n_shuf, n_comp_max=min(50, n_cells),
                                        normalize=normalize,
                                        random_state=random_state, verbose=verbose)
        num_comp = max(2, int(np.ceil(dim_info['dimensionality_corr'])))
        if verbose:
            print(f"  auto num_comp = {num_comp} "
                  f"(dimensionality_corr = {dim_info['dimensionality_corr']:.2f})")

    # 5. fit
    fit_fn = {'nmf': f_NMF, 'pca': f_PCA, 'spca': f_sparsePCA}.get(dred_method.lower())
    if fit_fn is None:
        raise ValueError(f"Unknown dred_method: {dred_method!r}")
    dred_kwargs = dict(dred_kwargs or {})
    dred_kwargs.setdefault('random_state', random_state)
    dred = fit_fn(X_norm.T, num_comp, **dred_kwargs)

    # 6. MATLAB orientation
    coeffs = dred['components'].T            # (n_cells, k)
    scores = dred['scores'].T                # (k, n_t)

    # 7. ordering by coefficient magnitude (cheap proxy for the MATLAB
    # f_hcluster_wrap step; consumer can re-sort with their own preference)
    ord_cell = np.argsort(-np.max(coeffs, axis=1))

    # 8. extract
    if extraction == 'thresh':
        thresh_c, thresh_s = f_ens_get_thresh(
            X_norm, coeffs, scores, mode=thresh_mode,
            signal_z_thresh=signal_z_thresh, n_shuf=n_shuf,
            dred_method=dred_method, dred_kwargs=dred_kwargs,
            random_state=random_state, verbose=verbose,
        )
        ens_out = f_apply_thresh(coeffs, scores, thresh_c, thresh_s)
        ens_out['thresh_coeffs'] = thresh_c
        ens_out['thresh_scores'] = thresh_s
    elif extraction == 'clust':
        ens_out = f_extract_clust(coeffs, scores, X_norm, num_comp,
                                   random_state=random_state, verbose=verbose)
    else:
        raise ValueError(f"Unknown extraction: {extraction!r}")

    ens_out.update({
        'coeffs':           coeffs,
        'scores':           scores,
        'dred':             dred,
        'ord_cell':         ord_cell,
        'num_comps':        num_comp,
        'extraction_method': extraction,
        'ensemble_method':  dred_method,
        'active_cells_mask': active,
    })
    if return_intermediates and dim_info is not None:
        ens_out['dim_info'] = dim_info
    return ens_out


# =============================================================================
# Behavior-clamped helpers — exploratory.
# =============================================================================

def f_residualize_on_behavior(X, B, ridge_alpha=1e-3):
    """
    Residualize neural data X on behavior features B via ridge regression.

    Solves W = argmin ||X - W B||_F^2 + α ||W||_F^2, then returns X - W B.

    Parameters
    ----------
    X : (n_cells, n_t) ndarray
    B : (n_feat, n_t) ndarray
        Behavior features aligned to imaging frames (built via
        f_feature_helpers.build_feature_blocks +
        f_cebra_helpers.make_cebra_supervision; pass `.T` if you built
        as (n_t, n_feat)).
    ridge_alpha : float
        L2 penalty on the regression weights.

    Returns
    -------
    X_resid : (n_cells, n_t) ndarray — X minus the behavior-explainable part.
    W : (n_cells, n_feat) ndarray — regression weights.
    """
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)
    # sklearn Ridge fits Y = X β; here Y = X.T (samples × cells), X = B.T (samples × features)
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(B.T, X.T)
    X_pred = ridge.predict(B.T).T                # (n_cells, n_t)
    return X - X_pred, ridge.coef_              # W shape (n_cells, n_feat)


def f_NMF_constrained(X, n_free, H_clamp=None, max_iter=300, tol=1e-4,
                      ridge=1.0, random_state=None, verbose=False,
                      eps=1e-9):
    """
    Sketch of semi-supervised NMF with a clamped block of components.

    Decomposes X (n_cells, n_t) ≈ W [H_free; H_clamp], where H_clamp is
    fixed (rows = behavior targets) and H_free is learned (n_free, n_t).
    Both halves of W are learned with multiplicative updates. All entries
    held non-negative via max(0, ·) clipping.

    Parameters
    ----------
    X : (n_cells, n_t) — non-negative
    n_free : int
    H_clamp : (n_clamp, n_t) or None
        If None, falls back to plain NMF with k = n_free.
    ridge : float
        L2 penalty on W.

    Returns
    -------
    dict with 'W' (n_cells, n_free + n_clamp), 'H' (full), 'H_free', 'H_clamp',
    'recon_err', 'n_free', 'n_clamp', 'algo'.
    """
    X = np.maximum(np.asarray(X, dtype=float), 0)
    n_cells, n_t = X.shape
    rng = np.random.default_rng(random_state)

    # Degenerate case (no clamp) falls through to the multiplicative loop
    # below with n_clamp = 0 — equivalent to plain NMF with random init.
    if H_clamp is None:
        H_clamp_local = np.zeros((0, n_t))
    else:
        H_clamp_local = np.maximum(np.asarray(H_clamp, dtype=float), 0)

    n_clamp = H_clamp_local.shape[0]
    k_total = n_free + n_clamp

    W = rng.random((n_cells, k_total)).astype(float)
    H_free = rng.random((n_free, n_t)).astype(float)

    err_hist = []
    for it in range(max_iter):
        H = np.vstack([H_free, H_clamp_local]) if n_clamp else H_free

        # W update: multiplicative, L2-regularized
        WH = W @ H + eps
        num_W = X @ H.T
        den_W = WH @ H.T + ridge * W + eps
        W = W * num_W / den_W
        W = np.maximum(W, 0)

        # H_free update only (H_clamp fixed)
        WH = W @ H + eps
        W_free = W[:, :n_free]
        num_H = W_free.T @ X
        den_H = W_free.T @ WH + eps
        H_free = H_free * num_H / den_H
        H_free = np.maximum(H_free, 0)

        H = np.vstack([H_free, H_clamp_local]) if n_clamp else H_free
        err = np.linalg.norm(X - W @ H, 'fro') / max(np.linalg.norm(X, 'fro'), eps)
        err_hist.append(err)
        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(f"  cNMF it {it}/{max_iter}  rel_err={err:.4f}")
        if it > 5 and abs(err_hist[-2] - err_hist[-1]) < tol:
            break

    return {
        'W':           W,
        'H':           H,
        'H_free':      H_free,
        'H_clamp':     H_clamp_local,
        'recon_err':   err_hist,
        'n_free':      n_free,
        'n_clamp':     n_clamp,
        'algo':        'NMF_constrained',
    }
