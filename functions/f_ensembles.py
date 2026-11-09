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

from sklearn.decomposition import NMF, PCA, SparsePCA, MiniBatchSparsePCA, TruncatedSVD
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
    # nndsvda (SVD-based init) requires num_comp <= min(n_samples, n_features); fall back to
    # 'random' for the over-complete case so it doesn't hard-error.
    _init = 'nndsvda' if num_comp <= min(X.shape) else 'random'
    model = NMF(
        n_components=num_comp,
        init=_init,
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


def f_SVD(X, num_comp, random_state=None, n_iter=7, tol=0.0):
    """UNCENTERED truncated SVD wrapper (no mean subtraction) — same return-dict shape as f_PCA but
    the low-rank reconstruction KEEPS the data mean (X ≈ scores @ components), unlike PCA which
    centers per feature. Use for neural cleanup where per-timepoint/mean centering is unwanted.
    TruncatedSVD has no `mean_`, so _get_factors yields mu=0 (correct uncentered LNO)."""
    start_time = time.perf_counter()
    model = TruncatedSVD(n_components=num_comp, random_state=random_state, n_iter=n_iter, tol=tol)
    X_reduced = model.fit_transform(X)
    return {
        'algo':         'SVD',
        'num_comp':     num_comp,
        'components':   model.components_,
        'scores':       X_reduced,
        'model':        model,
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


def f_nmf_consensus(X, num_comp, n_runs=30, normalize='norm_mean_std', smooth_sigma_bins=0,
                    dred_kwargs=None, random_state=42):
    """Consensus NMF (cNMF; Brunet 2004 / Kotliar 2019) — which components are STABLE across runs.
    Fit NMF `n_runs` times with different random inits on normalized(+smoothed) `X` (n_cells, T),
    L2-normalize every component loading vector, POOL all of them (n_cells × k·n_runs), and cluster
    (hierarchical, cosine/average) into `num_comp` groups. Each cluster's mean = a CONSENSUS
    component; a STABLE component is a tight cluster reproduced by ~all runs, a run-to-run artifact
    is a loose cluster from only a few runs. Returns dict:
      consensus (n_cells, k)   mean loading per cluster (stability-sorted, desc)
      stability (k,)           mean within-cluster |corr| (1 = identical across runs)
      silhouette (k,)          mean cosine silhouette of the cluster
      n_contrib (k,)           # distinct runs contributing (≈ n_runs ⇒ stable)
      cophenetic (scalar)      clustering-quality / k-selection metric (higher = cleaner)
      labels, all_coeffs, run_id"""
    from scipy.cluster.hierarchy import cophenet
    from sklearn.metrics import silhouette_samples
    X = np.asarray(X, float)
    Xsm = f_gauss_smooth(X, sigma_frames=smooth_sigma_bins) if smooth_sigma_bins else X
    Xn = f_normalize_rows(Xsm, normalize)
    dk = dict(dred_kwargs or {})
    k = int(num_comp)
    cols, run_id = [], []
    for r in range(int(n_runs)):
        W = f_NMF(Xn.T, k, random_state=random_state + r, **dk)['components'].T   # (n_cells, k)
        nrm = np.linalg.norm(W, axis=0, keepdims=True); nrm[nrm == 0] = 1.0
        cols.append(W / nrm); run_id += [r] * k
    allC = np.concatenate(cols, axis=1)                       # (n_cells, k*n_runs)
    run_id = np.asarray(run_id)
    D = pdist(allC.T, metric='cosine')
    Z = sp_linkage(D, method='average')
    labels = fcluster(Z, t=k, criterion='maxclust')
    coph = float(cophenet(Z, D)[0])
    try:
        sil = silhouette_samples(allC.T, labels, metric='cosine')
    except Exception:
        sil = np.full(allC.shape[1], np.nan)
    uniq = np.unique(labels)
    consensus = np.zeros((X.shape[0], len(uniq)))
    stability = np.full(len(uniq), np.nan); n_contrib = np.zeros(len(uniq), int)
    sil_c = np.full(len(uniq), np.nan)
    for j, c in enumerate(uniq):
        m = np.where(labels == c)[0]
        cc = allC[:, m]
        consensus[:, j] = cc.mean(1)
        if m.size > 1:
            C = np.abs(np.corrcoef(cc.T)); iu = np.triu_indices(m.size, 1)
            stability[j] = float(np.nanmean(C[iu]))
        n_contrib[j] = int(np.unique(run_id[m]).size)
        sil_c[j] = float(np.nanmean(sil[m]))
    order = np.argsort(-np.nan_to_num(stability, nan=-1.0))    # stability-sorted, desc
    return {'consensus': consensus[:, order], 'stability': stability[order],
            'silhouette': sil_c[order], 'n_contrib': n_contrib[order], 'n_runs': int(n_runs),
            'cophenetic': coph, 'labels': labels, 'all_coeffs': allC, 'run_id': run_id}


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

    # Guard per-cell denominators (constant / silent cells have std=rms=0 → replace with 1 so the
    # row comes out 0 instead of NaN, and no divide-by-zero warning is raised).
    if mode == 'norm_mean_std':
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xn = (X - mu) / sd
    elif mode == 'norm_mean':
        Xn = X - X.mean(axis=1, keepdims=True)
    elif mode == 'norm_std':
        sd = X.std(axis=1, keepdims=True, ddof=0)
        Xn = X / np.where(sd == 0, 1.0, sd)
    elif mode == 'norm_rms':
        rms = np.sqrt((X ** 2).mean(axis=1, keepdims=True))
        Xn = X / np.where(rms == 0, 1.0, rms)
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
                     chunk_size=None, random_state=None,
                     fold_scheme='contiguous', n_blocks=None):
    """
    Contiguous k-fold partition over n_t time bins. Port of MATLAB
    f_make_crossval_groups.

    fold_scheme='interleaved' ("chop", same as the decoder): split time into `n_blocks` small
    contiguous blocks round-robin'd to folds, so every fold's test set spans the whole session
    (no fold lands on a single quiet/odd epoch). n_blocks default = k_folds*10.

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
    if fold_scheme == 'interleaved':                 # "chop" — round-robin small blocks to folds
        nb = int(n_blocks or (k_folds * 10))
        block_size = max(1, n_t // nb)
        fold_of = np.minimum(np.arange(n_t) // block_size, nb - 1) % k_folds
        train_idx = [np.where(fold_of != f)[0] for f in range(k_folds)]
        test_idx = [np.where(fold_of == f)[0] for f in range(k_folds)]
        return {'train_idx': train_idx, 'test_idx': test_idx, 'n_t': n_t, 'k_folds': k_folds}
    elif fold_scheme != 'contiguous':
        raise ValueError(f"fold_scheme must be 'contiguous' or 'interleaved', got {fold_scheme!r}")

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

    fit_fn = {'nmf': f_NMF, 'pca': f_PCA, 'svd': f_SVD, 'spca': f_sparsePCA,
              'mspca': f_mini_batch_sparsePCA}.get(method.lower())
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
# Dim-red PARAMS struct + generic single-axis sweep (mirror of the encode/decode
# f_default_*_params / f_*_sweep architecture). Optimizes the neural dim-red by a
# leave-one-NEURON-out CV score; the axis can be an algo PARAM or the algo itself.
# =============================================================================

def f_dred_lno_r2(X, method='nmf', num_comp=20, smooth_sigma_bins=0,
                  normalize='norm_mean_std', k_folds=5, chunked_shuffle=False,
                  dred_kwargs=None, random_state=None, fold_scheme='contiguous', n_blocks=None,
                  norm_scope='global', return_oof=False):
    """Leave-one-NEURON-out CV reconstruction QUALITY, per cell. k-fold over time (contiguous, or
    fold_scheme='interleaved' = chop into n_blocks small blocks round-robin'd to folds); per fold
    fit the dim-red on the TRAIN bins, then LNO-predict each held-out cell on the TEST bins via
    f_dred_test_lno. Pool OOF predictions across folds → per-cell OOF Pearson r + R². Model-free
    noise ceiling: how well each neuron is predicted from the REST of the population.

    norm_scope : smooth + per-neuron standardize the FULL matrix ONCE **before** chopping into folds
        ('global', default — train & test share one per-cell scale, so the train-fit basis applies
        to test consistently; matches the MATLAB) | 're-normalize each fold from its own stats
        ('per_fold' — fully leak-free but train/test scales can drift, degrading the LNO recon).

    Returns dict: r_per_cell, r2_per_cell (n_cells), test_err (aggregate Frobenius/n_cells over
    folds), r_med, r2_med."""
    X = np.asarray(X, float)
    n_cells, n_t = X.shape
    cv = f_make_cv_groups(n_t, k_folds=k_folds, chunked_shuffle=chunked_shuffle,
                          random_state=random_state, fold_scheme=fold_scheme, n_blocks=n_blocks)
    dred_kwargs = dict(dred_kwargs or {})
    dred_kwargs.setdefault('random_state', random_state)
    fit_fn = {'nmf': f_NMF, 'pca': f_PCA, 'svd': f_SVD, 'spca': f_sparsePCA,
              'mspca': f_mini_batch_sparsePCA}.get(method.lower())
    if fit_fn is None:
        raise ValueError(f"Unknown method for CV: {method!r}")

    # 'global': standardize (+smooth) the whole matrix ONCE, then chop — one consistent per-cell scale.
    Xn_full = None
    if norm_scope == 'global':
        Xsm = f_gauss_smooth(X, sigma_frames=smooth_sigma_bins) if smooth_sigma_bins else X
        Xn_full = f_normalize_rows(Xsm, normalize)
    elif norm_scope != 'per_fold':
        raise ValueError(f"norm_scope must be 'global' or 'per_fold', got {norm_scope!r}")

    yt = np.full((n_cells, n_t), np.nan)          # OOF normalized test target
    yh = np.full((n_cells, n_t), np.nan)          # OOF leave-neuron-out prediction
    errs = []
    for kk in range(k_folds):
        tr, te = cv['train_idx'][kk], cv['test_idx'][kk]
        if norm_scope == 'global':
            Xtr_n, Xte_n = Xn_full[:, tr], Xn_full[:, te]
        else:                                     # per_fold: normalize each split from its own stats
            Xtr = f_gauss_smooth(X[:, tr], sigma_frames=smooth_sigma_bins) if smooth_sigma_bins else X[:, tr]
            Xtr_n = f_normalize_rows(Xtr, normalize)
            Xte_n = f_normalize_rows(X[:, te], normalize)
        dred = fit_fn(Xtr_n.T, num_comp, **dred_kwargs)
        Ycs = f_dred_test_lno(Xte_n, dred, method=method)
        yt[:, te] = Xte_n
        yh[:, te] = Ycs
        errs.append(np.linalg.norm(Xte_n - Ycs) / n_cells)

    r = np.full(n_cells, np.nan)
    r2 = np.full(n_cells, np.nan)
    ss_res_pool = 0.0; ss_tot_pool = 0.0                        # for the pooled (variance-explained) R²
    for i in range(n_cells):
        m = np.isfinite(yt[i]) & np.isfinite(yh[i])
        if m.sum() > 2 and yt[i, m].std() > 0:
            a, b = yt[i, m], yh[i, m]
            r[i] = float(np.corrcoef(a, b)[0, 1])
            ss_res = float(np.sum((a - b) ** 2)); ss_tot = float(np.sum((a - a.mean()) ** 2))
            r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            ss_res_pool += ss_res; ss_tot_pool += ss_tot
    # pooled R² = fraction of TOTAL member variance the LNO recon explains (variance-weighted over
    # cells & OOF frames): 1 - Σ SS_res / Σ SS_tot. Unlike r2_med (median of per-cell R²) it is the
    # normalized twin of test_err — high-variance cells count more, and it is scale/size-free.
    r2_pool = (1.0 - ss_res_pool / ss_tot_pool) if ss_tot_pool > 0 else np.nan
    out = {'r_per_cell': r, 'r2_per_cell': r2, 'test_err': float(np.mean(errs)),
           'r_med': float(np.nanmedian(r)), 'r2_med': float(np.nanmedian(r2)),
           'r2_pool': float(r2_pool)}
    if return_oof:                     # OOF target + leave-neuron-out prediction (n_cells, n_t)
        out['yt'], out['yh'] = yt, yh
    return out


def f_default_dimred_params(**overrides):
    """PARAM-CREATION struct for the neural dim-red optimization (flat dict; mirrors
    f_default_decoder_params / f_default_encoder_params). Consumed by f_run_dimred / f_dimred_sweep
    / f_reduce_neural. Sweep any key as an algo PARAM, or `method` for different ALGOS.

    Keys: method ('svd'|'pca'|'nmf'|'spca'|'mspca'), num_comp, smooth_ms (Gaussian, ms → bins via
    fs), normalize (per-NEURON: 'none'|'norm_mean'|'norm_std'|'norm_mean_std'|'norm_rms'; for NMF
    'norm_rms'/'none' [non-negative] fit better than centered 'norm_mean_std'), NMF l1_ratio /
    beta_loss ('kullback-leibler' = more distinct parts) / alpha_W / alpha_H / max_iter / tol,
    sparse-PCA alpha / ridge_alpha, k_folds, cv_frames
    (subset # frames for the sweep; None=all), scores (subset of test_err / lno_r2_med / lno_r2_pool
    [pooled variance-explained fraction] / lno_r_med / frac_reliable), r_reliable (LNO-R² keep
    threshold for reliable_cells), seed."""
    cfg = {
        'method':      'svd',
        'num_comp':    20,
        'smooth_ms':   100,
        'normalize':   'norm_mean_std',
        'l1_ratio':    0.0,            # NMF: 0=L2/ridge, 1=L1/lasso, 0.5=elastic-net
        'beta_loss':   'frobenius',    # NMF ('kullback-leibler' → solver mu; better for rate data,
                                       #      gives more DISTINCT parts = less splitting)
        'alpha_W':     0.0,            # NMF cell-loadings sparsity (L1 can over-prune — CV it)
        'alpha_H':     'same',         # NMF temporal-scores sparsity ('same' mirrors alpha_W; a
                                       #      number sets it independently — anti-splitting lever)
        'max_iter':    500,            # NMF (bump to ~1500 if ConvergenceWarnings)
        'tol':         1e-4,           # NMF convergence tol
        'alpha':       1.0,            # sparse-PCA
        'ridge_alpha': 0.01,           # sparse-PCA
        'k_folds':     5,
        'fold_scheme': 'contiguous',   # 'contiguous' | 'interleaved' (chop → round-robin)
        'n_blocks':    None,           # interleaved: # small blocks (None → k_folds*10)
        'norm_scope':  'global',       # 'global' = standardize FULL data before chopping into folds
                                       # (train/test share one scale) | 'per_fold' (leak-free)
        'cv_frames':   None,
        'scores':      ['test_err', 'lno_r2_med'],
        'r_reliable':  0.05,
        'seed':        42,
    }
    cfg.update(overrides)
    return cfg


def _dimred_kwargs(config):
    """Assemble the method-specific dred_kwargs for a dimred config (whitelist per method, so a
    swept NMF param never reaches PCA etc.)."""
    m = config['method'].lower()
    if m == 'nmf':
        bl = config.get('beta_loss', 'frobenius')
        return {'l1_ratio': config.get('l1_ratio', 0.0), 'beta_loss': bl,
                'alpha_W': config.get('alpha_W', 0.0), 'alpha_H': config.get('alpha_H', 'same'),
                'max_iter': config.get('max_iter', 500), 'tol': config.get('tol', 1e-4),
                'solver': ('mu' if bl != 'frobenius' else 'cd')}
    if m in ('spca', 'mspca'):
        return {'alpha': config.get('alpha', 1.0), 'ridge_alpha': config.get('ridge_alpha', 0.01)}
    return {}


def f_run_dimred(config, X_neu, fs=None, verbose=False):
    """ONE CV-LNO evaluation of a dim-red `config` (see f_default_dimred_params) on neural
    `X_neu` (n_cells, T): smooth_ms→bins (needs `fs`), assemble the method's kwargs, run
    f_dred_lno_r2. Returns {scores:{…}, per_cell_r2, keep_mask}."""
    X = np.asarray(X_neu, float)
    cvf = config.get('cv_frames')
    if cvf and X.shape[1] > int(cvf):
        X = X[:, :int(cvf)]
    smooth_bins = (max(0, int(round(config['smooth_ms'] * fs / 1000.0)))
                   if config.get('smooth_ms') and fs else 0)
    out = f_dred_lno_r2(X, method=config['method'], num_comp=int(config['num_comp']),
                        smooth_sigma_bins=smooth_bins, normalize=config.get('normalize', 'norm_mean_std'),
                        k_folds=config.get('k_folds', 5), dred_kwargs=_dimred_kwargs(config),
                        random_state=config.get('seed', 42),
                        fold_scheme=config.get('fold_scheme', 'contiguous'),
                        n_blocks=config.get('n_blocks'),
                        norm_scope=config.get('norm_scope', 'global'))
    r2c = np.asarray(out['r2_per_cell'], float)
    which = tuple(config.get('scores', ('test_err', 'lno_r2_med')))
    scores = {}
    if 'test_err' in which:      scores['test_err'] = out['test_err']
    if 'lno_r2_med' in which:    scores['lno_r2_med'] = out['r2_med']
    if 'lno_r2_pool' in which:   scores['lno_r2_pool'] = out['r2_pool']   # variance-explained fraction
    if 'lno_r_med' in which:     scores['lno_r_med'] = out['r_med']
    if 'frac_reliable' in which: scores['frac_reliable'] = float(np.nanmean(r2c > config.get('r_reliable', 0.05)))
    if verbose:
        print(f"  [{config['method']} k={config['num_comp']} {config.get('normalize')}]  "
              + '  '.join(f'{k}={v:+.3f}' for k, v in scores.items()))
    return {'scores': scores, 'per_cell_r2': r2c,
            'keep_mask': r2c > config.get('r_reliable', 0.05)}


def f_dimred_sweep(config, sweep_var, sweep_values, X_neu, fs=None, verbose=True):
    """Vary ONE key of a dim-red `config` across `sweep_values` — an algo PARAM (num_comp /
    smooth_ms / normalize / l1_ratio / beta_loss / alpha_W / alpha / ridge_alpha) OR the algorithm
    itself (`sweep_var='method'`, values like ['pca','svd','nmf','spca','mspca']) — run f_run_dimred
    at each, collect tidy rows [{'x': value, <score>: …}]. Combos where the swept param doesn't
    apply to the config's method are skipped with a message. Twin of f_decoder_sweep; plot with
    f_decoding.f_plot_decoder_sweep (recognizes the DIMRED config)."""
    if sweep_var not in config and sweep_var != 'method':
        raise ValueError(f"sweep_var {sweep_var!r} not a dimred-params key: {list(config)}")
    _nmf_only = {'l1_ratio', 'beta_loss', 'alpha_W', 'alpha_H', 'max_iter', 'tol'}
    _spca_only = {'alpha', 'ridge_alpha'}
    rows = []
    for v in sweep_values:
        cfg = {**config, sweep_var: v}
        m = cfg['method'].lower()
        if sweep_var in _nmf_only and m != 'nmf':
            print(f"  skip {sweep_var}={v!r}: applies only to method='nmf' (is {m!r})"); continue
        if sweep_var in _spca_only and m not in ('spca', 'mspca'):
            print(f"  skip {sweep_var}={v!r}: applies only to sparse-PCA (method is {m!r})"); continue
        try:
            out = f_run_dimred(cfg, X_neu, fs=fs)
        except Exception as e:
            print(f"  skip {sweep_var}={v!r}: {type(e).__name__}: {e}"); continue
        rows.append({'x': v, **out['scores']})
        if verbose:
            print(f"  {sweep_var}={str(v):>14s}  "
                  + '  '.join(f'{k}={rows[-1][k]:+.3f}' for k in out['scores']))
    return rows


def f_dimred_sweep2d(config, outer_var, outer_values, sweep_var, sweep_values,
                     X_neu, fs=None, verbose=True):
    """Run f_dimred_sweep ONCE per `outer_var` value → a dict {label: rows} for overlay plotting
    with f_decoding.f_plot_sweep_curves. The outer axis picks the curves, the inner (`sweep_var`)
    is the shared x. Classic use: outer_var='method', outer_values=['nmf','svd','pca'],
    sweep_var='num_comp' → one #components curve per algorithm on the same axes. Any two dimred
    keys work (e.g. outer normalize × inner num_comp). Empty inner sweeps (all combos skipped) are
    dropped by the plotter."""
    curves = {}
    for ov in outer_values:
        cfg = {**config, outer_var: ov}
        if verbose:
            print(f"[{outer_var}={ov}]")
        curves[f'{outer_var}={ov}'] = f_dimred_sweep(cfg, sweep_var, sweep_values,
                                                     X_neu, fs=fs, verbose=verbose)
    return curves


def f_reduce_neural(config, X_neu, mode='svd_recon', fs=None):
    """Fit the config's dim-red ONCE on normalized(+smoothed) `X_neu` (n_cells, T) and return the
    reduced/low-rank neural as a (T, d) matrix ready to be the encode/decode `Y_neu`:
      'svd_recon'      → low-rank reconstruction (T, n_cells), denoised, per-cell (normalized space)
      'components'     → the latent scores (T, k)
      'reliable_cells' → the RAW cells whose LNO-R² > r_reliable (T, n_kept)
    Returns dict {low_rank (T,d), kind, dred, keep_mask (reliable_cells only)}."""
    X = np.asarray(X_neu, float)
    n_cells = X.shape[0]
    smooth_bins = (int(round(config['smooth_ms'] * fs / 1000.0))
                   if config.get('smooth_ms') and fs else 0)
    Xsm = f_gauss_smooth(X, sigma_frames=smooth_bins) if smooth_bins else X
    Xn = f_normalize_rows(Xsm, config.get('normalize', 'norm_mean_std'))
    fit_fn = {'nmf': f_NMF, 'pca': f_PCA, 'svd': f_SVD, 'spca': f_sparsePCA,
              'mspca': f_mini_batch_sparsePCA}[config['method'].lower()]
    dred = fit_fn(Xn.T, int(config['num_comp']), random_state=config.get('seed', 42),
                  **_dimred_kwargs(config))
    if mode == 'components':
        return {'low_rank': np.asarray(dred['scores'], np.float32), 'kind': 'components', 'dred': dred}
    if mode == 'reliable_cells':
        lno = f_dred_lno_r2(X, method=config['method'], num_comp=int(config['num_comp']),
                            smooth_sigma_bins=smooth_bins, normalize=config.get('normalize', 'norm_mean_std'),
                            k_folds=config.get('k_folds', 5), dred_kwargs=_dimred_kwargs(config),
                            random_state=config.get('seed', 42),
                            fold_scheme=config.get('fold_scheme', 'contiguous'),
                            n_blocks=config.get('n_blocks'),
                            norm_scope=config.get('norm_scope', 'global'))
        keep = np.asarray(lno['r2_per_cell'], float) > config.get('r_reliable', 0.05)
        return {'low_rank': X.T[:, keep].astype(np.float32), 'kind': 'reliable_cells',
                'keep_mask': keep, 'dred': dred}
    # 'svd_recon' (default): low-rank reconstruction in the normalized fit space
    recon = dred['scores'] @ dred['components']       # (T, n_cells)
    if 'min_val' in dred:
        recon = recon + dred['min_val']
    elif hasattr(dred['model'], 'mean_') and dred['model'].mean_ is not None:
        recon = recon + dred['model'].mean_
    return {'low_rank': recon.astype(np.float32), 'kind': 'recon', 'dred': dred}


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
# Per-ensemble CV score + one-call ensemble extractor (list of ensembles →
# member cells → CV score). Port of MATLAB f_evaluate_ens_cv.
# =============================================================================

def _ar1_teff(Xz):
    """Effective sample size for autocorrelated rows via a mean AR(1) lag-1 coefficient:
    T_eff = T·(1−ρ₁)/(1+ρ₁). `Xz` (k, T) row-standardized. Calcium is heavily autocorrelated, so
    the raw T over-counts independent samples — this deflates it before any random-matrix test."""
    k, T = Xz.shape
    if T < 3:
        return float(T)
    num = np.sum(Xz[:, :-1] * Xz[:, 1:], axis=1)
    den = np.sum(Xz[:, :-1] ** 2, axis=1)
    r1 = float(np.nanmean(num / np.where(den == 0, np.nan, den)))
    r1 = np.clip(r1 if np.isfinite(r1) else 0.0, 0.0, 0.999)
    return float(np.clip(T * (1.0 - r1) / (1.0 + r1), k + 1, T))


def f_mp_edge_test(Xm, smooth_sigma_bins=0, autocorr_correct=True):
    """Marchenko–Pastur / random-matrix significance for ONE candidate ensemble (the standard
    cell-assembly analytical test, cheap — one eigendecomposition). Row-standardize the member
    submatrix `Xm` (k, T), take the leading eigenvalue λ₁ of its k×k correlation matrix, compare to
    the MP upper edge λ₊ = (1+√q)², q = k/T_eff. `autocorr_correct` uses an AR(1) effective sample
    size (calcium is autocorrelated → raw T inflates significance). Returns
    {mp_lambda1, mp_edge, mp_ratio (λ₁/λ₊; >1 ⇒ real), T_eff, mp_sig}."""
    Xm = np.asarray(Xm, float)
    if smooth_sigma_bins:
        Xm = f_gauss_smooth(Xm, sigma_frames=smooth_sigma_bins)
    k, T = Xm.shape
    Xz = Xm - Xm.mean(1, keepdims=True)
    sd = Xz.std(1, keepdims=True)
    Xz = Xz / np.where(sd == 0, 1.0, sd)
    C = (Xz @ Xz.T) / T
    lam1 = float(np.linalg.eigvalsh(C)[-1])
    T_eff = _ar1_teff(Xz) if autocorr_correct else float(T)
    q = k / T_eff
    edge = float((1.0 + np.sqrt(q)) ** 2)
    return {'mp_lambda1': lam1, 'mp_edge': edge, 'mp_ratio': lam1 / edge if edge > 0 else np.nan,
            'T_eff': T_eff, 'mp_sig': bool(lam1 > edge)}


def f_score_ensembles_cv(ens_out, X, cv_method='pca', cv_num_comp=1, normalize='norm_mean_std',
                         smooth_sigma_bins=0, k_folds=5, fold_scheme='contiguous', n_blocks=None,
                         norm_scope='global', min_members=2, random_state=42,
                         sig_method='cv', sig_alpha=0.05, cv_sig_thresh=0.0,
                         n_shuffle=50, shuffle_min_shift=0):
    """Per-ensemble score + SIGNIFICANCE — extends MATLAB f_evaluate_ens_cv. For each ensemble in
    `ens_out['cells']['ens_list']`, subset the raster to its MEMBER cells and run leave-one-NEURON-out
    CV (f_dred_lno_r2, `cv_num_comp` comp(s), `cv_method`): do the members form a co-activation
    pattern that reconstructs held-out time? `X` (n_cells, T) MUST be aligned to the extraction cell
    space, i.e. `X[ens_out['active_cells_mask']]`. Ensembles < `min_members` cells score NaN.

    ALWAYS computes the cheap scores: cv_r2 (MEDIAN per-member LNO R²; higher=better), cv_r2_pool
    (POOLED / variance-explained fraction = 1 - ΣSS_res/ΣSS_tot over members & OOF frames; the
    normalized twin of test_err, scale/size-free), cv_test_err (lower=better) + cv_test_err_pn
    (× √n_members → member-count-independent), and the Marchenko–Pastur stats (mp_lambda1/mp_edge/
    mp_ratio via f_mp_edge_test).

    `sig_method` picks the significance verdict written to `sig`/`sig_p`/`sig_stat`/`sig_thresh`:
      'cv'      → sig = cv_r2 > `cv_sig_thresh` (0 by default). FREE: cross-validation's built-in
                  null — a spurious 1-D pattern gives OOF R² ≤ 0.  (sig_p = NaN)
      'mp'      → sig = λ₁ > MP edge (analytical random-matrix test; autocorr-corrected). (sig_p=NaN)
      'shuffle' → per-ensemble CIRCULAR-SHIFT null (`n_shuffle` reps; independently roll each member,
                  destroying cross-cell synchrony) → cv_r2_null95 / cv_r2_z / cv_r2_p (frac null ≥
                  real) + cv_test_err_null05 / cv_test_err_p; sig = cv_r2_p < `sig_alpha`. Expensive.

    Returns a list (aligned to ens_list) of dicts with the keys above (unavailable ones NaN)."""
    X = np.asarray(X, float)

    def _score(Xsub, seed):
        nc = int(min(cv_num_comp, max(1, Xsub.shape[0] - 1)))
        return f_dred_lno_r2(Xsub, method=cv_method, num_comp=nc, smooth_sigma_bins=smooth_sigma_bins,
                             normalize=normalize, k_folds=k_folds, fold_scheme=fold_scheme,
                             n_blocks=n_blocks, norm_scope=norm_scope, random_state=seed)

    out = []
    for ie, members in enumerate(ens_out['cells']['ens_list']):
        members = np.asarray(members, int)
        rec = {'cv_test_err': np.nan, 'cv_r2': np.nan, 'cv_r2_pool': np.nan, 'cv_test_err_pn': np.nan,
               'n_members': int(members.size),
               'mp_lambda1': np.nan, 'mp_edge': np.nan, 'mp_ratio': np.nan,
               'cv_r2_null95': np.nan, 'cv_r2_z': np.nan, 'cv_r2_p': np.nan,
               'cv_test_err_null05': np.nan, 'cv_test_err_p': np.nan,
               'sig': False, 'sig_p': np.nan, 'sig_stat': np.nan, 'sig_thresh': np.nan,
               'sig_method': sig_method}
        if members.size < min_members:
            out.append(rec); continue
        Xm = X[members]
        r = _score(Xm, random_state)
        rec['cv_test_err'], rec['cv_r2'] = r['test_err'], r['r2_med']
        rec['cv_r2_pool'] = r['r2_pool']                                # variance-explained fraction
        rec['cv_test_err_pn'] = r['test_err'] * np.sqrt(members.size)   # remove 1/√k size scaling
        mp = f_mp_edge_test(Xm, smooth_sigma_bins=smooth_sigma_bins)    # cheap → always
        rec.update({k_: mp[k_] for k_ in ('mp_lambda1', 'mp_edge', 'mp_ratio')})

        if sig_method == 'shuffle' and n_shuffle > 0:
            rng = np.random.default_rng((random_state or 0) + 1000 * (ie + 1))
            null_r2, null_te = [], []
            for _ in range(int(n_shuffle)):
                Xs = f_shuffle_data(Xm, mode='circ_shift', random_state=rng, min_shift=shuffle_min_shift)
                rs = _score(Xs, int(rng.integers(0, 2**31 - 1)))
                null_r2.append(rs['r2_med']); null_te.append(rs['test_err'])
            null_r2 = np.asarray(null_r2, float); null_te = np.asarray(null_te, float)
            n = int(np.isfinite(null_r2).sum())
            if n > 0:
                mu, sd = np.nanmean(null_r2), np.nanstd(null_r2)
                rec['cv_r2_null95'] = float(np.nanpercentile(null_r2, 95))
                rec['cv_r2_z'] = float((rec['cv_r2'] - mu) / sd) if sd > 0 else np.nan
                rec['cv_r2_p'] = float((1 + np.nansum(null_r2 >= rec['cv_r2'])) / (n + 1))
                rec['cv_test_err_null05'] = float(np.nanpercentile(null_te, 5))
                rec['cv_test_err_p'] = float((1 + np.nansum(null_te <= rec['cv_test_err'])) / (n + 1))
            rec.update({'sig_stat': rec['cv_r2'], 'sig_thresh': rec['cv_r2_null95'],
                        'sig_p': rec['cv_r2_p'],
                        'sig': bool(np.isfinite(rec['cv_r2_p']) and rec['cv_r2_p'] < sig_alpha)})
        elif sig_method == 'mp':
            rec.update({'sig_stat': mp['mp_lambda1'], 'sig_thresh': mp['mp_edge'],
                        'sig': mp['mp_sig']})
        else:  # 'cv' — cross-validation's built-in null at cv_sig_thresh
            rec.update({'sig_stat': rec['cv_r2'], 'sig_thresh': cv_sig_thresh,
                        'sig': bool(np.isfinite(rec['cv_r2']) and rec['cv_r2'] > cv_sig_thresh)})
        out.append(rec)
    return out


def f_cv_membership(X, dimred_params, ref_coeffs, signal_z_thresh=2.5, k_folds=5,
                    fold_scheme='contiguous', n_blocks=None, consensus_frac=0.5,
                    smooth_sigma_bins=0, random_state=42):
    """NESTED (cross-validated) ensemble MEMBERSHIP — reduces the double-dipping of selecting members
    on the SAME data used to score them. For each CV fold over time: refit the dim-red on the TRAIN
    bins, MATCH its components back to the canonical full-fit components `ref_coeffs` (n_cells, k) by
    loading correlation (Hungarian, so component identity is preserved without any re-ordering),
    threshold the matched loadings, and record which cells were selected. A cell is a CONSENSUS member
    of ensemble i if it is selected in ≥ `consensus_frac` of folds. `X` (n_cells, T) is the
    extraction-space (active) matrix. Returns {ens_list (consensus members per component), sel_freq
    (n_cells, k) per-cell selection frequency, consensus_frac}."""
    from scipy.optimize import linear_sum_assignment
    X = np.asarray(X, float)
    n_cells, n_t = X.shape
    k = np.asarray(ref_coeffs).shape[1]
    fit_fn = {'nmf': f_NMF, 'pca': f_PCA, 'svd': f_SVD, 'spca': f_sparsePCA,
              'mspca': f_mini_batch_sparsePCA}[dimred_params['method'].lower()]
    dk = _dimred_kwargs(dimred_params)
    norm = dimred_params.get('normalize', 'norm_mean_std')
    cv = f_make_cv_groups(n_t, k_folds=k_folds, fold_scheme=fold_scheme, n_blocks=n_blocks,
                          random_state=random_state)
    Rc = np.asarray(ref_coeffs, float) - np.asarray(ref_coeffs, float).mean(0, keepdims=True)
    sel = np.zeros((n_cells, k))
    for kk in range(k_folds):
        tr = cv['train_idx'][kk]
        Xtr = f_gauss_smooth(X[:, tr], sigma_frames=smooth_sigma_bins) if smooth_sigma_bins else X[:, tr]
        Xtr_n = f_normalize_rows(Xtr, norm)
        Wf = fit_fn(Xtr_n.T, k, random_state=random_state, **dk)['components'].T   # (n_cells, k)
        Wc = Wf - Wf.mean(0, keepdims=True)
        C = np.abs(np.corrcoef(np.vstack([Rc.T, Wc.T]))[:k, k:])                   # |corr| ref × fold
        C = np.nan_to_num(C, nan=0.0)
        ri, cj = linear_sum_assignment(-C)
        order = np.empty(k, int); order[ri] = cj                                  # ref i → fold col
        for i in range(k):
            col = Wf[:, order[i]]
            thr = _signal_z_thresh(col, signal_z_thresh)
            m = col > thr.max()
            if thr.size == 2:
                m = m | (col < thr.min())
            sel[m, i] += 1
    sel_freq = sel / k_folds
    ens_list = [np.where(sel_freq[:, i] >= consensus_frac)[0] for i in range(k)]
    return {'ens_list': ens_list, 'sel_freq': sel_freq, 'consensus_frac': consensus_frac}


def f_extract_ensembles(X, dimred_params, thresh_mode='signal_z', signal_z_thresh=2.5,
                        extraction='thresh', cv=True, cv_method='pca', cv_num_comp=1,
                        cell_ids=None, sig_method='cv', sig_alpha=0.05, cv_sig_thresh=0.0,
                        n_shuffle=50, membership='z_thresh', consensus_frac=0.5, membership_folds=None,
                        fs=None, verbose=False):
    """ONE call → (ens_table, ens_out): the list of ENSEMBLES (dim-red components), the CELLS in
    each (thresholded coefficients), and a per-ensemble CV SCORE. Fits the dim-red per
    `dimred_params` (method default 'nmf', `num_comp` = # ensembles, `smooth_ms`, `normalize`) via
    f_ensemble_extract (membership = f_ens_get_thresh + f_apply_thresh), then f_score_ensembles_cv
    (member-cell leave-one-neuron-out, mirroring MATLAB f_evaluate_ens_cv).

    `X` = (n_cells, T) neural (whatever cell ORDER you extract in — the plot must use this same
    matrix). `cell_ids` : optional length-`X.shape[0]` array mapping each ROW of `X` to a true
    recording-neuron ID; when given, the table's `cells` column reports those IDs (else raw row
    positions in `X`).

    `sig_method` selects the per-ensemble significance verdict (`sig` column, see
    f_score_ensembles_cv): 'cv' (free — cv_r2 > `cv_sig_thresh`) | 'mp' (analytical Marchenko–Pastur
    edge) | 'shuffle' (circular-shift null, `n_shuffle` reps, `sig_alpha`). All cheap scores
    (cv_r2, cv_test_err, cv_test_err_pn [size-normalized], mp_ratio) are ALWAYS in the table; the
    circshift null columns (cv_r2_null95/z/p) appear only under 'shuffle'.

    `membership`: 'z_thresh' (default — members = threshold of the full-fit coefficients, per
    `thresh_mode`: median+z·std or shuffle) | 'cv_consensus' (aliases 'cv'/'cvr2': NESTED selection
    via f_cv_membership — refit per train fold, match to the full-fit components, keep cells selected
    in ≥ `consensus_frac` of `membership_folds` folds). 'cv_consensus' cuts the double-dipping (members
    no longer chosen on the same data used to score) and adds a `memb_frac` column (mean per-member
    selection frequency = membership stability); the full-fit coeffs/scores are kept for plotting.
    Returns:
      ens_table : pandas DataFrame, one row per ensemble — ensemble, n_cells, cells (true neuron IDs
                  if cell_ids given, else X-row positions), cv_r2, cv_test_err, cv_test_err_pn,
                  mp_ratio, frac_active, sig, sig_p (+ null columns under 'shuffle') — sorted by cv_r2
                  desc (so head(N) = top-N by score).
      ens_out   : full f_ensemble_extract dict (coeffs/scores/clust_ident/… for plots)."""
    method = str(dimred_params.get('method', 'nmf')).lower()
    if method not in ('nmf', 'pca', 'spca'):
        if verbose:
            print(f"  ensemble extraction: method {method!r} not supported (nmf/pca/spca) → nmf")
        method = 'nmf'
    smooth_bins = (int(round(dimred_params.get('smooth_ms', 0) * fs / 1000.0))
                   if dimred_params.get('smooth_ms') and fs else 0)
    ens_out = f_ensemble_extract(
        np.asarray(X, float), num_comp=int(dimred_params['num_comp']),
        smooth_sigma_bins=smooth_bins, dred_method=method, extraction=extraction,
        normalize=dimred_params.get('normalize', 'norm_mean_std'), thresh_mode=thresh_mode,
        signal_z_thresh=signal_z_thresh, dred_kwargs=_dimred_kwargs({**dimred_params, 'method': method}),
        random_state=dimred_params.get('seed', 42), verbose=verbose)

    _active_idx = np.flatnonzero(ens_out['active_cells_mask'])   # extraction-active → X-row index
    # X-row → true neuron ID (identity = X-row position if no map given)
    _ids = np.arange(np.asarray(X).shape[0]) if cell_ids is None else np.asarray(cell_ids)
    if _ids.shape[0] != np.asarray(X).shape[0]:
        raise ValueError(f"cell_ids length {_ids.shape[0]} != X rows {np.asarray(X).shape[0]}")
    _X_act = np.asarray(X, float)[ens_out['active_cells_mask']]
    n_t = ens_out['scores'].shape[1]

    # NESTED membership: reselect members by CV consensus (train-fold selection) to cut double-dipping
    memb_frac = None
    if str(membership).lower() in ('cv_consensus', 'cvr2', 'cv', 'consensus'):
        cvm = f_cv_membership(
            _X_act, {**dimred_params, 'method': method}, ens_out['coeffs'],
            signal_z_thresh=signal_z_thresh, k_folds=(membership_folds or dimred_params.get('k_folds', 5)),
            fold_scheme=dimred_params.get('fold_scheme', 'contiguous'), n_blocks=dimred_params.get('n_blocks'),
            consensus_frac=consensus_frac, smooth_sigma_bins=smooth_bins,
            random_state=dimred_params.get('seed', 42))
        ens_out['cells']['ens_list'] = cvm['ens_list']       # consensus members replace full-fit members
        ens_out['sel_freq'] = cvm['sel_freq']
        _ident = np.zeros(ens_out['coeffs'].shape[0], int)   # rebuild clust_ident / non_ens_list
        for i, ci in enumerate(cvm['ens_list']):
            _ident[np.asarray(ci, int)] = i + 1
        ens_out['cells']['clust_ident'] = _ident
        ens_out['cells']['non_ens_list'] = np.where(_ident == 0)[0]
        memb_frac = [float(cvm['sel_freq'][np.asarray(ci, int), i].mean()) if len(ci) else np.nan
                     for i, ci in enumerate(cvm['ens_list'])]   # mean selection freq of the members
    if cv:
        cv_rows = f_score_ensembles_cv(
            ens_out, _X_act, cv_method=cv_method, cv_num_comp=cv_num_comp,
            normalize=dimred_params.get('normalize', 'norm_mean_std'), smooth_sigma_bins=smooth_bins,
            k_folds=dimred_params.get('k_folds', 5), fold_scheme=dimred_params.get('fold_scheme', 'contiguous'),
            n_blocks=dimred_params.get('n_blocks'), norm_scope=dimred_params.get('norm_scope', 'global'),
            random_state=dimred_params.get('seed', 42), sig_method=sig_method, sig_alpha=sig_alpha,
            cv_sig_thresh=cv_sig_thresh, n_shuffle=n_shuffle)
    else:
        cv_rows = [{'cv_test_err': np.nan, 'cv_r2': np.nan, 'n_members': len(m)}
                   for m in ens_out['cells']['ens_list']]

    # always-present cheap scores + significance; circshift-null columns only under 'shuffle'
    _base_keys = ('cv_r2', 'cv_r2_pool', 'cv_test_err', 'cv_test_err_pn', 'mp_ratio', 'sig', 'sig_p', 'sig_method')
    _null_keys = ('cv_r2_null95', 'cv_r2_z', 'cv_r2_p', 'cv_test_err_null05', 'cv_test_err_p')
    rows = []
    for i, members in enumerate(ens_out['cells']['ens_list']):
        members = np.asarray(members, int)
        frac_active = (len(ens_out['trials']['ens_list'][i]) / n_t
                       if i < len(ens_out['trials']['ens_list']) else np.nan)
        row = {'ensemble': i, 'n_cells': int(members.size),
               'cells': _ids[_active_idx[members]].tolist()}   # true neuron IDs (or X-row pos)
        for kk in _base_keys:
            row[kk] = cv_rows[i].get(kk, np.nan)
        row['frac_active'] = float(frac_active)
        if memb_frac is not None:                              # CV-membership consensus strength
            row['memb_frac'] = memb_frac[i]
        if sig_method == 'shuffle' and n_shuffle > 0:
            for kk in _null_keys:
                row[kk] = cv_rows[i].get(kk, np.nan)
        rows.append(row)
    ens_table = pd.DataFrame(rows).sort_values('cv_r2', ascending=False,
                                               na_position='last').reset_index(drop=True)
    return ens_table, ens_out


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
