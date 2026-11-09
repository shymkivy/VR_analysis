# -*- coding: utf-8 -*-
"""
f_ensemble_pipeline.py — reusable ensemble-workflow orchestration glue.

The ensemble analysis in VR_ca_cebra.py was spread across several inline cells. The
low-level compute already lives in f_ensembles.py / f_ensemble_plots.py, but the glue
BETWEEN those calls — fit+threshold, membership re-derivation + write-back, the tidy
per-ensemble table with the extraction-idx→true-neuron-ID mapping, and the
`neural_source` dispatch that turns ensembles into the decoder/encoder input — was
copy-paste-prone. VR_ca_dimred.py already grew a second copy of the extract glue; more
copies would drift. This module factors that glue into four drivers so any script gets
the whole ensemble workflow from one import (`import f_ensemble_pipeline as enspipe`):

    f_ensemble_apply   — fit dim-red + threshold → ens_out (the expensive FIT step)
    f_ensemble_stats   — membership + per-ensemble CV score/significance → ens_table
                         (cheap; mutates ens_out in place with the chosen membership)
    f_run_ensembles    — apply → stats → optional plots, in one call
    f_neural_source    — ens_out/ens_table (or a dim-red mode) → the reassigned Y_neu

Since 2026-11-06 it also OWNS the ensemble settings: _ens_defaults() (top of this module,
every parameter + its options) → f_default_ens_config(**overrides) per script, plus the
neural-input recipe, the fit/stats wrapper and the per-dataset cache (f_ensembles_cached).
Plots are optional (behind a flag) and delegate to f_ensemble_plots.

Created 2026-07-16.
"""

import os

import numpy as np
import pandas as pd

from f_ensembles import (f_ensemble_extract, f_cv_membership, f_score_ensembles_cv,
                         f_reduce_neural, _dimred_kwargs)
from f_ensemble_plots import (f_plot_ens_scores, f_plot_ens_overview, f_plot_ensemble_fov)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE CONFIG DEFAULTS — edit here to change them for EVERY script (restart the kernel after).
# Values = VR_ca_cebra's settings on 2026-11-06. Changing any value changes the config tag,
# so cached ensembles (<features_dir>/<mouse>/ens_cache/) are refit, not reused.
#
# Override from a script instead (one run / one script; these defaults stay as they are):
#   import f_ensemble_pipeline as enspipe
#   ens_cfg = enspipe.f_default_ens_config()                          # all defaults
#   ens_cfg = enspipe.f_default_ens_config(num_comp=12)               # one key, by its name
#   ens_cfg = enspipe.f_default_ens_config(num_comp=12, alpha_W=1e-3,
#                                          membership='cv_consensus',
#                                          sigma_frames=0)            # keys from any section
#   ens_cfg = enspipe.f_default_ens_config(dimred={'num_comp': 12, 'max_iter': 3000},
#                                          stats={'sig_method': 'mp'})   # whole sections
#   ens_cfg = enspipe.f_default_ens_config(neural={'hclust_kw': {'metric': 'jsd'}})   # nested: merged
#   ens_cfg['dimred']['num_comp']                                     # read a value
# Unknown keys raise. Keys in only one section can be passed by name; pass the section dict otherwise.
# ─────────────────────────────────────────────────────────────────────────────
_ENS_SECTIONS = ('neural', 'dimred', 'apply', 'stats', 'select')


def _ens_defaults():
    return {
        # ── neural: spikes (est1['S']) → the matrix ensembles are extracted from ──────────────
        'neural': {
            'sigma_frames': 6,          # Gaussian smoothing σ in imaging frames | 0 = none
            'peak_norm':    True,       # True = divide each cell by its max | False = raw scale
            'sort':         'hclust',   # 'hclust' = cluster-sort cells (cebra) | 'none' = est1 order
            'hclust_kw': {              # only used when sort='hclust'
                'standardize': True,    # True | False — ignored for jsd/kl metrics (L1 row norm used)
                'metric':      'kl',    # 'kl' | 'symmetric_kl' | 'jsd' | any scipy pdist metric
                                        #   ('cosine', 'correlation', 'euclidean', …)
                'method':      'average',   # scipy linkage: 'average' | 'complete' | 'single' |
                                            #   'weighted' | 'centroid' | 'median' | 'ward'
                'similarity_transform': 'auto',  # 'auto' | 'none' | 'inverse_dist' | 'exp'
            },
        },
        # ── dimred: the dim-red fit (also cebra's dimred_params for neural cleanup) ───────────
        'dimred': {
            'method':      'nmf',       # ensembles: 'nmf' | 'pca' | 'spca' (anything else → nmf)
                                        #   cleanup also: 'svd' | 'mspca'
            'num_comp':    15,          # int — number of components = number of ensembles
            'smooth_ms':   0,           # extra Gaussian smoothing (ms) inside the fit | 0 = none
            'normalize':   'norm_mean_std',  # per-cell, before the fit:
                                        #   'none' | 'norm_mean' (center) | 'norm_std' (unit std) |
                                        #   'norm_mean_std' (z-score) | 'norm_rms' (÷ RMS, stays ≥ 0)
            'l1_ratio':    0.5,         # NMF: 0 = L2 (ridge) … 0.5 = elastic net … 1 = L1 (lasso)
            'beta_loss':   'kullback-leibler',  # NMF: 'frobenius' (solver cd) | 'kullback-leibler' |
                                        #   'itakura-saito' (solver mu; needs strictly positive data)
            'alpha_W':     1e-4,        # NMF sparsity on cell loadings | 0 = none
            'alpha_H':     'same',      # NMF sparsity on temporal scores: 'same' (= alpha_W) | float
            'max_iter':    5000,        # NMF iterations (raise if ConvergenceWarnings)
            'tol':         1e-4,        # NMF convergence tolerance
            'alpha':       1.0,         # sparse PCA ('spca'/'mspca') sparsity only
            'ridge_alpha': 0.01,        # sparse PCA ('spca'/'mspca') ridge only
            'k_folds':     5,           # CV folds (membership, significance, LNO scores)
            'fold_scheme': 'interleaved',    # 'contiguous' | 'interleaved' (small blocks round-robin)
            'n_blocks':    None,        # interleaved only: number of blocks | None = k_folds × 10
            'norm_scope':  'global',    # 'global' = normalize full data before folds | 'per_fold'
            'cv_frames':   5000,        # cleanup sweep only: first N frames | None = all
            'scores':      ['test_err', 'lno_r2_med', 'lno_r2_pool'],  # cleanup sweep scores, any of:
                                        #   'test_err' | 'lno_r2_med' | 'lno_r2_pool' | 'lno_r_med' |
                                        #   'frac_reliable'
            'r_reliable':  0.05,        # LNO-R² threshold for 'reliable_cells' / 'frac_reliable'
            'seed':        42,          # random state (NMF init, folds)
        },
        # ── apply: threshold each component's loadings into member cells ──────────────────────
        'apply': {
            'thresh_mode': 'signal_z',  # 'signal_z' = median + z·std of loadings |
                                        #   'shuff' = 95th percentile of shuffled refits (slower)
            'signal_z':    2.0,         # z for 'signal_z' (also the per-fold threshold in cv_consensus)
        },
        # ── stats: membership choice + per-ensemble CV score + significance ───────────────────
        'stats': {
            'cv_method':      'pca',    # per-ensemble CV model (1 comp): 'pca' | 'svd' | 'nmf' |
                                        #   'spca' | 'mspca'
            'membership':     'z_thresh',   # 'z_thresh' = threshold of the full fit (fast; cv_r2 optimistic) |
                                            #   'cv_consensus' = refit per fold, keep cells chosen in
                                            #   ≥ consensus_frac of folds (no double-dipping)
            'consensus_frac': 0.5,      # cv_consensus only: fraction of folds a cell must be chosen in
            'sig_method':     'cv',     # 'cv' = cv_r2 > cv_sig_thresh (free) |
                                        #   'mp' = Marchenko-Pastur edge, sig if mp_ratio > 1 (cheap) |
                                        #   'shuffle' = circshift null, sig if p < sig_alpha (slow)
            'sig_alpha':      0.05,     # 'shuffle' only: p threshold
            'cv_sig_thresh':  0.0,      # 'cv' only: cv_r2 threshold
            'n_shuffle':      50,       # 'shuffle' only: number of circshift nulls
        },
        # ── select: what downstream encode / decode / tuning receive ──────────────────────────
        'select': {
            'which': 'sig',             # 'sig' = significant ensembles | 'all'
            'trace': 'scores',          # 'scores' = each ensemble's NMF temporal score (only option)
        },
    }


def _smooth_bins(dimred_params, fs):
    """dimred_params['smooth_ms'] + fs → Gaussian sigma in bins (0 if no smoothing / no fs)."""
    ms = dimred_params.get('smooth_ms', 0)
    if ms and fs:
        return int(round(ms * fs / 1000.0))
    return 0


def _n_threads():
    return max(1, (os.cpu_count() or 2) - 1)


def f_ensemble_apply(X_raw, dimred_params, thresh_mode='signal_z', signal_z=2.0,
                     fs=None, dset_tag='', verbose=True):
    """FIT dim-red (dimred_params) + threshold membership → ens_out (NO CV scoring; that's
    f_ensemble_stats). `X_raw` = (T, N) raw neural (transposed to (N, T) for the extractor).
    Keeps a PRISTINE copy of the threshold membership in cells['ens_list_apply'] so the stats
    step can re-derive membership idempotently. Re-run only when the FIT changes (num_comp /
    method / normalize / thresh). Returns ens_out (see f_ensembles.f_ensemble_extract)."""
    _method = str(dimred_params.get('method', 'nmf')).lower()
    if _method not in ('nmf', 'pca', 'spca'):
        _method = 'nmf'
    _sb = _smooth_bins(dimred_params, fs)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=_n_threads()):
        ens_out = f_ensemble_extract(
            np.asarray(X_raw).T, num_comp=int(dimred_params['num_comp']), smooth_sigma_bins=_sb,
            dred_method=_method, extraction='thresh',
            normalize=dimred_params.get('normalize', 'norm_mean_std'), thresh_mode=thresh_mode,
            signal_z_thresh=signal_z,
            dred_kwargs=_dimred_kwargs({**dimred_params, 'method': _method}),
            random_state=dimred_params.get('seed', 42))
    ens_out['cells']['ens_list_apply'] = [np.asarray(c, int) for c in ens_out['cells']['ens_list']]
    if verbose:
        print(f'{dset_tag} — apply ({_method}, num_comp={dimred_params["num_comp"]}, '
              f'thresh={thresh_mode}): members ' + str([len(c) for c in ens_out['cells']['ens_list']]))
    return ens_out


def f_ensemble_stats(ens_out, X_raw, dimred_params, cell_ids=None, cv_method='pca',
                     membership='cv_consensus', consensus_frac=0.5, sig_method='cv',
                     sig_alpha=0.05, cv_sig_thresh=0.0, n_shuffle=50, signal_z=2.0,
                     fs=None, dset_tag='', verbose=True):
    """Membership choice + per-ensemble CV score/significance ON TOP of an ens_out (cheap to
    re-run while tuning sig_method / membership — no NMF re-fit). Membership always starts from
    the pristine APPLY copy, reselected by CV consensus if `membership='cv_consensus'`; the chosen
    membership is WRITTEN BACK into ens_out (cells['ens_list']/clust_ident/non_ens_list, sel_freq)
    so the plot cells + f_neural_source read it. Builds a tidy ens_table (one row per ensemble,
    sorted by cv_r2); `cells` = TRUE neuron IDs via `cell_ids` (extraction-input row → neuron id;
    None → np.arange). `X_raw` = (T, N) aligned to the extraction cell space. Returns ens_table."""
    X_raw = np.asarray(X_raw)
    if cell_ids is None:
        cell_ids = np.arange(X_raw.shape[1])
    cell_ids = np.asarray(cell_ids)
    _sb = _smooth_bins(dimred_params, fs)
    _active_idx = np.flatnonzero(ens_out['active_cells_mask'])
    _X_act = np.asarray(X_raw.T, float)[ens_out['active_cells_mask']]
    n_t = ens_out['scores'].shape[1]
    from threadpoolctl import threadpool_limits
    # membership: always start from the pristine APPLY threshold, reselect by CV consensus if asked
    _apply_members = ens_out['cells'].get('ens_list_apply', ens_out['cells']['ens_list'])
    memb_frac = None
    if str(membership).lower() in ('cv_consensus', 'cvr2', 'cv', 'consensus'):
        with threadpool_limits(limits=_n_threads()):
            _cvm = f_cv_membership(_X_act, {**dimred_params, 'method': ens_out.get('ensemble_method', 'nmf')},
                                   ens_out['coeffs'], signal_z_thresh=signal_z,
                                   k_folds=dimred_params.get('k_folds', 5),
                                   fold_scheme=dimred_params.get('fold_scheme', 'contiguous'),
                                   n_blocks=dimred_params.get('n_blocks'), consensus_frac=consensus_frac,
                                   smooth_sigma_bins=_sb, random_state=dimred_params.get('seed', 42))
        _members = _cvm['ens_list']
        memb_frac = [float(_cvm['sel_freq'][np.asarray(c, int), i].mean()) if len(c) else np.nan
                     for i, c in enumerate(_members)]
        ens_out['sel_freq'] = _cvm['sel_freq']
    else:
        _members = [np.asarray(c, int) for c in _apply_members]
    # write the chosen membership back to ens_out (rebuild clust_ident) so downstream uses it
    ens_out['cells']['ens_list'] = _members
    _ident = np.zeros(ens_out['coeffs'].shape[0], int)
    for i, c in enumerate(_members):
        _ident[np.asarray(c, int)] = i + 1
    ens_out['cells']['clust_ident'] = _ident
    ens_out['cells']['non_ens_list'] = np.where(_ident == 0)[0]
    # per-ensemble CV score + significance (this is the cheap-to-iterate part)
    with threadpool_limits(limits=_n_threads()):
        cv_rows = f_score_ensembles_cv(
            ens_out, _X_act, cv_method=cv_method, cv_num_comp=1,
            normalize=dimred_params.get('normalize', 'norm_mean_std'), smooth_sigma_bins=_sb,
            k_folds=dimred_params.get('k_folds', 5), fold_scheme=dimred_params.get('fold_scheme', 'contiguous'),
            n_blocks=dimred_params.get('n_blocks'), norm_scope=dimred_params.get('norm_scope', 'global'),
            random_state=dimred_params.get('seed', 42), sig_method=sig_method, sig_alpha=sig_alpha,
            cv_sig_thresh=cv_sig_thresh, n_shuffle=n_shuffle)
    # tidy table (cell_ids → true neuron IDs)
    _base = ('cv_r2', 'cv_r2_pool', 'cv_test_err', 'cv_test_err_pn', 'mp_ratio', 'sig', 'sig_p', 'sig_method')
    _null = ('cv_r2_null95', 'cv_r2_z', 'cv_r2_p', 'cv_test_err_null05', 'cv_test_err_p')
    _rows = []
    for i, c in enumerate(_members):
        c = np.asarray(c, int)
        _fa = (len(ens_out['trials']['ens_list'][i]) / n_t
               if i < len(ens_out['trials']['ens_list']) else np.nan)
        _row = {'ensemble': i, 'n_cells': int(c.size),
                'cells': cell_ids[_active_idx[c]].tolist()}
        for k in _base:
            _row[k] = cv_rows[i].get(k, np.nan)
        _row['frac_active'] = float(_fa)
        if memb_frac is not None:
            _row['memb_frac'] = memb_frac[i]
        if sig_method == 'shuffle' and n_shuffle > 0:
            for k in _null:
                _row[k] = cv_rows[i].get(k, np.nan)
        _rows.append(_row)
    ens_table = pd.DataFrame(_rows).sort_values('cv_r2', ascending=False,
                                                na_position='last').reset_index(drop=True)
    if verbose:
        _cols = ['ensemble', 'n_cells', 'cv_r2', 'cv_r2_pool', 'cv_test_err_pn', 'mp_ratio', 'frac_active', 'sig']
        if 'memb_frac' in ens_table.columns:
            _cols.insert(2, 'memb_frac')
        if sig_method == 'shuffle':
            _cols[-1:-1] = ['cv_r2_null95', 'cv_r2_z', 'cv_r2_p']
        _cols = [c for c in _cols if c in ens_table.columns]
        with pd.option_context('display.max_rows', None, 'display.width', 200):
            print(f'{dset_tag} — {dimred_params["method"]} ensembles (num_comp={dimred_params["num_comp"]}, '
                  f'memb={membership}, sig={sig_method}):')
            print(ens_table[_cols].to_string(index=False))
    return ens_table


def f_run_ensembles(X_raw, dimred_params, cell_ids=None, signal_z=2.0, thresh_mode='signal_z',
                    apply_kw=None, stats_kw=None, plots=False, plot_kw=None, fov=None,
                    dset_tag='', raster_cmap='viridis', fs=None, verbose=True):
    """One-call ensemble workflow: f_ensemble_apply → f_ensemble_stats (+ optional plots). The
    single entry for scripts that don't need the cheap-re-run split. `apply_kw`/`stats_kw` pass
    extra kwargs to the two stages. `plots=True` draws per-ensemble score bars + the sorted-raster
    overview (`raster_cmap`, plot_kw['vmax_pct'], and plot_kw['vmin']/['vmax'] for a fixed raster color
    scale); if `fov={'A','dims','active_ids'}` is given it also draws the FOV map. Returns
    {'ens_out', 'ens_table'}."""
    ens_out = f_ensemble_apply(X_raw, dimred_params, thresh_mode=thresh_mode, signal_z=signal_z,
                               fs=fs, dset_tag=dset_tag, verbose=verbose, **dict(apply_kw or {}))
    ens_table = f_ensemble_stats(ens_out, X_raw, dimred_params, cell_ids=cell_ids, signal_z=signal_z,
                                 fs=fs, dset_tag=dset_tag, verbose=verbose, **dict(stats_kw or {}))
    if plots:
        pk = dict(plot_kw or {})
        f_plot_ens_scores(ens_table, dset_tag=dset_tag)
        f_plot_ens_overview(ens_out, np.asarray(X_raw).T, mouse_dset_tag=dset_tag,
                            cmap=raster_cmap, vmax_pct=pk.get('vmax_pct', 99),
                            vmin=pk.get('vmin'), vmax=pk.get('vmax'), ens_table=ens_table)
        if fov is not None:
            f_plot_ensemble_fov(fov['A'], fov['dims'], fov['active_ids'], ens_table,
                                dset_tag=dset_tag, which=pk.get('fov_which', 'sig'),
                                thresh_frac=pk.get('fov_thresh_frac', 0.1), gamma=pk.get('fov_gamma', 0.7))
    return {'ens_out': ens_out, 'ens_table': ens_table}


def f_neural_source(neural_source, X_raw, dimred_params, ens_out=None, ens_table=None,
                    which='sig', fs=None):
    """Turn the chosen source into the reassigned neural input Y_neu (feeds encode/decode/CEBRA).
    `X_raw` = (T, N) raw neural. neural_source:
      'raw'            → X_raw unchanged
      'svd_recon' / 'components' / 'reliable_cells' → f_reduce_neural (dim-red reductions)
      'ens_components' → score traces of selected ensembles (T, n_ens)   [needs ens_out+ens_table]
      'ens_cells'      → raw cells that are MEMBERS of selected ensembles [needs ens_out+ens_table]
    `which` selects WHICH ensembles feed ens_*: 'sig' (ens_table['sig']) | 'all'. Returns
    (Y_neu (T,d) float32, info-string for the caller to print)."""
    X_raw = np.asarray(X_raw)
    if neural_source == 'raw':
        return X_raw, f'neural_source=raw → Y_neu {X_raw.shape} (raw, restored)'
    if neural_source in ('ens_components', 'ens_cells'):
        if ens_out is None or ens_table is None:
            raise ValueError(f'neural_source={neural_source} needs ens_out + ens_table')
        if which == 'all':
            _sel = list(range(len(ens_out['cells']['ens_list'])))
        else:
            _sel = ens_table.loc[ens_table['sig'].astype(bool), 'ensemble'].astype(int).tolist()
        if neural_source == 'ens_cells':                    # drop empty ensembles for the cell union
            _sel = [i for i in _sel if len(ens_out['cells']['ens_list'][i]) > 0]
        if not _sel:
            return X_raw, f'neural_source={neural_source} ({which}): NO ensembles selected → raw'
        if neural_source == 'ens_components':
            Y_neu = np.asarray(ens_out['scores'][np.asarray(_sel, int), :].T, np.float32)  # (T, n_ens)
            return Y_neu, (f'neural_source=ens_components ({which}): {len(_sel)} ensembles → '
                           f'Y_neu {X_raw.shape} → {Y_neu.shape}')
        # ens_cells — union of member cells of selected ensembles (extraction idx → raw col)
        _act = np.flatnonzero(ens_out['active_cells_mask'])
        _memb = sorted(set().union(*[set(np.asarray(ens_out['cells']['ens_list'][i], int).tolist())
                                     for i in _sel]))
        _cols = _act[np.asarray(_memb, int)]
        Y_neu = X_raw[:, _cols].astype(np.float32)
        return Y_neu, (f'neural_source=ens_cells ({which}): {len(_cols)} cells in {len(_sel)} '
                       f'ensembles → Y_neu {X_raw.shape} → {Y_neu.shape}')
    # dim-red reductions
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=_n_threads()):
        Y_lr = f_reduce_neural(dimred_params, X_raw.T, mode=neural_source, fs=fs)
    Y_neu = np.asarray(Y_lr['low_rank'], np.float32)      # (T, d) — reassigned neural input
    info = (f"neural_source={neural_source} ({dimred_params['method']}, k={dimred_params['num_comp']}, "
            f"{Y_lr['kind']}): Y_neu {X_raw.shape} → {Y_neu.shape}"
            + (f"  ({int(Y_lr['keep_mask'].sum())} reliable cells)" if 'keep_mask' in Y_lr else ''))
    return Y_neu, info


# ─────────────────────────────────────────────────────────────────────────────
# ONE ensemble-generation CONFIG reused by every script (PLAN_ensemble_config.md, 2026-11-06)
# ─────────────────────────────────────────────────────────────────────────────
# Defaults: _ens_defaults() at the TOP of this module.


def f_default_ens_config(**overrides):
    """Ensemble-generation config: {'neural','dimred','apply','stats','select'} (defaults = cebra).
    Override a whole section with a dict (merged, e.g. dimred={'num_comp': 12}) or a single key by its
    flat name (num_comp=12, membership='cv_consensus'). Unknown or ambiguous keys raise."""
    import copy
    cfg = _ens_defaults()
    for k, v in overrides.items():
        if k in _ENS_SECTIONS:
            if not isinstance(v, dict):
                raise ValueError(f'f_default_ens_config: section {k!r} needs a dict, got {type(v).__name__}')
            bad = [kk for kk in v if kk not in cfg[k]]
            if bad:
                raise ValueError(f'f_default_ens_config: unknown key(s) {bad} in {k!r}; '
                                 f'have {sorted(cfg[k])}')
            for kk, vv in copy.deepcopy(v).items():
                if isinstance(cfg[k][kk], dict) and isinstance(vv, dict):   # nested dict → merge
                    bad2 = [x for x in vv if x not in cfg[k][kk]]
                    if bad2:
                        raise ValueError(f'f_default_ens_config: unknown key(s) {bad2} in {k}.{kk}; '
                                         f'have {sorted(cfg[k][kk])}')
                    cfg[k][kk].update(vv)
                else:
                    cfg[k][kk] = vv
            continue
        hits = [s for s in _ENS_SECTIONS if k in cfg[s]]
        if len(hits) != 1:
            why = 'is unknown' if not hits else f'is ambiguous ({hits}); pass it inside its section dict'
            raise ValueError(f'f_default_ens_config: key {k!r} {why}')
        cfg[hits[0]][k] = copy.deepcopy(v)
    return cfg


def f_ens_config_tag(cfg, sections=_ENS_SECTIONS):
    """Short stable hash of the config (or of a subset of its sections) for cache keys / labels."""
    import hashlib
    import json
    blob = json.dumps({s: cfg[s] for s in sections}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def f_ens_neural_input(est1, cfg, verbose=True):
    """Neural input for ensemble extraction from est1, per cfg['neural']: drop silent cells →
    Gaussian smooth (sigma_frames) → per-cell peak-norm → optional hclust sort (cebra's recipe).
    Returns {'X' (T, N) float32, 'cell_ids' (est1 row per column), 'active', 'order', 'hclust'}."""
    from f_utils import f_neural_rate
    nc = cfg['neural']
    S_sm, active = f_neural_rate(est1['S'], sigma_frames=nc['sigma_frames'], peak_norm=nc['peak_norm'])
    hc = None
    if nc['sort'] == 'hclust':
        import f_analysis as ana
        hc = ana.f_hclust_firing_rates(S_sm, **nc['hclust_kw'])
        order = np.asarray(hc['res_order'])
    elif nc['sort'] in (None, 'none'):
        order = np.arange(S_sm.shape[0])
    else:
        raise ValueError(f"neural.sort must be 'hclust' or 'none', got {nc['sort']!r}")
    X = S_sm[order].T.astype(np.float32)
    if verbose:
        print(f'ensemble input: {X.shape[1]}/{active.size} active cells x {X.shape[0]} frames '
              f'(sigma={nc["sigma_frames"]}, peak_norm={nc["peak_norm"]}, sort={nc["sort"]})')
    return {'X': X, 'cell_ids': np.flatnonzero(active)[order], 'active': active, 'order': order,
            'hclust': hc}


def f_ensembles_from_config(X, cfg, cell_ids=None, fs=None, prior=None, dset_tag='', verbose=True):
    """Ensembles for ONE dataset from the config: f_ensemble_apply (fit) → f_ensemble_stats.
    `X` (T, N) from f_ens_neural_input. `prior` = an earlier result for the SAME X: if its fit
    sections (neural/dimred/apply) match, the NMF fit is reused and only stats re-run.
    Returns {'ens_out', 'ens_table', 'cfg', 'tag', 'fit_tag', 'cell_ids'}."""
    import copy
    X = np.asarray(X)
    fit_tag = f_ens_config_tag(cfg, ('neural', 'dimred', 'apply')) + f'_{X.shape[0]}x{X.shape[1]}'
    dp = cfg['dimred']
    if prior is not None and prior.get('fit_tag') == fit_tag:
        ens_out = prior['ens_out']
        if verbose:
            print(f'{dset_tag} — reusing NMF fit {fit_tag} (only stats re-run)')
    else:
        ens_out = f_ensemble_apply(X, dp, thresh_mode=cfg['apply']['thresh_mode'],
                                   signal_z=cfg['apply']['signal_z'], fs=fs, dset_tag=dset_tag,
                                   verbose=verbose)
    ens_table = f_ensemble_stats(ens_out, X, dp, cell_ids=cell_ids, signal_z=cfg['apply']['signal_z'],
                                 fs=fs, dset_tag=dset_tag, verbose=verbose, **cfg['stats'])
    return {'ens_out': ens_out, 'ens_table': ens_table, 'cfg': copy.deepcopy(cfg),
            'tag': f_ens_config_tag(cfg), 'fit_tag': fit_tag,
            'cell_ids': None if cell_ids is None else np.asarray(cell_ids)}


def f_ens_targets(ens):
    """Ensemble traces as encode/decode targets: {'ens_all': (Y (T, n), labels), 'ens_sig': (…)}.
    Trace = each ensemble's NMF temporal score (cfg['select']['trace']='scores'); 'ens_sig' is
    absent when no ensemble is significant."""
    tr = ens['cfg']['select']['trace']
    if tr != 'scores':
        raise ValueError(f"select.trace {tr!r} not supported yet (only 'scores')")
    sc = np.asarray(ens['ens_out']['scores'], np.float32)            # (n_ens, T)
    out = {'ens_all': (sc.T, np.arange(sc.shape[0]))}
    # ens_table order (cv_r2 descending) — same as f_neural_source / f_tuning.f_select_tuning_units
    sig = ens['ens_table'].loc[ens['ens_table']['sig'].astype(bool), 'ensemble'].astype(int).to_numpy()
    if sig.size:
        out['ens_sig'] = (sc[sig].T, sig)
    return out


# ── ensemble CACHE: one file per (dataset, config) so every script reloads the same ensembles ──

def f_ens_cache_path(features_dir, mouse_id, dset_name, tag):
    """<features_dir>/ens_cache/<mouse>_<dset>_<tag>_ens.joblib"""
    return os.path.join(features_dir, 'ens_cache', f'{mouse_id}_{dset_name}_{tag}_ens.joblib')


def f_ens_cache_save(features_dir, mouse_id, dset_name, ens, n_cells_total=None, verbose=True):
    """Write one f_ensembles_from_config result (+ dataset identity) atomically. Returns the path."""
    import joblib
    import time
    path = f_ens_cache_path(features_dir, mouse_id, dset_name, ens['tag'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {**ens, 'mouse_id': mouse_id, 'dset_name': dset_name, 'n_cells_total': n_cells_total,
           'n_frames': int(np.asarray(ens['ens_out']['scores']).shape[1]),
           'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
    tmp = path + '.tmp'
    joblib.dump(rec, tmp, compress=3)
    os.replace(tmp, path)
    if verbose:
        print(f'ensemble cache WROTE {path}')
    return path


def f_ens_cache_load(features_dir, mouse_id, dset_name, cfg, n_cells_total=None, verbose=True):
    """Load the cached result for this exact config, or None (missing / config or dataset mismatch)."""
    import joblib
    tag = f_ens_config_tag(cfg)
    path = f_ens_cache_path(features_dir, mouse_id, dset_name, tag)
    if not os.path.exists(path):
        if verbose:
            print(f'ensemble cache MISS {os.path.basename(path)}')
        return None
    try:
        rec = joblib.load(path)
    except Exception as e:
        print(f'ensemble cache unreadable ({type(e).__name__}: {e}) → refit')
        return None
    if f_ens_config_tag(rec['cfg']) != tag or rec.get('dset_name') != dset_name:
        print(f'ensemble cache {os.path.basename(path)}: config/dataset mismatch → refit')
        return None
    if n_cells_total is not None and rec.get('n_cells_total') not in (None, n_cells_total):
        print(f'ensemble cache: {rec.get("n_cells_total")} cells stored vs {n_cells_total} now → refit')
        return None
    if verbose:
        print(f'ensemble cache HIT {os.path.basename(path)} (saved {rec.get("saved")})')
    return rec


def f_ensembles_cached(est1, cfg, features_dir, mouse_id, dset_name, fs=None, force=False,
                       save=True, dset_tag='', verbose=True):
    """Load-or-build in one call: cache HIT → stored ensembles; else neural input → fit → stats →
    save. `force=True` refits and overwrites. Returns the f_ensembles_from_config dict, plus 'X'
    (the neural input) on a fresh build; on a hit, call f_ens_neural_input(est1, cfg) if X is needed."""
    n_tot = int(np.asarray(est1['S']).shape[0])
    if not force:
        rec = f_ens_cache_load(features_dir, mouse_id, dset_name, cfg, n_cells_total=n_tot,
                               verbose=verbose)
        if rec is not None:
            return rec
    ni = f_ens_neural_input(est1, cfg, verbose=verbose)
    res = f_ensembles_from_config(ni['X'], cfg, cell_ids=ni['cell_ids'], fs=fs,
                                  dset_tag=dset_tag, verbose=verbose)
    if save:
        f_ens_cache_save(features_dir, mouse_id, dset_name, res, n_cells_total=n_tot, verbose=verbose)
    return {**res, 'X': ni['X']}
