# -*- coding: utf-8 -*-
"""
f_ensemble_plots.py — MATLAB-style visuals for the ensemble pipeline.

Companion to f_ensembles.py (compute). One-way import — this module
imports from f_ensembles, never the reverse — mirroring the documented
f_render_diagnostics → f_functions split.

Plots:
    f_plot_cv_grid          — port of MATLAB f_plot_cv_error_3D.
    f_plot_dim_estimate     — Method B companion: real vs shuffle eigenvalues.
    f_plot_raster_mean      — heatmap + linked mean trace (port of
                              f_plot_raster_mean).
    f_plot_trial_indicator  — colored bands above raster.
    f_plot_ensemble_deets   — per-ensemble 4-panel detail figure (port of
                              f_plot_ensemble_deets).
    f_plot_comp_scatter     — 2D/3D component scatter coloured by ensemble
                              (port of f_plot_comp_scatter).
    f_plot_ens_overview     — convenience: sorted raster + per-ensemble
                              loop, mirroring the MATLAB driver's end-of-
                              script figure set.

All plots take an optional `title` / `tag` kwarg for the figure suptitle
and an `ax`/`fig` kwarg for embedding inside an existing layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d' projection)


# Default colour cycle (matches MATLAB's typical {r,g,b,c,m,y,k} list
# the original f_plot_comp_scatter falls back to).
_DEFAULT_COLORS = ['#d62728', '#2ca02c', '#1f77b4', '#17becf',
                   '#9467bd', '#bcbd22', '#000000']


# =============================================================================
# Method A companion — CV error surface (port of f_plot_cv_error_3D).
# =============================================================================

def f_plot_cv_grid(cv_df, cv_df_shuf=None, x='smooth_SD', y='num_comp',
                   z='test_err', ax=None, title=None):
    """
    Plot CV grid error over (x, y) → z. If both x and y have >1 unique
    value, produces a 3D surface; otherwise a 1D line with errorbars.
    Real data in blue; shuffled in black (if provided).

    Parameters
    ----------
    cv_df : pandas.DataFrame
        From f_cv_estimate_grid. Must have columns x, y, z, 'rep'.
    cv_df_shuf : pandas.DataFrame or None
        Same shape; overlaid as black.
    """
    def _grid(df):
        # average across reps; std for errorbar
        g = df.groupby([x, y])[z].agg(['mean', 'std']).reset_index()
        xs = np.sort(g[x].unique())
        ys = np.sort(g[y].unique())
        Z_mean = np.full((len(ys), len(xs)), np.nan)
        Z_std = np.full((len(ys), len(xs)), np.nan)
        for _, row in g.iterrows():
            ix = np.where(xs == row[x])[0][0]
            iy = np.where(ys == row[y])[0][0]
            Z_mean[iy, ix] = row['mean']
            Z_std[iy, ix] = row['std']
        return xs, ys, Z_mean, Z_std

    xs, ys, Zm, Zs = _grid(cv_df)
    one_d = (len(xs) == 1) or (len(ys) == 1)

    if one_d:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure
        if len(xs) == 1:
            xv, zv = ys, Zm[:, 0]
            ev = Zs[:, 0]
            xl = y
        else:
            xv, zv = xs, Zm[0, :]
            ev = Zs[0, :]
            xl = x
        ax.errorbar(xv, zv, yerr=ev, fmt='-o', color='b', label='data', lw=2)
        if cv_df_shuf is not None:
            xs2, ys2, Zm2, Zs2 = _grid(cv_df_shuf)
            if len(xs2) == 1:
                xv2, zv2, ev2 = ys2, Zm2[:, 0], Zs2[:, 0]
            else:
                xv2, zv2, ev2 = xs2, Zm2[0, :], Zs2[0, :]
            ax.errorbar(xv2, zv2, yerr=ev2, fmt='-o', color='k',
                        label='shuf', lw=2, alpha=0.6)
            ax.legend()
        ax.set_xlabel(xl)
        ax.set_ylabel(z)
        if title:
            ax.set_title(title)
        return fig, ax

    # 3D surface
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, Zm, cmap='viridis', alpha=0.85, edgecolor='b', lw=0.3)
    if cv_df_shuf is not None:
        xs2, ys2, Zm2, _ = _grid(cv_df_shuf)
        X2, Y2 = np.meshgrid(xs2, ys2)
        ax.plot_wireframe(X2, Y2, Zm2, color='k', lw=0.6, alpha=0.6,
                          label='shuf')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    if title:
        ax.set_title(title)
    return fig, ax


# =============================================================================
# Method B companion — auto-num-comp via shuffle eigenvalues.
# =============================================================================

def f_plot_dim_estimate(dim_info, ax=None, title=None, max_show=30):
    """
    Real explained-variance bar + shuffle-max shaded band + dim_corr line.

    `dim_info` is the dict returned by f_estimate_dim_corr.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    de_real = dim_info['d_explained_real']
    de_shuf = dim_info['d_explained_shuf']
    n_show = min(max_show, de_real.size)
    ks = np.arange(1, n_show + 1)

    ax.bar(ks, de_real[:n_show], color='#1f77b4', alpha=0.85, label='real')
    shuf_mean = de_shuf[:, :n_show].mean(axis=0)
    shuf_lo = np.percentile(de_shuf[:, :n_show], 5, axis=0)
    shuf_hi = np.percentile(de_shuf[:, :n_show], 95, axis=0)
    ax.plot(ks, shuf_mean, 'k-', lw=1, label='shuf mean')
    ax.fill_between(ks, shuf_lo, shuf_hi, color='k', alpha=0.15,
                    label='shuf 5–95%')
    ax.axhline(dim_info['max_lamb_shuf'].mean(), color='gray', ls=':',
               lw=1, label='mean shuf max')
    ax.axvline(dim_info['dimensionality_corr'], color='r', ls='--', lw=1.5,
               label=f"dim_corr = {dim_info['dimensionality_corr']:.2f}")
    ax.set_xlabel('component')
    ax.set_ylabel('explained variance ratio')
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig, ax


# =============================================================================
# Trial indicator (port of f_plot_trial_indicator3).
# =============================================================================

def f_plot_trial_indicator(ax, trial_ind, colors=None, position='top',
                            height_frac=0.04):
    """
    Draw coloured horizontal bands above (or below) `ax` corresponding to
    a length-T trial-type indicator. 0 → no trial, 1..N → trial type.
    Rendered via a PatchCollection of Rectangles for speed.
    """
    trial_ind = np.asarray(trial_ind).ravel()
    n_t = trial_ind.size
    types = np.unique(trial_ind[trial_ind > 0])
    if types.size == 0:
        return
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = {t: cmap((i % 10) / 10.0) for i, t in enumerate(types)}
    else:
        colors = {t: colors[i % len(colors)] for i, t in enumerate(types)}

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if position == 'top':
        band_y = y1
        band_h = span * height_frac
        new_lim = (y0, y1 + band_h)
    else:
        band_h = span * height_frac
        band_y = y0 - band_h
        new_lim = (y0 - band_h, y1)

    rects, face = [], []
    for t in types:
        idx = np.where(trial_ind == t)[0]
        if idx.size == 0:
            continue
        # group consecutive runs into single rectangles
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        ends = np.concatenate([idx[breaks], [idx[-1]]])
        for s, e in zip(starts, ends):
            rects.append(Rectangle((s, band_y), e - s + 1, band_h))
            face.append(colors[t])

    pc = PatchCollection(rects, facecolors=face, edgecolors='none')
    ax.add_collection(pc)
    ax.set_ylim(new_lim)


# =============================================================================
# Raster + mean (port of f_plot_raster_mean).
# =============================================================================

def f_plot_raster_mean(raster, plot_mean=True, xlabels=None, trial_ind=None,
                       colors=None, cmap='viridis', invert_cmap=False,
                       vmax_pct=99, vmin=None, vmax=None, title=None, fig=None):
    """
    Heatmap of `raster` (n_cells, n_t) above a population-mean trace.
    x-axes shared. Optional trial-indicator bands above raster.
    `vmax_pct` percentile-clips the color scale for brightness (lower = brighter).
    Explicit `vmin`/`vmax` override the percentile scale (fixed, comparable color range).
    """
    raster = np.asarray(raster, dtype=float)
    if invert_cmap:
        raster = -raster
    rn, _imkw = _raster_disp(raster, cmap, vmax_pct, vmin=vmin, vmax=vmax)

    if xlabels is None:
        xlabels = np.arange(raster.shape[1])
    extent = [xlabels[0], xlabels[-1], raster.shape[0] - 0.5, -0.5]

    if not plot_mean:
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 3.5))
        else:
            ax = fig.add_subplot(111)
        ax.imshow(rn, extent=extent, **_imkw)
        ax.set_ylabel('cells')
        ax.set_xlabel('time (frames)')
        if trial_ind is not None:
            f_plot_trial_indicator(ax, trial_ind, colors=colors,
                                   position='top')
        if title:
            ax.set_title(title)
        return fig, [ax]

    if fig is None:
        fig = plt.figure(figsize=(8, 5.5))
    gs = GridSpec(4, 1, figure=fig, hspace=0.08)
    ax_r = fig.add_subplot(gs[:3, 0])
    ax_m = fig.add_subplot(gs[3, 0], sharex=ax_r)
    ax_r.imshow(rn, extent=extent, **_imkw)
    ax_r.set_ylabel('cells')
    ax_r.tick_params(labelbottom=False)
    if trial_ind is not None:
        f_plot_trial_indicator(ax_r, trial_ind, colors=colors, position='top')
    ax_m.plot(xlabels, raster.mean(axis=0), 'k', lw=1)
    ax_m.set_ylabel('population avg')
    ax_m.set_xlabel('time (frames)')
    if title:
        fig.suptitle(title)
    return fig, [ax_r, ax_m]


# =============================================================================
# Per-ensemble detail figure (port of f_plot_ensemble_deets).
# =============================================================================

def f_plot_ensemble_deets(firing_rate, cells_i, trials_i, score_i, coeffs_i,
                          title=None, fig=None, cmap='viridis', vmax_pct=99,
                          vmin=None, vmax=None):
    """
    4-panel per-ensemble figure mirroring MATLAB f_plot_ensemble_deets.

    Panels (GridSpec 3 rows × 11 cols, plus a 5-row outer layout for the
    score trace):
        - main raster: cells in this ensemble, sorted by coeff descending.
        - sorted-coefficient line plot on the right (sharey with raster).
        - mean-of-ensemble trace below raster, pink rectangles at trials.
        - ensemble score trace at the bottom, same pink background.
    All time axes share x.

    Parameters
    ----------
    firing_rate : (n_cells, n_t)
    cells_i : array of cell indices in this ensemble
    trials_i : array of trial / frame indices where ensemble is active
    score_i : (n_t,) — ensemble score
    coeffs_i : (n_cells_in_ensemble,) — loadings of cells_i on the component
    title : str or None
    """
    firing_rate = np.asarray(firing_rate, dtype=float)
    cells_i = np.asarray(cells_i).ravel()
    trials_i = np.asarray(trials_i).ravel()
    score_i = np.asarray(score_i, dtype=float).ravel()
    coeffs_i = np.asarray(coeffs_i, dtype=float).ravel()

    if cells_i.size == 0:
        # empty ensemble — render a stub figure so the loop still works
        if fig is None:
            fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, '(empty ensemble)', ha='center', va='center')
        if title:
            fig.suptitle(title)
        return fig, []

    order = np.argsort(-coeffs_i)
    cells_sorted = cells_i[order]
    raster_ens = firing_rate[cells_sorted, :]
    n_t = firing_rate.shape[1]
    mean_trace = firing_rate[cells_i, :].mean(axis=0)

    if fig is None:
        fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(5, 11, figure=fig, hspace=0.12, wspace=0.6)
    ax_rast = fig.add_subplot(gs[0:3, 0:10])
    ax_coef = fig.add_subplot(gs[0:3, 10], sharey=ax_rast)
    ax_mean = fig.add_subplot(gs[3, 0:10], sharex=ax_rast)
    ax_score = fig.add_subplot(gs[4, 0:10], sharex=ax_rast)

    # raster (percentile-clipped for brightness; vmin/vmax override for a fixed scale)
    rn, _imkw = _raster_disp(raster_ens, cmap, vmax_pct, vmin=vmin, vmax=vmax)
    ax_rast.imshow(rn, extent=[0, n_t, len(cells_sorted) - 0.5, -0.5], **_imkw)
    ax_rast.set_ylabel('cells (sorted by coeff)')
    ax_rast.tick_params(labelbottom=False)

    # coeffs
    ax_coef.plot(coeffs_i[order], np.arange(len(order)), '-ok', ms=4, lw=1.2)
    ax_coef.set_xlabel('coeff')
    ax_coef.invert_yaxis()
    ax_coef.tick_params(labelleft=False)
    ax_coef.axvline(0, color='gray', lw=0.5)
    if coeffs_i.min() >= 0:
        ax_coef.set_xlim(left=0)

    # red background for active (significant) trial frames (group consecutive runs)
    def _trial_bg(ax, idx, color='red', alpha=0.55):
        if idx.size == 0:
            return
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        ends = np.concatenate([idx[breaks], [idx[-1]]])
        for s, e in zip(starts, ends):
            ax.axvspan(s, e + 1, color=color, alpha=alpha, lw=0)

    # mean trace
    _trial_bg(ax_mean, trials_i)
    ax_mean.plot(np.arange(n_t), mean_trace, 'k', lw=1)
    ax_mean.set_ylabel('mean(cells)')
    ax_mean.tick_params(labelbottom=False)

    # ensemble score
    _trial_bg(ax_score, trials_i)
    ax_score.plot(np.arange(n_t), score_i, 'k', lw=1)
    ax_score.set_ylabel('ens score')
    ax_score.set_xlabel('frames')
    if score_i.size:
        smin, smax = score_i.min(), score_i.max()
        ax_score.set_ylim(1.2 * min(smin, 0), 1.2 * max(smax, 1e-6))

    if title:
        fig.suptitle(title)
    return fig, [ax_rast, ax_coef, ax_mean, ax_score]


# =============================================================================
# Component scatter (port of f_plot_comp_scatter).
# =============================================================================

def f_plot_comp_scatter(scores, ens_list, dim=None, plot_mean=True,
                        colors=None, ax=None, title=None,
                        marker_size=20, mean_marker_size=200):
    """
    Scatter cells in coefficient space, coloured by ensemble label.

    Parameters
    ----------
    scores : (n_cells, n_dim) ndarray
        Per-cell coordinates in component space. For NMF, pass `coeffs`
        directly (n_cells × k). The first 2 or 3 columns are used.
    ens_list : (n_cells,) ndarray
        0 = non-responsive (rendered black). 1..K = ensemble labels.
    dim : int or None
        2 or 3. If None, picks `min(3, scores.shape[1])`.
    plot_mean : bool
        Add group-mean markers (* size mean_marker_size).
    """
    scores = np.asarray(scores)
    ens_list = np.asarray(ens_list).ravel()
    n_cols = scores.shape[1]
    if dim is None:
        dim = 2 if n_cols < 3 else 3
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    colors = colors or _DEFAULT_COLORS

    if ax is None:
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)
    else:
        fig = ax.figure

    # non-responsive
    nonresp = ens_list == 0
    if nonresp.any():
        if dim == 3:
            ax.scatter(scores[nonresp, 0], scores[nonresp, 1], scores[nonresp, 2],
                       c='k', s=marker_size, alpha=0.3, label='nonresp')
        else:
            ax.scatter(scores[nonresp, 0], scores[nonresp, 1],
                       c='k', s=marker_size, alpha=0.3, label='nonresp')

    groups = np.sort(np.unique(ens_list[ens_list > 0]))
    for j, g in enumerate(groups):
        m = ens_list == g
        c = colors[j % len(colors)]
        if dim == 3:
            ax.scatter(scores[m, 0], scores[m, 1], scores[m, 2],
                       c=c, s=marker_size, label=f'ens {g}')
            if plot_mean:
                ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(),
                           scores[m, 2].mean(),
                           c=c, marker='*', s=mean_marker_size,
                           edgecolors='k', linewidths=0.8)
        else:
            ax.scatter(scores[m, 0], scores[m, 1],
                       c=c, s=marker_size, label=f'ens {g}')
            if plot_mean:
                ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(),
                           c=c, marker='*', s=mean_marker_size,
                           edgecolors='k', linewidths=0.8)

    ax.set_xlabel('comp 1')
    ax.set_ylabel('comp 2')
    if dim == 3:
        ax.set_zlabel('comp 3')
    ax.legend(fontsize=8, loc='best')
    if title:
        ax.set_title(title)
    return fig, ax


# =============================================================================
# Top-level convenience — mirrors MATLAB driver end-of-script figure set.
# =============================================================================

def f_plot_ens_overview(ens_out, firing_rate, mouse_dset_tag=None,
                        cmap='viridis', vmax_pct=99, vmin=None, vmax=None,
                        ens_table=None, skip_empty=True):
    """
    Reproduce the MATLAB driver's end-of-script figure set:

      1. Sorted-cell raster + mean trace (figure 1).
      2. Per-ensemble detail figures (one per non-empty ensemble).

    Parameters
    ----------
    ens_out : dict
        From f_ensemble_extract.
    firing_rate : (n_cells, n_t)
        Original (or smoothed) firing rates to render.
    mouse_dset_tag : str or None
        Optional prefix used in figure titles.
    skip_empty : bool
        Skip detail figures for ensembles with no cells.

    Returns
    -------
    figs : list of matplotlib Figure
    """
    figs = []
    tag = f"{mouse_dset_tag} — " if mouse_dset_tag else ""
    method = ens_out.get('ensemble_method', 'dred')
    extr = ens_out.get('extraction_method', 'thresh')

    # use the original cells (active_cells_mask) for plotting
    mask = ens_out.get('active_cells_mask',
                       np.ones(firing_rate.shape[0], dtype=bool))
    fr = firing_rate[mask, :]

    # 1. sorted raster
    order = ens_out['ord_cell']
    fig1, _ = f_plot_raster_mean(
        fr[order, :], plot_mean=True, cmap=cmap, vmax_pct=vmax_pct, vmin=vmin, vmax=vmax,
        title=f"{tag}{method} ({extr}) — sorted raster")
    figs.append(fig1)

    # per-ensemble scores + significance keyed by ensemble index, if a table is given
    score_by_ens = {}
    if ens_table is not None:
        for _, r in ens_table.iterrows():
            score_by_ens[int(r['ensemble'])] = r

    # 2. per-ensemble detail
    ens_lists = ens_out['cells']['ens_list']
    trial_lists = ens_out['trials']['ens_list']
    coeffs = ens_out['coeffs']
    scores = ens_out['scores']
    for i, (ci, ti) in enumerate(zip(ens_lists, trial_lists)):
        if skip_empty and len(ci) == 0:
            continue
        _sc = score_by_ens.get(i)
        _scstr = ''
        if _sc is not None:
            if np.isfinite(_sc.get('cv_r2', np.nan)):
                _scstr += f", cv_r2={_sc['cv_r2']:.3f}"
            if np.isfinite(_sc.get('cv_r2_pool', np.nan)):
                _scstr += f", var.expl={_sc['cv_r2_pool']:.1%}"
            if np.isfinite(_sc.get('mp_ratio', np.nan)):
                _scstr += f", MP={_sc['mp_ratio']:.2f}"
            if np.isfinite(_sc.get('cv_test_err', np.nan)):
                _scstr += f", test_err={_sc['cv_test_err']:.2f}"
            if 'sig' in _sc.index:
                _scstr += f"  [{'SIG' if bool(_sc['sig']) else 'n.s.'}"
                _scstr += (f", p={_sc['sig_p']:.3f}]" if np.isfinite(_sc.get('sig_p', np.nan)) else ']')
        fig_i, _ = f_plot_ensemble_deets(
            fr, ci, ti, scores[i, :], coeffs[ci, i],
            title=f"{tag}{method} ensemble {i + 1}  "
                  f"({len(ci)} cells, {len(ti)} active frames{_scstr})",
            cmap=cmap, vmax_pct=vmax_pct, vmin=vmin, vmax=vmax,
        )
        figs.append(fig_i)
    return figs


_ENS_SCORE_PANELS = [   # (column, label, color, per-ensemble-threshold column, constant threshold)
    ('cv_r2',          'cv R² median  (higher=better)',  'seagreen',  'cv_r2_null95', None),
    ('cv_r2_pool',     'cv R² pooled = var. explained',  'mediumseagreen', None, 0.0),
    ('cv_test_err_pn', 'cv test_err ·√n  (lower=better)', 'indianred', 'cv_test_err_null05', None),
    ('cv_test_err',    'cv test_err  (lower=better)',    'indianred', 'cv_test_err_null05', None),
    ('mp_ratio',       'MP λ₁/edge  (>1 ⇒ real)',        'darkorange', None, 1.0),
    ('cv_r2_z',        'cv R² z  (vs shuffle)',          'slateblue', None, 0.0),
    ('n_cells',        '# member cells',                'steelblue', None, None),
    ('frac_active',    'frac active frames',            'mediumpurple', None, None),
]


def f_plot_ens_scores(ens_table, dset_tag='', cols=None, title=None):
    """Per-ensemble summary bars from an f_extract_ensembles table — one panel PER available score
    (cv_r2, size-normalized cv_test_err·√n, MP λ₁/edge, cv_r2 z, # member cells, frac_active; pass
    `cols` to override). x = ensemble index (sorted ascending, stable across runs). Significant
    ensembles (the table's `sig` column) get a '*' on the cv_r2 panel; per-ensemble shuffle-null
    thresholds (cv_r2_null95 / cv_test_err_null05) draw as black ticks and the MP panel draws its
    λ₁/edge=1 line. Returns None (no Spyder Figure echo)."""
    if ens_table is None or len(ens_table) == 0:
        print('f_plot_ens_scores: empty table'); return None
    t = ens_table.sort_values('ensemble').reset_index(drop=True)
    x = np.arange(len(t))
    xl = [str(int(e)) for e in t['ensemble']]
    sig = (t['sig'].to_numpy(dtype=bool) if 'sig' in t.columns else np.zeros(len(t), bool))
    # default: every score panel whose column exists; drop raw cv_test_err if the normalized one is present
    panels = [p for p in _ENS_SCORE_PANELS if (cols is None or p[0] in cols) and p[0] in t.columns]
    if any(p[0] == 'cv_test_err_pn' for p in panels):
        panels = [p for p in panels if p[0] != 'cv_test_err']
    if not panels:
        print('f_plot_ens_scores: no score columns'); return None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.3), squeeze=False)
    for ax, (col, lab, color, nullcol, cthresh) in zip(axes[0], panels):
        vals = t[col].to_numpy(dtype=float)
        ax.bar(x, np.nan_to_num(vals, nan=0.0), color=color)
        ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=8)
        ax.set_xlabel('ensemble'); ax.set_title(lab, fontsize=9); ax.set_ylabel(col)
        ax.axhline(0, color='gray', lw=0.6)
        if cthresh is not None:                                # constant significance line (MP=1, z=0)
            ax.axhline(cthresh, color='k', ls='--', lw=1.0, label='threshold')
            ax.legend(fontsize=7, loc='best')
        if nullcol and nullcol in t.columns and np.isfinite(t[nullcol].to_numpy(float)).any():
            nv = t[nullcol].to_numpy(dtype=float)              # per-ensemble shuffle-null threshold
            for xi, th in zip(x, nv):
                if np.isfinite(th):
                    ax.plot([xi - 0.4, xi + 0.4], [th, th], color='k', lw=1.4,
                            label='shuffle null' if xi == x[0] else None)
            ax.legend(fontsize=7, loc='best')
        if col == 'cv_r2':                                     # star the significant ensembles
            for xi, v, s in zip(x, vals, sig):
                if s and np.isfinite(v):
                    ax.text(xi, v, '*', ha='center', va='bottom', fontsize=12, color='k')
        if col not in ('n_cells', 'frac_active'):              # mark NaN (unscored) ensembles
            for xi, v in zip(x, vals):
                if not np.isfinite(v):
                    ax.text(xi, 0, 'n/a', ha='center', va='bottom', fontsize=7, color='gray')
    _sm = str(ens_table['sig_method'].iloc[0]) if 'sig_method' in ens_table.columns else ''
    fig.suptitle(title or f'{dset_tag} — per-ensemble scores'
                 + (f'  (* = significant, method={_sm})' if _sm else ''))
    fig.tight_layout()
    return None


def f_plot_nmf_stability(res, dset_tag='', title=None):
    """Consensus-NMF stability bars from f_nmf_consensus: per consensus component, the within-cluster
    |corr| (stability), the fraction of runs contributing, and the cosine silhouette — one panel each,
    x = component (stability-sorted). A stable component has stability≈1, runs≈all, high silhouette;
    a run-to-run artifact scores low on all three. Returns None."""
    import numpy as _np
    st = _np.asarray(res['stability'], float); nc = _np.asarray(res['n_contrib'], float)
    nr = float(res['n_runs']); sil = _np.asarray(res['silhouette'], float)
    x = _np.arange(len(st))
    panels = [(st, 'within-cluster |corr| (stability)', 'seagreen', (0, 1.02)),
              (nc / nr, 'fraction of runs contributing', 'steelblue', (0, 1.02)),
              (sil, 'cosine silhouette', 'slateblue', None)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (v, lab, col, ylim) in zip(axes, panels):
        ax.bar(x, _np.nan_to_num(v, nan=0.0), color=col)
        ax.set_xlabel('consensus component'); ax.set_xticks(x); ax.set_title(lab, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(title or f'{dset_tag} — NMF consensus stability (k={len(st)}, {int(nr)} runs, '
                 f'cophenetic={res["cophenetic"]:.3f})')
    fig.tight_layout()
    return None


def f_plot_ens_cv_trace(ens_out, X, ens_table, dimred_params, fs=None, which='sig',
                        cv_method='pca', cv_num_comp=1, frame_t=None, dset_tag='', max_panels=8):
    """Per-ensemble CROSS-VALIDATED reconstruction trace — visualizes what `cv_r2` measures. For each
    selected ensemble, run member-cell leave-one-NEURON-out CV (return_oof) and overlay, over time,
    the mean-member OOF **target** (true co-activation) vs the mean-member OOF **LNO prediction**
    (reconstructed from the other members). Close traces ⇒ high cv_r2. `X` (n_cells, T) = the
    extraction-space matrix (Y_neu_raw.T). `which`: 'sig' (significant rows of ens_table) | 'all' |
    an explicit list of ensemble indices. Significant frames shaded pink. Returns None."""
    from f_ensembles import f_dred_lno_r2
    X = np.asarray(X, float)
    if which == 'sig' and 'sig' in ens_table.columns:
        idx = ens_table.loc[ens_table['sig'].astype(bool), 'ensemble'].astype(int).tolist()
    elif which == 'all':
        idx = list(range(len(ens_out['cells']['ens_list'])))
    else:
        idx = list(which)
    idx = [i for i in idx if len(ens_out['cells']['ens_list'][i]) >= 2][:max_panels]
    if not idx:
        print('f_plot_ens_cv_trace: no eligible ensembles (need ≥2 members)'); return None
    sb = (int(round(dimred_params.get('smooth_ms', 0) * fs / 1000.0))
          if dimred_params.get('smooth_ms') and fs else 0)
    r2_by = {int(r['ensemble']): r.get('cv_r2', np.nan) for _, r in ens_table.iterrows()}
    t = np.asarray(frame_t, float) if (frame_t is not None and len(frame_t) == X.shape[1]) else np.arange(X.shape[1])

    def _shade(ax, ti):                                # pink bands over significant frames
        ti = np.asarray(ti, int)
        if ti.size == 0:
            return
        br = np.where(np.diff(ti) > 1)[0]
        for s, e in zip(np.concatenate([[ti[0]], ti[br + 1]]), np.concatenate([ti[br], [ti[-1]]])):
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color='red', alpha=0.12, lw=0)

    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 1.9 * len(idx)), sharex=True, squeeze=False)
    for ax, i in zip(axes[:, 0], idx):
        members = np.asarray(ens_out['cells']['ens_list'][i], int)
        out = f_dred_lno_r2(X[members], method=cv_method,
                            num_comp=int(min(cv_num_comp, max(1, members.size - 1))),
                            smooth_sigma_bins=sb, normalize=dimred_params.get('normalize', 'norm_mean_std'),
                            k_folds=dimred_params.get('k_folds', 5),
                            fold_scheme=dimred_params.get('fold_scheme', 'contiguous'),
                            n_blocks=dimred_params.get('n_blocks'),
                            norm_scope=dimred_params.get('norm_scope', 'global'),
                            random_state=dimred_params.get('seed', 42), return_oof=True)
        yt = np.nanmean(out['yt'], 0); yh = np.nanmean(out['yh'], 0)     # mean over members
        _shade(ax, ens_out['trials']['ens_list'][i])
        ax.plot(t, yt, 'k', lw=0.8, label='member mean (OOF target)')
        ax.plot(t, yh, 'r', lw=0.8, alpha=0.85, label='LNO OOF prediction')
        ax.set_ylabel(f'ens {i}', fontsize=8)
        ax.set_title(f'ensemble {i}: {members.size} cells, cv_r2={r2_by.get(i, np.nan):.3f}', fontsize=8)
        if ax is axes[0, 0]:
            ax.legend(fontsize=7, loc='upper right')
    axes[-1, 0].set_xlabel('imaging time (s)' if (frame_t is not None and len(frame_t) == X.shape[1]) else 'frame')
    fig.suptitle(f'{dset_tag} — per-ensemble CV reconstruction (member-mean, out-of-fold)')
    fig.tight_layout()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Neural-cleanup (dim-red) diagnostics — moved out of VR_ca_cebra.py.
# ─────────────────────────────────────────────────────────────────────────────

_RASTER_DIVERGING = ('rdbu_r', 'rdbu', 'coolwarm', 'bwr', 'seismic', 'piyg', 'prgn')


def _raster_disp(raster, cmap, vmax_pct=99, vmin=None, vmax=None):
    """Display matrix + imshow color kwargs for a raster, percentile-clipped for brightness
    (mirrors f_plot_dimred_rasters, cell 441). Lower `vmax_pct` = brighter (clips more). A diverging
    cmap → per-row z-score + symmetric scale; otherwise raw values with vmin=0, vmax=percentile.
    Explicit `vmin`/`vmax` (either not None) OVERRIDE the percentile scale for the non-diverging
    branch (e.g. vmin=0, vmax=0.5 for a fixed, comparable, better-looking scale)."""
    raster = np.asarray(raster, float)
    if str(cmap).lower() in _RASTER_DIVERGING:
        mu = raster.mean(1, keepdims=True); sd = raster.std(1, keepdims=True)
        disp = (raster - mu) / np.where(sd == 0, 1.0, sd)
        v = float(np.nanpercentile(np.abs(disp), vmax_pct)) or 1.0
        return disp, dict(cmap=cmap, vmin=-v, vmax=v, aspect='auto', interpolation='none')
    lo = 0.0 if vmin is None else float(vmin)
    hi = (float(np.nanpercentile(raster, vmax_pct)) or 1.0) if vmax is None else float(vmax)
    return raster, dict(cmap=cmap, vmin=lo, vmax=hi, aspect='auto', interpolation='none')


def f_plot_source_raster(Y_neu_raw, Y_neu, source_label='', normalize='norm_mean_std',
                         fs=None, smooth_ms=0, frame_t=None, cmap='viridis', norm_orig=True,
                         norm_sel=True, colorbar=False, dset_tag=''):
    """Two-panel raster of the ORIGINAL raw neural (top) vs the SELECTED neural input `Y_neu` (bottom)
    — the matrix that actually feeds encode/decode/CEBRA, WHATEVER neural_source produced (svd_recon /
    components / reliable_cells / ens_components / ens_cells / raw). NO refit: both matrices are
    plotted as given. `Y_neu_raw` (T, n_cells), `Y_neu` (T, d). Rows are per-row z-scored for display
    (symmetric, so component scores and raw cells both read on one scale; `norm_orig=False` /
    `norm_sel=False` keep the original / selected panel in raw non-negative scale). `cmap` = raster
    colormap, `smooth_ms` optionally blurs the
    original for display only. Returns None."""
    from f_ensembles import f_normalize_rows
    from f_utils import f_gauss_smooth
    R = np.asarray(Y_neu_raw, float).T            # (n_cells, T)
    S = np.asarray(Y_neu, float).T                # (d, T) selected input
    sb = int(round(smooth_ms * fs / 1000.0)) if (smooth_ms and fs) else 0
    if sb:
        R = f_gauss_smooth(R, sigma_frames=sb)

    def _disp(M, force_z):                        # per-row z-score (symmetric) or raw non-neg
        if force_z:
            mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True)
            Mz = (M - mu) / np.where(sd == 0, 1.0, sd)
            v = float(np.nanpercentile(np.abs(Mz), 99)) or 1.0
            return Mz, dict(cmap=cmap, vmin=-v, vmax=v, aspect='auto', interpolation='none')
        v = float(np.nanpercentile(M, 99)) or 1.0
        return M, dict(cmap=cmap, vmin=0.0, vmax=v, aspect='auto', interpolation='none')

    top, topkw = _disp(f_normalize_rows(R, normalize) if norm_orig else R, norm_orig)
    sel, selkw = _disp(S, norm_sel)               # z-score selected (norm_sel=False → raw scale)
    have_t = frame_t is not None and len(frame_t) == R.shape[1]
    t0t1 = [float(frame_t[0]), float(frame_t[-1])] if have_t else None
    mats = [('original' + ('' if norm_orig else ' (raw)'), top, topkw, 'cell'),
            (f'selected: {source_label} ({S.shape[0]} rows)', sel, selkw,
             'component' if 'component' in str(source_label) else 'unit')]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    for ax, (lab, m, kw, yl) in zip(axes, mats):
        ext = (t0t1 + [m.shape[0], 0]) if t0t1 else None
        ax.imshow(m, extent=ext, **kw); ax.set_ylabel(yl); ax.set_title(lab, fontsize=9)
    axes[-1].set_xlabel('imaging time (s)' if have_t else 'frame')
    fig.suptitle(f'{dset_tag} — neural raster: original vs selected input ({source_label})')
    fig.tight_layout()
    if colorbar:
        import matplotlib as _mpl
        _sm = _mpl.cm.ScalarMappable(norm=_mpl.colors.Normalize(vmin=selkw['vmin'], vmax=selkw['vmax']),
                                     cmap=cmap)
        cfig, cax = plt.subplots(figsize=(1.3, 4))
        cfig.colorbar(_sm, cax=cax, label='z-score'); cfig.tight_layout()
    return None


def f_plot_dimred_rasters(X, dimred_params, num_comps=None, mode='svd_recon', fs=None,
                          frame_t=None, cmap='viridis', norm_orig=True, colorbar=False,
                          precomputed=None, dset_tag=''):
    """ORIGINAL neural vs the REDUCED raster(s) for a dim-red config. `X` = (n_cells, T) raw neural
    (cells in the order you want them displayed). Fits the dim-red at each rank in `num_comps`
    (default [dimred_params['num_comp']]) via f_reduce_neural(mode=…) and stacks the original on top
    of one reduced panel per rank, sharing the x-axis (seconds via `frame_t` if given). `mode` picks
    what the lower panels show (matches neural_source):
      'svd_recon'      → low-rank reconstruction, per-cell (already in normalized space)
      'components'     → the k latent scores (k rows; per-row z-scored for display)
      'reliable_cells' → the RAW cells with LNO-R² > r_reliable (per-row z-scored for display)
    `precomputed` : optional dict {num_comp: f_reduce_neural-result} to REUSE an already-fit reduction
    (e.g. the assignment cell's `Y_lr`) instead of re-fitting — ranks not in it are fit as usual. Must
    match `mode`. `norm_orig=False` → original in raw smoothed scale. `cmap`: a diverging name uses a
    symmetric scale (the raw-original panel falls back to viridis). `colorbar=True` → scale as a
    SEPARATE figure. Returns None."""
    from f_ensembles import f_normalize_rows, f_reduce_neural
    from f_utils import f_gauss_smooth
    X = np.asarray(X, float)
    precomputed = precomputed or {}
    num_comps = list(num_comps) if num_comps else [dimred_params['num_comp']]
    sb = (int(round(dimred_params.get('smooth_ms', 0) * fs / 1000.0))
          if dimred_params.get('smooth_ms') and fs else 0)
    Xsm = f_gauss_smooth(X, sigma_frames=sb) if sb else X
    orig = f_normalize_rows(Xsm, dimred_params.get('normalize', 'norm_mean_std'))
    method = dimred_params["method"]

    def _rowz(M):                                  # per-row z-score for display (raw-unit panels)
        mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True)
        return (M - mu) / np.where(sd == 0, 1.0, sd)

    _tag = {'svd_recon': 'recon', 'components': 'components', 'reliable_cells': 'reliable cells'}[mode]
    _yl = 'component' if mode == 'components' else 'cell'
    recons = []                                    # (label, display matrix, ylabel)
    for k in num_comps:
        out = precomputed.get(k)                    # reuse an already-fit reduction if provided
        if out is None:
            out = f_reduce_neural({**dimred_params, 'num_comp': k}, X, mode=mode, fs=fs)
        M = np.asarray(out['low_rank'], float).T                             # (rows, T)
        disp = M if mode == 'svd_recon' else _rowz(M)   # svd_recon already normalized; else z per row
        _reused = ' [reused]' if k in precomputed else ''
        recons.append((f'{method} k={k} {_tag} ({M.shape[0]}){_reused}', disp, _yl))
    div = str(cmap).lower() in _RASTER_DIVERGING
    orig_disp = orig if norm_orig else Xsm
    mats = [('original' + ('' if norm_orig else ' (raw)'), orig_disp, norm_orig, 'cell')] + \
           [(lab, m, True, yl) for lab, m, yl in recons]
    vN = float(np.nanpercentile(np.abs(orig), 99)) or 1.0
    normkw = dict(aspect='auto', interpolation='none', cmap=cmap,
                  vmin=(-vN if div else float(min(0.0, np.nanpercentile(orig, 1)))), vmax=vN)
    rawkw = dict(aspect='auto', interpolation='none', cmap=('viridis' if div else cmap),
                 vmin=0.0, vmax=float(np.nanpercentile(Xsm, 99)) or 1.0)
    t0t1 = ([float(frame_t[0]), float(frame_t[-1])] if frame_t is not None else None)
    xlab = 'imaging time (s)' if frame_t is not None else 'frame'
    fig, axes = plt.subplots(len(mats), 1, figsize=(12, 2.8 * len(mats)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, m, isn, yl) in zip(axes, mats):
        ext_i = (t0t1 + [m.shape[0], 0]) if t0t1 else None   # y-extent per panel (row counts differ)
        ax.imshow(m, extent=ext_i, **(normkw if isn else rawkw))
        ax.set_ylabel(yl); ax.set_title(lab, fontsize=9)
    axes[-1].set_xlabel(xlab)
    fig.suptitle(f'{dset_tag} — neural raster: original vs {_tag}  '
                 f'({method}, {dimred_params["normalize"]})')
    fig.tight_layout()
    if colorbar:                                   # color scale (normalized panels) as its own figure
        import matplotlib as _mpl
        _sm = _mpl.cm.ScalarMappable(norm=_mpl.colors.Normalize(vmin=normkw['vmin'], vmax=normkw['vmax']),
                                     cmap=normkw['cmap'])
        cfig, cax = plt.subplots(figsize=(1.3, 4))
        cfig.colorbar(_sm, cax=cax, label=('z-score' if div else 'a.u.'))
        cfig.tight_layout()
    return None


def f_plot_lno_ceiling(X, dimred_params, fs=None, dset_tag='', ens_lno=None, ens_label='ensembles'):
    """Per-cell leave-one-neuron-out R² ECDF (population noise ceiling) for a dim-red config.
    `X` = (n_cells, T). Runs f_run_dimred → per-cell R²; plots the ECDF with the r_reliable line.
    Optionally overlays a SECOND ECDF `ens_lno` (per-ENSEMBLE LNO R², e.g. ens_table['cv_r2'] =
    each ensemble's within-ensemble member reconstruction) so cell vs ensemble reliability show on
    one axis. Returns the per-cell R² array."""
    from f_ensembles import f_run_dimred
    out = f_run_dimred(dimred_params, np.asarray(X, float), fs=fs)
    v = np.asarray(out['per_cell_r2'], float); v = v[np.isfinite(v)]
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.2))
    xs = np.sort(v)
    ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where='post', color='C0',
            label=f'cells (n={v.size}, med {np.nanmedian(v):+.3f})')
    if ens_lno is not None:                              # overlay per-ensemble LNO ECDF
        e = np.asarray(ens_lno, float); e = e[np.isfinite(e)]
        if e.size:
            xe = np.sort(e)
            ax.step(xe, np.arange(1, xe.size + 1) / xe.size, where='post', color='C1',
                    label=f'{ens_label} (n={e.size}, med {np.nanmedian(e):+.3f})')
    ax.axvline(0, color='gray', lw=0.6)
    ax.axvline(dimred_params['r_reliable'], color='C3', ls='--', lw=0.8,
               label=f"r_reliable={dimred_params['r_reliable']}")
    ax.set_xlabel('leave-one-neuron-out R²'); ax.set_ylabel('cumulative fraction')
    ax.set_ylim(0, 1); ax.legend(fontsize=8, loc='upper left')
    _ttl = (f'{dset_tag} — LNO-R² ECDF: cells vs {ens_label}' if ens_lno is not None
            else f'{dset_tag} — population LNO-R² ceiling (median {np.nanmedian(v):+.3f}, '
                 f'{int((v > dimred_params["r_reliable"]).sum())} reliable cells)')
    ax.set_title(_ttl)
    fig.tight_layout()
    return out['per_cell_r2']


def f_plot_ensemble_fov(A, dims, active_ids, ens_table, dset_tag='', thresh_frac=0.1,
                        cmap='tab10', order='F', gray='0.5', which='sig', ax=None,
                        gamma=0.7, bg_gain=0.9, ens_gain=1.0, outline=False, ax_bg='white',
                        interp='nearest'):
    """Field-of-view map of WEIGHTED cell-ROI projections, ensemble members colored. Each footprint in
    `A` is painted at its actual intensity (normalized per cell), alpha-composited onto a white
    background: every SELECTED cell (`active_ids`) that is NOT in a colored ensemble contributes to a
    GRAY projection; cells in each SIGNIFICANT ensemble contribute to that ensemble's COLOR channel.
      A          — est1['A'], sparse (n_comp, n_pixels); rows align with est1['S'] rows.
      dims       — est1['dims'] (d1, d2) FOV shape; footprints reshaped with `order` ('F' for CaImAn).
      active_ids — 1-D array of true neuron IDs (np.flatnonzero(active_cells)).
      ens_table  — the stats-cell table; 'cells' = true IDs, 'ensemble' = index, 'sig' = significance.
      which      — 'sig' (only significant ensembles) | 'all' (every ensemble).
      thresh_frac— per-cell floor (fraction of the cell's max) zeroed before projecting → cleans tails.
      gamma      — intensity gamma (<1 boosts dim footprint edges); bg_gain/ens_gain scale opacity.
      outline    — also draw a thin footprint outline per ensemble cell (delineates overlaps).
      interp     — imshow interpolation ('nearest' = true pixels/full res; 'bilinear'/'gaussian' smooth).
    Footprints are painted at FULL est1['dims'] resolution (no downsampling). Returns {ens: color}."""
    import matplotlib.colors as mcolors
    d1, d2 = int(dims[0]), int(dims[1])
    Acsr = A.tocsr() if hasattr(A, 'tocsr') else A

    def _foot(cid):                                            # normalized (d1,d2) footprint in [0,1]
        f = np.asarray(Acsr[int(cid)].todense()).ravel() if hasattr(Acsr, 'todense') \
            else np.asarray(Acsr[int(cid)]).ravel()
        mx = f.max()
        if not np.isfinite(mx) or mx <= 0:
            return None
        fm = f.reshape((d1, d2), order=order) / mx
        fm[fm < thresh_frac] = 0.0                             # floor tails
        return fm

    # which ensembles + a color each
    if which == 'all':
        ens_sel = ens_table['ensemble'].astype(int).tolist()
    else:
        ens_sel = ens_table.loc[ens_table['sig'].astype(bool), 'ensemble'].astype(int).tolist()
    cmap_o = plt.get_cmap(cmap)
    colors = {e: cmap_o(k % cmap_o.N) for k, e in enumerate(ens_sel)}
    # cell → ensemble (a member cell painted in its ensemble's color; later ens wins on overlap)
    cell2ens = {}
    for e in ens_sel:
        rowc = ens_table.loc[ens_table['ensemble'] == e, 'cells']
        for cid in (rowc.iloc[0] if len(rowc) else []):
            cell2ens[int(cid)] = e

    active_ids = [int(c) for c in np.asarray(active_ids).ravel()]
    # intensity projections (max across cells within a group so bright somata dominate their own pixels)
    G = np.zeros((d1, d2))                                     # gray (non-ensemble) projection
    ens_int = {e: np.zeros((d1, d2)) for e in ens_sel}
    for cid in active_ids:
        fm = _foot(cid)
        if fm is None:
            continue
        if cid in cell2ens:
            e = cell2ens[cid]; np.maximum(ens_int[e], fm, out=ens_int[e])
        else:
            np.maximum(G, fm, out=G)

    # composite RGB: white → gray layer → ensemble color layers (alpha = intensity**gamma · gain)
    img = np.ones((d1, d2, 3), float)
    gcol = np.array(mcolors.to_rgb(gray))
    a = (np.clip(G, 0, 1) ** gamma)[..., None] * bg_gain
    img = img * (1 - a) + gcol * a
    for e in ens_sel:
        col = np.array(mcolors.to_rgb(colors[e][:3]))
        a = (np.clip(ens_int[e], 0, 1) ** gamma)[..., None] * ens_gain
        img = img * (1 - a) + col * a

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5 * d1 / max(d2, 1)))
    ax.set_facecolor(ax_bg)
    ax.imshow(np.clip(img, 0, 1), origin='upper', interpolation=interp)
    if outline:                                                # thin outline per ensemble cell
        for cid, e in cell2ens.items():
            if cid not in active_ids:
                continue
            fm = _foot(cid)
            if fm is not None and fm.any():
                ax.contour((fm > 0).astype(float), levels=[0.5], colors=[colors[e]], linewidths=0.6)
    handles = [Rectangle((0, 0), 1, 1, fc=colors[e], ec=colors[e]) for e in ens_sel]
    labels = [f'ens {e}' for e in ens_sel]
    if handles:
        ax.legend(handles, labels, fontsize=7, loc='upper right', framealpha=0.9,
                  title=('significant' if which == 'sig' else 'all'))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{dset_tag} — FOV: {len(active_ids)} cells, '
                 f'{len(ens_sel)} {"sig " if which == "sig" else ""}ensembles colored')
    if ax.figure is not None:
        ax.figure.tight_layout()
    return colors
