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
                       title=None, fig=None):
    """
    Heatmap of `raster` (n_cells, n_t) above a population-mean trace.
    x-axes shared. Optional trial-indicator bands above raster.
    """
    raster = np.asarray(raster, dtype=float)
    if invert_cmap:
        raster = -raster
    rmin, rmax = raster.min(), raster.max()
    rn = (raster - rmin) / max(rmax - rmin, 1e-12)

    if xlabels is None:
        xlabels = np.arange(raster.shape[1])
    extent = [xlabels[0], xlabels[-1], raster.shape[0] - 0.5, -0.5]

    if not plot_mean:
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 3.5))
        else:
            ax = fig.add_subplot(111)
        ax.imshow(rn, aspect='auto', cmap=cmap, interpolation='none',
                  extent=extent)
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
    ax_r.imshow(rn, aspect='auto', cmap=cmap, interpolation='none',
                extent=extent)
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
                          title=None, fig=None, cmap='viridis'):
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

    # raster
    rmin, rmax = raster_ens.min(), raster_ens.max()
    rn = (raster_ens - rmin) / max(rmax - rmin, 1e-12)
    ax_rast.imshow(rn, aspect='auto', cmap=cmap, interpolation='none',
                   extent=[0, n_t, len(cells_sorted) - 0.5, -0.5])
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

    # pink background for active trial frames (group consecutive runs)
    def _trial_bg(ax, idx, color='pink', alpha=0.35):
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
                        cmap='viridis', skip_empty=True):
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
        fr[order, :], plot_mean=True, cmap=cmap,
        title=f"{tag}{method} ({extr}) — sorted raster")
    figs.append(fig1)

    # 2. per-ensemble detail
    ens_lists = ens_out['cells']['ens_list']
    trial_lists = ens_out['trials']['ens_list']
    coeffs = ens_out['coeffs']
    scores = ens_out['scores']
    for i, (ci, ti) in enumerate(zip(ens_lists, trial_lists)):
        if skip_empty and len(ci) == 0:
            continue
        fig_i, _ = f_plot_ensemble_deets(
            fr, ci, ti, scores[i, :], coeffs[ci, i],
            title=f"{tag}{method} ensemble {i + 1}  "
                  f"({len(ci)} cells, {len(ti)} active frames)",
            cmap=cmap,
        )
        figs.append(fig_i)
    return figs
