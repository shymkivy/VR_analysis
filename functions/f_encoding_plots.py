# -*- coding: utf-8 -*-
"""
f_encoding_plots.py — encoding-arm visuals (sub-plan P2, D1). The RF weight-viewer is the
build's CANARY: render a fitted per-neuron weight vector AS A MONITOR FRAME (one frame per
lag = a short weight-movie), so a planted/real receptive field is visible the moment the
encoder fits. Compute lives in f_encoding.py (f_encode_fit_weights / f_rf_pixel_frames /
f_rf_grid_location). Plots return None (Spyder-echo safe). Created 2026-07-03.
"""

import numpy as np


def f_plot_rf(rf, lags, fs=None, cmap='RdBu_r', symmetric=True, title='', max_cols=6):
    """Plot a spatiotemporal RF `(n_lags, H, W)` as ONE frame per lag. Signed weights →
    diverging cmap centered at 0 (red +, blue −); pass symmetric=False + a sequential cmap
    (e.g. 'magma') for a magnitude location map. `lags` = frame lags; `fs` (Hz) → ms titles.
    Returns None."""
    import matplotlib.pyplot as plt
    rf = np.asarray(rf, float)
    n = rf.shape[0]
    vmax = float(np.nanmax(np.abs(rf))) or 1.0
    vmin = -vmax if symmetric else float(np.nanmin(rf))
    if not symmetric:
        vmax = float(np.nanmax(rf)) or 1.0
    ncol = min(max_cols, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes[j // ncol][j % ncol]
        if j < n:
            im = ax.imshow(rf[j], cmap=cmap, vmin=vmin, vmax=vmax)
            lbl = f'{lags[j] / fs * 1000:+.0f} ms' if fs else f'lag {lags[j]}'
            ax.set_title(lbl, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    if n and 'im' in dir():
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    if title:
        fig.suptitle(title)
    return None


def f_plot_rf_cells(fit, cells, hw, fs=None, from_pixels=True, grid=None, n_chan=None,
                    labels=None, suptitle=''):
    """Convenience: RF weight-movie for several cells (rows = cells, cols = lags) in one
    figure. `cells` = kept-cell indices into fit['W']. `from_pixels` → f_rf_pixel_frames
    (needs `hw`); else the feature-grid location map (needs `grid`,`n_chan`). Returns None."""
    import matplotlib.pyplot as plt
    from f_encoding import f_rf_pixel_frames, f_rf_grid_location
    rfs = [f_rf_pixel_frames(fit, c, hw) if from_pixels
           else f_rf_grid_location(fit, c, grid, n_chan) for c in cells]
    n_lag = rfs[0].shape[0]
    vmax = max((float(np.nanmax(np.abs(r))) for r in rfs), default=1.0) or 1.0
    fig, axes = plt.subplots(len(cells), n_lag, figsize=(2.1 * n_lag, 2.1 * len(cells)),
                             squeeze=False)
    for i, rf in enumerate(rfs):
        for j in range(n_lag):
            ax = axes[i][j]
            if from_pixels:
                ax.imshow(rf[j], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(rf[j], cmap='magma', vmin=0, vmax=vmax)
            if i == 0:
                lbl = f'{fit["lags"][j] / fs * 1000:+.0f}ms' if fs else f'lag {fit["lags"][j]}'
                ax.set_title(lbl, fontsize=8)
            if j == 0:
                ax.set_ylabel((labels[i] if labels else f'cell {cells[i]}'), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return None
