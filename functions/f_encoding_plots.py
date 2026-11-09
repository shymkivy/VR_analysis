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


# ─────────────────────────────────────────────────────────────────────────────
# Feature-tuning visuals (reverse-correlation on conv Gabor/meng filters, f_tuning.py).
# `readout` = f_tuning.f_tuning_readout output; `u` = unit position (kept order, aligned to W).
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_tuning_heatmap(readout, u, dset_tag='', cmap='magma', zscore=False, axis='auto'):
    """One unit's tuning as an orientation × (spatial- or temporal-freq) grid, |tuning| at the unit's
    peak lag. `axis`: 'sf' → orientation × spatial-freq (at the pref TF for meng); 'tf' → orientation
    × temporal-freq (at the pref SF); 'auto' → SF for gabor, TF for meng. The unfixed axis is held at
    the unit's preferred value. `zscore=True` z-scores the profile ACROSS filters (grid in SD units,
    diverging 0-centered) so cross-frequency magnitude differences don't bias the picture."""
    import matplotlib.pyplot as plt
    W = readout['W']; fdf = readout['filt_df']; pl = int(readout['peak_lag_idx'][u])
    tun = np.abs(W[:, pl, u])
    if zscore:                                           # z-score across filters (scale-free)
        _sd = tun.std()
        tun = (tun - tun.mean()) / _sd if _sd > 0 else tun * 0.0
    od = fdf['odeg'].to_numpy(float)
    # SF/TF axes in PHYSICAL units (cyc/deg + Hz) when present, matching the readout's uniq_* axes.
    sfx = fdf['sf_cyc_deg'].to_numpy(float) if 'sf_cyc_deg' in fdf.columns else fdf['wavelength'].to_numpy(float)
    tfx = fdf['tf_hz'].to_numpy(float) if 'tf_hz' in fdf.columns else fdf['tfreq'].to_numpy(float)
    sf_label = readout.get('sf_label', 'wavelength (px)'); tf_label = readout.get('tf_label', 'tfreq')
    uo = readout['uniq_odeg']; has_tf = np.any(~np.isnan(tfx))
    r = readout['table'].iloc[u]
    sf_pref = float(r.get('pref_sf_cyc_deg', r['pref_wavelength']))
    tf_pref = float(r.get('pref_tf_hz', r.get('pref_tfreq', np.nan)))
    _axis = 'sf' if not has_tf else ('tf' if axis == 'auto' else axis)
    if _axis == 'sf':                                    # orientation × spatial-freq (@ pref TF if meng)
        uw = readout['uniq_wavelength']
        G = np.full((len(uo), len(uw)), np.nan)
        for i, o in enumerate(uo):
            for j, w in enumerate(uw):
                m = (od == o) & (sfx == w)
                if has_tf:
                    m = m & (tfx == tf_pref)
                if m.any():
                    G[i, j] = tun[m].mean()
        xlab = sf_label + (f' (@ tf={tf_pref:g})' if has_tf else ''); xt = uw
    else:                                                # orientation × temporal-freq (@ pref SF)
        ut = readout['uniq_tfreq']
        G = np.full((len(uo), len(ut)), np.nan)
        for i, o in enumerate(uo):
            for j, t in enumerate(ut):
                m = (od == o) & (tfx == t) & (sfx == sf_pref)
                if m.any():
                    G[i, j] = tun[m].mean()
        xlab, xt = f'{tf_label} (@ sf={sf_pref:g})', ut
    fig, ax = plt.subplots(figsize=(5, 4))
    _kw = {}                                             # z-scored → diverging, symmetric about 0
    if zscore:
        _vm = float(np.nanmax(np.abs(G))) or 1.0
        _kw = dict(vmin=-_vm, vmax=_vm)
    im = ax.imshow(G, aspect='auto', cmap=('RdBu_r' if zscore else cmap), origin='lower',
                   interpolation='none', **_kw)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels([f'{v:g}' for v in xt]); ax.set_xlabel(xlab)
    ax.set_yticks(range(len(uo))); ax.set_yticklabels([f'{o:g}' for o in uo]); ax.set_ylabel('orientation (deg)')
    fig.colorbar(im, ax=ax, label='|tuning| z-score' if zscore else '|tuning|')
    _mm = f' [{readout.get("method", "")}]' if readout.get("method") else ''
    _sfp = (f'sf{r["pref_sf_cyc_deg"]:.3g}cyc/deg' if 'pref_sf_cyc_deg' in r
            else f'wl{r["pref_wavelength"]:g}')
    _tfp = f' tf{r["pref_tf_hz"]:g}Hz' if ('pref_tf_hz' in r and not np.isnan(r["pref_tf_hz"])) else ''
    ax.set_title(f'{dset_tag} unit {r["unit"]}: pref {r["pref_odeg"]:.0f}° {_sfp}{_tfp} '
                 f'lag{r["peak_lag_ms"]:.0f}ms OSI{r["osi"]:.2f}{_mm}', fontsize=8)
    fig.tight_layout()
    return None


def f_plot_tuning_marginals(readout, u=None, dset_tag=''):
    """Marginal tuning curves (|tuning| vs orientation / wavelength / temporal-freq) for unit `u`
    (kept order) or the population mean (`u=None`). Returns None."""
    import matplotlib.pyplot as plt
    uo, uw, ut = readout['uniq_odeg'], readout['uniq_wavelength'], readout['uniq_tfreq']

    def _get(name):
        A = readout[name]
        if A is None or not np.size(A):
            return None
        return A[u] if u is not None else np.nanmean(A, axis=0)

    have_tf = readout['by_tfreq'] is not None
    n = 2 + (1 if have_tf else 0)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), squeeze=False)
    axes = axes[0]
    sf_label = readout.get('sf_label', 'wavelength (px)'); tf_label = readout.get('tf_label', 'temporal freq')
    axes[0].plot(uo, _get('by_odeg'), '-o'); axes[0].set_xlabel('orientation (deg)')
    axes[0].set_ylabel('|tuning|'); axes[0].set_title('orientation')
    axes[1].plot(uw, _get('by_wavelength'), '-o'); axes[1].set_xlabel(sf_label)
    axes[1].set_title('spatial freq')
    if have_tf:
        axes[2].plot(ut, _get('by_tfreq'), '-o'); axes[2].set_xlabel(tf_label)
        axes[2].set_title('temporal freq')
    _who = 'population mean' if u is None else f"unit {readout['table'].iloc[u]['unit']}"
    _mm = f' [{readout.get("method", "")}]' if readout.get("method") else ''
    fig.suptitle(f'{dset_tag} — tuning marginals ({_who}){_mm}')
    fig.tight_layout()
    return None


def f_plot_tuning_marginals_all(readout, units=None, dset_tag='', normalize=True, max_legend=15):
    """Marginal tuning curves (|tuning| vs orientation / spatial-freq / temporal-freq) overlaid, ONE
    LINE PER UNIT (e.g. every significant ensemble) + the population mean in bold. `units` = kept-unit
    indices (else all in the readout). `normalize=True` peak-normalizes each curve per variable so
    tuning SHAPE compares regardless of magnitude. Returns None."""
    import matplotlib.pyplot as plt
    uo, uw, ut = readout['uniq_odeg'], readout['uniq_wavelength'], readout['uniq_tfreq']
    tbl = readout['table']
    if units is None:
        units = list(range(tbl.shape[0]))
    sf_label = readout.get('sf_label', 'wavelength (px)'); tf_label = readout.get('tf_label', 'temporal freq')
    have_tf = readout['by_tfreq'] is not None and np.size(readout['by_tfreq'])
    panels = [('by_odeg', uo, 'orientation (deg)', 'orientation'),
              ('by_wavelength', uw, sf_label, 'spatial freq')]
    if have_tf:
        panels.append(('by_tfreq', ut, tf_label, 'temporal freq'))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.4), squeeze=False)
    axes = axes[0]

    def _norm(y):
        m = np.nanmax(np.abs(y))
        return (y / m if m > 0 else y) if normalize else y

    for ai, (key, xax, xlab, title) in enumerate(panels):
        A = np.asarray(readout[key], float)                   # (n_units, n_level)
        for u in units:
            axes[ai].plot(xax, _norm(A[u]), '-', lw=0.8, alpha=0.45, label=str(tbl.iloc[u]['unit']))
        axes[ai].plot(xax, _norm(np.nanmean(A[units], axis=0)), 'k-', lw=2.4, label='mean')
        axes[ai].set_xlabel(xlab); axes[ai].set_title(title)
        axes[ai].set_ylabel('norm |tuning|' if normalize else '|tuning|')
    if len(units) <= max_legend:
        axes[-1].legend(fontsize=6, ncol=2, title='unit')
    _mm = f' [{readout.get("method", "")}]' if readout.get("method") else ''
    fig.suptitle(f'{dset_tag} — per-unit tuning marginals ({len(units)} units){_mm}')
    fig.tight_layout()
    return None


def f_plot_tuning_population(table, dset_tag='', method=''):
    """Population tuning summary: preferred-orientation histogram, preferred-wavelength (and -tfreq)
    bars, OSI ECDF, peak-lag histogram. `table` = readout['table']. Returns None."""
    import matplotlib.pyplot as plt
    # prefer PHYSICAL pref columns (cyc/deg + Hz) when present, else the legacy px / cyc-frame
    sf_col = 'pref_sf_cyc_deg' if 'pref_sf_cyc_deg' in table.columns else 'pref_wavelength'
    tf_col = 'pref_tf_hz' if 'pref_tf_hz' in table.columns else 'pref_tfreq'
    sf_lab = 'pref SF (cyc/deg)' if sf_col == 'pref_sf_cyc_deg' else 'wavelength (px)'
    tf_lab = 'pref TF (Hz)' if tf_col == 'pref_tf_hz' else 'temporal freq'
    has_tf = tf_col in table.columns and table[tf_col].notna().any()
    n = 4 + (1 if has_tf else 0)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    axes = axes[0]
    axes[0].hist(table['pref_odeg'].dropna(), bins=np.arange(-22.5, 203, 45))
    axes[0].set_xlabel('pref orientation (deg)'); axes[0].set_title('preferred orientation')
    vw = table[sf_col].value_counts().sort_index()
    axes[1].bar(range(len(vw)), vw.values, color='steelblue')
    axes[1].set_xticks(range(len(vw))); axes[1].set_xticklabels([f'{w:g}' for w in vw.index])
    axes[1].set_xlabel(sf_lab); axes[1].set_title('preferred spatial freq')
    xs = np.sort(table['osi'].dropna().values)
    if xs.size:
        axes[2].plot(xs, np.linspace(0, 1, xs.size))
    axes[2].set_xlabel('OSI'); axes[2].set_ylabel('ECDF'); axes[2].set_title('orientation selectivity')
    axes[3].hist(table['peak_lag_ms'].dropna(), color='slateblue')
    axes[3].set_xlabel('peak lag (ms)'); axes[3].set_title('peak lag')
    if has_tf:
        vt = table[tf_col].dropna().value_counts().sort_index()
        axes[4].bar(range(len(vt)), vt.values, color='darkorange')
        axes[4].set_xticks(range(len(vt))); axes[4].set_xticklabels([f'{t:g}' for t in vt.index])
        axes[4].set_xlabel(tf_lab); axes[4].set_title('preferred temporal freq')
    fig.suptitle(f'{dset_tag} — population tuning ({len(table)} units)'
                 + (f' [{method}]' if method else ''))
    fig.tight_layout()
    return None


def f_plot_tuning_compare(readout_sta, readout_ridge, dset_tag=''):
    """STA-vs-ridge agreement (unit-aligned): preferred orientation, preferred wavelength, and OSI
    scatters. Ridge de-correlates the filters → usually higher OSI (sharper). Returns None."""
    import matplotlib.pyplot as plt
    ts, tr = readout_sta['table'], readout_ridge['table']
    m = min(len(ts), len(tr))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].scatter(ts['pref_odeg'][:m], tr['pref_odeg'][:m], s=10)
    axes[0].plot([0, 180], [0, 180], 'k--', lw=0.6)
    axes[0].set_xlabel('STA pref odeg'); axes[0].set_ylabel('ridge pref odeg'); axes[0].set_title('preferred orientation')
    _sfc = 'pref_sf_cyc_deg' if 'pref_sf_cyc_deg' in ts.columns else 'pref_wavelength'
    _sl = 'pref SF (cyc/deg)' if _sfc == 'pref_sf_cyc_deg' else 'pref wl (px)'
    axes[1].scatter(ts[_sfc][:m], tr[_sfc][:m], s=10)
    axes[1].set_xlabel(f'STA {_sl}'); axes[1].set_ylabel(f'ridge {_sl}'); axes[1].set_title('preferred spatial freq')
    axes[2].scatter(ts['osi'][:m], tr['osi'][:m], s=10)
    axes[2].plot([0, 1], [0, 1], 'k--', lw=0.6)
    axes[2].set_xlabel('STA OSI'); axes[2].set_ylabel('ridge OSI'); axes[2].set_title('OSI (ridge sharpens)')
    fig.suptitle(f'{dset_tag} — STA vs ridge tuning')
    fig.tight_layout()
    return None


def f_plot_tuning_lags(readout, units=None, dset_tag='', max_units=8):
    """Tuning strength as a function of LAG (calcium lags the stimulus). Panel 1: max_filter|W| vs
    lag_ms for a few top units (by peak_abs) + the population mean (bold), each unit's peak lag marked.
    Panel 2: population lag×orientation heatmap (mean_units mean_filter|W| per orientation, per lag) so
    you see whether the preferred orientation drifts with lag. `units` = explicit kept-unit indices
    (else the top `max_units` by peak_abs). Returns None."""
    import matplotlib.pyplot as plt
    by_lag = readout.get('by_lag'); lags_ms = np.asarray(readout.get('lags_ms'), float)
    if by_lag is None or not np.size(by_lag):
        return None
    by_lag = np.asarray(by_lag, float)                        # (n_units, n_lags)
    n_u = by_lag.shape[0]
    tbl = readout['table']
    if units is None:
        order = np.argsort(tbl['peak_abs'].to_numpy(float))[::-1]
        units = list(order[:min(max_units, n_u)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for u in units:
        lab = tbl.iloc[u]['unit']
        axes[0].plot(lags_ms, by_lag[u], '-o', ms=3, lw=0.8, alpha=0.6, label=str(lab))
        pl = int(readout['peak_lag_idx'][u])
        axes[0].plot(lags_ms[pl], by_lag[u, pl], 'k*', ms=9)
    pop = np.nanmean(by_lag, axis=0)
    axes[0].plot(lags_ms, pop, 'k-', lw=2.2, label='pop mean')
    axes[0].set_xlabel('lag (ms)'); axes[0].set_ylabel('max|tuning|')
    axes[0].set_title(f'tuning strength vs lag ({len(units)} units)')
    if len(units) <= 12:
        axes[0].legend(fontsize=6, ncol=2)

    # lag × orientation heatmap (population): mean over units of per-orientation mean|W| at each lag
    W = np.abs(np.asarray(readout['W'], float))               # (n_filter, n_lags, n_units)
    uo = readout['uniq_odeg']; od = readout['filt_df']['odeg'].to_numpy(float)
    if uo.size:
        Wm = W.mean(axis=2)                                   # (n_filter, n_lags)
        LO = np.array([Wm[od == o].mean(axis=0) for o in uo])  # (n_odeg, n_lags)
        im = axes[1].imshow(LO, aspect='auto', origin='lower', cmap='magma',
                            extent=[lags_ms.min(), lags_ms.max(), 0, uo.size])
        axes[1].set_yticks(np.arange(uo.size) + 0.5)
        axes[1].set_yticklabels([f'{o:g}' for o in uo])
        axes[1].set_xlabel('lag (ms)'); axes[1].set_ylabel('orientation (deg)')
        axes[1].set_title('pop lag × orientation')
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    _mm = f' [{readout.get("method", "")}]' if readout.get("method") else ''
    fig.suptitle(f'{dset_tag} — tuning vs lag{_mm}')
    fig.tight_layout()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tuning verification views — all-filters overview + monitor-space maps (location / STA).
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_tuning_allfilters(readout, u, dset_tag='', zscore=False):
    """Per-unit tuning across EVERY filter: |tuning| at the unit's peak lag, one bar per filter
    (labeled o{deg}wl{λ}[tf]), the SELECTED (preferred) filter in red. Shows the whole profile so
    you see how peaked/broad the tuning is and exactly which filter won. `zscore=True` plots the
    profile z-scored ACROSS filters (peak in SD units — scale-free, so cross-frequency magnitude
    differences don't bias the comparison). Returns None."""
    import matplotlib.pyplot as plt
    W = readout['W']; fdf = readout['filt_df']; pl = int(readout['peak_lag_idx'][u])
    tun = np.abs(W[:, pl, u]); n = len(tun)
    if zscore:                                            # z-score across filters (peak vs the distribution)
        _sd = tun.std()
        tun = (tun - tun.mean()) / _sd if _sd > 0 else tun * 0.0
    _has_sf = 'sf_cyc_deg' in fdf.columns; _has_hz = 'tf_hz' in fdf.columns   # physical labels if present
    labs = []
    for _, r in fdf.iterrows():
        s = f"o{int(r['odeg'])}" + (f"sf{r['sf_cyc_deg']:g}" if _has_sf else f"wl{r['wavelength']:g}")
        _t = r.get('tf_hz', np.nan) if _has_hz else r.get('tfreq', np.nan)
        if not np.isnan(_t):
            s += (f"tf{_t:g}Hz" if _has_hz else f"tf{_t:g}")
        labs.append(s)
    pref = int(np.argmax(tun))
    colors = ['crimson' if i == pref else 'steelblue' for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, 0.24 * n), 3.6))
    ax.bar(range(n), tun, color=colors)
    if zscore:
        ax.axhline(0, color='gray', lw=0.6)
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_ylabel('|tuning| z-score' if zscore else '|tuning|')
    rr = readout['table'].iloc[u]
    _mm = f' [{readout.get("method", "")}]' if readout.get("method") else ''
    _pz = f", peak_z {rr['peak_z']:.1f}" if 'peak_z' in rr.index and np.isfinite(rr.get('peak_z', np.nan)) else ''
    _sfp = (f'sf{rr["pref_sf_cyc_deg"]:.3g}' if 'pref_sf_cyc_deg' in rr.index else f'wl{rr["pref_wavelength"]:g}')
    ax.set_title(f'{dset_tag} unit {rr["unit"]} — all filters (red = selected: o{rr["pref_odeg"]:.0f} '
                 f'{_sfp}, lag {rr["peak_lag_ms"]:.0f}ms{_pz}){_mm}', fontsize=8)
    fig.tight_layout()
    return None


def f_plot_tuning_monitor(M, dset_tag='', title='', diverging=False, n_monitors=None,
                          labels=None, monitor_aspect=16 / 9, zscore=False):
    """Show a monitor-space map (ny×nx single monitor, or the wide L‖R map from side='both') as a
    heatmap with the peak marked. Each monitor is forced to `monitor_aspect` (16:9 W:H) regardless of
    the data lattice, outlined with a black rectangle, and labeled in its top-left corner (`labels`,
    default ['L','R'] for two monitors). `n_monitors` (default: infer 2 if the map is wide, else 1)
    splits the width into equal panels. `diverging=True` → signed (corr), symmetric RdBu_r; else
    magnitude (magma). `zscore=True` → z-score the map across pixels (SD units, forces diverging) so
    absolute magnitude doesn't set the scale. Returns None."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    M = np.asarray(M, float)
    if zscore:                                            # z-score across pixels → scale-free, signed
        _f = np.isfinite(M); _sd = float(M[_f].std()) if _f.any() else 0.0
        if _sd > 0:
            M = (M - float(M[_f].mean())) / _sd
        diverging = True
    nrows, ncols = M.shape
    if n_monitors is None:
        n_monitors = 2 if ncols > 1.5 * nrows else 1
    per_nx = ncols / n_monitors
    if labels is None:
        labels = ['L', 'R'][:n_monitors] if n_monitors == 2 else ['']
    fig, ax = plt.subplots(figsize=(min(15, 2.4 * monitor_aspect * n_monitors + 1.6), 4.0))
    if diverging:
        v = float(np.nanpercentile(np.abs(M), 99)) or 1.0
        im = ax.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
    else:
        im = ax.imshow(M, cmap='magma', interpolation='nearest')
    # force each monitor to `monitor_aspect` (W:H) on screen: data-aspect A = per_nx/(nrows·aspect)
    ax.set_aspect((per_nx / nrows) / monitor_aspect)
    for m in range(n_monitors):                               # black outline + corner label per monitor
        x0 = m * per_nx - 0.5
        ax.add_patch(Rectangle((x0, -0.5), per_nx, nrows, fill=False, edgecolor='k', lw=1.8))
        if m < len(labels) and labels[m]:
            ax.text(x0 + 0.5, -0.5 + 0.5, labels[m], color='k', fontsize=11, fontweight='bold',
                    ha='left', va='top',
                    bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.7))
    pk = np.unravel_index(int(np.nanargmax(np.abs(M))), M.shape)
    ax.plot(pk[1], pk[0], 'c+', ms=13, mew=2)
    ax.set_xlim(-0.5, ncols - 0.5); ax.set_ylim(nrows - 0.5, -0.5)   # tight, origin='upper'
    fig.colorbar(im, ax=ax, fraction=0.025, label=('z-score' if zscore else None))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{dset_tag} {title}', fontsize=9)
    fig.tight_layout()
    return None


def f_plot_sta_triggers(activity, frame_t=None, thresh_pct=90, dset_tag='', title=''):
    """STA 'trigger' view: the unit's activity trace with the high-activity frames that DOMINATE the
    STA marked in red. The STA weights EVERY frame by its rectified activity (continuous, not a hard
    spike set), so the top-(100-`thresh_pct`)% frames contribute most — this shows what the average is
    effectively built from. The dashed line is the threshold. Returns None."""
    import matplotlib.pyplot as plt
    a = np.asarray(activity, float); T = a.size
    use_t = frame_t is not None and len(frame_t) == T
    t = np.asarray(frame_t, float) if use_t else np.arange(T)
    w = np.clip(a, 0, None)                                    # the STA weight (rectified activity)
    thr = float(np.percentile(w[w > 0], thresh_pct)) if np.any(w > 0) else 0.0
    hi = w >= thr
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, a, 'k', lw=0.5, label='activity')
    ax.plot(t[hi], a[hi], 'r.', ms=3, label=f'STA triggers (top {100 - thresh_pct:.0f}%)')
    ax.axhline(thr, color='r', lw=0.6, ls='--')
    ax.set_xlabel('imaging time (s)' if use_t else 'frame'); ax.set_ylabel('activity')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(title or f'{dset_tag} — STA trigger frames ({int(hi.sum())}/{T} above {thresh_pct}th pct)')
    fig.tight_layout()
    return None


def f_plot_encode_sweep(df, metric='r_med', x=None, line=None, band=True, logx='auto', title=''):
    """Line plot of an f_encoding.f_encode_sweep table: `metric` vs the `x` axis, one line per value
    of `line`. Defaults: x = first column of df, line = the second sweep axis, else 'target' when there
    are several targets. `band` shades p25–p75 across units for 'r_med' / 'r2_med'. Non-numeric x
    (lists, names) is plotted at even positions with its values as tick labels. `logx`: 'auto' (log
    when x > 0 spans ≥ 2 decades) | True | False. Returns None."""
    import matplotlib.pyplot as plt
    if df is None or len(df) == 0:
        print('nothing to plot'); return None
    axis_cols = list(df.columns[:df.columns.get_loc('target') + 1])      # sweep axes come first
    score_cols = set(df.columns) - set(axis_cols) - {'n_valid', 'enc_tag'}
    x = x or axis_cols[0]
    if line is None:
        others = [c for c in axis_cols if c != x]
        line = next((c for c in others if c != 'target'), None)
        if line is None and df['target'].nunique() > 1:
            line = 'target'
    if metric not in df:
        raise ValueError(f'metric {metric!r} not in the table; have {sorted(score_cols)}')
    xs = list(dict.fromkeys(map(str, df[x])))
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                  for v in df[x])
    lo, hi = {'r_med': ('r_p25', 'r_p75'), 'r2_med': ('r2_p25', 'r2_p75')}.get(metric, (None, None))
    fig, a = plt.subplots(figsize=(8, 5))
    groups = [(None, df)] if line is None else list(df.groupby(line, sort=False))
    for lv, g in groups:
        if numeric:
            g = g.sort_values(x)
            xv = g[x].to_numpy(float)
        else:
            xv = np.array([xs.index(str(v)) for v in g[x]], float)
        yv = g[metric].to_numpy(float)
        a.plot(xv, yv, marker='o', ms=4, label=(f'{line}={lv}' if line else metric))
        if band and lo in g and hi in g:
            a.fill_between(xv, g[lo].to_numpy(float), g[hi].to_numpy(float), alpha=0.15)
    if not numeric:
        a.set_xticks(range(len(xs)))
        a.set_xticklabels(xs, rotation=30, ha='right', fontsize=8)
    elif logx is True or (logx == 'auto' and (df[x] > 0).all()
                          and df[x].max() / max(df[x].min(), 1e-30) >= 100):
        a.set_xscale('log')
    a.set_xlabel(x)
    a.set_ylabel(metric + ('   (band = p25–p75 over units)' if band and lo in df else ''))
    a.axhline(0, color='gray', lw=0.6)
    a.grid(alpha=0.3)
    if line:
        a.legend(fontsize=8)
    a.set_title(title or f'encoding sweep: {metric} vs {x}')
    fig.tight_layout()
    return None
