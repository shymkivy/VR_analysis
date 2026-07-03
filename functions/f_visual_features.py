# -*- coding: utf-8 -*-
"""
Visual-cortex-style monitor feature banks (compute).

Builds features from the rendered monitor movie (`two_mon_frames` / per-side
`*_mon_frames`, ~101×101 per monitor in ANGULAR coords) that are closer to how
visual cortex encodes the scene: location-selective (retinotopic), edge/
orientation-selective, and direction-selective. Replaces / augments the
object-center `agg` block and the raw-pixel-PCA `pix` block.

Design (see PLAN_monitor_features.md, TODO #47):
    filter -> nonlinearity -> spatial/temporal pool -> compress -> linear decode

This module owns the COMPUTE half (extractors + a registry-driven build driver).
Visualization will live in a separate feature-analysis module (planned).

────────────────────────────────────────────────────────────────────────────
Extractor interface (every feature type implements this)
────────────────────────────────────────────────────────────────────────────
    def f_<name>_features(movie, **params) -> (X, names)
        movie : ndarray (T, H, W)   one monitor's movie on the BEHAVIOR clock,
                                    temporally ordered (so motion/flow extractors
                                    can diff consecutive frames). uint8/float ok.
        returns
            X     : ndarray (T, d)  features on the SAME (behavior) clock.
            names : list[str] (d)   base channel names (no side/type prefix;
                                    the build driver adds those).

Extractors are PURE spatial/spatiotemporal ops — no clock alignment, no PCA,
no per-side logic. The driver `build_visual_blocks` handles per-side iteration,
resampling to the imaging clock, and optional PCA compression, and returns
blocks in the same shape as `build_feature_blocks` (`{'X','names',...}`) so they
merge straight into `built_blocks`.

New feature types plug in by (1) writing an extractor and (2) adding one entry
to VISUAL_FEATURE_REGISTRY — that registry is the seed of the eventual
feature-selection module.

Created 2026-06-06 — Tier 1 (grid). Tiers 2 (flow), 3 (gabor; +optional phase) and
4a (meng motion energy) done; Tier 4b (cnn) registers here when implemented.
"""

import time

import numpy as np

from sklearn.decomposition import PCA

from f_feature_helpers import f_resample_to_imaging


# ─────────────────────────────────────────────────────────────────────────────
# Shared spatial helper — retinotopic grid (reused by every pooling extractor)
# ─────────────────────────────────────────────────────────────────────────────

def f_grid_bounds(H, W, grid):
    """Integer row/col cell boundaries for an (ny, nx) tiling of an H×W frame.

    Returns (ys, xs) with len ny+1 / nx+1; cell (iy,ix) spans
    rows ys[iy]:ys[iy+1], cols xs[ix]:xs[ix+1]. Edge cells absorb the remainder.
    """
    ny, nx = grid
    ys = np.linspace(0, H, ny + 1).astype(int)
    xs = np.linspace(0, W, nx + 1).astype(int)
    return ys, xs


def _pool_sum(F, ys, xs):
    """Sum a 2-D field F over the (ny, nx) cells defined by bounds ys, xs.

    Vectorized block-reduce via np.add.reduceat (handles uneven edge cells).
    Returns (ny, nx). Divide by cell area for the mean.
    """
    return np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=0), xs[:-1], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — retinotopic grid pooling (#48)
# ─────────────────────────────────────────────────────────────────────────────

def f_grid_features(movie, grid=(6, 6), stats=('occupancy', 'mean', 'edge')):
    """Per-cell summary statistics over a retinotopic grid — Tier 1.

    Converts the single object-CENTER representation into a population "what's in
    each part of the visual field." For each grid cell, one channel per requested
    statistic:
      'occupancy' — fraction of cell pixels that are non-zero (object/terrain
                    present in that part of the field).
      'mean'      — mean pixel intensity (luminance) in the cell.
      'edge'      — mean Sobel-gradient magnitude in the cell (local edge/contrast
                    energy). Computed per cell (no full-frame gradient → memory-
                    light).

    Channel order is cell-major (all stats for cell (0,0), then (0,1), ...).
    Column names: e.g. 'occ_r0c0', 'mean_r0c0', 'edge_r0c0', 'occ_r0c1', ...

    Parameters
    ----------
    movie : ndarray (T, H, W)   one monitor's movie (behavior clock).
    grid : (ny, nx)             tiling resolution.
    stats : tuple of str        subset of {'occupancy', 'mean', 'edge'}.

    Returns
    -------
    X : ndarray (T, ny*nx*len(stats)) float32
    names : list[str]
    """
    movie = np.asarray(movie)
    T, H, W = movie.shape
    ys, xs = f_grid_bounds(H, W, grid)
    ny, nx = grid
    feats, names = [], []
    for iy in range(ny):
        for ix in range(nx):
            cell = movie[:, ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
            if 'occupancy' in stats:
                feats.append((cell > 0).mean(axis=(1, 2)).astype(np.float32))
                names.append(f'occ_r{iy}c{ix}')
            if 'mean' in stats:
                feats.append(cell.mean(axis=(1, 2), dtype=np.float32))
                names.append(f'mean_r{iy}c{ix}')
            if 'edge' in stats:
                c = cell.astype(np.float32)
                gy = np.gradient(c, axis=1)
                gx = np.gradient(c, axis=2)
                feats.append(np.sqrt(gx * gx + gy * gy).mean(axis=(1, 2)).astype(np.float32))
                names.append(f'edge_r{iy}c{ix}')
    return np.column_stack(feats), names


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — optical flow pooled to a retinotopic grid (#49)
# ─────────────────────────────────────────────────────────────────────────────

def f_flow_features(movie, grid=(6, 6), stats=('speed', 'dir', 'div'),
                    method='lucas_kanade', reg=1e-3, farneback=None):
    """Optical-flow features pooled over a retinotopic grid — Tier 2.

    Direction-of-edge-motion at each retinotopic location. One flow vector per
    grid cell per frame; per cell, channels for each requested statistic:
      'speed' — motion magnitude in the cell (motion energy).
      'dir'   — net flow DIRECTION as a sin/cos PAIR (two channels: dirsin,
                dircos; from the cell flow vector, no ±π wrap).
      'div'   — flow divergence (∂u/∂x + ∂v/∂y): local expansion/LOOMING (>0)
                vs contraction (<0).
    First frame has zero flow (no preceding frame).

    method
    ------
    'lucas_kanade' (default, numpy-only, NO extra deps): per-cell Lucas–Kanade
        least-squares from pooled spatial+temporal gradients. One vector per
        cell (aperture-limited, coarse) but dependency-free — works in any env.
    'farneback' (needs opencv-python / cv2; lazy-imported): dense Farneback flow
        then mean-pooled per cell. Denser/better; set method='farneback' in the
        visual_specs entry once cv2 is installed in the kernel env.

    Channel order is cell-major. Names e.g. 'speed_r0c0', 'dirsin_r0c0',
    'dircos_r0c0', 'div_r0c0', 'speed_r0c1', ...

    Parameters
    ----------
    movie : ndarray (T, H, W)   one monitor's movie (behavior clock, ordered).
    grid : (ny, nx)             tiling resolution.
    stats : tuple of str        subset of {'speed', 'dir', 'div'}.
    method : 'lucas_kanade' | 'farneback'
    reg : float                 Tikhonov term on the LK 2x2 system (stabilizes
                                low-texture cells / the aperture problem).
    farneback : dict | None     overrides for cv2.calcOpticalFlowFarneback.

    Returns
    -------
    X : ndarray (T, ny*nx*nch) float32 ; names : list[str]
    """
    movie = np.asarray(movie)
    T, H, W = movie.shape
    ys, xs = f_grid_bounds(H, W, grid)
    ny, nx = grid
    area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)
    do_div = ('div' in stats) and ny > 1 and nx > 1   # cell-grid divergence needs >1

    u_g  = np.zeros((T, ny, nx), np.float32)   # per-cell flow x
    v_g  = np.zeros((T, ny, nx), np.float32)   # per-cell flow y
    sp_g = np.zeros((T, ny, nx), np.float32)   # per-cell speed (motion energy)
    dv_g = np.zeros((T, ny, nx), np.float32)   # per-cell divergence

    if method == 'farneback':
        import cv2   # lazy — only this method needs opencv-python
        fb = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                  poly_n=5, poly_sigma=1.2, flags=0)
        if farneback:
            fb.update(farneback)
        prev = movie[0].astype(np.uint8)
        for t in range(1, T):
            cur = movie[t].astype(np.uint8)
            fl = cv2.calcOpticalFlowFarneback(
                prev, cur, None, fb['pyr_scale'], fb['levels'], fb['winsize'],
                fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags'])
            vx, vy = fl[..., 0], fl[..., 1]
            u_g[t]  = _pool_sum(vx, ys, xs) / area
            v_g[t]  = _pool_sum(vy, ys, xs) / area
            sp_g[t] = _pool_sum(np.sqrt(vx * vx + vy * vy), ys, xs) / area
            if do_div:
                dv_g[t] = _pool_sum(np.gradient(vx, axis=1) + np.gradient(vy, axis=0),
                                    ys, xs) / area
            prev = cur
    elif method == 'lucas_kanade':
        prev = movie[0].astype(np.float32)
        for t in range(1, T):
            cur = movie[t].astype(np.float32)
            Ix = np.gradient(cur, axis=1)
            Iy = np.gradient(cur, axis=0)
            It = cur - prev
            Sxx = _pool_sum(Ix * Ix, ys, xs); Syy = _pool_sum(Iy * Iy, ys, xs)
            Sxy = _pool_sum(Ix * Iy, ys, xs)
            Sxt = _pool_sum(Ix * It, ys, xs); Syt = _pool_sum(Iy * It, ys, xs)
            det = (Sxx + reg) * (Syy + reg) - Sxy * Sxy
            # solve A·(u,v) = -(Sxt, Syt), A = [[Sxx,Sxy],[Sxy,Syy]] (+reg on diag)
            u = (-(Syy + reg) * Sxt + Sxy * Syt) / det
            v = (Sxy * Sxt - (Sxx + reg) * Syt) / det
            u_g[t]  = u; v_g[t] = v
            sp_g[t] = np.sqrt(u * u + v * v)
            if do_div:
                dv_g[t] = np.gradient(u, axis=1) + np.gradient(v, axis=0)
            prev = cur
    else:
        raise ValueError(f"unknown flow method {method!r} "
                         "('lucas_kanade' | 'farneback')")

    feats, names = [], []
    for iy in range(ny):
        for ix in range(nx):
            if 'speed' in stats:
                feats.append(sp_g[:, iy, ix]); names.append(f'speed_r{iy}c{ix}')
            if 'dir' in stats:
                ang = np.arctan2(v_g[:, iy, ix], u_g[:, iy, ix])
                feats.append(np.sin(ang).astype(np.float32)); names.append(f'dirsin_r{iy}c{ix}')
                feats.append(np.cos(ang).astype(np.float32)); names.append(f'dircos_r{iy}c{ix}')
            if 'div' in stats:
                feats.append(dv_g[:, iy, ix]); names.append(f'div_r{iy}c{ix}')
    return np.column_stack(feats).astype(np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Gabor magnitude/phase bank pooled to a retinotopic grid (#50)
# ─────────────────────────────────────────────────────────────────────────────

def _gabor_pair(theta, wavelength, sigma, gamma=1.0):
    """One quadrature Gabor pair (even=cosine, odd=sine phase) at orientation
    `theta` (rad) and `wavelength` (pixels/cycle). `sigma` = envelope SD (px),
    `gamma` = aspect ratio (envelope elongation ⟂ to the bars). The even kernel is
    DC-removed so it doesn't respond to flat luminance. Returns (even, odd) float32."""
    n = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    xr =  x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2.0 * sigma ** 2))
    even = env * np.cos(2.0 * np.pi * xr / wavelength)
    odd  = env * np.sin(2.0 * np.pi * xr / wavelength)
    even = even - even.mean()                       # DC removal (edge, not luminance)
    return even.astype(np.float32), odd.astype(np.float32)


def f_gabor_bank(n_orient=4, wavelengths=(6, 12), sigma_scale=0.5, gamma=1.0):
    """The (even, odd) Gabor kernels for the whole bank — for visualization / sanity.

    Returns list of dicts {io, isf, theta, wavelength, sigma, even, odd}, in the
    same (orientation × SF) order the extractor uses."""
    thetas = np.arange(n_orient) * np.pi / n_orient
    bank = []
    for io, th in enumerate(thetas):
        for isf, lam in enumerate(wavelengths):
            sigma = sigma_scale * lam
            ev, od = _gabor_pair(th, lam, sigma, gamma)
            bank.append({'io': io, 'isf': isf, 'theta': float(th),
                         'wavelength': float(lam), 'sigma': float(sigma),
                         'even': ev, 'odd': od})
    return bank


def _spatial_fftconv_chunked(movie, kernel, chunk_t):
    """FFT-convolve a (T,H,W) stack with a 2-D kernel over the spatial axes, in
    T-chunks to bound memory. mode='same'. Returns (T,H,W) float32."""
    from scipy.signal import fftconvolve
    T = movie.shape[0]
    out = np.empty(movie.shape, np.float32)
    k = kernel[None]
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        out[s:e] = fftconvolve(movie[s:e], k, mode='same', axes=(1, 2))
    return out


def f_gabor_features(movie, grid=(6, 6), n_orient=4, wavelengths=(6, 12),
                     sigma_scale=0.5, gamma=1.0, stats=('magnitude',), chunk_t=2000):
    """Gabor bank pooled over a retinotopic grid — Tier 3.

    Oriented-edge representation at each retinotopic location and spatial frequency.
    For each (orientation × wavelength), a quadrature Gabor pair (even=cos, odd=sin
    phase) is convolved with every frame, giving the complex response R = even +
    i·odd at each pixel. The requested `stats` are mean-pooled into the (ny, nx) grid:
      'magnitude' — |R| = √(even²+odd²) (phase-INVARIANT amount of oriented edge;
                    1 channel). Locally invariant to where the edge sits within the
                    patch; the grid pooling adds the larger-scale spatial integration.
      'phase'     — unit phase of R (cosφ, sinφ; 2 channels; 0 where |R| ≈ 0).
                    φ = atan2(odd, even) = WHERE the edge sits within the carrier;
                    its TEMPORAL change (via the FIR lags downstream) is the local
                    edge velocity (normal component).
      'even'      — rectified even-phase response |even| (phase-SENSITIVE; 1 channel).

    Channel order is cell-major. Names carry a stat token:
    'gabM_o{io}sf{isf}_r{iy}c{ix}' (magnitude), 'gabPc_'/'gabPs_' (phase cos/sin),
    'gabEv_' (even).

    Parameters
    ----------
    movie : ndarray (T, H, W)   one monitor's movie (behavior clock).
    grid : (ny, nx)             pooling tiling.
    n_orient : int              orientations evenly spaced in [0, π).
    wavelengths : tuple         Gabor wavelengths in PIXELS/cycle (pixel ≈ 1° here;
                                small low-SF bank for mouse acuity, e.g. (6, 12)).
    sigma_scale : float         envelope SD = sigma_scale * wavelength.
    gamma : float               envelope aspect ratio (⟂ elongation).
    stats : tuple of str        subset of {'magnitude', 'phase', 'even'}.
    chunk_t : int               frames per FFT-conv chunk (memory bound).

    Returns
    -------
    X : ndarray (T, ny*nx*n_orient*len(wavelengths)*nch) float32 ; names : list[str]
    (nch = #channels implied by stats: magnitude→1, phase→2, even→1.)
    """
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    ys, xs = f_grid_bounds(H, W, grid)
    ny, nx = grid
    area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)
    wavelengths = np.atleast_1d(wavelengths)             # tolerate a scalar (e.g. (6))
    n_sf = len(wavelengths)
    thetas = np.arange(n_orient) * np.pi / n_orient

    def _pool(F):                                    # (T,H,W) -> (T,ny,nx) mean
        return (np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=1), xs[:-1], axis=2)
                / area[None])

    # (token, fn(re, ro, mag) -> (T,H,W)) per requested stat
    chan = []
    if 'magnitude' in stats:
        chan.append(('M', lambda re, ro, mag: mag))
    if 'phase' in stats:
        chan.append(('Pc', lambda re, ro, mag: np.where(mag > 1e-6, re / np.maximum(mag, 1e-6), 0.0)))
        chan.append(('Ps', lambda re, ro, mag: np.where(mag > 1e-6, ro / np.maximum(mag, 1e-6), 0.0)))
    if 'even' in stats:
        chan.append(('Ev', lambda re, ro, mag: np.abs(re)))
    if not chan:
        raise ValueError(f"no valid stats in {stats!r} (magnitude/phase/even)")

    pooled = {}                                      # (io, isf, token) -> (T,ny,nx)
    for io, th in enumerate(thetas):
        for isf, lam in enumerate(wavelengths):
            ev, od = _gabor_pair(th, lam, sigma_scale * lam, gamma)
            re = _spatial_fftconv_chunked(movie, ev, chunk_t)
            ro = _spatial_fftconv_chunked(movie, od, chunk_t)
            mag = np.sqrt(re * re + ro * ro)
            for tok, fn in chan:
                pooled[(io, isf, tok)] = _pool(fn(re, ro, mag)).astype(np.float32)

    feats, names = [], []
    for iy in range(ny):
        for ix in range(nx):
            for io in range(n_orient):
                for isf in range(n_sf):
                    for tok, _ in chan:
                        feats.append(pooled[(io, isf, tok)][:, iy, ix])
                        names.append(f'gab{tok}_o{io}sf{isf}_r{iy}c{ix}')
    return np.column_stack(feats).astype(np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4a — Adelson–Bergen spatiotemporal motion energy pooled to a grid (#51)
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_quad(tfreq, support, sigma):
    """Quadrature temporal filter pair (even=cos, odd=sin) at `tfreq` cycles/frame,
    Gaussian envelope SD `sigma`, length `support`. Even is DC-removed so a static
    image produces ~0 motion energy. Returns (te, to) float32."""
    tau = np.arange(support) - (support - 1) / 2.0
    env = np.exp(-tau ** 2 / (2.0 * sigma ** 2))
    te = env * np.cos(2.0 * np.pi * tfreq * tau)
    to = env * np.sin(2.0 * np.pi * tfreq * tau)
    te = te - te.mean()
    return te.astype(np.float32), to.astype(np.float32)


def f_motion_energy_features(movie, grid=(6, 6), n_orient=4, wavelengths=(8,),
                             temporal_freqs=(0.15,), temporal_support=11,
                             temporal_sigma=None, sigma_scale=0.5, gamma=1.0,
                             output='opponent', chunk_t=1000):
    """Adelson–Bergen spatiotemporal motion energy pooled to a grid — Tier 4a.

    Direction- AND speed-selective energy (the principled motion model; what static
    Gabor energy + a linear readout cannot build). Per (orientation × spatial
    wavelength × temporal frequency): the complex spatial Gabor response
    Cs = G_even + i·G_odd is formed per frame, then temporally filtered with a
    complex temporal Gabor (T_even + i·T_odd) for one direction and its conjugate
    for the opposite; directional energy = |·|². `output`:
      'opponent' — E_dir1 − E_dir2 (signed direction-selective; 1 channel).
      'both'     — E_dir1 and E_dir2 separately (2 channels).
    Mean-pooled into the grid. Motion is along each orientation's normal; the
    orientation bank tiles direction, temporal_freqs/wavelengths set speed tuning
    (speed ≈ tfreq·wavelength px/frame). Time-chunked with a halo for memory.

    Parameters
    ----------
    movie : ndarray (T, H, W)        one monitor's movie (behavior clock, ordered).
    grid : (ny, nx)                  pooling tiling.
    n_orient : int                   orientations in [0, π).
    wavelengths : tuple              spatial Gabor wavelengths (px/cycle).
    temporal_freqs : tuple           temporal frequencies (cycles/frame).
    temporal_support : int           temporal kernel length (frames).
    temporal_sigma : float | None    temporal envelope SD (default support/4).
    sigma_scale, gamma : float       spatial Gabor envelope (as f_gabor_features).
    output : {'opponent', 'both'}
    chunk_t : int                    frames per time chunk (memory bound).

    Returns
    -------
    X : ndarray (T, ny*nx*n_orient*n_sf*n_tf*nout) float32 ; names : list[str]
    (nout = 1 for 'opponent', 2 for 'both'.)
    """
    from scipy.signal import fftconvolve
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    ys, xs = f_grid_bounds(H, W, grid)
    ny, nx = grid
    area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)
    thetas = np.arange(n_orient) * np.pi / n_orient
    wavelengths = np.atleast_1d(wavelengths)             # tolerate scalars (e.g. (8))
    temporal_freqs = np.atleast_1d(temporal_freqs)
    n_sf, n_tf = len(wavelengths), len(temporal_freqs)
    halo = temporal_support // 2
    tsig = temporal_sigma if temporal_sigma else temporal_support / 4.0

    def _pool(F):                                    # (n,H,W) -> (n,ny,nx) mean
        return (np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=1), xs[:-1], axis=2)
                / area[None])

    sgab = {(io, isf): _gabor_pair(thetas[io], wavelengths[isf],
                                   sigma_scale * wavelengths[isf], gamma)
            for io in range(n_orient) for isf in range(n_sf)}
    tk = {itf: _temporal_quad(tf, temporal_support, tsig)
          for itf, tf in enumerate(temporal_freqs)}
    n_out = 1 if output == 'opponent' else 2
    store = {(io, isf, itf, d): np.empty((T, ny, nx), np.float32)
             for io in range(n_orient) for isf in range(n_sf)
             for itf in range(n_tf) for d in range(n_out)}

    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        lo, hi = max(0, s - halo), min(T, e + halo)
        sub = movie[lo:hi]
        vs, ve = s - lo, s - lo + (e - s)            # valid slice within the chunk
        for (io, isf), (ev, od) in sgab.items():
            re = fftconvolve(sub, ev[None], mode='same', axes=(1, 2))
            ro = fftconvolve(sub, od[None], mode='same', axes=(1, 2))
            Cs = (re + 1j * ro).astype(np.complex64)
            for itf in range(n_tf):
                te, to = tk[itf]
                R = fftconvolve(Cs, (te + 1j * to).reshape(-1, 1, 1), mode='same', axes=0)
                L = fftconvolve(Cs, (te - 1j * to).reshape(-1, 1, 1), mode='same', axes=0)
                eR = (R.real ** 2 + R.imag ** 2).astype(np.float32)[vs:ve]
                eL = (L.real ** 2 + L.imag ** 2).astype(np.float32)[vs:ve]
                if output == 'opponent':
                    store[(io, isf, itf, 0)][s:e] = _pool(eR - eL)
                else:
                    store[(io, isf, itf, 0)][s:e] = _pool(eR)
                    store[(io, isf, itf, 1)][s:e] = _pool(eL)

    dtoks = ['opp'] if output == 'opponent' else ['d0', 'd1']
    feats, names = [], []
    for iy in range(ny):
        for ix in range(nx):
            for io in range(n_orient):
                for isf in range(n_sf):
                    for itf in range(n_tf):
                        for d, dt in enumerate(dtoks):
                            feats.append(store[(io, isf, itf, d)][:, iy, ix])
                            names.append(f'meng{dt}_o{io}sf{isf}tf{itf}_r{iy}c{ix}')
    return np.column_stack(feats).astype(np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Feature-type registry (seed of the feature-selection module)
# ─────────────────────────────────────────────────────────────────────────────
# name -> {fn: extractor, defaults: param dict, desc: str}. Choose feature types
# by name + params via build_visual_blocks(specs=...). Tiers 2–4 add entries.
VISUAL_FEATURE_REGISTRY = {
    'grid': {
        'fn': f_grid_features,
        'defaults': {'grid': (6, 6), 'stats': ('occupancy', 'mean', 'edge')},
        'desc': 'Tier 1 — retinotopic grid pooling (occupancy/mean/edge energy).',
    },
    'flow': {
        'fn': f_flow_features,
        'defaults': {'grid': (6, 6), 'stats': ('speed', 'dir', 'div'),
                     'method': 'lucas_kanade'},
        'desc': ('Tier 2 — optical flow pooled to the grid (speed/dir/divergence). '
                 "method='lucas_kanade' (numpy) or 'farneback' (needs cv2)."),
    },
    'gabor': {
        'fn': f_gabor_features,
        'defaults': {'grid': (6, 6), 'n_orient': 4, 'wavelengths': (6, 12),
                     'sigma_scale': 0.5, 'gamma': 1.0, 'stats': ('magnitude',),
                     'chunk_t': 2000},
        'desc': ('Tier 3 — Gabor bank pooled to the grid. stats: magnitude '
                 '√(even²+odd²), phase (cos/sin φ → velocity via FIR), even (|even|).'),
    },
    'meng': {
        'fn': f_motion_energy_features,
        'defaults': {'grid': (6, 6), 'n_orient': 4, 'wavelengths': (8,),
                     'temporal_freqs': (0.15,), 'temporal_support': 11,
                     'sigma_scale': 0.5, 'gamma': 1.0, 'output': 'opponent',
                     'chunk_t': 1000},
        'desc': ('Tier 4a — Adelson–Bergen spatiotemporal motion energy pooled to '
                 'the grid (direction+speed selective; opponent or both directions).'),
    },
    # 'cnn'  : Tier 4b — pretrained-CNN feature maps (#52)           [planned]
}


def _pca_fit(X, n_comp, random_state=42):
    """PCA-compress X by `n_comp`: int = #components, float in (0,1) = smallest k
    reaching that cumulative variance fraction, None/0/≥n_features = no PCA.
    Returns (scores, pca_or_None)."""
    d = X.shape[1]
    if not n_comp:
        return X, None
    if isinstance(n_comp, float) and 0 < n_comp < 1:        # variance-fraction
        pca = PCA(n_components=n_comp, svd_solver='full', random_state=random_state)
        return pca.fit_transform(X), pca
    n_comp = int(n_comp)
    if n_comp >= d:
        return X, None
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X), pca


def f_apply_block_pca(X_img, raw_names, fname, n_pca):
    """Apply a block's `n_pca` spec to RAW channels → (X, names, pca, pca_groups).

    `n_pca`: int / float(0,1) / None/0 → PCA on the WHOLE block; dict {prefix: ncomp} →
    SEPARATE PCA per channel-name-prefix group (side-stripped), concatenated. Factored out
    of build_visual_blocks so the pooled feature-recon path applies the SAME (grouped)
    PCA — no drift. `raw_names` are the pre-PCA channel names."""
    raw_dim = X_img.shape[1]
    stripped = [n.split('_', 1)[1] if '_' in n else n for n in raw_names]
    if isinstance(n_pca, dict):
        parts, out_names, pca = [], [], {}
        pca_groups, assigned = {}, [False] * raw_dim
        for prefix, ncomp in n_pca.items():
            cols = [i for i, s in enumerate(stripped)
                    if s.startswith(prefix) and not assigned[i]]
            if not cols:
                raise ValueError(
                    f"{fname}: no channels match PCA-group prefix {prefix!r} "
                    f"(prefixes present: {sorted({s.split('_o')[0] for s in stripped})})")
            for i in cols:
                assigned[i] = True
            Xg, p_g = _pca_fit(X_img[:, cols], ncomp)
            s0 = sum(part.shape[1] for part in parts)
            parts.append(Xg)
            pca[prefix] = p_g
            pca_groups[prefix] = {'src_cols': cols, 'xslice': (s0, s0 + Xg.shape[1])}
            out_names += ([f'{fname}_{prefix}_PC{i + 1}' for i in range(Xg.shape[1])]
                          if p_g is not None else [f'{fname}_{stripped[c]}' for c in cols])
        if not all(assigned):
            miss = sorted({stripped[i].split('_o')[0]
                           for i in range(raw_dim) if not assigned[i]})
            raise ValueError(f"{fname}: channels not covered by any PCA-group "
                             f"prefix: {miss}")
        return np.concatenate(parts, axis=1), out_names, pca, pca_groups
    X_out, pca = _pca_fit(X_img, n_pca)
    names = ([f'{fname}_PC{i + 1}' for i in range(X_out.shape[1])]
             if pca is not None else list(raw_names))
    return X_out, names, pca, None


def build_visual_blocks(movies, side_tags, beh_t, frame_t, pulse_delay,
                        specs, default_n_pca=None, resample_kind='linear'):
    """Build selected visual feature blocks, registry-driven.

    For each requested feature type: run its registered extractor per monitor
    side, concatenate sides (side tag in the channel names), resample from the
    behavior clock to the imaging clock, and optionally PCA-compress. Returns a
    dict in the same shape as build_feature_blocks output, so the caller can do
    `built_blocks.update(build_visual_blocks(...))`.

    Parameters
    ----------
    movies : list[ndarray (T_beh, H, W)]   per-side monitor movies (behavior clock).
    side_tags : list[str]                  e.g. ['L','R'] or ['R'] — matches movies.
    beh_t, frame_t, pulse_delay            clock-alignment args (as build_feature_blocks).
    specs : dict {feat_name: params_dict|None}
        Which feature types to build and with what params (override the registry
        defaults). The per-type 'n_pca' key controls PCA compression and may be:
          int        — that many components on the WHOLE block.
          float(0,1) — components reaching that cumulative variance fraction.
          None/0     — no PCA (raw channels).
          dict       — SEPARATE PCA per channel-group, keyed by a channel-name
                       PREFIX (matched against the side-stripped name), each value
                       an int / float(0,1) / None. e.g. {'gabM': 20, 'gabP': 0.9}
                       PCAs magnitude and phase separately so neither scale washes
                       the other out, then concatenates the per-group PCs. Every
                       channel must match exactly one prefix.
    default_n_pca : int | None             fallback if a spec omits 'n_pca'.
    resample_kind : str                    interp kind for f_resample_to_imaging.

    Returns
    -------
    dict {feat_name: {'X': (T_img, d), 'names': list, 'pca': PCA|dict|None,
                      'raw_dim': int, 'raw_names': list, 'params': dict,
                      'pca_groups': dict|None}}
    ('pca' is a dict and 'pca_groups' is set when grouped PCA was used.)
    """
    blocks = {}
    for fname, params in specs.items():
        if fname not in VISUAL_FEATURE_REGISTRY:
            raise ValueError(f'unknown visual feature {fname!r}; '
                             f'registered: {list(VISUAL_FEATURE_REGISTRY)}')
        reg = VISUAL_FEATURE_REGISTRY[fname]
        p = dict(reg['defaults'])
        p.update(params or {})
        n_pca = p.pop('n_pca', default_n_pca)

        _t0 = time.perf_counter()
        Xs, names = [], []
        for movie, tag in zip(movies, side_tags):
            Xb, base = reg['fn'](movie, **p)
            Xs.append(Xb)
            names.extend(f'{tag}_{b}' for b in base)
        X_beh = np.concatenate(Xs, axis=1)
        print(f'    ⏱ visual extract {fname!r}: {time.perf_counter() - _t0:.2f}s')

        X_img = f_resample_to_imaging(beh_t, X_beh, frame_t, pulse_delay,
                                      kind=resample_kind)
        raw_dim = X_img.shape[1]
        raw_names = list(names)        # pre-PCA channel names (for reconstruction)
        X_img, names, pca, pca_groups = f_apply_block_pca(X_img, raw_names, fname, n_pca)

        blocks[fname] = {'X': X_img, 'names': names, 'pca': pca,
                         'raw_dim': raw_dim, 'params': p, 'raw_names': raw_names,
                         'pca_groups': pca_groups}
    return blocks
