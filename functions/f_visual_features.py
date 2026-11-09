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

from scipy.ndimage import gaussian_filter1d
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


def _sample_idx(n, ds, center=True):
    """Point-sample pixel indices along a length-`n` axis at stride `ds`. center=True (default) anchors a
    sample on the axis CENTER — offset = round((n-1)/2) % ds — so the middle sample lands at the middle of
    the screen (rounded for even n); center=False = the legacy [0::ds]. Returned as arange(off, n, ds) so
    callers can slice `[:, off::ds]` (mmap-efficient) or index directly.

    NOT symmetric in general (measured 2026-07-28, docstring previously claimed it was): arange walks UP
    from the centre-congruent offset, so a mirrored sample landing at a negative index is simply absent
    while its far-edge partner survives. Only bites when ds approaches n — at lat=150, ds=75 it gives
    x=[74, 149]: the centre plus a sample sitting ON the last column with no left-edge partner, and that
    one has ~half its Gaussian envelope (sigma=0.5*lambda=75px) out in the zero pad. Negligible at fine SF
    (2 of 630 columns); at cyc_fov=1 it is 1 of 2. Parity also hinges on banker's rounding of (n-1)/2 for
    even n (150 -> 74, not 75), which flips which edge gets the extra sample."""
    ds = max(1, int(ds))
    off = int(round((n - 1) / 2.0)) % ds if center else 0
    return np.arange(off, n, ds)


def _pool_sum(F, ys, xs):
    """Sum a 2-D field F over the (ny, nx) cells defined by bounds ys, xs.

    Vectorized block-reduce via np.add.reduceat (handles uneven edge cells).
    Returns (ny, nx). Divide by cell area for the mean.
    """
    return np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=0), xs[:-1], axis=1)


def _make_reducer(H, W, grid, pool, sample_ds):
    """Spatial reducer for the gabor/meng extractors. Returns (ny, nx, reduce_fn) where
    reduce_fn maps a dense (T,H,W) field → (T,ny,nx). pool='grid' → MEAN-pool into the (ny,nx)
    tiling (f_grid_bounds + area-normalized reduceat); pool='sample' → POINT-sample every
    `sample_ds` pixels (⌈H/ds⌉×⌈W/ds⌉; ds=1 → every pixel = full response)."""
    if pool == 'grid':
        ys, xs = f_grid_bounds(H, W, grid)
        ny, nx = grid
        area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)

        def _reduce(F):                                  # (T,H,W) -> (T,ny,nx) mean
            return (np.add.reduceat(np.add.reduceat(F, ys[:-1], axis=1), xs[:-1], axis=2)
                    / area[None])
        return ny, nx, _reduce
    if pool == 'sample':
        ds_h, ds_w = ((int(sample_ds[0]), int(sample_ds[1]))       # (ds_h, ds_w) anisotropic
                      if isinstance(sample_ds, (tuple, list)) else (int(sample_ds), int(sample_ds)))
        rows, cols = _sample_idx(H, ds_h), _sample_idx(W, ds_w)    # CENTERED on the screen middle
        oh, ow = int(rows[0]), int(cols[0])
        ny, nx = len(rows), len(cols)

        def _reduce(F):                                  # (T,H,W) -> (T,ny,nx) centered point-sample
            return F[:, oh::ds_h, ow::ds_w]
        return ny, nx, _reduce
    raise ValueError(f"pool must be 'grid' or 'sample', got {pool!r}")


EDGE_MIN_FRAC = 0.45          # default on-screen fraction below which a sample is dropped


def f_edge_axis_frac(n, ds, sigma, n_sigma=3.0):
    """Fraction of a filter's envelope energy staying on a length-`n` axis, per sample of the
    centred lattice. |env|^2 is Gaussian with SD sigma/sqrt(2) on a +-ceil(n_sigma*sigma) support."""
    from scipy.special import erf
    m = float(np.ceil(float(n_sigma) * float(sigma)))
    pos = _sample_idx(int(n), int(ds)).astype(float)
    lo = np.maximum(-m, -pos - 0.5)
    hi = np.minimum(m, float(n) - 0.5 - pos)
    return np.clip((erf(hi / sigma) - erf(lo / sigma)) / (2.0 * erf(m / sigma)), 0.0, 1.0)


def f_edge_keep_mask(H, W, ds, sigma, min_frac=EDGE_MIN_FRAC, n_sigma=3.0):
    """Boolean (ny, nx) over the point-sample lattice: True where at least `min_frac` of the
    filter's envelope energy is on screen. The envelope is isotropic, so the 2-D fraction is the
    product of the two axes (verified against measured kernel energies to 0.0%). Pure geometry —
    known before any convolution."""
    ds_h, ds_w = ((int(ds[0]), int(ds[1])) if isinstance(ds, (tuple, list)) else (int(ds), int(ds)))
    fy = f_edge_axis_frac(H, ds_h, sigma, n_sigma)
    fx = f_edge_axis_frac(W, ds_w, sigma, n_sigma)
    return np.outer(fy, fx) >= float(min_frac)


def f_subsample_grid_block(X, names, factor):
    """Coarsen a sampled/grid feature matrix by keeping every `factor`-th sample in each axis
    (columns named `..._r{iy}c{ix}`). Because strides compose exactly ([::ds][::k]==[::ds·k]),
    this derives coarser resolutions from ONE fine extraction without re-convolving. r/c indices
    are renumbered to the coarse lattice; column order (cell-major) is preserved. Returns
    (X_sub, names_sub)."""
    import re
    pat = re.compile(r'^(.*_r)(\d+)c(\d+)$')
    keep, new_names = [], []
    for j, nm in enumerate(names):
        m = pat.match(nm)
        if m is None:
            raise ValueError(f'name {nm!r} lacks an _r{{iy}}c{{ix}} suffix')
        iy, ix = int(m.group(2)), int(m.group(3))
        if iy % factor == 0 and ix % factor == 0:
            keep.append(j)
            new_names.append(f'{m.group(1)}{iy // factor}c{ix // factor}')
    return np.asarray(X)[:, keep], new_names


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


def _gauss_env(sigma, gamma=1.0):
    """Carrier-free Gaussian envelope kernel (unit-sum, NOT DC-removed) = the Kay/Gallant luminance-only
    wavelet. Same ±3σ support as _gabor_pair so it matches the bank's spatial extent; projecting a frame
    onto it gives a Gaussian-weighted local mean LUMINANCE (the DC term the gabors deliberately drop)."""
    n = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    env = np.exp(-(x ** 2 + (gamma * y) ** 2) / (2.0 * sigma ** 2))
    env = env / env.sum()                            # unit-sum → weighted mean luminance
    return env.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3b — windowed patch-FFT spectrum on the gabor bank's own axes (#73)
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_window(sigma, ny_half, nx_half):
    """UN-normalized isotropic Gaussian window at explicit half-widths → (2*ny_half+1, 2*nx_half+1).
    Matches `_gabor_pair`'s envelope exactly (which is un-normalized and multiplied by the carrier),
    NOT `_gauss_env` (unit-sum). Isotropic only — see f_fft_features on why gamma must be 1."""
    y, x = np.mgrid[-ny_half:ny_half + 1, -nx_half:nx_half + 1]
    return np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def _snap_patch(n_want, lam, n_cap):
    """Pick the patch half-width whose extent N=2n+1 puts the target frequency 1/lam CLOSEST to an
    exact rfft lattice point. The lattice resolves k/N, so the target sits at k = N/lam and the
    quantization error is frac = |k - round(k)|; MEASURED, that error dominates fft-vs-gabor
    agreement (corr(frac, agreement) = -0.88; frac<0.15 -> 0.96, frac>0.35 -> 0.39, and frac~0.03
    -> 0.9998). N is ours to choose — the window is the Gaussian at sigma either way, so moving N a
    few px only trims near-zero tails. Searches +-15% around `n_want`, capped at `n_cap`."""
    n_want, n_cap = int(n_want), int(n_cap)
    if n_want >= n_cap:                                   # cap binds → no freedom
        return n_cap
    lo, hi = max(1, int(n_want * 0.85)), min(n_cap, int(np.ceil(n_want * 1.15)))
    cand = np.arange(lo, hi + 1)
    k = (2 * cand + 1) / float(lam)
    return int(cand[np.argmin(np.abs(k - np.round(k)))])


def _fft_patch_geom(lam, sigma_scale, H, W, patch_snap=True):
    """Patch geometry for one pyramid level. Half-width = ceil(3*sigma) (the SAME support as
    _gabor_pair), then CAPPED per axis at H-1 / W-1: a patch centred anywhere in the frame can only
    reach H-1 rows / W-1 cols before everything is zero pad, so the cap is exact (same argument as
    f_conv_gpu._crop_kernel) and it keeps the low-SF levels tractable (sigma=75 would give 451x451
    on a 101x150 frame). `patch_snap` then nudges each axis so the target frequency lands on a
    lattice point (_snap_patch) — off by default only for parity testing.
    Returns (sigma, ny_half, nx_half, Ny, Nx)."""
    sigma = float(sigma_scale) * float(lam)
    n = int(np.ceil(3 * sigma))
    if patch_snap:
        ny_half, nx_half = _snap_patch(n, lam, H - 1), _snap_patch(n, lam, W - 1)
    else:
        ny_half, nx_half = min(n, H - 1), min(n, W - 1)
    return sigma, ny_half, nx_half, 2 * ny_half + 1, 2 * nx_half + 1


def _fft_sector_map(Ny, Nx, thetas, wavelengths):
    """Map the rfft2 lattice of an (Ny,Nx) patch onto the bank's (orientation x SF) sectors.

    rfft2 gives ky = fftfreq(Ny) (signed) and kx = rfftfreq(Nx) (>=0) — the half-plane, which covers
    orientation [0,180) exactly like the gabor bank. Per coefficient: frequency (fx,fy) cyc/px,
    |f| = hypot, orientation atan2(fy,fx) mod pi (the gabor convention: carrier along
    xr = x cos(theta) + y sin(theta), so the wavevector points at theta).

    Sector EDGES (not nearest-neighbour over the whole plane, which would sweep DC and Nyquist junk
    into the end bins): angular edges at midpoints between adjacent thetas; radial edges at GEOMETRIC
    midpoints between adjacent target frequencies, with the outer edges half an octave beyond the
    ends. Coefficients outside the bank's SF range are DROPPED.

    Returns (cols, nearest) — cols[(io,isf)] = flat lattice indices in that sector (possibly empty);
    nearest[(io,isf)] = the single flat index closest to that sector's exact (theta, lambda) target
    in (log-SF, angle), or None if the sector is empty. Frame-independent → compute once per level."""
    fy = np.fft.fftfreq(Ny)[:, None]                      # cycles/px, signed
    fx = np.fft.rfftfreq(Nx)[None, :]                     # cycles/px, >= 0
    rad = np.hypot(fx, fy)                                # |f|
    ang = np.mod(np.arctan2(fy, fx * np.ones_like(fy)), np.pi)   # [0, pi)
    thetas = np.mod(np.asarray(thetas, float), np.pi)
    targ = 1.0 / np.asarray(wavelengths, float)           # target |f| per SF, cyc/px
    order = np.argsort(targ)                              # radial edges need ascending frequency
    ts = targ[order]
    edges = np.empty(len(ts) + 1)                         # geometric midpoints + half-octave ends
    edges[1:-1] = np.sqrt(ts[:-1] * ts[1:])
    edges[0], edges[-1] = ts[0] / np.sqrt(2.0), ts[-1] * np.sqrt(2.0)
    isf_of_bin = np.empty(len(ts), int)                   # ascending-bin -> original SF index
    isf_of_bin[np.arange(len(ts))] = order
    with np.errstate(invalid='ignore'):
        rbin = np.digitize(rad, edges) - 1                # -1 / len(ts) = outside the bank
    dang = np.abs(ang[..., None] - thetas[None, None, :])            # angular distance, wrapped
    dang = np.minimum(dang, np.pi - dang)
    obin = np.argmin(dang, axis=-1)
    half = (np.pi / len(thetas)) / 2.0 if len(thetas) > 1 else np.pi
    valid = (rbin >= 0) & (rbin < len(ts)) & (np.min(dang, axis=-1) <= half + 1e-9)
    valid &= rad > 0                                                  # never bin DC
    lrad = np.log(np.maximum(rad, 1e-12))
    cols, nearest = {}, {}
    for io in range(len(thetas)):
        for ib in range(len(ts)):
            isf = int(isf_of_bin[ib])
            sel = valid & (obin == io) & (rbin == ib)
            idx = np.flatnonzero(sel.ravel())
            cols[(io, isf)] = idx
            if idx.size:                                  # closest to the exact (theta, lambda) target
                d = ((lrad.ravel()[idx] - np.log(ts[ib])) ** 2
                     + (dang[..., io].ravel()[idx] / half) ** 2 * 1e-3)
                nearest[(io, isf)] = int(idx[np.argmin(d)])
            else:
                nearest[(io, isf)] = None
    return cols, nearest


def _expand_targets(values, n_sub, log=True):
    """Subdivide each target into `n_sub` sub-targets (geometric for SF, linear for angle) → a longer
    target list. n_sub=1 returns the input unchanged, so the default keeps index-for-index
    correspondence with the gabor bank."""
    v = np.asarray(values, float)
    if int(n_sub) <= 1:
        return v
    n_sub = int(n_sub)
    if log:                                               # geometric sub-steps within +-1/2 spacing
        step = (v[1] / v[0]) if len(v) > 1 else 2.0
        f = step ** (np.linspace(-0.5, 0.5, n_sub + 1)[:-1] + 0.5 / n_sub)
        return np.sort(np.concatenate([vi * f for vi in v]))
    step = (v[1] - v[0]) if len(v) > 1 else np.pi
    off = step * (np.linspace(-0.5, 0.5, n_sub + 1)[:-1] + 0.5 / n_sub)
    return np.sort(np.concatenate([vi + off for vi in v]))


def f_fft_features(movie, grid=(6, 6), n_orient=4, wavelengths=(6, 12), sigma_scale=0.5, gamma=1.0,
                   stats=('magnitude',), chunk_t=200, pool='sample', sample_ds=1,
                   presmooth_sigma=None, backend='cpu', device=None, odegs=None,
                   n_sf_sub=1, n_ori_sub=1, patch_snap=True, edge_min_frac=None):
    """Windowed patch-FFT spectrum on the gabor bank's own axes — Tier 3b (#73).

    A Gabor response at (x0,y0,theta,lambda) IS one Fourier coefficient of the Gaussian-windowed
    patch at that position, at frequency f = (cos(theta)/lambda, sin(theta)/lambda). This extractor
    takes the SAME window at the SAME positions and keeps the whole lattice instead of one
    coefficient per filter, then bins it back onto the bank's (orientation x SF) sectors. So the
    gabor bank is the lossy subsample and this is the fuller description of the same space-frequency
    plane. Constant-Q is preserved by using ONE PATCH SIZE PER LEVEL (patch tied to lambda), matching
    the wavelet pyramid rather than a single constant-bandwidth FFT.

    Geometry is shared with f_gabor_features so the two are directly comparable: sigma =
    sigma_scale*lambda; patch half-width ceil(3*sigma) capped per axis at H-1/W-1 (exact — beyond that
    is all zero pad); patch centres from `_sample_idx` (identical points to pool='sample'); borders
    zero-padded as F.conv2d does. With `n_sf_sub`=`n_ori_sub`=1 the output is column-for-column
    aligned with the gabor bank: fftM_o{io}sf{isf}_r{iy}c{ix} and gabM_o{io}sf{isf}_r{iy}c{ix} name
    the same (theta,lambda) at the same location, differing only in that fft pools the whole sector.

    `stats`:
      'magnitude'  — mean |F| over the sector's lattice coefficients ('fftM'; 1 channel). Each |F|
                     IS that frequency's gabor magnitude, so this is the mean gabor magnitude over
                     the sector rather than the single filter the bank samples.
      'quadrature' — Re{F}, Im{F} at the SINGLE coefficient nearest the sector's (theta,lambda)
                     ('fftQe'/'fftQo'; 2 channels). Not a sector mean: averaging complex values
                     cancels phase. Approximately reproduces gabQe/gabQo — raise `n_sf_sub` /
                     `n_ori_sub` to resolve more of the sector as separate columns.
    `n_sf_sub` / `n_ori_sub` subdivide each SF / orientation bin (geometric / linear), emitting the
    finer targets as their own sf{isf} / o{io} indices — the knob that makes this strictly richer
    than the bank. NOTE index correspondence with gabor holds only at 1.

    `gamma` must be 1: with an anisotropic envelope the window is orientation-DEPENDENT, so one patch
    FFT cannot serve all orientations. `pool` must be 'sample' (patch centres are points, there is no
    dense field to mean-pool); `grid` is accepted and ignored so the spec plumbing is uniform.

    Returns X (T, ny*nx*n_orient*n_sf*nch) float32, names."""
    if float(gamma) != 1.0:
        raise ValueError(f'f_fft_features needs gamma=1 (isotropic window); got {gamma}. An '
                         'anisotropic envelope differs per orientation, so one patch FFT cannot '
                         'serve the whole bank.')
    if pool != 'sample':
        raise ValueError(f"f_fft_features supports pool='sample' only (patch centres are points, "
                         f"there is no dense field to mean-pool); got pool={pool!r}. In a feature "
                         f"spec set reduce='sample' (+ ds), not reduce='grid'.")
    movie = np.asarray(movie)
    if presmooth_sigma:
        movie = gaussian_filter1d(np.asarray(movie, np.float32), presmooth_sigma, axis=1, mode='nearest')
        movie = gaussian_filter1d(movie, presmooth_sigma, axis=2, mode='nearest')
    T, H, W = movie.shape
    wavelengths = np.atleast_1d(wavelengths).astype(float)
    if len(wavelengths) != 1:
        raise ValueError('f_fft_features takes ONE wavelength per call (patch size is tied to it — '
                         'the pyramid is built by calling once per level); got '
                         f'{len(wavelengths)}. Use f_pyramid_specs to emit one spec per SF.')
    thetas = (np.deg2rad(np.atleast_1d(np.asarray(odegs, float))) if odegs is not None
              else np.arange(n_orient) * np.pi / n_orient)
    lam = float(wavelengths[0])
    sigma, ny_half, nx_half, Ny, Nx = _fft_patch_geom(lam, sigma_scale, H, W, patch_snap)

    th_t = _expand_targets(np.sort(thetas), n_ori_sub, log=False) if n_ori_sub > 1 else thetas
    wl_t = _expand_targets(wavelengths, n_sf_sub, log=True) if n_sf_sub > 1 else wavelengths
    n_o, n_sf = len(th_t), len(wl_t)

    toks = []
    if 'magnitude' in stats:
        toks.append('M')
    if 'quadrature' in stats:
        toks += ['Qe', 'Qo']
    if not toks:
        raise ValueError(f"no valid stats in {stats!r} (magnitude / quadrature)")

    ds_h, ds_w = ((int(sample_ds[0]), int(sample_ds[1])) if isinstance(sample_ds, (tuple, list))
                  else (int(sample_ds), int(sample_ds)))
    rows, cols_i = _sample_idx(H, ds_h), _sample_idx(W, ds_w)      # SAME centres as gabor 'sample'
    ny, nx = len(rows), len(cols_i)
    keep = (f_edge_keep_mask(H, W, sample_ds, sigma, edge_min_frac) if edge_min_frac else None)
    win = _gauss_window(sigma, ny_half, nx_half)
    sec_cols, sec_near = _fft_sector_map(Ny, Nx, th_t, wl_t)
    # rfft2 references phase to the patch CORNER (index 0); the gabor references its CENTRE. Undo the
    # ramp so Re/Im mean the same thing as gabQe/gabQo (a fixed rotation, so |F| is unaffected).
    _ph = np.exp(2j * np.pi * (np.fft.fftfreq(Ny)[:, None] * ny_half
                               + np.fft.rfftfreq(Nx)[None, :] * nx_half)).astype(np.complex64)
    _ph = _ph.ravel()

    out = {(io, isf, tk): np.empty((T, ny, nx), np.float32)
           for io in range(n_o) for isf in range(n_sf) for tk in toks}
    empty = [(io, isf) for io in range(n_o) for isf in range(n_sf) if sec_cols[(io, isf)].size == 0]
    if empty:
        print(f'  [fft] lam={lam:.1f}px patch={Ny}x{Nx}: {len(empty)}/{n_o * n_sf} sectors have NO '
              f'lattice coefficient (patch too small to resolve that SF) -> zero-filled')
    from numpy.lib.stride_tricks import sliding_window_view
    import scipy.fft as _sfft
    # scipy.fft (not np.fft) because numpy always computes in DOUBLE and single-threaded: on the
    # cyc_fov=4 unit that one call was 6.60 s/chunk vs 0.19 s here (complex64 + workers=-1), 35x.
    # The phase de-ramp only matters for Re/Im — |F*ph| == |F| — so magnitude-only skips it entirely.
    want_phase = ('Qe' in toks) or ('Qo' in toks)
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        pad = np.zeros((e - s, H + 2 * ny_half, W + 2 * nx_half), np.float32)
        pad[:, ny_half:ny_half + H, nx_half:nx_half + W] = movie[s:e]
        pw = sliding_window_view(pad, (Ny, Nx), axis=(1, 2))        # (t,H,W,Ny,Nx) view
        # index BOTH lattice axes at once: the two-step pw[:,rows][:,:,cols_i] materializes a
        # (t,ny,W,Ny,Nx) intermediate first — 7.7 GB vs 0.36 GB at cyc_fov=4, and 6x slower.
        patches = pw[:, rows[:, None], cols_i[None, :]] * win       # (t,ny,nx,Ny,Nx) windowed copy
        F = _sfft.rfft2(patches, axes=(-2, -1), workers=-1).reshape(e - s, ny, nx, -1)
        if want_phase:
            F = F * _ph
        for tk in toks:
            for io in range(n_o):
                for isf in range(n_sf):
                    if tk == 'M':                                   # index BEFORE abs: 26 of 6441 cols
                        idx = sec_cols[(io, isf)]
                        out[(io, isf, tk)][s:e] = (np.abs(F[..., idx]).mean(-1) if idx.size else 0.0)
                    else:
                        j = sec_near[(io, isf)]
                        v = F[..., j] if j is not None else 0.0
                        out[(io, isf, tk)][s:e] = (0.0 if j is None else
                                                   (v.real if tk == 'Qe' else v.imag))
        del pad, pw, patches, F

    feats, names = [], []
    for iy in range(ny):
        for ix in range(nx):
            if keep is not None and not keep[iy, ix]:
                continue
            for io in range(n_o):
                for isf in range(n_sf):
                    for tk in toks:
                        feats.append(out[(io, isf, tk)][:, iy, ix])
                        names.append(f'fft{tk}_o{io}sf{isf}_r{iy}c{ix}')
    return np.column_stack(feats).astype(np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3c — global (unwindowed) FFT harmonics: no patches, no envelope
# ─────────────────────────────────────────────────────────────────────────────

def f_fft_global_features(movie, n_orient=4, wavelengths=(6, 12), stats=('magnitude',),
                          chunk_t=200, odegs=None, n_sf_sub=1, n_ori_sub=1, **_ignored):
    """PLAIN rfft2 of each whole frame — the patch-free limit of f_fft_features. No window, so perfect
    frequency resolution and ZERO spatial localization: position is carried entirely in the PHASE
    (shifting the image multiplies coefficient k by exp(-i2*pi*k*shift/N), leaving |F| unchanged).

    The contrast against the gabor bank: same (orientation x SF) target axes via _fft_sector_map, but
    ONE value per target instead of a retinotopic lattice. Names carry NO `_r{iy}c{ix}` — there is no
    lattice — which downstream name parsing tolerates (r/c are optional in f_pca_group_cols) but the
    image-space viewers cannot render.

    stats: 'magnitude' → sector-mean |F| (shift-invariant, position discarded);
           'quadrature' → Re/Im at the nearest coefficient (keeps the phase, so position survives).
    `n_sf_sub` / `n_ori_sub` subdivide each sector, as in f_fft_features.
    Returns X (T, n_orient*n_sf*nch), names."""
    movie = np.asarray(movie)
    T, H, W = movie.shape
    wavelengths = np.atleast_1d(wavelengths).astype(float)
    thetas = (np.deg2rad(np.atleast_1d(np.asarray(odegs, float))) if odegs is not None
              else np.arange(n_orient) * np.pi / n_orient)
    th_t = _expand_targets(np.sort(thetas), n_ori_sub, log=False) if n_ori_sub > 1 else np.sort(thetas)
    wl_t = _expand_targets(wavelengths, n_sf_sub, log=True) if n_sf_sub > 1 else wavelengths
    n_o, n_sf = len(th_t), len(wl_t)
    toks = (['M'] if 'magnitude' in stats else []) + (['Qe', 'Qo'] if 'quadrature' in stats else [])
    if not toks:
        raise ValueError(f"no valid stats in {stats!r} (magnitude / quadrature)")
    # SAME sector map as the windowed path, sized to the WHOLE frame instead of a patch
    sec_cols, sec_near = _fft_sector_map(H, W, th_t, wl_t)
    out = {(io, isf, tk): np.zeros(T, np.float32) for io in range(n_o)
           for isf in range(n_sf) for tk in toks}
    for t0 in range(0, T, chunk_t):
        blk = np.asarray(movie[t0:t0 + chunk_t], np.float32)
        F = np.fft.rfft2(blk, axes=(1, 2)).reshape(blk.shape[0], -1)   # NO window
        for io in range(n_o):
            for isf in range(n_sf):
                cols = sec_cols.get((io, isf))
                if 'M' in toks:
                    out[(io, isf, 'M')][t0:t0 + blk.shape[0]] = (
                        np.abs(F[:, cols]).mean(axis=1) if cols is not None and len(cols)
                        else 0.0)
                if 'Qe' in toks:
                    k = sec_near.get((io, isf))
                    v = F[:, k] if k is not None else np.zeros(blk.shape[0], complex)
                    out[(io, isf, 'Qe')][t0:t0 + blk.shape[0]] = np.real(v)
                    out[(io, isf, 'Qo')][t0:t0 + blk.shape[0]] = np.imag(v)
    feats, names = [], []
    for io in range(n_o):
        for isf in range(n_sf):
            for tk in toks:
                feats.append(out[(io, isf, tk)])
                names.append(f'fftg{tk}_o{io}sf{isf}')     # no r/c — there is no lattice
    return np.column_stack(feats).astype(np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3d — separable wavelet transform (DWT critically sampled | SWT shift-invariant)
# ─────────────────────────────────────────────────────────────────────────────

# Orthonormal analysis lowpass taps; highpass by the QMF relation h[n] = (-1)^n g[N-1-n].
_WAVELET_TAPS = {
    'haar': np.array([1.0, 1.0]) / np.sqrt(2.0),
    'db2':  np.array([1 + np.sqrt(3), 3 + np.sqrt(3), 3 - np.sqrt(3), 1 - np.sqrt(3)]) / (4 * np.sqrt(2)),
    'db4':  np.array([0.23037781330886, 0.71484657055254, 0.63088076792959, -0.02798376941698,
                      -0.18703481171888, 0.03084138183599, 0.03288301166698, -0.01059740178500]),
}
WAVELET_SUBBANDS = ('LH', 'HL', 'HH')      # the 3 detail orientations of a separable 2-D wavelet


def _wavelet_filters(family):
    """(lowpass, highpass) analysis taps for `family`."""
    if family not in _WAVELET_TAPS:
        raise ValueError(f'unknown wavelet family {family!r}; have {list(_WAVELET_TAPS)}')
    g = np.asarray(_WAVELET_TAPS[family], float)
    h = g[::-1].copy()
    h[1::2] *= -1.0                                   # QMF: highpass from the reversed lowpass
    return g, h


def _sep_filt(X, k, axis, decimate):
    """Filter (T,H,W) along `axis` with 1-D kernel `k`, periodic wrap (keeps sizes exact), then
    decimate by 2 when asked. Wrap makes the transform exactly invertible for the DWT."""
    n = X.shape[axis]
    idx = (np.arange(n)[:, None] + np.arange(len(k))[None, :]) % n
    Y = np.moveaxis(np.moveaxis(X, axis, -1)[..., idx] @ k, -1, axis)
    if decimate:
        Y = Y[(slice(None),) * axis + (slice(None, None, 2),)]
    return Y


def f_wavelet_level_for(wl_px):
    """Pixel wavelength → DWT level (1 = finest). Detail level j passes wavelengths ~2^j..2^(j+1) px,
    so level = round(log2(lambda)) - 1. Lets the wavelet sit on the pyramid's SF axis: at the rig's
    150px FOV, cyc_fov (1,2,4,8,16) → lambda (150,75,37.5,18.8,9.4)px → levels (6,5,4,3,2)."""
    return int(max(1, round(np.log2(max(float(wl_px), 2.0))) - 1))


def f_wavelet_features(movie, family='db2', n_levels=3, levels=None, decimate=True,
                       chunk_t=500, odegs=None, n_orient=3, wavelengths=None, **_ignored):
    """Separable 2-D wavelet transform of each frame — Tier 3d, the critically-sampled counterpart to
    the (redundant) gabor bank.

    `decimate=True` → DWT: downsample by 2 per level, so the whole transform has exactly as many
    coefficients as pixels (orthonormal, perfectly invertible). Efficient, but SHIFT-VARIANT — a 1-px
    translation reshuffles the coefficients, which usually hurts an encoding model.
    `decimate=False` → SWT / a-trous: same filters, no downsampling (kernels dilated per level), so
    every level is full resolution. Redundant (3 x n_levels) but shift-INVARIANT.

    The 3 detail subbands LH / HL / HH ARE the orientations (horizontal / vertical / diagonal detail),
    so `io` ∈ 0..2 and there is no free orientation axis — `odegs` must be None or length 3.
    `levels` selects which levels to emit (1 = finest); default = 1..n_levels. Level j is the SF axis
    (`sf{isf}`), so marginalize / layer procs / f_score_grid all work unchanged. On the PYRAMID path
    `wavelengths` (px, from sf_cyc_deg) arrives instead and is mapped by f_wavelet_level_for, so a
    cyc_fov list selects levels directly — one level per call, matching gabor/fft.
    Names `wv{LH|HL|HH}_o{io}sf{isf}_r{iy}c{ix}`. Returns X (T, sum over levels), names."""
    if odegs is not None and len(np.atleast_1d(odegs)) != 3:
        raise ValueError(f'f_wavelet_features has exactly 3 subband orientations (LH/HL/HH); got '
                         f'odegs={odegs!r}. A separable wavelet has no free orientation axis — use '
                         f'kind="gabor"/"fft" for arbitrary angles.')
    if n_orient not in (3, None):
        raise ValueError(f'f_wavelet_features gives 3 orientations (LH/HL/HH); got n_orient={n_orient}')
    movie = np.asarray(movie)
    T = movie.shape[0]
    g, h = _wavelet_filters(family)
    if levels is not None:
        lv = sorted({int(v) for v in np.atleast_1d(levels)})
    elif wavelengths is not None:                     # pyramid path: cyc_fov -> lambda px -> level
        lv = sorted({f_wavelet_level_for(w) for w in np.atleast_1d(wavelengths)})
    else:
        lv = list(range(1, int(n_levels) + 1))
    if min(lv) < 1:
        raise ValueError(f'wavelet levels are 1-based (1 = finest); got {lv}')
    store = {}                                        # (isf, io) -> (T, ny, nx)
    for t0 in range(0, T, chunk_t):
        A = np.asarray(movie[t0:t0 + chunk_t], np.float32)
        for j in range(1, max(lv) + 1):
            if decimate:
                gj, hj = g, h
            else:                                     # a-trous: dilate the kernels by 2^(j-1)
                up = 2 ** (j - 1)
                gj = np.zeros(len(g) * up - (up - 1)); gj[::up] = g
                hj = np.zeros(len(h) * up - (up - 1)); hj[::up] = h
            L = _sep_filt(A, gj, 2, decimate)         # along W
            Hh = _sep_filt(A, hj, 2, decimate)
            LL = _sep_filt(L, gj, 1, decimate)        # along H
            LH = _sep_filt(L, hj, 1, decimate)        # H-detail (rows) x lowpass cols
            HL = _sep_filt(Hh, gj, 1, decimate)
            HH = _sep_filt(Hh, hj, 1, decimate)
            if j in lv:
                isf = lv.index(j)
                for io, (sub, arr) in enumerate(zip(WAVELET_SUBBANDS, (LH, HL, HH))):
                    store.setdefault((isf, io, sub), []).append(arr.astype(np.float32))
            A = LL
    feats, names = [], []
    for (isf, io, sub) in sorted(store):
        arr = np.concatenate(store[(isf, io, sub)], axis=0)
        ny, nx = arr.shape[1], arr.shape[2]
        for iy in range(ny):
            for ix in range(nx):
                feats.append(arr[:, iy, ix])
                names.append(f'wv{sub}_o{io}sf{isf}_r{iy}c{ix}')
    return np.column_stack(feats).astype(np.float32), names


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
                     sigma_scale=0.5, gamma=1.0, stats=('magnitude',), chunk_t=2000,
                     pool='grid', sample_ds=1, presmooth_sigma=None, backend='cpu', device=None,
                     odegs=None, edge_min_frac=None):
    """Gabor bank pooled over a retinotopic grid — Tier 3.

    Two spatial-reduction modes (`pool`): 'grid' (default) MEAN-pools the dense response into
    the (ny, nx) grid cells; 'sample' POINT-samples it on a lattice every `sample_ds` pixels
    (⌈H/ds⌉×⌈W/ds⌉ samples; ds=1 → the full per-pixel response) — the sampling locations ARE the
    "grid". Sampled output is grid-shaped (r{iy}c{ix} names over the sampled shape), so PCA /
    decode / viz treat it identically. `presmooth_sigma` optionally Gaussian-blurs the movie
    before convolution (smoothing-then-gabor; also anti-aliases the point-sampling at large ds).

    Oriented-edge representation at each retinotopic location and spatial frequency.
    For each (orientation × wavelength), a quadrature Gabor pair (even=cos, odd=sin
    phase) is convolved with every frame, giving the complex response R = even +
    i·odd at each pixel. The requested `stats` are reduced (mean-pool or point-sample) to (ny,nx):
      'magnitude' — |R| = √(even²+odd²) (phase-INVARIANT amount of oriented edge;
                    1 channel). Locally invariant to where the edge sits within the
                    patch; the grid pooling adds the larger-scale spatial integration.
      'quadrature'— the SIGNED even/odd responses themselves (re, ro; 2 channels), NO magnitude
                    taken and NO normalization. Since re = |R|cosφ and ro = |R|sinφ this keeps
                    magnitude AND phase jointly — |R| = √(re²+ro²) is recoverable, so it is a
                    strict superset of 'magnitude' and 'phase' and is the LINEAR gabor response a
                    ridge encoder can use directly (rectifying/squaring is a nonlinearity better
                    left to the model). Preferred over 'phase' where |R| is small: no division, so
                    empty screen regions contribute ~0 instead of a full-amplitude arbitrary unit
                    vector. Not phase-invariant → keep the Nyquist λ/2 sampling.
      'phase'     — unit phase of R (cosφ, sinφ; 2 channels; 0 where |R| ≈ 0). CAUTION: re/mag
                    EXPLODES as |R|→0 and the gate is an absolute `mag > 1e-6`, so where the screen
                    is empty these channels emit unit-magnitude values driven by rounding, not
                    stimulus — measured to differ by corr 0.86 between conv engines. Prefer
                    'quadrature' unless the magnitude-invariance is specifically wanted.
                    φ = atan2(odd, even) = WHERE the edge sits within the carrier;
                    its TEMPORAL change (via the FIR lags downstream) is the local
                    edge velocity (normal component).
      'even'      — rectified even-phase response |even| (phase-SENSITIVE; 1 channel).
      'luminance' — Kay/Gallant luminance-only wavelet: √2·|Gaussian-envelope projection| (carrier-free,
                    NOT DC-removed; 1 channel per SF, orientation-INDEPENDENT → emitted once at io=0).
                    The DC term the oriented gabors drop; request at grid=(1,1) for the whole-FOV mean.

    Channel order is cell-major. Names carry a stat token:
    'gabM_o{io}sf{isf}_r{iy}c{ix}' (magnitude), 'gabPc_'/'gabPs_' (phase cos/sin),
    'gabQe_'/'gabQo_' (signed quadrature even/odd), 'gabEv_' (even),
    'gabLum_o0sf{isf}_r{iy}c{ix}' (luminance, io=0 only).

    Parameters
    ----------
    movie : ndarray (T, H, W)   one monitor's movie (behavior clock).
    grid : (ny, nx)             pooling tiling.
    n_orient : int              orientations evenly spaced in [0, π).
    wavelengths : tuple         Gabor wavelengths in PIXELS/cycle (pixel ≈ 1° here;
                                small low-SF bank for mouse acuity, e.g. (6, 12)).
    sigma_scale : float         envelope SD = sigma_scale * wavelength.
    gamma : float               envelope aspect ratio (⟂ elongation).
    stats : tuple of str        subset of {'magnitude', 'phase', 'even', 'luminance'}.
    chunk_t : int               frames per FFT-conv chunk (memory bound).

    Returns
    -------
    X : ndarray (T, ny*nx*n_orient*len(wavelengths)*nch) float32 ; names : list[str]
    (nch = #channels implied by stats: magnitude→1, phase→2, even→1; luminance→1 per SF at io=0.)
    """
    movie = np.asarray(movie, np.float32)
    if presmooth_sigma:                              # optional pre-blur (smoothing-then-gabor)
        movie = gaussian_filter1d(movie, presmooth_sigma, axis=1, mode='nearest')
        movie = gaussian_filter1d(movie, presmooth_sigma, axis=2, mode='nearest')
    T, H, W = movie.shape
    wavelengths = np.atleast_1d(wavelengths)             # tolerate a scalar (e.g. (6))
    n_sf = len(wavelengths)
    if odegs is not None:                                # explicit orientation angles (deg) override n_orient
        thetas = np.deg2rad(np.atleast_1d(np.asarray(odegs, float)))
        n_orient = len(thetas)
    else:
        thetas = np.arange(n_orient) * np.pi / n_orient  # evenly spaced in [0, π)
    ny, nx, _pool = _make_reducer(H, W, grid, pool, sample_ds)   # grid mean-pool | point-sample
    # edge guard: drop samples whose window is mostly zero pad (largest sigma if multi-SF)
    keep = (f_edge_keep_mask(H, W, sample_ds, sigma_scale * float(np.max(wavelengths)),
                             edge_min_frac) if (edge_min_frac and pool == 'sample') else None)

    # (token, fn(re, ro, mag) -> (T,H,W)) per requested stat
    chan = []
    if 'magnitude' in stats:
        chan.append(('M', lambda re, ro, mag: mag))
    if 'phase' in stats:
        chan.append(('Pc', lambda re, ro, mag: np.where(mag > 1e-6, re / np.maximum(mag, 1e-6), 0.0)))
        chan.append(('Ps', lambda re, ro, mag: np.where(mag > 1e-6, ro / np.maximum(mag, 1e-6), 0.0)))
    if 'quadrature' in stats:                        # SIGNED even+odd, no magnitude taken (2 channels)
        chan.append(('Qe', lambda re, ro, mag: re))
        chan.append(('Qo', lambda re, ro, mag: ro))
    if 'even' in stats:
        chan.append(('Ev', lambda re, ro, mag: np.abs(re)))
    want_lum = 'luminance' in stats                  # DC/luminance-only wavelet (orientation-independent)
    if not chan and not want_lum:
        raise ValueError(f"no valid stats in {stats!r} (magnitude/phase/even/luminance)")

    pooled = {}                                      # (io, isf, token) -> (T,ny,nx)
    if backend == 'gpu':                             # GPU conv + GPU stat+pool (transfers only pooled)
        import f_conv_gpu as _cg
        pooled = _cg.f_gabor_pooled_gpu(movie, thetas, wavelengths, [tok for tok, _ in chan],
                                        pool, grid, sample_ds, sigma_scale=sigma_scale, gamma=gamma,
                                        chunk_t=chunk_t, device=device, lum=want_lum)
    else:
        if chan:
            for io, th in enumerate(thetas):
                for isf, lam in enumerate(wavelengths):
                    ev, od = _gabor_pair(th, lam, sigma_scale * lam, gamma)
                    re = _spatial_fftconv_chunked(movie, ev, chunk_t)
                    ro = _spatial_fftconv_chunked(movie, od, chunk_t)
                    mag = np.sqrt(re * re + ro * ro)
                    for tok, fn in chan:
                        pooled[(io, isf, tok)] = _pool(fn(re, ro, mag)).astype(np.float32)
        if want_lum:                                 # carrier-free envelope projection, once per SF (io=0)
            for isf, lam in enumerate(wavelengths):
                proj = _spatial_fftconv_chunked(movie, _gauss_env(sigma_scale * lam, gamma), chunk_t)
                pooled[(0, isf, 'Lum')] = _pool(np.sqrt(2.0) * np.abs(proj)).astype(np.float32)

    feats, names = [], []
    for iy in range(ny):
        for ix in range(nx):
            if keep is not None and not keep[iy, ix]:
                continue
            for io in range(n_orient):
                for isf in range(n_sf):
                    for tok, _ in chan:
                        feats.append(pooled[(io, isf, tok)][:, iy, ix])
                        names.append(f'gab{tok}_o{io}sf{isf}_r{iy}c{ix}')
                    if want_lum and io == 0:         # luminance is orientation-independent → emit once
                        feats.append(pooled[(0, isf, 'Lum')][:, iy, ix])
                        names.append(f'gabLum_o0sf{isf}_r{iy}c{ix}')
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
                             output='opponent', chunk_t=1000,
                             pool='grid', sample_ds=1, presmooth_sigma=None,
                             backend='cpu', device=None, odegs=None, edge_min_frac=None):
    """Adelson–Bergen spatiotemporal motion energy pooled to a grid — Tier 4a.

    `pool`/`sample_ds`/`presmooth_sigma` behave as in f_gabor_features: 'grid' MEAN-pools the
    dense per-pixel opponent/directional energy into the (ny,nx) tiling; 'sample' POINT-samples
    it every `sample_ds` pixels (grid-shaped output, r{iy}c{ix} names); `presmooth_sigma`
    Gaussian-blurs the movie before convolution.

    Direction- AND speed-selective energy (the principled motion model; what static
    Gabor energy + a linear readout cannot build). Per (orientation × spatial
    wavelength × temporal frequency): the complex spatial Gabor response
    Cs = G_even + i·G_odd is formed per frame, then temporally filtered with a
    complex temporal Gabor (T_even + i·T_odd) for one direction and its conjugate
    for the opposite; directional energy = |·|². `output`:
      'opponent' — E_dir1 − E_dir2 (signed direction-selective; 1 channel, 'mengopp').
      'both'     — E_dir1 and E_dir2 separately (2 channels, 'mengd0'/'mengd1'). Keeps the SUM
                   (total energy, direction-invariant) as well as the difference, so it separates
                   the three cases opponent maps to ~0: empty screen (E1=E2=0), static edge
                   (E1=E2 small), opposing motion / flicker (E1=E2 large). 2x the columns.
    Direction d0/d1 is per orientation (motion along the bar normal: θ=0 → horizontal,
    θ=90° → vertical); which of d0/d1 is which screen direction follows the kernel sign
    convention — verify with a drifting bar if the sign matters.
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
    if presmooth_sigma:                              # optional pre-blur (smoothing-then-gabor)
        movie = gaussian_filter1d(movie, presmooth_sigma, axis=1, mode='nearest')
        movie = gaussian_filter1d(movie, presmooth_sigma, axis=2, mode='nearest')
    T, H, W = movie.shape
    if odegs is not None:                                # explicit orientation angles (deg) override n_orient
        thetas = np.deg2rad(np.atleast_1d(np.asarray(odegs, float)))
        n_orient = len(thetas)
    else:
        thetas = np.arange(n_orient) * np.pi / n_orient  # evenly spaced in [0, π)
    wavelengths = np.atleast_1d(wavelengths)             # tolerate scalars (e.g. (8))
    temporal_freqs = np.atleast_1d(temporal_freqs)
    n_sf, n_tf = len(wavelengths), len(temporal_freqs)
    halo = temporal_support // 2
    tsig = temporal_sigma if temporal_sigma else temporal_support / 4.0
    ny, nx, _pool = _make_reducer(H, W, grid, pool, sample_ds)   # grid mean-pool | point-sample
    keep = (f_edge_keep_mask(H, W, sample_ds, sigma_scale * float(np.max(wavelengths)),
                             edge_min_frac) if (edge_min_frac and pool == 'sample') else None)

    n_out = 1 if output == 'opponent' else 2
    if backend == 'gpu':                             # GPU conv + GPU energy+pool (transfers only pooled)
        import f_conv_gpu as _cg
        store = _cg.f_meng_pooled_gpu(movie, thetas, wavelengths, temporal_freqs, pool, grid, sample_ds,
                                      temporal_support=temporal_support, temporal_sigma=tsig,
                                      sigma_scale=sigma_scale, gamma=gamma, output=output,
                                      chunk_t=chunk_t, device=device)
    else:
        sgab = {(io, isf): _gabor_pair(thetas[io], wavelengths[isf],
                                       sigma_scale * wavelengths[isf], gamma)
                for io in range(n_orient) for isf in range(n_sf)}
        tk = {itf: _temporal_quad(tf, temporal_support, tsig)
              for itf, tf in enumerate(temporal_freqs)}
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
#
# ADDING A KIND — this registry is not enough on its own. Checklist of every list that also needs the
# name (missing #2 is how 'fft' silently never built: no vis_spec, no render, then a KeyError):
#   1. this registry                             extractor fn + defaults + desc
#   2. f_feature_source._VISUAL_SOURCES          route via build_visual_blocks (not the cheap agg/pix
#                                                path) and force the monitor render (_RENDER_SOURCES)
#   3. f_feature_source.NO_TF_KINDS              no temporal axis -> carries `stats`, one spec per SF
#   4. f_feature_source.PYRAMID_KINDS            buildable by f_pyramid_grid (has an SF axis)
#   5. f_feature_source.KIND_KNOBS               kind-specific knobs, for the strict validation
#   6. f_feature_source._normalize_spec          which extra keys pass through to the extractor
#   7. f_feature_source._visual_spec_from_input  `reduce` -> pool/sample_ds/grid (only lattice kinds)
#   8. f_conv_autosave.AUTOSAVE_KINDS            conv-heavy enough to cache to SSD
#   9. f_conv_autosave.f_spec_fname              put `stats` in the cache filename
#  10. build_visual_blocks (below)               accepts backend=/device=; edge_min_frac applies
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
                 '√(even²+odd²), phase (cos/sin φ → velocity via FIR), even (|even|), '
                 'luminance (√2·|Gaussian-env proj|, DC term, io=0).'),
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
    'fft': {
        'fn': f_fft_features,
        'defaults': {'n_orient': 4, 'wavelengths': (6,), 'sigma_scale': 0.5, 'gamma': 1.0,
                     'stats': ('magnitude',), 'pool': 'sample', 'sample_ds': 4, 'chunk_t': 200},
        'desc': ('Tier 3b — windowed patch-FFT spectrum binned onto the gabor bank axes (#73). '
                 'Same window/positions as the gabor bank, but keeps the whole lattice per patch '
                 'instead of one coefficient per filter. stats: magnitude (sector-mean |F|), '
                 'quadrature (Re/Im at the nearest coefficient). ONE wavelength per call.'),
    },
    'fftglobal': {
        'fn': f_fft_global_features,
        'defaults': {'n_orient': 4, 'wavelengths': (6,), 'stats': ('magnitude',), 'chunk_t': 200},
        'desc': ('Tier 3c — PLAIN rfft2 of the whole frame: no patches, no envelope. Perfect '
                 'frequency resolution, ZERO localization — position lives entirely in the phase, so '
                 'stats=magnitude is shift-invariant and quadrature keeps position. ONE value per '
                 '(orientation x SF) target; names carry no r/c because there is no lattice.'),
    },
    'wavelet': {
        'fn': f_wavelet_features,
        'defaults': {'family': 'db2', 'n_levels': 3, 'decimate': True, 'chunk_t': 500},
        'desc': ('Tier 3d — separable 2-D wavelet. decimate=True → DWT, critically sampled '
                 '(coefficients == pixels, orthonormal, invertible) but shift-VARIANT; '
                 'decimate=False → SWT/a-trous, full resolution per level, redundant but '
                 'shift-INVARIANT. The 3 detail subbands LH/HL/HH ARE the orientations, so there is '
                 'no free orientation axis. Level (1=finest) is the SF axis.'),
    },
    # 'cnn'  : Tier 4b — pretrained-CNN feature maps (#52)           [planned]
}


def _pca_fit(X, n_comp, random_state=42):
    """PCA-compress X by `n_comp`: int = #components; float in (0,1) = smallest k reaching that
    cumulative variance fraction; `(frac, cap)` tuple = variance-fraction CAPPED at `cap` components
    (min of the two); None/0/≥n_features = no PCA. Returns (scores, pca_or_None)."""
    X = np.asarray(X)
    # Downcast float64→float32 to halve PCA peak RAM + speed the SVD, but only when SAFE: float32
    # resolves ~1.2e-7·max|X|, so it's lossless when that quantum is tiny vs the spread PCA fits (std)
    # and there's no overflow. Check on a row-subsample so we don't allocate a full-matrix temporary.
    if X.dtype == np.float64 and X.size:
        Xs = X[::max(1, X.shape[0] // 10000)]                    # strided view — no copy
        amax = float(max(abs(Xs.max()), abs(Xs.min())))
        std = float(Xs.std())
        quantum = np.finfo(np.float32).eps * amax               # abs resolution near the largest value
        if amax < np.finfo(np.float32).max / 8 and (std == 0.0 or quantum < 1e-4 * std):
            X = X.astype(np.float32)
        else:
            print(f'    [_pca_fit] range too wide for f32 → keeping float64 '
                  f'(max|X|={amax:.3g}, std={std:.3g})')
    d = X.shape[1]
    if not n_comp:
        return X, None
    if isinstance(n_comp, (tuple, list)) and len(n_comp) == 2:   # (variance-fraction, cap)
        frac, cap = float(n_comp[0]), min(int(n_comp[1]), d)
        pca = PCA(n_components=cap, random_state=random_state).fit(X)
        k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), frac) + 1)
        k = min(max(k, 1), cap)
        if k >= d:
            return X, None
        if k < cap:                                          # truncate the fitted PCA to k (no 2nd fit)
            pca.components_ = pca.components_[:k]
            pca.explained_variance_ = pca.explained_variance_[:k]
            pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:k]
            pca.singular_values_ = pca.singular_values_[:k]
            pca.n_components_ = k
        return pca.transform(X), pca
    if isinstance(n_comp, float) and 0 < n_comp < 1:        # variance-fraction
        pca = PCA(n_components=n_comp, svd_solver='full', random_state=random_state)
        return pca.fit_transform(X), pca
    n_comp = int(n_comp)
    if n_comp >= d:
        return X, None
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X), pca


def _svd_fit(X, n_comp, random_state=42):
    """UNCENTERED truncated SVD — the drop-in twin of `_pca_fit` that does NOT subtract the mean.

    PCA centers first, so its components describe FLUCTUATION about the time-average and the constant
    baseline is discarded. For a strictly non-negative block (gabor/fft/meng magnitude) that baseline is
    a real part of the signal — "nothing on screen" is 0, not the session mean — and centering moves the
    features onto a signed scale where that reference is gone. TruncatedSVD keeps the origin, so its
    first component typically points along the mean/energy direction.

    Same `n_comp` forms as _pca_fit (int | float variance-fraction | (frac, cap) | None/0 = identity)
    and the same (scores, obj_or_None) return, so every consumer that reads `.components_`,
    `.explained_variance_ratio_` or calls `.transform()` works unchanged.

    CAVEAT on the variance columns: sklearn defines TruncatedSVD's `explained_variance_ratio_` as the
    variance of the PROJECTED data over the total (centered) variance of X. The near-constant first
    SVD component therefore carries a large share of the NORM but little VARIANCE, so `var_kept` for an
    svd block is not on the same footing as for a pca block — compare svd to svd, pca to pca."""
    from sklearn.decomposition import TruncatedSVD
    X = np.asarray(X)
    d = X.shape[1]
    if not n_comp:
        return X, None
    if isinstance(n_comp, (tuple, list)) and len(n_comp) == 2:      # (variance-fraction, cap)
        frac, cap = float(n_comp[0]), min(int(n_comp[1]), d - 1)
        svd = TruncatedSVD(n_components=max(1, cap), random_state=random_state).fit(X)
        k = int(np.searchsorted(np.cumsum(svd.explained_variance_ratio_), frac) + 1)
        k = min(max(k, 1), cap)
        if k < cap:                                                 # truncate in place (no 2nd fit)
            svd.components_ = svd.components_[:k]
            svd.explained_variance_ = svd.explained_variance_[:k]
            svd.explained_variance_ratio_ = svd.explained_variance_ratio_[:k]
            svd.singular_values_ = svd.singular_values_[:k]
            svd.n_components = k
        return svd.transform(X), svd
    if isinstance(n_comp, float) and 0 < n_comp < 1:                # variance-fraction
        cap = max(1, d - 1)                                         # TruncatedSVD cannot take all d
        svd = TruncatedSVD(n_components=cap, random_state=random_state).fit(X)
        cum = np.cumsum(svd.explained_variance_ratio_)
        if cum[-1] < float(n_comp):        # d-1 comps do not reach the fraction -> keep all d, as
            return X, None                 # _pca_fit would (else svd always returns one comp fewer)
        k = int(np.searchsorted(cum, float(n_comp)) + 1)
        k = min(max(k, 1), cap)
        svd.components_ = svd.components_[:k]
        svd.explained_variance_ = svd.explained_variance_[:k]
        svd.explained_variance_ratio_ = svd.explained_variance_ratio_[:k]
        svd.singular_values_ = svd.singular_values_[:k]
        svd.n_components = k
        return svd.transform(X), svd
    n_comp = int(n_comp)
    if n_comp >= d:                                                 # identity (TruncatedSVD needs < d)
        return X, None
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    return svd.fit_transform(X), svd


def f_pca_group_spec(raw_names, granularity, n_pca):
    """Build the `{prefix: n_pca}` grouping dict for f_apply_block_pca from sampled-block column names
    (`gab{tok}_o{io}sf{isf}_r{iy}c{ix}` / `meng{dt}_o{io}sf{isf}tf{itf}_r{iy}c{ix}`, side-prefixed). The
    side (L/R) is stripped exactly as f_apply_block_pca does, so each group spans BOTH monitors.
      'filter' → one group per distinct filter (tok, orientation, SF[, TF]) — prefix = the name minus
                 its `_r{iy}c{ix}` sample suffix (e.g. 'gabM_o0sf0', 'mengopp_o0sf0tf0').
      'orient' → one group per (tok, orientation), pooling that orientation's SFs/TFs — prefix
                 '{tok}_o{io}sf' (the trailing 'sf' keeps o0 from matching o10).
    Each group gets `n_pca` (int | float | (frac,cap)). Returns an ordered dict; groups are mutually
    exclusive by construction (no prefix is a prefix of another)."""
    import re
    prefixes = []
    for n in raw_names:
        s = n.split('_', 1)[1] if '_' in n else n           # strip side prefix
        if granularity == 'filter':
            pre = re.sub(r'_r\d+c\d+$', '', s)
        elif granularity == 'orient':
            m = re.match(r'(.+?_o\d+)sf', s)
            pre = (m.group(1) + 'sf') if m else re.sub(r'_r\d+c\d+$', '', s)
        else:
            raise ValueError(f"granularity must be 'filter' or 'orient', got {granularity!r}")
        if pre not in prefixes:
            prefixes.append(pre)
    return {pre: n_pca for pre in prefixes}


def f_pca_group_cols(raw_names, pool_axis, n_pca):
    """POOL-AWAY grouping of sampled-block columns → an ordered {label: {'cols':[i…], 'n': n_pca}}.
    `pool_axis` names the axis(es) to COLLAPSE: 'ori' (orientation) | 'sf'/'wl' (spatial freq) | 'tf'
    (temporal freq) | 'all' (pool o+sf+tf → one group) | 'none' (pool nothing → per filter). A tuple
    pools several (e.g. ('sf','tf') → per orientation). Columns are grouped by the distinct combos of
    the axes NOT pooled (spatial samples r/c always pooled; token gab*/meng* always kept distinct).
    Unlike f_pca_group_spec (leading-prefix startswith), this keys by EXPLICIT column indices, so it
    handles the NON-leading sf/tf axes. Names: gab{tok}_o{io}sf{isf}_r{iy}c{ix} /
    meng{dt}_o{io}sf{isf}tf{itf}_r{iy}c{ix} (side-prefixed). Drives f_apply_block_pca's ('pool',axis,n)."""
    import re
    from collections import OrderedDict
    if pool_axis == 'all':
        pool = {'ori', 'sf', 'tf'}
    elif pool_axis in (None, 'none'):
        pool = set()
    elif isinstance(pool_axis, (tuple, list, set)):
        pool = {('sf' if a == 'wl' else a) for a in pool_axis}
    else:
        pool = {'sf' if pool_axis == 'wl' else pool_axis}
    groups = OrderedDict()
    for i, n in enumerate(raw_names):
        s = n.split('_', 1)[1] if '_' in n else n            # strip side prefix
        m = re.match(r'(.+?)_o(\d+)', s)
        tok, io = (m.group(1), m.group(2)) if m else (s, '')
        msf = re.search(r'sf(\d+)', s)
        mtf = re.search(r'tf(\d+)', s)
        key = [tok]                                          # token always kept distinct
        if 'ori' not in pool:
            key.append(f'o{io}')
        if 'sf' not in pool and msf:
            key.append(f'sf{msf.group(1)}')
        if 'tf' not in pool and mtf:
            key.append(f'tf{mtf.group(1)}')
        groups.setdefault('_'.join(key), {'cols': [], 'n': n_pca})['cols'].append(i)
    return groups


def f_apply_block_pca(X_img, raw_names, fname, n_pca):
    """Apply a block's `n_pca` spec to RAW channels → (X, names, pca, pca_groups).

    `n_pca`: int / float(0,1) / None/0 → PCA on the WHOLE block; dict {prefix: ncomp} →
    SEPARATE PCA per channel-name-prefix group (side-stripped), concatenated. Factored out
    of build_visual_blocks so the pooled feature-recon path applies the SAME (grouped)
    PCA — no drift. `raw_names` are the pre-PCA channel names.

    n_pca may also be a `('filter'|'orient', ncomp)` tuple — a GRANULARITY marker (string first
    element) expanded here to the per-filter / per-orientation `{prefix: ncomp}` dict via
    f_pca_group_spec. This is the fresh-build 'separate' knob (vs an int = whole-block 'combined',
    None = 'raw'), so a decode/encode search expresses PCA granularity in the plan with no cache.
    Or a `('pool', axis, ncomp)` triple → POOL-AWAY grouped PCA keyed by explicit column indices via
    f_pca_group_cols (handles the non-leading sf/tf axes); each group's columns are PCA'd to ncomp."""
    # ('pool', axis, ncomp) → pool-away grouped PCA by EXPLICIT column indices (sf/tf aren't leading
    # name prefixes, so startswith can't group them). Same pca/pca_groups/xslice shape as the dict path.
    if isinstance(n_pca, (tuple, list)) and len(n_pca) == 3 and n_pca[0] == 'pool':
        col_groups = f_pca_group_cols(raw_names, n_pca[1], n_pca[2])
        parts, out_names, pca, pca_groups = [], [], {}, {}
        for label, info in col_groups.items():
            cols = info['cols']
            Xg, p_g = _pca_fit(X_img[:, cols], info['n'])
            s0 = sum(part.shape[1] for part in parts)
            parts.append(Xg)
            pca[label] = p_g
            pca_groups[label] = {'src_cols': cols, 'xslice': (s0, s0 + Xg.shape[1])}
            out_names += ([f'{fname}_{label}_PC{i + 1}' for i in range(Xg.shape[1])] if p_g is not None
                          else [f'{fname}_{raw_names[c].split("_", 1)[-1]}' for c in cols])
        return np.concatenate(parts, axis=1), out_names, pca, pca_groups
    # granularity tuple → grouped dict (string first element distinguishes it from a (frac,cap) tuple).
    if isinstance(n_pca, (tuple, list)) and len(n_pca) == 2 and isinstance(n_pca[0], str):
        n_pca = f_pca_group_spec(raw_names, n_pca[0], n_pca[1])
    raw_dim = X_img.shape[1]
    stripped = [n.split('_', 1)[1] if '_' in n else n for n in raw_names]
    if isinstance(n_pca, dict):
        parts, out_names, pca = [], [], {}
        pca_groups, assigned = {}, [False] * raw_dim
        for prefix, ncomp in n_pca.items():
            cols = [i for i, s in enumerate(stripped)
                    if s.startswith(prefix) and not assigned[i]]
            if not cols:
                # A requested group can be absent from the actual channels — e.g. 'gabP' (phase) when
                # gabor was LOADED from a conv_repo that stores magnitude only. Skip it (warn) rather
                # than crash; the block just keeps the groups that do have channels.
                print(f"  [f_apply_block_pca] {fname}: PCA-group prefix {prefix!r} has no channels "
                      f"(present: {sorted({s.split('_o')[0] for s in stripped})}) — skipped")
                continue
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


def f_combine_blocks_pca(built_blocks, block_names, n_pca, name='featjoint',
                         zscore=False, random_state=42):
    """Concatenate the columns of `block_names` and fit ONE joint PCA → a single combined block
    `name`. Use with the source blocks loaded RAW (their `n_pca=None`) to do the feature PCA in a
    SINGLE step and avoid the double-PCA data loss (per-block truncate at load, THEN a second PCA
    before the decoder — the first truncation discards each block's low-variance tail before the
    blocks are ever seen jointly). `zscore=True` z-scores the concatenated raw columns before PCA
    (put blocks on a common scale); else raw. Adds `built_blocks[name]` = {X (T,k) PC scores, names,
    pca (fitted), raw_names, raw_dim, source_blocks} and returns built_blocks. Downstream: set
    feature_block=name, n_feature_pcs=None (this IS the only PCA — a further truncation would just
    re-cut it)."""
    Xs, raw_names = [], []
    for b in block_names:
        if b not in built_blocks:
            raise ValueError(f"f_combine_blocks_pca: block {b!r} not in built_blocks")
        Xb = np.asarray(built_blocks[b]['X'], float)
        Xs.append(Xb)
        raw_names += [f'{b}:{nm}' for nm in built_blocks[b]['names']]
    X = np.concatenate(Xs, axis=1)
    Xf = np.where(np.isnan(X), 0.0, X)
    if zscore:
        mu = Xf.mean(0); sd = Xf.std(0); sd[sd == 0] = 1.0
        Xf = (Xf - mu) / sd
    scores, pca = _pca_fit(Xf, n_pca)
    k = scores.shape[1]
    built_blocks[name] = {'X': scores.astype(np.float32),
                          'names': [f'{name}_PC{i + 1}' for i in range(k)] if pca is not None else list(raw_names),
                          'pca': pca, 'raw_names': raw_names, 'raw_dim': X.shape[1],
                          'source_blocks': list(block_names)}
    return built_blocks


def build_visual_blocks(movies, side_tags, beh_t, frame_t, pulse_delay,
                        specs, default_n_pca=None, resample_kind='linear',
                        backend='cpu', device=None, verbose=True, autosave=None,
                        edge_min_frac=None):
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
    autosave : dict | None                 f_conv_autosave.f_autosave_ctx(...) → reload the pooled
        gabor/meng/fft block from SSD when one matching the resolved params exists, else build+save.
        Cached PRE-PCA so one block serves every PCA / proc / marginalize variant. None = off.
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
    # deferred: f_conv_autosave → f_conv_repo → f_visual_features would be a circular import
    _cas = None
    if autosave:
        import f_conv_autosave as _cas
    blocks = {}
    for fname, params in specs.items():
        p = dict(params or {})
        # `type` lets a spec KEY be an arbitrary block name (e.g. 'gaborS_ds4') mapping to a
        # registry extractor — so several configs of one extractor become distinct named blocks.
        ftype = p.pop('type', fname)
        if ftype not in VISUAL_FEATURE_REGISTRY:
            raise ValueError(f'unknown visual feature type {ftype!r} (spec {fname!r}); '
                             f'registered: {list(VISUAL_FEATURE_REGISTRY)}')
        reg = VISUAL_FEATURE_REGISTRY[ftype]
        merged = dict(reg['defaults'])
        merged.update(p)
        p = merged
        n_pca = p.pop('n_pca', default_n_pca)
        if ftype in ('gabor', 'meng'):               # GPU conv backend (only these extractors accept it)
            p.setdefault('backend', backend)
            p.setdefault('device', device)
        # edge guard: run-wide default from build_cfg; a per-spec edge_min_frac wins
        if edge_min_frac is not None and ftype in ('gabor', 'meng', 'fft'):
            p.setdefault('edge_min_frac', edge_min_frac)

        # SSD autosave (opt-in): the conv+pool is the expensive part and depends only on the resolved
        # params, so reload it when a matching block exists. Cached PRE-PCA / pre-proc, so one saved
        # block serves every downstream PCA / proc / marginalize variant. This is the single hook every
        # build path funnels through, so the search and cebra share one store.
        _as_dir = _as_path = _as_key = None
        X_img = None
        if autosave and ftype in _cas.AUTOSAVE_KINDS:
            _as_dir = autosave['dir']
            _fn, _as_key = _cas.f_spec_fname(fname, ftype, p, side_tags,
                                             [m.shape for m in movies], len(np.asarray(frame_t).ravel()))
            _as_path = _as_dir / _fn
            _hit = _cas.f_autosave_load(_as_dir, _as_path)
            if _hit is not None:
                X_img, raw_names = np.asarray(_hit[0]['X']), list(_hit[0]['raw_names'])
                if autosave.get('verbose', True):
                    print(f'    autosave HIT {fname!r} ({_fn})', flush=True)

        if X_img is None:
            _t0 = time.perf_counter()
            Xs, names = [], []
            for movie, tag in zip(movies, side_tags):
                Xb, base = reg['fn'](np.asarray(movie), **p)     # real pixels (renders a lazy movie)
                Xs.append(Xb)
                names.extend(f'{tag}_{b}' for b in base)
            X_beh = np.concatenate(Xs, axis=1)
            if verbose:
                print(f'    ⏱ visual extract {fname!r}: {time.perf_counter() - _t0:.2f}s')

            X_img = f_resample_to_imaging(beh_t, X_beh, frame_t, pulse_delay,
                                          kind=resample_kind)
            raw_names = list(names)    # pre-PCA channel names (for reconstruction)
            if _as_path is not None:   # persist for later configs / runs
                _cas.f_autosave_put(_as_dir, _as_path,
                                    {'X': X_img, 'names': raw_names, 'raw_names': raw_names,
                                     'raw_dim': X_img.shape[1], 'kind': fname},
                                    frame_t, _as_key, _as_key,
                                    verbose=autosave.get('verbose', True))
        raw_dim = X_img.shape[1]
        X_img, names, pca, pca_groups = f_apply_block_pca(X_img, raw_names, fname, n_pca)

        blocks[fname] = {'X': X_img, 'names': names, 'pca': pca,
                         'raw_dim': raw_dim, 'params': p, 'raw_names': raw_names,
                         'pca_groups': pca_groups}
    return blocks
