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

Created 2026-06-06 — Tier 1 (grid). Tiers 2–4 (flow / gabor / motion-energy /
cnn) register here as they're implemented.
"""

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
    # 'gabor': Tier 3 — Gabor energy bank + grid (#50)               [planned]
    # 'meng' : Tier 4a — Adelson–Bergen motion energy (#51)          [planned]
    # 'cnn'  : Tier 4b — pretrained-CNN feature maps (#52)           [planned]
}


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
        defaults). A per-type 'n_pca' key (in params) triggers PCA compression of
        that block; else default_n_pca.
    default_n_pca : int | None             fallback PCA dim if a spec omits 'n_pca'.
    resample_kind : str                    interp kind for f_resample_to_imaging.

    Returns
    -------
    dict {feat_name: {'X': (T_img, d), 'names': list, 'pca': PCA|None,
                      'raw_dim': int, 'params': dict}}
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

        Xs, names = [], []
        for movie, tag in zip(movies, side_tags):
            Xb, base = reg['fn'](movie, **p)
            Xs.append(Xb)
            names.extend(f'{tag}_{b}' for b in base)
        X_beh = np.concatenate(Xs, axis=1)

        X_img = f_resample_to_imaging(beh_t, X_beh, frame_t, pulse_delay,
                                      kind=resample_kind)
        raw_dim = X_img.shape[1]
        raw_names = list(names)        # pre-PCA channel names (for reconstruction)
        pca = None
        if n_pca and raw_dim > n_pca:
            pca = PCA(n_components=n_pca, random_state=42)
            X_img = pca.fit_transform(X_img)
            names = [f'{fname}_PC{i + 1}' for i in range(n_pca)]

        blocks[fname] = {'X': X_img, 'names': names, 'pca': pca,
                         'raw_dim': raw_dim, 'params': p, 'raw_names': raw_names}
    return blocks
