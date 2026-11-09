# -*- coding: utf-8 -*-
"""
Feature visualization — reconstruct a monitor movie from a feature block.

Turns a (PCA-compressed) feature block back into an image-space movie so you can
SEE what each representation keeps, and how that degrades with the number of
retained PCs. The start of the feature-analysis/visualization module planned in
PLAN_monitor_features.md (TODO #47).

Per method ("kind"):
  'pix'  — true pixel reconstruction: inverse-PCA of the pix block → reshape to
           frames (a denoised / rank-k version of the actual monitor movie).
  'grid' — paint each retinotopic cell with a chosen stat (occ/mean/edge):
           a blocky low-res view of what the grid representation encodes.
  'flow' — paint each cell with a chosen flow stat (speed/div): the motion-
           energy field the flow representation encodes.
  'gabor'— paint each cell with one orientation×SF magnitude channel (or phase/even);
           channel='gabM_o0sf0' selects it ('auto' = first).
  'meng' — paint each cell with one direction×SF×temporal-freq motion-energy channel;
           channel='mengopp_o0sf0tf0' selects it ('auto' = first).

All use rank-k reconstruction: zero the PCA scores beyond `n_pcs`, then
inverse_transform. Movies come out on the imaging clock (the block's clock).
Save with f_functions.f_save_mon_movie (multi-page TIFF → ImageJ).

Created 2026-06-09.
"""

import re

import numpy as np

from f_visual_features import (f_grid_bounds, _gabor_pair, _gauss_env, _temporal_quad,
                               _spatial_fftconv_chunked, f_gabor_bank, _sample_idx)
from f_feature_helpers import f_resample_to_imaging

_CELL_RE = re.compile(r'r(\d+)c(\d+)$')   # trailing grid-cell id in any channel name


def f_resolve_recon_n_pcs(n_pcs, pca):
    """Resolve an n_pcs request to an integer component count.

    Mirrors sklearn's n_components convention so the recon cell can be driven by
    a variance target instead of a hard count:
      int k        → use k components (rank-k), as before.
      float (0,1)  → smallest k whose cumulative variance reaches that fraction of
                     the block's RETAINED variance (e.g. 0.9 → enough PCs for 90%
                     of what this already-compressed block holds, so 'all' = 1.0).
                     Normalized by the kept PCs' total so it's meaningful even when
                     the block itself only captures part of the original variance.
      1.0 or None  → all available components.
    A fraction needs a fitted PCA (pca is None for raw/uncompressed blocks); in
    that case the request is returned unchanged (None → all raw channels).
    """
    if pca is None or n_pcs is None:
        return n_pcs
    if isinstance(n_pcs, float) and 0.0 < n_pcs < 1.0:
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr) / np.sum(evr)              # fraction of RETAINED var
        k = int(np.searchsorted(cum, n_pcs) + 1)        # first k reaching target
        return min(k, pca.n_components_)
    return int(n_pcs)                                    # int, or float 1.0 → all


def _truncated_inverse(scores, pca, n_pcs=None):
    """Rank-k inverse-PCA: keep the first n_pcs score columns, zero the rest,
    then inverse_transform. n_pcs=None → use all components. (Resolve fractional
    variance targets with f_resolve_recon_n_pcs before calling.)"""
    s = scores
    if n_pcs is not None and n_pcs < s.shape[1]:
        s = s.copy()
        s[:, n_pcs:] = 0.0
    return pca.inverse_transform(s)


def _recon_raw(block, n_pcs):
    """Inverse-PCA a visual block back to its raw per-cell channels (T, raw_dim),
    handling single PCA, grouped (per-stat) PCA, or no PCA. raw_names give the
    column layout."""
    pca = block.get('pca')
    groups = block.get('pca_groups')
    X = np.asarray(block['X'])
    if isinstance(pca, dict) and groups:                # grouped PCA: per-group inverse
        raw = np.zeros((X.shape[0], block['raw_dim']), np.float32)
        for prefix, p_g in pca.items():
            s0, s1 = groups[prefix]['xslice']
            cols = groups[prefix]['src_cols']
            Xg = X[:, s0:s1]
            raw[:, cols] = (_truncated_inverse(Xg, p_g, f_resolve_recon_n_pcs(n_pcs, p_g))
                            if p_g is not None else Xg)
        return raw
    if pca is not None:
        return _truncated_inverse(X, pca, f_resolve_recon_n_pcs(n_pcs, pca))
    return X                                            # uncompressed → X is raw


def _minmax255(a):
    """Scale an array to uint8 [0, 255] by global min–max (flat → zeros)."""
    lo = float(np.nanmin(a)); hi = float(np.nanmax(a))
    if hi <= lo:
        return np.zeros(a.shape, np.float32)
    return (a - lo) / (hi - lo) * 255.0


def _parse_cell(base_name):
    """Grid cell (iy, ix) from a channel name's TRAILING 'rNcM'. Works for any
    prefix: 'mean_r2c3', 'speed_r0c1', 'gabM_o0sf0_r0c0', 'mengopp_o0sf0tf0_r0c0'.
    Returns None if there's no trailing cell id."""
    m = _CELL_RE.search(base_name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _cell_prefix(base_name):
    """Channel prefix = name minus the trailing 'rNcM' (e.g. 'gabM_o0sf0_r0c0' →
    'gabM_o0sf0'). None if no trailing cell id."""
    m = _CELL_RE.search(base_name)
    return base_name[:m.start()].rstrip('_') if m else None


def _render_cells(raw_side, base_names, grid, mon_hw, channel):
    """Paint each grid cell with its `channel` value → (T, H, W) blocky movie.

    raw_side : (T, d) reconstructed per-side feature columns.
    base_names : list[str] d channel names (e.g. 'mean_r0c0'), no side prefix.
    channel : prefix to render ('occ' | 'mean' | 'edge' | 'speed' | 'div').
    """
    T = raw_side.shape[0]
    H, W = mon_hw
    ys, xs = f_grid_bounds(H, W, grid)
    frame = np.zeros((T, H, W), np.float32)
    painted = 0
    for li, bn in enumerate(base_names):
        if not bn.startswith(channel + '_'):
            continue
        cell = _parse_cell(bn)
        if cell is None:
            continue
        iy, ix = cell
        frame[:, ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]] = raw_side[:, li][:, None, None]
        painted += 1
    if painted == 0:
        raise ValueError(f"no channel {channel!r} columns found; "
                         f"available prefixes: "
                         f"{sorted({n.split('_', 1)[0] for n in base_names})}")
    return frame


def _grid_from_names(names):
    """Infer the (ny, nx) sampling lattice from `..._r{iy}c{ix}` channel names = (max iy+1,
    max ix+1). Works for grid-pooled AND point-sampled blocks; returns None if no name matches."""
    import re
    pat = re.compile(r'_r(\d+)c(\d+)$')
    iy = ix = -1
    for nm in names:
        m = pat.search(nm)
        if m:
            iy = max(iy, int(m.group(1)))
            ix = max(ix, int(m.group(2)))
    return (iy + 1, ix + 1) if iy >= 0 else None


def f_reconstruct_feature_movie(block, block_name, mon_hw, side_tags,
                                n_pcs=None, channel='auto', normalize='auto',
                                frame_idx=None):
    """Reconstruct an image-space monitor movie from a feature block.

    Parameters
    ----------
    block : dict                 a built_blocks entry (pix / grid / flow / gabor / meng).
    block_name : str             'pix' | 'grid' | 'flow' | 'gabor' | 'meng'.
    mon_hw : (H, W)              ONE monitor's frame size (e.g. (num_samp, num_samp)).
    side_tags : list[str]        e.g. ['R'] or ['L', 'R']; sides are concatenated
                                 along width in the output.
    n_pcs : int | float | None   # PCs for the rank-k reconstruction. int = that
                                 many; float in (0,1) = enough PCs for that
                                 cumulative variance fraction (e.g. 0.9); None /
                                 1.0 = all. See f_resolve_recon_n_pcs.
    channel : str                grid: 'occ'|'mean'|'edge'; flow: 'speed'|'div';
                                 gabor: 'gabM_o{io}sf{isf}' (or gabPc/gabPs/gabEv);
                                 meng: 'meng{opp|d0|d1}_o{io}sf{isf}tf{itf}'.
                                 'auto' → 'mean' (grid) / 'speed' (flow) / first
                                 channel prefix (gabor/meng). Ignored for pix.
    normalize : 'auto'|'clip'|'minmax'
                                 'clip' (pix default) keeps native 0–255; 'minmax'
                                 (grid/flow default) rescales to 0–255.
    frame_idx : sequence | None  if given, reconstruct ONLY these frames (slices the
                                 block scores first) → returns (len(frame_idx), H, W*ns).
                                 Cheap for a few-frame overview / comparison.

    Returns
    -------
    movie : ndarray (T, H, W*n_sides) uint8  — pass to f_save_mon_movie.
    """
    n_sides = len(side_tags)
    H, W = mon_hw
    if frame_idx is not None:                  # reconstruct only the requested frames
        block = {**block, 'X': np.asarray(block['X'])[list(frame_idx)]}

    if block_name == 'pix':
        k = f_resolve_recon_n_pcs(n_pcs, block['pca_pix'])
        raw = _truncated_inverse(block['X'], block['pca_pix'], k)   # (T, H*W*n_sides)
        T = raw.shape[0]
        mov = raw.reshape(T, H, W * n_sides)
        mov = np.clip(mov, 0, 255) if normalize in ('auto', 'clip') else _minmax255(mov)
        return mov.astype(np.uint8)

    # visual blocks (grid / flow / gabor ...): inverse-PCA → per-side render
    raw = _recon_raw(block, n_pcs)
    raw_names = block.get('raw_names')
    if raw_names is None:
        raise ValueError(f"block {block_name!r} has no 'raw_names'; rebuild it "
                         "with the current build_visual_blocks to enable recon.")
    # Prefer the (iy,ix) lattice inferred from the r{iy}c{ix} names — correct for BOTH grid-pooled
    # and point-SAMPLED blocks (where params['grid'] is the ignored default). Fall back to params.
    grid = _grid_from_names(raw_names) or block.get('params', {}).get('grid', (6, 6))
    if channel == 'auto':
        if block_name == 'grid':
            channel = 'mean'
        elif block_name == 'flow':
            channel = 'speed'
        else:                                  # gabor/meng/...: first channel prefix
            channel = _cell_prefix(raw_names[0].split('_', 1)[1])

    psd = raw.shape[1] // n_sides                       # per-side dim
    side_frames = []
    for s in range(n_sides):
        cols = slice(s * psd, (s + 1) * psd)
        base = [n.split('_', 1)[1] for n in raw_names[cols]]   # strip side tag
        side_frames.append(_render_cells(raw[:, cols], base, grid, mon_hw, channel))
    mov = np.concatenate(side_frames, axis=2)          # concat sides on width
    mov = np.clip(mov, 0, 255) if normalize == 'clip' else _minmax255(mov)
    return mov.astype(np.uint8)


def f_reconstruct_gabor_phase_movie(block, mon_hw, side_tags, io=0, isf=0,
                                    n_pcs=None, value_pct=99.0, cmap='hsv',
                                    frame_idx=None):
    """Phase movie for a gabor block, one orientation (io) × SF (isf).

    Maps φ = atan2(sinφ, cosφ) (from the gabPs / gabPc channels) to color per grid
    cell, darkened by magnitude (gabM) so empty cells (no edge) are BLACK and phase
    appears only where there's an edge. As an edge drifts the color cycles → motion
    shows. `cmap`:
      'hsv'   — cyclic hue = φ, value = magnitude (default; no wrap discontinuity).
      else    — any matplotlib colormap applied to φ∈[0,1] (e.g. 'gray', 'twilight'),
                then multiplied by normalized magnitude. NOTE for a non-cyclic map
                like 'gray' the ±π wrap is a black↔white seam; 'twilight' is cyclic.
    Companion to the grayscale MAGNITUDE movie (channel='gabM_o{io}sf{isf}' via
    f_reconstruct_feature_movie). Needs stats=('magnitude','phase').

    Parameters
    ----------
    block : dict                 a built_blocks['gabor'] entry.
    mon_hw : (H, W)              one monitor's frame size.
    side_tags : list[str]        sides; concatenated along width.
    io, isf : int                orientation / SF index to render.
    n_pcs : int | float | None   rank for the inverse-PCA (per group).
    value_pct : float            percentile of magnitude mapped to full brightness.
    cmap : str                   'hsv' (cyclic) or any matplotlib colormap name.

    Returns
    -------
    movie : ndarray (T, H, W*n_sides, 3) uint8 RGB — pass to f_save_mon_movie.
    """
    from matplotlib.colors import hsv_to_rgb
    from matplotlib import colormaps
    n_sides = len(side_tags)
    if frame_idx is not None:                  # reconstruct only the requested frames
        block = {**block, 'X': np.asarray(block['X'])[list(frame_idx)]}
    raw = _recon_raw(block, n_pcs)
    raw_names = block['raw_names']
    grid = block.get('params', {}).get('grid', (6, 6))
    pm, pc, ps = (f'gabM_o{io}sf{isf}', f'gabPc_o{io}sf{isf}', f'gabPs_o{io}sf{isf}')

    psd = raw.shape[1] // n_sides
    huevals = []
    for s in range(n_sides):
        cols = slice(s * psd, (s + 1) * psd)
        base = [n.split('_', 1)[1] for n in raw_names[cols]]
        Rsub = raw[:, cols]
        M = _render_cells(Rsub, base, grid, mon_hw, pm)
        C = _render_cells(Rsub, base, grid, mon_hw, pc)
        S = _render_cells(Rsub, base, grid, mon_hw, ps)
        hue = (np.arctan2(S, C) + np.pi) / (2 * np.pi)   # φ → [0,1)
        huevals.append((hue, M))
    vmax = np.percentile(np.concatenate([m for _, m in huevals], axis=2), value_pct)
    vmax = vmax if vmax > 0 else 1.0
    frames = []
    for hue, M in huevals:
        v = np.clip(M / vmax, 0, 1)                       # magnitude brightness
        if cmap == 'hsv':
            rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), v], axis=-1))
        else:
            rgb = colormaps[cmap](hue)[..., :3] * v[..., None]   # phase→cmap, ×magnitude
        frames.append(rgb)                               # (T,H,W,3) in [0,1]
    rgb = np.concatenate(frames, axis=2)                 # concat sides on width
    return (rgb * 255).astype(np.uint8)


def f_gabor_flow_field(block, side_tags, isf=0, n_pcs=None, reg=1e-2, sign=-1.0,
                       frame_range=None):
    """Decode a per-cell 2-D motion field from a gabor block's phase dynamics.

    Phase-based optic flow: for each orientation io the wrapped temporal change of
    phase dφ/dt (from the gabPc/gabPs unit phase) gives the NORMAL flow component
    (across the bars): v·n_io = s_io, with n_io = (cosθ, sinθ),
    s_io = sign · (dφ/dt) · λ / (2π) (px/frame). Combine the orientations by
    magnitude-weighted least squares per cell/frame → the full (vx, vy), resolving
    the aperture problem. Needs stats=('magnitude','phase') and ≥2 orientations.

    sign : ±1 — overall direction convention; flip if arrows point the wrong way.
    reg  : Tikhonov term on the 2×2 solve (stabilizes low-energy / single-orient cells).
    frame_range : (start, stop) | None — compute flow over a CONTIGUOUS slice only
                  (needed for long sessions; flow is a temporal derivative so the slice
                  must be contiguous).

    Returns dict: vx, vy, speed, mag each (n_sides, T, ny, nx) [px/frame]; grid; wavelength.
    """
    if frame_range is not None:
        block = {**block, 'X': np.asarray(block['X'])[frame_range[0]:frame_range[1]]}
    raw = _recon_raw(block, n_pcs)
    raw_names = block['raw_names']
    params = block.get('params', {})
    grid = params.get('grid', (6, 6)); ny, nx = grid
    n_orient = params.get('n_orient', 4)
    waves = np.atleast_1d(params.get('wavelengths', (6,)))
    lam = float(waves[isf])
    thetas = np.arange(n_orient) * np.pi / n_orient
    n_sides, T = len(side_tags), raw.shape[0]
    psd = raw.shape[1] // n_sides
    ct = np.cos(thetas)[None, :, None, None]
    st = np.sin(thetas)[None, :, None, None]

    out = {k: [] for k in ('vx', 'vy', 'speed', 'mag')}
    for s in range(n_sides):
        Rsub = raw[:, s * psd:(s + 1) * psd]
        idx = {n.split('_', 1)[1]: i for i, n in enumerate(raw_names[s * psd:(s + 1) * psd])}
        M = np.zeros((T, n_orient, ny, nx), np.float32)
        C = np.zeros_like(M); S = np.zeros_like(M)
        for io in range(n_orient):
            for iy in range(ny):
                for ix in range(nx):
                    M[:, io, iy, ix] = Rsub[:, idx[f'gabM_o{io}sf{isf}_r{iy}c{ix}']]
                    C[:, io, iy, ix] = Rsub[:, idx[f'gabPc_o{io}sf{isf}_r{iy}c{ix}']]
                    S[:, io, iy, ix] = Rsub[:, idx[f'gabPs_o{io}sf{isf}_r{iy}c{ix}']]
        # wrapped dφ = angle(u_t · conj(u_{t-1})), u = cosφ + i sinφ
        dphi = np.zeros_like(M)
        dphi[1:] = np.arctan2(S[1:] * C[:-1] - C[1:] * S[:-1],
                              C[1:] * C[:-1] + S[1:] * S[:-1])
        s_norm = sign * dphi * lam / (2 * np.pi)         # normal speed per orient
        w = M                                            # magnitude = confidence
        A00 = (w * ct * ct).sum(1); A01 = (w * ct * st).sum(1); A11 = (w * st * st).sum(1)
        b0 = (w * s_norm * ct).sum(1); b1 = (w * s_norm * st).sum(1)
        det = A00 * A11 - A01 * A01 + reg
        vx = (A11 * b0 - A01 * b1) / det
        vy = (A00 * b1 - A01 * b0) / det
        out['vx'].append(vx); out['vy'].append(vy)
        out['speed'].append(np.sqrt(vx * vx + vy * vy)); out['mag'].append(M.mean(1))
    for k in out:
        out[k] = np.stack(out[k])                        # (n_sides, T, ny, nx)
    out['grid'] = grid; out['wavelength'] = lam
    return out


def f_render_flow_quiver_movie(flow, mon_hw, side_tags, stride=1, frame_range=None,
                               scale=None, min_speed=0.0, bg_gain=1.0, dpi=100,
                               mag_gate=0.15, speed_pct=98.0, norm='movie',
                               arrow_frac=0.8, arrow_width=0.004):
    """Quiver movie of a decoded flow field (f_gabor_flow_field) — one arrow per grid
    cell over a faint magnitude background; arrow length/color = speed, direction =
    motion. Sides tiled along width. Returns (T', Hpx, Wpx, 3) uint8 RGB.

    Arrows are GATED by gabor magnitude (a cell with magnitude < `mag_gate` × reference
    gets NO arrow — kills the spurious arrows in no-edge cells, whose decoded velocity is
    magnitude-independent) and drawn at a FIXED scale set from the `speed_pct` speed
    percentile, so a given speed is the same arrow length in EVERY frame (no per-frame
    autoscale blow-up). `norm` = the window for the magnitude reference + the speed scale:
    'movie' (global, default) or 'frame' (per-frame, the old look). A numeric `scale`
    overrides the auto scale; `mag_gate=0` disables gating. `arrow_frac` = how many cell
    widths a `speed_pct` arrow spans (raise it — e.g. 1.5/2.5 — for bigger arrows);
    `arrow_width` = shaft thickness (raise — e.g. 0.008 — for fatter arrows)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    H, W = mon_hw
    ny, nx = flow['grid']
    n_sides = len(side_tags)
    Wn = W * n_sides
    ys, xs = f_grid_bounds(H, W, (ny, nx))
    yc = (ys[:-1] + ys[1:]) / 2.0
    xc = (xs[:-1] + xs[1:]) / 2.0
    # cell-center pixel coords for all sides
    X = np.concatenate([np.add.outer(np.ones(ny), xc + s * W).ravel()
                        for s in range(n_sides)])
    Y = np.concatenate([np.add.outer(yc, np.ones(nx)).ravel() for _ in range(n_sides)])

    T = flow['vx'].shape[1]
    sel = list(range(*(frame_range or (0, T))))[::stride]
    magn = flow['mag']                                   # (n_sides, T, ny, nx)
    speed = flow['speed']
    arrow_px = arrow_frac * float(xs[1] - xs[0])         # spd_ref arrow ≈ arrow_frac cells

    def _refs(mag_arr, spd_arr):
        """Magnitude ref + speed ref over a window — the speed ref is taken over the
        GATED (high-magnitude) cells ONLY, so spurious no-signal velocities don't
        inflate the scale and shrink the real arrows."""
        mr = float(np.percentile(mag_arr, 99)) or 1.0
        gated = spd_arr[mag_arr >= mag_gate * mr]
        sr = float(np.percentile(gated if gated.size else spd_arr, speed_pct))
        return mr, max(sr, 1e-6)

    mref_g, sref_g = _refs(magn, speed)                  # global (full movie)

    fig = Figure(figsize=(Wn / dpi, H / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    out = []
    for t in sel:
        mref, sref = _refs(magn[:, t], speed[:, t]) if norm == 'frame' else (mref_g, sref_g)
        qscale = scale if scale is not None else (sref / arrow_px)

        ax.clear(); ax.set_axis_off()
        ax.set_xlim(0, Wn); ax.set_ylim(H, 0)            # image coords (y down)
        bg = np.concatenate([np.kron(magn[s, t], np.ones((ys[1] - ys[0], xs[1] - xs[0])))
                             [:H, :W] for s in range(n_sides)], axis=1)
        ax.imshow(np.clip(bg / mref * bg_gain, 0, 1), cmap='gray', vmin=0, vmax=1,
                  extent=[0, Wn, H, 0], aspect='auto', zorder=0)
        U = np.concatenate([flow['vx'][s, t].ravel() for s in range(n_sides)])
        V = np.concatenate([flow['vy'][s, t].ravel() for s in range(n_sides)])
        cmag = np.concatenate([magn[s, t].ravel() for s in range(n_sides)])
        spd = np.sqrt(U * U + V * V)
        m = (spd >= min_speed) & (cmag >= mag_gate * mref)   # gate by magnitude + speed
        ax.quiver(X[m], Y[m], U[m], V[m], spd[m], cmap='autumn', angles='xy',
                  scale_units='xy', scale=qscale, width=arrow_width, zorder=1)  # fixed len+dir
        canvas.draw()
        out.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return np.stack(out)


def f_gabor_flow_dense(movie, n_orient=4, wavelength=8.0, sigma_scale=0.5, gamma=1.0,
                       smooth_t=2.0, smooth_xy=1.0, reg=1e-2, sign=-1.0, chunk_t=1500):
    """FULL-RESOLUTION phase-based optic flow from one monitor movie (T, H, W).

    Recomputes the Gabor responses at EVERY pixel (not the pooled grid), so the flow
    is high-res and free of the feature block's PCA-reconstruction noise. Per
    orientation, dφ/dt (temporally smoothed) → normal flow; magnitude-weighted least
    squares across orientations → (vx, vy) per pixel; optional spatial smoothing.
    Slice the movie to a frame range first to keep the convolution cost down.

    Returns dict: vx, vy, speed, mag — each (T, H, W) float32 (px/frame).
    """
    from scipy.ndimage import gaussian_filter1d, gaussian_filter
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    thetas = np.arange(n_orient) * np.pi / n_orient
    A00 = np.zeros((T, H, W), np.float32)
    A01 = np.zeros_like(A00); A11 = np.zeros_like(A00)
    b0 = np.zeros_like(A00); b1 = np.zeros_like(A00); magsum = np.zeros_like(A00)
    for th in thetas:
        ev, od = _gabor_pair(th, wavelength, sigma_scale * wavelength, gamma)
        re = _spatial_fftconv_chunked(movie, ev, chunk_t)
        ro = _spatial_fftconv_chunked(movie, od, chunk_t)
        dphi = np.zeros((T, H, W), np.float32)           # wrapped dφ via u_t·conj(u_{t-1})
        dphi[1:] = np.arctan2(ro[1:] * re[:-1] - re[1:] * ro[:-1],
                              re[1:] * re[:-1] + ro[1:] * ro[:-1])
        if smooth_t:
            dphi = gaussian_filter1d(dphi, smooth_t, axis=0)
        s_norm = sign * dphi * wavelength / (2 * np.pi)
        w = np.sqrt(re * re + ro * ro)
        ct, stt = np.cos(th), np.sin(th)
        A00 += w * ct * ct; A01 += w * ct * stt; A11 += w * stt * stt
        b0 += w * s_norm * ct; b1 += w * s_norm * stt; magsum += w
    det = A00 * A11 - A01 * A01 + reg
    vx = (A11 * b0 - A01 * b1) / det
    vy = (A00 * b1 - A01 * b0) / det
    if smooth_xy:
        vx = gaussian_filter(vx, (0, smooth_xy, smooth_xy))
        vy = gaussian_filter(vy, (0, smooth_xy, smooth_xy))
    return {'vx': vx.astype(np.float32), 'vy': vy.astype(np.float32),
            'speed': np.sqrt(vx * vx + vy * vy).astype(np.float32),
            'mag': (magsum / n_orient).astype(np.float32)}


def f_render_flow_color(flow, speed_pct=98.0, mag_gate=True):
    """Optical-flow COLOR for a dense flow dict (f_gabor_flow_dense): hue = motion
    DIRECTION, brightness = speed (×magnitude when mag_gate, so noise in empty
    regions stays dark). Smooth and full-resolution. Returns (T, H, W, 3) uint8 RGB."""
    from matplotlib.colors import hsv_to_rgb
    vx, vy, spd, mag = flow['vx'], flow['vy'], flow['speed'], flow['mag']
    hue = (np.arctan2(vy, vx) + np.pi) / (2 * np.pi)
    vmax = np.percentile(spd, speed_pct) or 1.0
    val = np.clip(spd / vmax, 0, 1)
    if mag_gate:
        val = val * np.clip(mag / (np.percentile(mag, 99) or 1.0), 0, 1)
    hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
    return (hsv_to_rgb(hsv) * 255).astype(np.uint8)


def f_render_flow_quiver_dense(flow, arrow_step=6, stride=1, frame_range=None,
                               scale=None, min_speed=0.0, bg_gain=1.0, dpi=100):
    """Quiver of a dense flow dict — arrows every `arrow_step` pixels over the
    full-res magnitude background (finer + smoother than the grid quiver).
    Returns (T', Hpx, Wpx, 3) uint8 RGB."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    vx, vy, mag = flow['vx'], flow['vy'], flow['mag']
    T, H, W = vx.shape
    yy, xx = np.mgrid[arrow_step // 2:H:arrow_step, arrow_step // 2:W:arrow_step]
    X, Y = xx.ravel(), yy.ravel()
    sel = list(range(*(frame_range or (0, T))))[::stride]
    mref = np.percentile(mag, 99) or 1.0
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    out = []
    for t in sel:
        ax.clear(); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.imshow(np.clip(mag[t] / mref * bg_gain, 0, 1), cmap='gray',
                  extent=[0, W, H, 0], zorder=0)
        U = vx[t, yy, xx].ravel(); V = vy[t, yy, xx].ravel()
        spd = np.sqrt(U * U + V * V); m = spd >= min_speed
        ax.quiver(X[m], Y[m], U[m], V[m], spd[m], cmap='autumn', angles='xy',
                  scale=scale, width=0.003, zorder=1)
        canvas.draw()
        out.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return np.stack(out)


def f_stack_recon_over_original(mov_recon, orig_frames, beh_t, frame_t, pulse_delay,
                                gap_px=4, normalize='clip', resample_kind='nearest'):
    """Stack a reconstructed movie OVER the matched original monitor movie.

    Top panel = mov_recon (reconstruction, already on the imaging clock); bottom
    panel = the source monitor movie resampled to the SAME imaging clock so the
    two play frame-for-frame in sync. A black gap separates them. Use to eyeball
    what a representation keeps vs. the real scene.

    The original is on the behavior clock and the reconstruction on the imaging
    clock (the pix block PCAs behavior-clock frames, then resamples the scores),
    so we resample the raw frames here with the same f_resample_to_imaging the
    build cell uses — otherwise the two panels would drift out of sync.

    Parameters
    ----------
    mov_recon : ndarray (T_img, H, Wn) uint8   f_reconstruct_feature_movie output.
    orig_frames : ndarray (T_beh, H, Wn)       source monitor movie, behavior
                                               clock (e.g. two_mon_frames). Width
                                               must match mov_recon (same sides).
    beh_t, frame_t, pulse_delay                clock-alignment (same as build cell).
    gap_px : int                               black separator rows between panels.
    normalize : 'clip' | 'minmax'              map the original to uint8.
    resample_kind : str                        interp kind for f_resample_to_imaging.

    Returns
    -------
    stacked : ndarray (T_img, H*2 + gap_px, Wn) uint8 — pass to f_save_mon_movie.
    """
    Tr, Hr, Wr = mov_recon.shape
    orig = np.asarray(orig_frames)
    Tb, Ho, Wo = orig.shape
    if (Ho, Wo) != (Hr, Wr):
        raise ValueError(f"original frame size {(Ho, Wo)} != reconstruction "
                         f"{(Hr, Wr)}; check side selection / mon_hw.")

    # behavior clock → imaging clock (match mov_recon's frames exactly)
    flat = orig.reshape(Tb, -1).astype(np.float32)
    flat_img = f_resample_to_imaging(beh_t, flat, frame_t, pulse_delay,
                                     kind=resample_kind)
    orig_img = flat_img.reshape(-1, Hr, Wr)
    if orig_img.shape[0] != Tr:
        raise ValueError(f"resampled original has {orig_img.shape[0]} frames but "
                         f"reconstruction has {Tr}; frame_t mismatch.")
    orig_img = (np.clip(orig_img, 0, 255) if normalize == 'clip'
                else _minmax255(orig_img)).astype(np.uint8)

    gap = np.zeros((Tr, gap_px, Wr), np.uint8)
    return np.concatenate([mov_recon, gap, orig_img], axis=1)


def f_plot_movie_frames(mov, frame_idx=None, n=6, cmap='gray', titles=None,
                        suptitle='', max_cols=6, vmin=None, vmax=None):
    """Show a few example frames from a movie for eyeballing / picking frames to
    analyze. Handles grayscale (T,H,W) and RGB (T,H,W,3). `frame_idx` = explicit
    indices; None → `n` evenly-spaced. Returns (selected_frames, idx) so the frames
    can be reused downstream. Pass the SAME frame_idx to compare movies at one time."""
    import matplotlib.pyplot as plt
    mov = np.asarray(mov)
    T = mov.shape[0]
    if frame_idx is None:
        frame_idx = list(np.unique(np.linspace(0, T - 1, min(n, T)).round().astype(int)))
    frame_idx = [int(i) for i in frame_idx]
    k = len(frame_idx)
    ncol = min(max_cols, k)
    nrow = int(np.ceil(k / ncol))
    rgb = (mov.ndim == 4)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), squeeze=False)
    for j, fi in enumerate(frame_idx):
        ax = axes[j // ncol][j % ncol]
        if rgb:
            ax.imshow(mov[fi])
        else:
            ax.imshow(mov[fi], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(titles[j] if titles is not None else f't={fi}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(k, nrow * ncol):           # blank any unused panels
        axes[j // ncol][j % ncol].axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return mov[frame_idx], frame_idx


def f_plot_movie_frames_compare(frame_stacks, frame_idx, labels=None, cmaps=None,
                                suptitle='', field=None, field_idx=None, overlay_rows=None,
                                mon_hw=None, side_tags=None, mag_gate=0.15, speed_pct=98.0,
                                norm='movie', arrow_frac=1.5, arrow_width=0.004):
    """Compare several already-sampled frame stacks at the SAME timepoints: a grid with one
    ROW per stack and one COLUMN per frame. Each stack is (n,H,W) grayscale or (n,H,W,3) RGB
    with the SAME n; `frame_idx` (len n) labels the columns. `labels` name the rows.

    Optional GENERIC quiver overlay: if `field` (a motion-field dict {vx,vy,speed,mag,grid}
    from ANY source — e.g. f_gabor_flow_at) is given, draw vector arrows on the rows named by
    `overlay_rows` (None/'all' = every row; or a list of row indices / label strings) at flow
    position `field_idx[c]` per column. Needs `mon_hw` + `side_tags`. Gating/scale match the
    quiver movie (`mag_gate`, `speed_pct` over gated cells, `arrow_frac`)."""
    import matplotlib.pyplot as plt
    stacks = [np.asarray(s) for s in frame_stacks]
    nrow, ncol = len(stacks), len(frame_idx)
    labels = labels or [f'movie {i}' for i in range(nrow)]
    cmaps = cmaps or ['gray'] * nrow

    ov = set()
    if field is not None:
        ov = (set(range(nrow)) if overlay_rows in (None, 'all')
              else {it if isinstance(it, int) else labels.index(it) for it in overlay_rows})
        H, W = mon_hw
        ny, nx = field['grid']
        n_sides = len(side_tags)
        Wn = W * n_sides
        ys, xs = f_grid_bounds(H, W, (ny, nx))
        yc = (ys[:-1] + ys[1:]) / 2.0
        xc = (xs[:-1] + xs[1:]) / 2.0
        Xg = np.concatenate([np.add.outer(np.ones(ny), xc + s * W).ravel() for s in range(n_sides)])
        Yg = np.concatenate([np.add.outer(yc, np.ones(nx)).ravel() for _ in range(n_sides)])
        magn, spd_all = field['mag'], field['speed']
        arrow_px = arrow_frac * float(xs[1] - xs[0])
        fq = [int(i) for i in (field_idx if field_idx is not None else frame_idx)]

        def _refs(m, sp):
            mr = float(np.percentile(m, 99)) or 1.0
            g = sp[m >= mag_gate * mr]
            return mr, max(float(np.percentile(g if g.size else sp, speed_pct)), 1e-6)
        mref_g, sref_g = _refs(magn, spd_all)

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), squeeze=False)
    for r, s in enumerate(stacks):
        rgb = (s.ndim == 4)
        for c in range(ncol):
            ax = axes[r][c]
            frame = s[c] if c < s.shape[0] else np.zeros_like(s[0])
            _ext = [0, Wn, H, 0] if (field is not None and r in ov) else None
            if rgb:
                ax.imshow(frame, extent=_ext)
            else:
                ax.imshow(frame, cmap=cmaps[r], extent=_ext)
            if field is not None and r in ov:
                t = fq[c]
                mref, sref = _refs(magn[:, t], spd_all[:, t]) if norm == 'frame' else (mref_g, sref_g)
                U = np.concatenate([field['vx'][sd, t].ravel() for sd in range(n_sides)])
                V = np.concatenate([field['vy'][sd, t].ravel() for sd in range(n_sides)])
                cm = np.concatenate([magn[sd, t].ravel() for sd in range(n_sides)])
                mm = cm >= mag_gate * mref
                ax.quiver(Xg[mm], Yg[mm], U[mm], V[mm], np.sqrt(U[mm]**2 + V[mm]**2),
                          cmap='autumn', angles='xy', scale_units='xy', scale=sref / arrow_px,
                          width=arrow_width, zorder=2)
                ax.set_xlim(0, Wn); ax.set_ylim(H, 0)
            if r == 0:
                ax.set_title(f't={int(frame_idx[c])}', fontsize=8)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def f_gabor_flow_at(block, side_tags, frame_idx, isf=0, n_pcs=None, sign=-1.0, win=1):
    """Per-patch motion field at a SET of (possibly non-contiguous) frames — each computed
    from a local window (f-win, f+1) via f_gabor_flow_field, taking its last frame. Returns
    the standard field dict {vx,vy,speed,mag,grid,wavelength} with T=len(frame_idx).
    Reconstructs only ~len(frame_idx)*(win+1) frames — never a big block (for long sessions)."""
    fields = {k: [] for k in ('vx', 'vy', 'speed', 'mag')}
    grid = wl = None
    for f in frame_idx:
        a = max(0, int(f) - win)
        fl = f_gabor_flow_field(block, side_tags, isf=isf, n_pcs=n_pcs, sign=sign,
                                frame_range=(a, int(f) + 1))
        for k in fields:
            fields[k].append(fl[k][:, -1])          # (n_sides, ny, nx) at frame f
        grid, wl = fl['grid'], fl['wavelength']
    out = {k: np.stack(v, axis=1) for k, v in fields.items()}   # (n_sides, T, ny, nx)
    out['grid'], out['wavelength'] = grid, wl
    return out


def f_pick_motion_frames(built_blocks, n=6, mode='spread', block='pix', frame_range=None,
                         min_gap=None, stride=1, win=1):
    """Frame indices at high MOTION, from a cheap per-frame score = ‖Δ block['X']‖ over time
    (block='pix' → change in the pixel-PCA scores = any on-screen motion; 'motion'/'meng'
    also work). `mode='spread'` → top-n frames spread across the session (≥ min_gap apart;
    default span//(2n)); `mode='event'` → a run of n frames (step `stride`) centered on the
    peak-motion frame — 'a few frames THROUGH one motion event'. Frames < `win` are excluded
    (the quiver needs a predecessor). Returns sorted frame indices."""
    X = np.asarray(built_blocks[block]['X'], dtype=float)
    X = np.nan_to_num(X)
    T = X.shape[0]
    score = np.zeros(T)
    score[1:] = np.linalg.norm(np.diff(X, axis=0), axis=1)
    a, b = frame_range or (0, T)
    valid = np.arange(max(a, win), min(b, T))
    if valid.size == 0:
        return list(range(min(n, T)))
    if mode == 'event':
        peak = int(valid[np.argmax(score[valid])])
        idx = np.clip(peak + (np.arange(n) - n // 2) * stride, int(valid[0]), T - 1)
        return sorted(set(int(i) for i in idx))
    if min_gap is None:
        min_gap = max(1, (int(valid[-1]) - int(valid[0])) // (2 * max(n, 1)))
    picked = []
    for f in valid[np.argsort(score[valid])[::-1]]:
        if all(abs(int(f) - p) >= min_gap for p in picked):
            picked.append(int(f))
            if len(picked) >= n:
                break
    return sorted(picked)


def f_plot_conv_hist(samples, df=None, bins=80, logy=True, max_panels=16, suptitle=''):
    """Per-filter value HISTOGRAMS from f_conv_repo.f_conv_repo_scan (`samples` = {name: 1-D values}).
    One panel per filter (up to `max_panels`), log-y by default so heavy outlier tails are visible;
    median-|value| (green) and 99.9th-|value| (red) marked. If `df` (the scan table) is given, the
    panel title shows the tail ratio (absmax / p50_abs). Returns None (Spyder-safe)."""
    import matplotlib.pyplot as plt
    names = list(samples)[:max_panels]
    if not names:
        print('f_plot_conv_hist: no samples'); return None
    tail = {r['name']: r.get('tail') for _, r in df.iterrows()} if df is not None else {}
    n = len(names); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow), squeeze=False)
    for j, nm in enumerate(names):
        ax = axes[j // ncol][j % ncol]
        v = np.asarray(samples[nm], float)
        ax.hist(v, bins=bins, color='steelblue')
        if logy:
            ax.set_yscale('log')
        av = np.abs(v)
        ax.axvline(np.median(av), color='seagreen', lw=0.8)       # median |value|
        ax.axvline(np.percentile(av, 99.9), color='C3', lw=0.8, ls='--')   # 99.9th |value|
        ax.axvline(-np.percentile(av, 99.9), color='C3', lw=0.8, ls='--')
        _t = f'{nm}' + (f'  tail={tail[nm]:.0f}' if tail.get(nm) is not None else '')
        ax.set_title(_t, fontsize=6.5)
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(suptitle or 'conv_repo per-filter value histograms (log-y; green=median|·|, red=99.9%)')
    fig.tight_layout()
    return None


def f_plot_conv_activity(df, ts, by='tfreq', kind=None, reduce_label='|response|',
                         suptitle='', max_points=2000):
    """Diagnostic plot of conv_repo feature ACTIVITY (from f_conv_repo.f_conv_activity): a 2-panel
    figure grouped by `by` ('odeg' | 'wavelength' | 'tfreq'), aggregating (mean) over all other
    axes incl. sides — (L) mean response-magnitude trace over time per group; (R) one overall
    number (time × space mean) per group. `kind` filters to 'gabor'|'meng' (use kind='meng',
    by='tfreq' to pick temporal freqs). Rows with a NaN `by` (e.g. tfreq on gabor) are dropped.
    Returns None (Spyder-safe)."""
    import matplotlib.pyplot as plt
    d = df.reset_index(drop=True)
    keep = np.ones(len(d), bool)
    if kind is not None:
        keep &= (d['kind'] == kind).to_numpy()
    keep &= d[by].notna().to_numpy()
    d = d[keep].reset_index(drop=True)
    tsk = np.asarray(ts)[keep]
    if len(d) == 0:
        print(f'f_plot_conv_activity: no rows for by={by!r} kind={kind!r}')
        return None
    groups = sorted(d[by].unique())
    by_vals = d[by].to_numpy()
    scal = d['scalar'].to_numpy()
    T = tsk.shape[1]
    step = max(1, T // max_points)
    xt = np.arange(0, T, step)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(groups))))
    bar_h = []
    for g, c in zip(groups, cmap):
        m = by_vals == g
        ax[0].plot(xt, tsk[m].mean(axis=0)[::step], color=c, lw=1.0, label=f'{by}={g}')
        bar_h.append(float(scal[m].mean()))
    ax[0].set_xlabel('frame'); ax[0].set_ylabel(f'mean {reduce_label} over space')
    ax[0].set_title('response magnitude over time'); ax[0].legend(fontsize=7)
    ax[1].bar([str(g) for g in groups], bar_h, color=cmap)
    ax[1].set_xlabel(by); ax[1].set_ylabel(f'mean {reduce_label} (time × space)')
    ax[1].set_title(f'overall activity per {by}')
    # context: which OTHER axes' values this panel averaged over (odeg / wavelength / tfreq).
    _ctx = []
    for _pair in [('odeg',), ('wavelength', 'isf'), ('tfreq', 'itf')]:
        if by in _pair:
            continue
        _col = next((c for c in _pair if c in d.columns and d[c].notna().any()), None)
        if _col is not None:
            _ctx.append(f'{_col} {sorted(d[_col].dropna().unique())}')
    _ctx = ('  (avg over ' + '; '.join(_ctx) + ')') if _ctx else ''
    fig.suptitle((suptitle or f'conv activity by {by}' + (f' ({kind})' if kind else '')) + _ctx)
    fig.tight_layout()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Driver helpers — original-frame lookup + movie-pair save (used by VR_feature_viz_cache.py)
# Parameterized (no module-global state) so the driver script stays thin.
# ─────────────────────────────────────────────────────────────────────────────

def f_orig_for_side(arr, side_tags):
    """Crop an L|R original-frame array to the recon's side(s) so widths match. `arr` (T,H,W|2W);
    returns it unchanged for two-sided (L,R) recon, else the matching half."""
    if arr is None or len(side_tags) == 2:
        return arr
    _h = arr.shape[2] // 2
    return arr[:, :, :_h] if side_tags[0] == 'L' else arr[:, :, _h:]


def f_orig_frames(pidx, orig_mov, side_tags, example_frames=None, example_frame_idx=None):
    """Original frames at imaging-frame indices `pidx`, cropped to side: from the cheap saved
    `example_frames` when pidx == example_frame_idx, else from the full `orig_mov`, else None."""
    if pidx is None:
        return None
    if example_frames is not None and list(pidx) == list(example_frame_idx):
        return f_orig_for_side(np.asarray(example_frames), side_tags)
    if orig_mov is not None:
        return f_orig_for_side(np.asarray(orig_mov)[list(pidx)], side_tags)
    return None


def f_frame_indices(n_total, want, n, example_frames=None, example_frame_idx=None):
    """Indices to display/save: `want` (explicit list) → as-is; else the saved example_frame_idx;
    else `n` evenly-spaced over n_total."""
    if want is not None:
        return [int(i) for i in want]
    if example_frames is not None:
        return list(example_frame_idx)
    return list(np.unique(np.linspace(0, n_total - 1, min(n, n_total)).round().astype(int)))


def f_save_movie_pair(feat_mov, stem, out_dir, ds, mov_tag='', orig_mov=None, side_tags=None,
                      save_idx=None, save_orig=False, pair=True):
    """Save an already-frame-subset feature movie as <out_dir>/<stem>_<ds><_tag>.tif, and (when
    save_orig and pair) the matching ORIGINAL as <...>_ORIG.tif at the SAME frames. `save_idx` =
    the imaging-frame indices `feat_mov` was built at (for the original). Prints what it wrote."""
    from f_functions import f_save_mon_movie
    _mt = f'_{mov_tag}' if mov_tag else ''
    f_save_mon_movie(feat_mov, f'{out_dir}/{stem}_{ds}{_mt}.tif', gap_px=0)
    msg = f'saved {stem} → {out_dir}/{stem}_{ds}{_mt}.tif'
    if pair and save_orig:
        if orig_mov is None or save_idx is None:
            msg += '  (save_orig set but no original movie — run the Original-movie cell)'
        else:
            _o = f_orig_for_side(np.asarray(orig_mov)[list(save_idx)], side_tags)
            f_save_mon_movie(_o, f'{out_dir}/{stem}_{ds}{_mt}_ORIG.tif', gap_px=0)
            msg += '  (+_ORIG)'
    print(msg)


def f_block_activity(built_blocks, block_names=None, reduce='absmean', repo_dir=None):
    """Approximate feature ACTIVITY from ALREADY-LOADED sampled blocks — NO disk re-read (reuses the
    data f_load_conv_blocks already pulled in, so it also honours whatever load filters were applied).
    Frame-mean over each filter's SAMPLED cells (a subsample of monitor space ≈ the full-space mean).
    For each conv sampled block (gaborS_ds*/mengS_ds*): inverse-PCA → raw per-cell values, group
    channels by (stat, odeg, isf, itf) over sides+cells, reduce per frame → trace, mean over time →
    scalar. Returns (df, ts) keyed by odeg / isf / itf (SF & TF INDICES). Pass `repo_dir` to also map
    isf→wavelength and itf→tfreq (from the manifest) → `wavelength`/`tfreq` columns with the ACTUAL
    px/cyc VALUES, so meng can be plotted by real temporal freq (not just the index). Plot with
    f_plot_conv_activity(..., by='odeg' | 'isf' | 'itf' | 'wavelength' | 'tfreq')."""
    import pandas as pd
    _pat = re.compile(r'^(?P<stat>.+?)_o(?P<odeg>\d+)sf(?P<isf>\d+)(?:tf(?P<itf>\d+))?$')
    if block_names is None:
        block_names = [b for b, blk in built_blocks.items()
                       if blk.get('params', {}).get('pool') == 'sample']
    _wl, _tf = {}, {}                                       # (kind,isf)→wavelength, (kind,itf)→tfreq
    if repo_dir is not None:
        from f_conv_repo import f_list_conv_filters
        for e in f_list_conv_filters(repo_dir):
            _wl[(e['kind'], e['isf'])] = e['wavelength']
            if e.get('itf') is not None:
                _tf[(e['kind'], e['itf'])] = e.get('tfreq')
    rows, traces = [], []
    for bn in block_names:
        blk = built_blocks[bn]
        kind = blk.get('kind')
        raw = _recon_raw(blk, None)                        # (T, raw_dim) per-cell values
        groups = {}                                        # channel prefix (no side/cell) → col idxs
        for j, nm in enumerate(blk['raw_names']):
            groups.setdefault(_cell_prefix(nm.split('_', 1)[1]), []).append(j)
        for pfx, cols in groups.items():
            m = _pat.match(pfx or '')
            if not m:
                continue
            x = raw[:, cols]                               # (T, n over sides+cells)
            if reduce == 'mean':
                tr = x.mean(axis=1)
            elif reduce == 'rms':
                tr = np.sqrt((x * x).mean(axis=1))
            else:
                tr = np.abs(x).mean(axis=1)
            isf = int(m['isf'])
            itf = int(m['itf']) if m['itf'] is not None else None
            rows.append({'block': bn, 'kind': kind, 'stat': m['stat'], 'odeg': int(m['odeg']),
                         'isf': isf, 'itf': itf,
                         'wavelength': _wl.get((kind, isf)),
                         'tfreq': (None if itf is None else _tf.get((kind, itf))),
                         'scalar': float(tr.mean())})
            traces.append(tr.astype(np.float32))
    return pd.DataFrame(rows), (np.vstack(traces) if traces else np.zeros((0, 0), np.float32))


# ── feature-KERNEL overlay (verify gabor/meng size + sampling vs the monitor) ─────────────────────
def f_kernel_geom(inp, cam_params, lat_samp, vert_samp, sigma_scale=0.5):
    """Physical geometry of ONE normalized feature spec `inp` (from f_feature_source._normalize_spec) on a
    (vert_samp × lat_samp) monitor. Uses the SAME conversions as extraction (f_deg_per_px / f_ds_from_wl):
    sf_cyc_deg → wavelength px → σ px → kernel support; ds (Nyquist or literal) → sample lattice. Returns a
    readout dict (no plotting) for the size/sampling troubleshooting table."""
    from f_conv_repo import f_deg_per_px, f_ds_from_wl
    dpp = f_deg_per_px(cam_params, lat_samp, vert_samp)          # deg/px (same as extraction)
    sf = float(np.atleast_1d(inp['sf_cyc_deg'])[0])
    wl_px = 1.0 / (sf * dpp)
    sigma = sigma_scale * wl_px
    support = 2 * int(np.ceil(3 * sigma)) + 1
    if inp.get('ds') == 'nyquist' or inp.get('ds_factor') is not None:
        ds = f_ds_from_wl(wl_px, factor=float(inp.get('ds_factor', 0.5)))
    else:
        ds = int(np.atleast_1d(inp.get('ds', 1))[0])
    ny, nx = len(_sample_idx(vert_samp, ds)), len(_sample_idx(lat_samp, ds))   # CENTERED lattice
    return {'source': inp.get('source'), 'sf_cyc_deg': sf, 'wl_px': wl_px, 'sigma_px': sigma,
            'support_px': support, 'ds': ds, 'n_samples': (ny, nx), 'n_tiles': ny * nx,
            'tf_hz': (float(np.atleast_1d(inp['tf_hz'])[0]) if inp.get('tf_hz') is not None else None)}


def f_plot_filter_kernel(frame, inp, cam_params, lat_samp, vert_samp, *, draw='even', io=0,
                         sigma_scale=0.5, gamma=1.0, sigma_extent=3.0, show_points=True,
                         show_grid=False, bg=None, ax=None, title=None):
    """Show ONE filter on a monitor `frame` (H×W, one monitor): mark ALL sample locations (where the filter
    is sampled from) as points, and draw the actual gabor TO SCALE at the sample nearest screen center. The
    kernel is built by the SHARED builders (change them → this changes): draw='even'/'odd' → _gabor_pair
    (θ = odegs[io]° or io·180/n_orient); 'env' → _gauss_env (luminance). Drawn out to ±`sigma_extent`·σ
    (2σ = bulk of the weighting). Sample lattice from ds (Nyquist or literal). `bg` = plain background
    instead of the frame: a matplotlib color ('black'/'white'/'gray') or a 0-255 gray scalar; None = show
    the movie `frame` (which may then be None). Returns f_kernel_geom's readout. For verifying gabor SIZE +
    SAMPLING density."""
    import matplotlib.pyplot as plt
    g = f_kernel_geom(inp, cam_params, lat_samp, vert_samp, sigma_scale)
    H, W = (frame.shape[:2] if frame is not None else (vert_samp, lat_samp))
    odegs = inp.get('odegs'); n_orient = inp.get('n_orient') or 1
    theta_deg = float(np.atleast_1d(odegs)[io]) if odegs is not None else io * 180.0 / n_orient
    theta = np.deg2rad(theta_deg)
    g['theta_deg'] = theta_deg
    if draw == 'env':                                           # luminance-only wavelet (Gaussian env)
        kern = _gauss_env(g['sigma_px'], gamma)
    else:                                                       # full gabor pair; pick even (cos) / odd (sin)
        ev, od = _gabor_pair(theta, g['wl_px'], g['sigma_px'], gamma)
        kern = ev if draw == 'even' else od
    m = max(1, int(round(sigma_extent * g['sigma_px'])))        # crop to ±sigma_extent·σ (drawn extent)
    c0 = kern.shape[0] // 2
    kern = kern[max(0, c0 - m):c0 + m + 1, max(0, c0 - m):c0 + m + 1]
    kh, kw = kern.shape
    ds = g['ds']
    rows = list(_sample_idx(H, ds)); cols = list(_sample_idx(W, ds))   # CENTERED on screen middle
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5 * H / W + 0.4))
    if bg is not None:                                          # plain background (choose color) — no frame
        ax.set_facecolor((float(bg) / 255,) * 3 if isinstance(bg, (int, float)) else bg)
        ax.set_aspect('equal')
    else:                                                       # overlay on the real monitor frame
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255, origin='upper', aspect='equal')
    if show_grid:
        for r in rows:
            ax.axhline(r, color='cyan', lw=0.3, alpha=0.4)
        for cc in cols:
            ax.axvline(cc, color='cyan', lw=0.3, alpha=0.4)
    if show_points:                                             # ALL sample locations this filter reads from
        Xg, Yg = np.meshgrid(cols, rows)
        ax.scatter(Xg.ravel(), Yg.ravel(), s=6, c='yellow', edgecolors='k', linewidths=0.2, zorder=2)
    cy = rows[int(np.argmin([abs(r - H / 2) for r in rows]))]   # sample nearest screen center
    cx = cols[int(np.argmin([abs(cc - W / 2) for cc in cols]))]
    amax = float(np.abs(kern).max()) or 1.0                     # ONE gabor to scale at that sample.
    rgba = plt.get_cmap('bwr')(np.clip(kern / amax * 0.5 + 0.5, 0, 1))   # explicit RGBA (per-pixel alpha
    rgba[..., 3] = np.clip(np.abs(kern) / amax, 0, 1)           # works on ALL mpl; 2D `alpha=` needs ≥3.4)
    ax.imshow(rgba, zorder=3, origin='upper',
              extent=[cx - kw / 2, cx + kw / 2, cy + kh / 2, cy - kh / 2])
    ax.scatter([cx], [cy], marker='+', c='lime', s=80, linewidths=1.2, zorder=4)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    _tf = '' if g['tf_hz'] is None else f"  tf={g['tf_hz']:g}Hz"
    ax.set_title(title or (f"{g['source']} θ={theta_deg:g}°  sf={g['sf_cyc_deg']:g}cyc/deg{_tf}\n"
                           f"λ={g['wl_px']:.1f}px  σ={g['sigma_px']:.1f}px  ±{sigma_extent:g}σ={kh}px  "
                           f"ds={ds}px  n={g['n_samples'][0]}×{g['n_samples'][1]}"), fontsize=8)
    return g


def f_kernel_gallery(n_orient=4, wavelengths=(6, 12), sigma_scale=0.5, gamma=1.0, which='even',
                     suptitle='Gabor bank kernels'):
    """Gallery of the raw bank kernels (orientation × SF) via f_gabor_bank — the 'here are the filters'
    panel. `which` = 'even' | 'odd'. wavelengths in PIXELS (convert cyc/deg → px first for a physical bank)."""
    import matplotlib.pyplot as plt
    bank = f_gabor_bank(n_orient=n_orient, wavelengths=wavelengths, sigma_scale=sigma_scale, gamma=gamma)
    n_sf = len(np.atleast_1d(wavelengths))
    fig, axes = plt.subplots(n_orient, n_sf, figsize=(2.2 * n_sf, 2.2 * n_orient), squeeze=False)
    for k in bank:
        ax = axes[k['io']][k['isf']]
        ker = k['even'] if which == 'even' else k['odd']
        v = float(np.abs(ker).max()) or 1.0
        ax.imshow(ker, cmap='bwr', vmin=-v, vmax=v)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"θ={np.degrees(k['theta']):.0f}°  λ={k['wavelength']:.0f}px ({ker.shape[0]}px)",
                     fontsize=8)
    fig.suptitle(suptitle); fig.tight_layout()
    return bank
