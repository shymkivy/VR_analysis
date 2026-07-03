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

from f_visual_features import (f_grid_bounds, _gabor_pair,
                               _spatial_fftconv_chunked)
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
    grid = block.get('params', {}).get('grid', (6, 6))
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
