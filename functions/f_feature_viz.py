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

All use rank-k reconstruction: zero the PCA scores beyond `n_pcs`, then
inverse_transform. Movies come out on the imaging clock (the block's clock).
Save with f_functions.f_save_mon_movie (multi-page TIFF → ImageJ).

Created 2026-06-09.
"""

import numpy as np

from f_visual_features import f_grid_bounds
from f_feature_helpers import f_resample_to_imaging


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


def _minmax255(a):
    """Scale an array to uint8 [0, 255] by global min–max (flat → zeros)."""
    lo = float(np.nanmin(a)); hi = float(np.nanmax(a))
    if hi <= lo:
        return np.zeros(a.shape, np.float32)
    return (a - lo) / (hi - lo) * 255.0


def _parse_cell(base_name):
    """'mean_r2c3' / 'speed_r0c1' → (iy, ix). Returns None if not a cell name."""
    try:
        rc = base_name.split('_', 1)[1]           # 'r2c3'
        ci = rc.index('c')
        return int(rc[1:ci]), int(rc[ci + 1:])
    except (IndexError, ValueError):
        return None


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
                                n_pcs=None, channel='auto', normalize='auto'):
    """Reconstruct an image-space monitor movie from a feature block.

    Parameters
    ----------
    block : dict                 a built_blocks entry (pix / grid / flow / ...).
    block_name : str             'pix' | 'grid' | 'flow' (+ future visual kinds).
    mon_hw : (H, W)              ONE monitor's frame size (e.g. (num_samp, num_samp)).
    side_tags : list[str]        e.g. ['R'] or ['L', 'R']; sides are concatenated
                                 along width in the output.
    n_pcs : int | float | None   # PCs for the rank-k reconstruction. int = that
                                 many; float in (0,1) = enough PCs for that
                                 cumulative variance fraction (e.g. 0.9); None /
                                 1.0 = all. See f_resolve_recon_n_pcs.
    channel : str                grid: 'occ'|'mean'|'edge'; flow: 'speed'|'div';
                                 'auto' → 'mean' (grid) / 'speed' (flow). Ignored
                                 for pix.
    normalize : 'auto'|'clip'|'minmax'
                                 'clip' (pix default) keeps native 0–255; 'minmax'
                                 (grid/flow default) rescales to 0–255.

    Returns
    -------
    movie : ndarray (T, H, W*n_sides) uint8  — pass to f_save_mon_movie.
    """
    n_sides = len(side_tags)
    H, W = mon_hw

    if block_name == 'pix':
        k = f_resolve_recon_n_pcs(n_pcs, block['pca_pix'])
        raw = _truncated_inverse(block['X'], block['pca_pix'], k)   # (T, H*W*n_sides)
        T = raw.shape[0]
        mov = raw.reshape(T, H, W * n_sides)
        mov = np.clip(mov, 0, 255) if normalize in ('auto', 'clip') else _minmax255(mov)
        return mov.astype(np.uint8)

    # visual blocks (grid / flow / gabor ...): inverse-PCA → per-side render
    pca = block.get('pca')
    k = f_resolve_recon_n_pcs(n_pcs, pca)
    raw = (_truncated_inverse(block['X'], pca, k) if pca is not None
           else np.asarray(block['X']))                # uncompressed → X is raw
    raw_names = block.get('raw_names')
    if raw_names is None:
        raise ValueError(f"block {block_name!r} has no 'raw_names'; rebuild it "
                         "with the current build_visual_blocks to enable recon.")
    grid = block.get('params', {}).get('grid', (6, 6))
    if channel == 'auto':
        channel = 'mean' if block_name == 'grid' else 'speed'

    psd = raw.shape[1] // n_sides                       # per-side dim
    side_frames = []
    for s in range(n_sides):
        cols = slice(s * psd, (s + 1) * psd)
        base = [n.split('_', 1)[1] for n in raw_names[cols]]   # strip side tag
        side_frames.append(_render_cells(raw[:, cols], base, grid, mon_hw, channel))
    mov = np.concatenate(side_frames, axis=2)          # concat sides on width
    mov = np.clip(mov, 0, 255) if normalize == 'clip' else _minmax255(mov)
    return mov.astype(np.uint8)


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
