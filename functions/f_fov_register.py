# -*- coding: utf-8 -*-
"""
f_fov_register.py — 2p FOV ↔ widefield registration + neuron-location helpers (created 2026-07-17).

Shared registration toolkit factored out of f_encode_scan so the encode-scan viewer
(VR_encode_scan_plots.py), the cell-projection atlas (VR_register_2pfov_cells.py), and any future
consumer use ONE implementation:
  - footprint centroids from the CaImAn sparse `A` + `dims`,
  - the 4-corner FOV homography (tp_fov_locs → wf_locs, from VR_data.xlsx) + apply,
  - centroids → widefield coords,
  - a rotation helper that keeps overlaid points aligned with an np.rot90'd background,
  - a generic score-on-FOV scatter.
Registration values verified against VR_register_2pfov.py's FOV quads (cells land inside the FOV rect).
"""

import json

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Neuron footprint locations + widefield registration
# ─────────────────────────────────────────────────────────────────────────────

def f_footprint_centroids(A, dims, cell_ids=None, order='F'):
    """Per-neuron footprint center-of-mass from the sparse CaImAn `A` (n_comp, n_pix) reshaped to
    `dims`=(d1,d2). `order='F'` matches the CaImAn footprint reshape (f_ensemble_plots). Returns
    (centroids (n,2) as [x=col, y=row], areas (n,) = # nonzero pixels), one row per cell_id (all rows
    if cell_ids is None)."""
    d1, d2 = int(dims[0]), int(dims[1])
    rows = np.arange(A.shape[0]) if cell_ids is None else np.asarray(cell_ids, int)
    Rgrid, Cgrid = np.indices((d1, d2))                             # row (y), col (x)
    cent = np.full((len(rows), 2), np.nan)
    area = np.zeros(len(rows))
    for i, cid in enumerate(rows):
        fm = np.asarray(A[int(cid)].todense()).ravel().reshape((d1, d2), order=order)
        w = np.clip(fm, 0, None)
        s = w.sum()
        if s > 0:
            cent[i, 0] = float((w * Cgrid).sum() / s)              # x
            cent[i, 1] = float((w * Rgrid).sum() / s)              # y
            area[i] = float((w > 0).sum())
    return cent, area


def f_fov_homography(src, dst):
    """3×3 projective homography H mapping `src` (N,2) → `dst` (N,2), N≥4, via DLT (SVD least squares).
    Use to map 2p-image FOV-corner landmarks (tp_fov_locs) onto widefield landmarks (wf_locs)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError(f'need matching (N>=4, 2) src/dst, got {src.shape} / {dst.shape}')
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows, float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def f_apply_homography(pts, H):
    """Apply a 3×3 homography to `pts` (N,2) → (N,2)."""
    pts = np.asarray(pts, float)
    ph = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, float).T
    return ph[:, :2] / ph[:, 2:3]


def f_load_fov_landmarks(vr_data_xlsx, mouse_id, dset_name):
    """Read the 4-corner FOV landmarks for one dataset from VR_data.xlsx (as VR_register_2pfov.py):
    tp_fov_locs (2p-image space) + wf_locs (widefield space) + wf_im (widefield image filename).
    Returns {'tp_locs' (4,2), 'wf_locs' (4,2), 'wf_im': str|None}. Raises if the row or landmarks
    are missing/blank."""
    vr = pd.read_excel(vr_data_xlsx)
    row = vr[(vr.mouse_id == mouse_id) & (vr.dset_name == dset_name)]
    if row.empty:
        raise ValueError(f'{mouse_id}/{dset_name}: no VR_data.xlsx row')
    row = row.iloc[0]
    tp, wf = row.get('tp_fov_locs'), row.get('wf_locs')
    if not isinstance(tp, str) or not isinstance(wf, str):
        raise ValueError(f'{mouse_id}/{dset_name}: tp_fov_locs / wf_locs not set in VR_data.xlsx')
    return {'tp_locs': np.array(json.loads(tp), float), 'wf_locs': np.array(json.loads(wf), float),
            'wf_im': (row.get('wf_im') if isinstance(row.get('wf_im'), str) else None)}


def f_neuron_widefield_coords(est1, cell_ids, tp_locs, wf_locs, order='F'):
    """Footprint centroids of `cell_ids` mapped into the mouse widefield frame via the 4-corner
    tp→wf homography. Returns (wf_coords (n,2), centroids_px (n,2), areas (n,))."""
    cent, area = f_footprint_centroids(est1['A'], est1['dims'], cell_ids, order=order)
    H = f_fov_homography(tp_locs, wf_locs)
    return f_apply_homography(cent, H), cent, area


def f_bg_image(est1, order='F'):
    """Reconstruct a STRUCTURAL 2p background image (shows blood vessels) from the CaImAn background
    b @ mean(f) → (d1, d2). `b` = spatial background (n_pix, nb) [or its transpose], `f` = temporal
    (nb, T). Returns the (d1,d2) mean-background image, or None if b/f are absent (loaded by
    f_load_caim_data). Use as a vessel reference to check 2p↔widefield orientation."""
    b, ff, dims = est1.get('b'), est1.get('f'), est1['dims']
    if b is None or ff is None:
        return None
    b = np.asarray(b, float); ff = np.asarray(ff, float)
    d1, d2 = int(dims[0]), int(dims[1]); npix = d1 * d2
    fmean = ff.mean(axis=1) if ff.ndim == 2 else np.atleast_1d(np.asarray(ff, float).mean())
    if b.ndim == 1:
        bg = b * float(np.ravel(fmean)[0])
    elif b.shape[0] == npix:
        bg = b @ fmean
    elif b.shape[1] == npix:
        bg = b.T @ fmean
    else:
        return None
    if bg.size != npix:
        return None
    return np.asarray(bg, float).reshape((d1, d2), order=order)


def f_warp_to_widefield(img2p, tp_locs, wf_locs, out_hw, order=1):
    """Perspective-warp a 2p-space image `img2p` (d1,d2) into the widefield frame via the tp→wf
    homography → a full `out_hw`=(H,W) widefield-sized image, NaN outside the FOV. Inverse-maps each
    widefield pixel back to 2p coords and samples (scipy.ndimage.map_coordinates) — no cv2. `order`=1
    bilinear (intensity images) | 0 nearest (label/index maps). Overlay it (semi-transparent) on the
    widefield to compare vessels / check orientation, or warp a footprint-label map to project ROIs."""
    from scipy.ndimage import map_coordinates
    img = np.asarray(img2p, float)
    d1, d2 = img.shape
    Hinv = f_fov_homography(wf_locs, tp_locs)                 # widefield (X,Y) → 2p (x,y)
    Hout, Wout = int(out_hw[0]), int(out_hw[1])
    XX, YY = np.meshgrid(np.arange(Wout), np.arange(Hout))
    src = f_apply_homography(np.column_stack([XX.ravel(), YY.ravel()]), Hinv)   # (N,2) 2p x,y
    sx, sy = src[:, 0], src[:, 1]
    vals = map_coordinates(np.nan_to_num(img), [sy, sx], order=order, mode='constant', cval=0.0)
    vals[(sx < 0) | (sx > d2 - 1) | (sy < 0) | (sy > d1 - 1)] = np.nan    # mask outside the 2p FOV
    return vals.reshape((Hout, Wout))


def f_footprint_label_map(A, dims, cell_ids=None, order='F', transpose=False, thresh=0.2):
    """Paint each cell's footprint into a 2p-space image so ROIs (not just centroids) can be projected
    onto the widefield. Returns (label (d1,d2) int, intensity (d1,d2) float): `label` = the LOCAL index
    (0..n-1, position in cell_ids) of the max-intensity footprint covering each pixel (kept where
    footprint > `thresh`*its own max), -1 where empty; `intensity` = that winning footprint value.
    Transposed to the tp_im TIFF orientation when `transpose` (matches f_bg_image / the centroid swap).
    Warp `label` with f_warp_to_widefield(order=0) and `intensity` with order=1."""
    d1, d2 = int(dims[0]), int(dims[1])
    rows = np.arange(A.shape[0]) if cell_ids is None else np.asarray(cell_ids, int)
    label = np.full((d1, d2), -1, int)
    best = np.zeros((d1, d2))
    for i, cid in enumerate(rows):
        fm = np.clip(np.asarray(A[int(cid)].todense()).ravel().reshape((d1, d2), order=order), 0, None)
        mx = fm.max()
        if mx <= 0:
            continue
        m = (fm > thresh * mx) & (fm > best)
        label[m] = i
        best[m] = fm[m]
    if transpose:
        label = label.T.copy(); best = best.T.copy()
    return label, best


# ─────────────────────────────────────────────────────────────────────────────
# Rotation (keep overlaid coords aligned with an np.rot90'd background)
# ─────────────────────────────────────────────────────────────────────────────

def f_rot_xy(x, y, k, H, W):
    """Rotate (x=col, y=row) coords to match np.rot90(image, k) on an (H, W) image, so overlaid points
    stay aligned with the rotated background. k in 0..3 (×90° CCW, np.rot90 default axes)."""
    k = int(k) % 4
    x = np.asarray(x, float); y = np.asarray(y, float)
    if k == 0:
        return x, y
    if k == 1:                                   # CCW: new (W,H); (x,y)->(y, W-1-x)
        return y, (W - 1) - x
    if k == 2:                                   # 180°
        return (W - 1) - x, (H - 1) - y
    return (H - 1) - y, x                         # k==3 (CW): new (W,H)


def f_rot_for_plot(wf_im, rot_k, use_widefield=True):
    """Return (image_to_show, xyfun) for the requested rotation. `xyfun(x, y)` rotates coords to match
    np.rot90(wf_im, rot_k); identity when not rotating / no widefield image."""
    if not (use_widefield and wf_im is not None and rot_k):
        return (wf_im if use_widefield else None), (lambda x, y: (np.asarray(x, float), np.asarray(y, float)))
    H, W = np.asarray(wf_im).shape[:2]
    return np.rot90(np.asarray(wf_im), int(rot_k)), (lambda x, y: f_rot_xy(x, y, rot_k, H, W))


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_neuron_scores_fov(coords, scores, *, wf_im=None, dset_tag='', title='', cmap='viridis',
                             vmin=None, vmax=None, s=18, rot_k=0):
    """Scatter neurons at `coords` (N,2 [x,y]) colored by `scores` (N,), optional widefield image
    background `wf_im` (2D array), rotated by `rot_k` (image + coords together). NaN scores dropped."""
    import matplotlib.pyplot as plt
    coords = np.asarray(coords, float); scores = np.asarray(scores, float)
    ok = np.isfinite(scores) & np.isfinite(coords).all(axis=1)
    im, R = f_rot_for_plot(wf_im, rot_k, True)
    cx, cy = R(coords[:, 0], coords[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    if im is not None:
        ax.imshow(np.asarray(im), cmap='gray', aspect='auto')
    sc = ax.scatter(cx[ok], cy[ok], c=scores[ok], s=s, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
    fig.colorbar(sc, ax=ax, label='R²')
    ax.set_title(f'{dset_tag} {title}'.strip(), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if wf_im is None:
        ax.invert_yaxis()
    fig.tight_layout()
    return None


def f_plot_neuron_field_fov(coords, scores, *, wf_im=None, dset_tag='', title='', cmap='viridis',
                            vmin=None, vmax=None, sigma=25.0, grid_res=300, density_floor=0.05,
                            field_alpha=0.85, contours=6, show_dots=False, dot_size=3, rot_k=0):
    """Smooth topographic map of per-neuron `scores` over (widefield) space. Gaussian-convolves the
    scattered (coords, scores) into a continuous weighted-mean field (Nadaraya-Watson: convolve the
    score-sum and the count, then divide), masks regions with < `density_floor`·max smoothed coverage,
    and imshows it (alpha `field_alpha`) over the optional `wf_im` background with `contours` topo
    lines. `sigma` = smoothing bandwidth in coord (pixel) units. Set `show_dots` to overlay the cells.
    coords (N,2 [x,y]); NaN scores dropped; rotated by `rot_k` (image + coords together)."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    coords = np.asarray(coords, float); scores = np.asarray(scores, float)
    ok = np.isfinite(scores) & np.isfinite(coords).all(axis=1)
    coords, scores = coords[ok], scores[ok]
    im, R = f_rot_for_plot(wf_im, rot_k, True)
    cx, cy = R(coords[:, 0], coords[:, 1])
    if im is not None:                                    # grid matches the (rotated) background pixels
        H, W = np.asarray(im).shape[:2]
        x0, y0, dx, dy, nx, ny = 0.0, 0.0, 1.0, 1.0, int(W), int(H)
    else:                                                 # grid = padded point bounding box
        pad = 4.0 * sigma
        x0, x1 = cx.min() - pad, cx.max() + pad
        y0, y1 = cy.min() - pad, cy.max() + pad
        nx = ny = int(grid_res)
        dx = (x1 - x0) / nx; dy = (y1 - y0) / ny
    ix = np.clip(((cx - x0) / dx).astype(int), 0, nx - 1)
    iy = np.clip(((cy - y0) / dy).astype(int), 0, ny - 1)
    num = np.zeros((ny, nx)); den = np.zeros((ny, nx))
    np.add.at(num, (iy, ix), scores)                      # score-sum per cell
    np.add.at(den, (iy, ix), 1.0)                         # count per cell
    sig_px = (sigma / dy, sigma / dx)                     # (row, col) bandwidth in grid cells
    num_s = gaussian_filter(num, sig_px, mode='constant')
    den_s = gaussian_filter(den, sig_px, mode='constant')
    field = np.divide(num_s, den_s, out=np.full_like(num_s, np.nan), where=den_s > 0)
    if den_s.max() > 0:
        field[den_s < density_floor * den_s.max()] = np.nan   # hide sparsely-covered regions
    cmap_o = plt.get_cmap(cmap).copy(); cmap_o.set_bad(alpha=0.0)
    extent = None if im is not None else [x0, x0 + nx * dx, y0 + ny * dy, y0]
    fig, ax = plt.subplots(figsize=(6, 5))
    if im is not None:
        ax.imshow(np.asarray(im), cmap='gray', aspect='auto')
    mp = ax.imshow(field, origin='upper', extent=extent, cmap=cmap_o, alpha=field_alpha,
                   vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
    if contours:
        xg = x0 + (np.arange(nx) + 0.5) * dx; yg = y0 + (np.arange(ny) + 0.5) * dy
        with np.errstate(invalid='ignore'):
            ax.contour(xg, yg, field, levels=int(contours), colors='k', linewidths=0.4, alpha=0.5)
    if show_dots:
        ax.scatter(cx, cy, c=scores, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
    fig.colorbar(mp, ax=ax, label='R²')
    ax.set_title(f'{dset_tag} {title}'.strip(), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])   # extent (no-wf) / origin='upper' (wf) already orient y down
    fig.tight_layout()
    return None
