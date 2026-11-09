# -*- coding: utf-8 -*-
"""
VR_register_2pfov_cells.py — project 2p CELL ROIs onto the widefield visual-cortex map.

Sister to VR_register_2pfov.py: instead of plotting just the 2p-FOV rectangles, this loads EVERY 2p
dataset of a mouse, computes each cell's footprint CENTROID (from est1['A']/dims), and maps it onto the
mouse's common widefield map via the same 4-corner FOV landmarks (tp_fov_locs → wf_locs, VR_data.xlsx).
So all cells from all sessions land on ONE widefield atlas — a coverage view, and (optionally) colored
by per-cell scores from EITHER the VR_encode_scan store (R² for a block×model) OR the VR_tuning_scan
store (a tuning metric for a feat_set), with score→brightness so strong cells pop (Section 4).

cell_mode='dots' draws centroid markers; cell_mode='roi' warps each cell's actual A footprint onto the
widefield (painted label map → per-dataset tint in Section 3 / per-cell R² image in Section 4).
transpose_2p reconciles the order='F' A/b reshape with the tp_im TIFF orientation the landmarks were
digitized in — applied to centroids, background, AND footprints together so they stay locked.

Registration compute is reused from functions/f_fov_register.py (f_footprint_centroids /
f_fov_homography / f_load_fov_landmarks / f_rot_for_plot). Spyder cell-based. Created 2026-07-17.
"""

import sys
import os
import json
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from f_pipeline import f_load_session
import f_fov_register as fov          # registration + rotation + fov helpers
import f_encode_scan as es            # encode-scan store (scores) for the optional score-coloring cell

try:
    import tifffile as tiff
except Exception:
    tiff = None


#%% =========================================================================
# SECTION 1 · CONFIG — mouse, paths
# ==========================================================================
mouse_id     = 'L'
data_dir     = 'F:/VR/data_proc/' + mouse_id
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
wf_dir       = 'F:/VR/data_proc/2p_fov_reg/'      # widefield images referenced by VR_data.xlsx wf_im
active_only  = True                               # cells with any activity (S.sum>0) — matches encode scan
foot_order   = 'F'                                # CaImAn footprint reshape order (as f_ensemble_plots)
rot_k        = 1      
show_cells   = True                               # draw the cells (off = backgrounds only)
cell_mode    = 'dots'                               # 'dots' = centroid markers | 'roi' = warped A footprints  ('dots' | 'roi')
roi_thresh   = 0.2                                 # footprint kept where A > roi_thresh*its max (roi mode)
roi_alpha    = 0.7                                 # opacity of the projected ROI footprints (roi mode)
show_bg      = True                               # overlay the 2p BACKGROUND (b@mean(f) = vessels) warped
                                                  #   onto the widefield — to check 2p↔widefield orientation
wf_alpha     = 1                                # widefield image opacity (<0.5 = >50% transparent)
bg_alpha     = 1                                # 2p-background overlay opacity
bg_cmap      = 'inferno'                           # colormap for the 2p-background overlay (vs gray widefield)
bg_darkness  = 0.0                                # 0..1 dim the small 2p-FOV backgrounds (0=full, 1=black)
show_date    = True                               # label each FOV with its dset_name (date) under the top edge
transpose_2p = True                               # tp_fov_locs were digitized on the tp_im TIFF (x=col,y=row),
                                                  #   but A/b reshape order='F' is its transpose — swap centroid
                                                  #   x<->y AND transpose the bg so BOTH match the landmarks



#%% SECTION 2 · BUILD — per dataset: footprint centroids → widefield coords (all do_proc datasets)
# For each do_proc dataset: load calcium (fast), compute footprint centroids for the kept cells, and
# map them to the mouse widefield frame via the dataset's tp→wf landmarks. Datasets missing landmarks
# are kept in session-pixel space only (wf_x/wf_y NaN) + logged. Also collects the FOV corner quads.
vr = pd.read_excel(vr_data_xlsx)
vr2 = vr[(vr.mouse_id == mouse_id) & (vr.do_proc == 1)].reset_index(drop=True)
dset_names = list(vr2.dset_name)
print(f'{mouse_id}: {len(dset_names)} do_proc datasets → {dset_names}')

rows, fov_quads, wf_im, bg_store, roi_store = [], {}, None, {}, {}
for nd, dname in enumerate(dset_names):
    try:
        session = f_load_session(mouse_id, nd, data_dir, data_dir_bh, vr_data_xlsx,
                                 behavior=False, verbose=False)
        est1 = session['est1']
        S = np.asarray(est1['S'])
        cell_ids = (np.flatnonzero(S.sum(axis=1) > 0) if active_only
                    else np.arange(S.shape[0]))
        cent, area = fov.f_footprint_centroids(est1['A'], est1['dims'], cell_ids, order=foot_order)
        if transpose_2p:                                      # match the tp_im TIFF orientation (see knob)
            cent = cent[:, ::-1]
        wf_xy = np.full_like(cent, np.nan)
        try:                                                  # register to widefield (if landmarks set)
            lm = fov.f_load_fov_landmarks(vr_data_xlsx, mouse_id, dname)
            H = fov.f_fov_homography(lm['tp_locs'], lm['wf_locs'])
            wf_xy = fov.f_apply_homography(cent, H)
            fov_quads[dname] = np.asarray(lm['wf_locs'], float)
            if wf_im is None and tiff is not None and lm['wf_im']:
                _p = os.path.join(wf_dir, lm['wf_im'])
                wf_im = tiff.imread(_p) if os.path.exists(_p) else None
            if show_bg:                                       # 2p background (vessels) → warp later
                _bg2p = fov.f_bg_image(est1, order=foot_order)
                if _bg2p is not None:
                    bg_store[dname] = (_bg2p.T if transpose_2p else _bg2p,
                                       lm['tp_locs'], lm['wf_locs'])
            if cell_mode == 'roi':                            # paint footprints → warp later (roi mode)
                lab2p, int2p = fov.f_footprint_label_map(est1['A'], est1['dims'], cell_ids,
                                                         order=foot_order, transpose=transpose_2p,
                                                         thresh=roi_thresh)
                roi_store[dname] = (lab2p, int2p, lm['tp_locs'], lm['wf_locs'], len(rows))
        except Exception as _e:
            print(f'  {dname}: no widefield registration ({type(_e).__name__}: {_e})')
        for i, cid in enumerate(cell_ids):
            rows.append({'mouse': mouse_id, 'n_dset': nd, 'dset_name': dname, 'neuron': int(cid),
                         'x': cent[i, 0], 'y': cent[i, 1], 'area': float(area[i]),
                         'wf_x': wf_xy[i, 0], 'wf_y': wf_xy[i, 1]})
        print(f'  {dname}: {len(cell_ids)} cells'
              + ('  (+widefield)' if np.isfinite(wf_xy).any() else '  (px only)')
              + ('  (+bg)' if dname in bg_store else ''))
        del session
    except Exception as e:
        print(f'  {dname}: SKIPPED — {type(e).__name__}: {e}')

cells = pd.DataFrame(rows)
n_wf = int(cells['wf_x'].notna().sum()) if not cells.empty else 0
print(f'\ntotal {len(cells)} cells across {cells["dset_name"].nunique() if not cells.empty else 0} '
      f'datasets ({n_wf} mapped to widefield); wf image {"loaded" if wf_im is not None else "MISSING"}')

# composite the per-dataset 2p backgrounds into ONE widefield-sized image (nanmean over datasets)
bg_wf = None
if show_bg and wf_im is not None and bg_store:
    _Hh, _Ww = np.asarray(wf_im).shape[:2]
    _acc = np.zeros((_Hh, _Ww)); _cnt = np.zeros((_Hh, _Ww))
    for dn, (b2p, tp, wf) in bg_store.items():
        w = fov.f_warp_to_widefield(b2p, tp, wf, (_Hh, _Ww))
        m = np.isfinite(w)
        _acc[m] += w[m]; _cnt[m] += 1
    bg_wf = np.where(_cnt > 0, _acc / np.maximum(_cnt, 1), np.nan)
    print(f'2p-background overlay composited from {len(bg_store)} datasets '
          f'({int((_cnt > 0).sum())} widefield px covered)')

# warp painted footprints into the widefield: per-dataset intensity mask (Section-3 dataset coloring)
# + one GLOBAL cells-row label map (Section-4 per-cell score coloring). One warp per dataset.
roi_int_wf, roi_label_wf = {}, None
if cell_mode == 'roi' and wf_im is not None and roi_store:
    _Hh, _Ww = np.asarray(wf_im).shape[:2]
    roi_label_wf = np.full((_Hh, _Ww), -1, int)
    for dn, (lab2p, int2p, tp, wf, start) in roi_store.items():
        roi_int_wf[dn] = fov.f_warp_to_widefield(int2p, tp, wf, (_Hh, _Ww), order=1)
        labw = fov.f_warp_to_widefield(lab2p.astype(float), tp, wf, (_Hh, _Ww), order=0)
        m = np.isfinite(labw) & (labw >= 0)
        roi_label_wf[m] = start + labw[m].astype(int)         # local idx → global cells-row
    print(f'ROI footprints warped for {len(roi_int_wf)} datasets '
          f'({int((roi_label_wf >= 0).sum())} widefield px covered)')


def _bg_disp(img, rot_k, darkness):
    """Rotate bg_wf for plotting + normalize to [0,1] and dim by `darkness` (0=full, 1=black).
    Returns (image, vmin, vmax) so imshow keeps a fixed scale; NaN pixels stay transparent."""
    im = np.rot90(img, rot_k) if rot_k else img
    lo, hi = np.nanmin(im), np.nanmax(im)
    im = (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)
    return im * (1.0 - float(np.clip(darkness, 0, 1))), 0.0, 1.0


#%% SECTION 3 · PLOT — all cell ROIs on the widefield VC map, colored by DATASET (coverage view)
# Cells from every session on one widefield atlas; dot size ∝ √footprint-area. FOV rectangles overlaid
# for context (as VR_register_2pfov.py). Set use_widefield=False to see raw 2p-pixel coords instead.
if 1:
    use_widefield = True
    size_by_area  = True
    dsets = sorted(cells['dset_name'].unique())
    fig, ax = plt.subplots(figsize=(8, 7))
    _im, _R = fov.f_rot_for_plot(wf_im, rot_k, use_widefield)      # rotate image + coords together
    if _im is not None:
        ax.imshow(np.asarray(_im), cmap='gray', aspect='auto', alpha=wf_alpha)
    if show_bg and use_widefield and bg_wf is not None:           # 2p vessels warped onto widefield
        _bgim, _v0, _v1 = _bg_disp(bg_wf, rot_k, bg_darkness)
        ax.imshow(_bgim, cmap=bg_cmap, aspect='auto', alpha=bg_alpha, vmin=_v0, vmax=_v1)
    xcol, ycol = ('wf_x', 'wf_y') if use_widefield else ('x', 'y')
    _cmap = plt.get_cmap('tab20')
    _roi = show_cells and cell_mode == 'roi' and use_widefield and roi_int_wf
    for i, ds in enumerate(dsets):
        g = cells[cells['dset_name'] == ds]
        ok = (g[xcol].notna() & g[ycol].notna()).to_numpy()
        col = _cmap(i % 20)
        if _roi and ds in roi_int_wf:                         # warped footprints, tinted by dataset
            iw = roi_int_wf[ds]
            iw = np.rot90(iw, rot_k) if (rot_k and use_widefield) else iw
            m = np.isfinite(iw) & (iw > 0)
            rgba = np.zeros(iw.shape + (4,))
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = col[0], col[1], col[2]
            rgba[..., 3] = np.where(m, roi_alpha, 0.0)
            ax.imshow(rgba, aspect='auto')
            ax.plot([], [], 's', color=col, label=f'{ds} ({int(ok.sum())})')   # legend proxy
        elif show_cells:
            rx, ry = _R(g[xcol].to_numpy(), g[ycol].to_numpy())
            s = (3 + 0.15 * np.sqrt(g['area'].to_numpy()[ok])) if size_by_area else 8
            ax.scatter(rx[ok], ry[ok], s=s, color=col,
                       alpha=0.5, edgecolors='none', label=f'{ds} ({int(ok.sum())})')
    if use_widefield:                                         # FOV rectangles (corners 0-1-3-2-0)
        for ds, q in fov_quads.items():
            qx, qy = _R(q[:, 0], q[:, 1])
            for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
                ax.plot([qx[a], qx[b]], [qy[a], qy[b]], color='k', lw=0.6, alpha=0.4)
            if show_date:                                     # date under the visually-topmost edge
                edges = ((0, 1), (1, 3), (3, 2), (2, 0))
                a, b = min(edges, key=lambda e: (qy[e[0]] + qy[e[1]]) / 2)
                mx, my = (qx[a] + qx[b]) / 2, (qy[a] + qy[b]) / 2
                ax.text(mx, my + 6, ds, ha='center', va='top', fontsize=6, color='k',
                        alpha=0.8, clip_on=True)
    ax.set_title(f'{mouse_id} — 2p cell ROIs on widefield ({len(cells)} cells, {len(dsets)} datasets)',
                 fontsize=10)
    if show_cells:
        ax.legend(fontsize=6, ncol=2, markerscale=2, loc='upper right')
    if use_widefield:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.invert_yaxis()
    fig.tight_layout()


#%% SECTION 4 · COLOR cells by ENCODER or TUNING-scan score on the widefield
# Loads either the VR_encode_scan store (per-cell R² for a block×model) OR the VR_tuning_scan store
# (per-cell tuning metric for a feat_set), joins by (dset_name, neuron), and colors each neuron by that
# score. `bright_by_score` maps the score to BRIGHTNESS (per-neuron alpha) so strong cells pop on the
# widefield; the colormap encodes value too. Works for cell_mode 'dots' and 'roi'. Cells not in the
# store (or NaN) are drawn faint gray. Ensembles have no footprint → use score_source='cells'.
if 1:
    import matplotlib.colors as mcolors
    import matplotlib.cm as _mcm

    score_src     = 'tuning'         # 'encode' (VR_encode_scan) | 'tuning' (VR_tuning_scan)
    score_source  = 'cells'          # unit source ('cells'; 'ensembles' have no location)
    score_feat_root = 'F:/VR/features/'
    # -- encoder knobs (score_src='encode') --
    ENC_TAG       = 'enc_7_17_26'
    enc_block     = 'gaborS+mengS'
    enc_model     = 'ridge'
    # -- tuning knobs (score_src='tuning') --
    TUN_TAG       = 'tun_7_17_26'
    tun_feat_set  = 'meng'           # 'gabor' | 'meng' | 'pix' | 'agg' | 'self_mot'
    tun_metric    = 'peak_abs'       # 'peak_abs' | 'peak_z' | 'osi' | 'pref_odeg' | 'pref_tf_hz'
    # -- coloring --
    score_cmap    = 'magma'          # brightness ramp (use 'twilight' for the circular pref_odeg)
    score_clip    = (None, None)     # (vmin, vmax); None → 2nd/98th percentile of the scored cells
    bright_by_score = True           # score → per-neuron ALPHA (brightness) on top of the widefield
    alpha_floor   = 0.15             # min alpha for the lowest-scored neuron (bright_by_score)

    if score_src == 'encode':
        store = es.f_encode_scan_load(es.f_encode_scan_path(score_feat_root, ENC_TAG))
        if store is None:
            raise FileNotFoundError('no encode-scan store — run VR_encode_scan.py first')
        tbl = es.f_encode_scan_table(store)
        sub = tbl[(tbl['source'] == score_source) & (tbl['block'] == enc_block)
                  & (tbl['model'] == enc_model)].rename(columns={'r2': 'score'})
        score_label = f'{enc_block}·{enc_model} R²'
    else:
        import f_tuning_scan as ts
        store = ts.f_tuning_scan_load(ts.f_tuning_scan_path(score_feat_root, TUN_TAG))
        if store is None:
            raise FileNotFoundError('no tuning-scan store — run VR_tuning_scan.py first')
        tbl = ts.f_tuning_scan_table(store)
        sub = tbl[(tbl['source'] == score_source)
                  & (tbl['feat_set'] == tun_feat_set)].rename(columns={tun_metric: 'score'})
        score_label = f'{tun_feat_set} {tun_metric}'

    merged = cells.merge(sub[['mouse', 'dset_name', 'unit', 'score']],
                         left_on=['mouse', 'dset_name', 'neuron'],
                         right_on=['mouse', 'dset_name', 'unit'], how='left')
    sval = merged['score'].to_numpy(float)
    ok = merged['wf_x'].notna().to_numpy()
    _fin = sval[np.isfinite(sval)]
    vmin = score_clip[0] if score_clip[0] is not None else (np.nanpercentile(_fin, 2) if _fin.size else 0.0)
    vmax = score_clip[1] if score_clip[1] is not None else (np.nanpercentile(_fin, 98) if _fin.size else 1.0)
    nrm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(score_cmap)

    fig, ax = plt.subplots(figsize=(8, 7))
    _im, _R = fov.f_rot_for_plot(wf_im, rot_k, True)               # rotate image + coords together
    if _im is not None:
        ax.imshow(np.asarray(_im), cmap='gray', aspect='auto', alpha=wf_alpha)
    if show_bg and bg_wf is not None:
        _bgim, _v0, _v1 = _bg_disp(bg_wf, rot_k, bg_darkness)
        ax.imshow(_bgim, cmap=bg_cmap, aspect='auto', alpha=bg_alpha, vmin=_v0, vmax=_v1)
    rx, ry = _R(merged['wf_x'].to_numpy(), merged['wf_y'].to_numpy())
    _hit = ok & np.isfinite(sval)

    if show_cells and cell_mode == 'roi' and roi_label_wf is not None:  # warped footprints, brightness by score
        lab = roi_label_wf
        score_wf = np.full(lab.shape, np.nan)
        valid = lab >= 0
        score_wf[valid] = sval[lab[valid]]                    # global cells-row → its score
        score_wf = np.rot90(score_wf, rot_k) if rot_k else score_wf
        m = np.isfinite(score_wf)
        alpha_img = np.zeros(score_wf.shape)
        alpha_img[m] = roi_alpha * ((alpha_floor + (1 - alpha_floor) * nrm(score_wf[m]))
                                    if bright_by_score else 1.0)
        ax.imshow(np.ma.masked_invalid(score_wf), cmap=cmap, norm=nrm, aspect='auto', alpha=alpha_img)
    elif show_cells:
        if _hit.sum() < ok.sum():                             # faint gray for present-but-unscored cells
            _nan = ok & ~np.isfinite(sval)
            ax.scatter(rx[_nan], ry[_nan], s=6, color='0.5', alpha=0.25, edgecolors='none', label='not scored')
        rgba = cmap(nrm(sval))
        if bright_by_score:                                   # score → per-neuron brightness (alpha)
            rgba[:, 3] = alpha_floor + (1 - alpha_floor) * nrm(sval)
        ax.scatter(rx[_hit], ry[_hit], s=16, c=rgba[_hit], edgecolors='none')
        if _hit.sum() < ok.sum():
            ax.legend(fontsize=7, loc='upper right')
    _sm = _mcm.ScalarMappable(norm=nrm, cmap=cmap); _sm.set_array([])
    fig.colorbar(_sm, ax=ax, label=score_label)
    ax.set_title(f'{mouse_id} — {score_src}: {score_label} on widefield '
                 f'({int(_hit.sum())} scored cells)', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
