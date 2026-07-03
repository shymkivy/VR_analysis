# -*- coding: utf-8 -*-
"""
VR_feature_viz.py — visualize the FEATURES WE DECODE FROM (TODO #47/#49/#66).

Loads a feature cache and reconstructs the image-space decoding blocks back to monitor
movies, judged against the original input:
  - pix / grid / flow / gabor / meng reconstruction  (f_reconstruct_feature_movie)
  - gabor magnitude + phase                          (f_reconstruct_gabor_phase_movie)
  - gabor per-PATCH phase-quiver (motion from phase)  (f_gabor_flow_field + quiver)
All read built_blocks only. The ORIGINAL input is the comparison baseline — shown from
the few `example_frames` saved in the cache (cheap), or the full movie if cached / rendered.

Each feature cell BUILDS its movie, plots a few frames inline (vs original), and only
SAVES the multi-page TIFF when make_movie=True. `save_orig` then also writes the matching
ORIGINAL movie (same frames + filename tag) so they pair up in ImageJ. To raise a
feature's fidelity, increase its grid / n_pca in visual_specs and REGENERATE — the viz
reflects whatever resolution the features were built at.

Split out of VR_ca_cebra.py on 2026-06-29. Spyder cell-based: run the load cell, then any
viz cell.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import numpy as np

from f_feature_cache import (f_load_features, f_resolve_feature_file,
                             f_list_feature_files)
from f_functions import f_save_mon_movie
from f_feature_viz import (f_plot_movie_frames, f_plot_movie_frames_compare,
                           f_reconstruct_feature_movie, f_reconstruct_gabor_phase_movie,
                           f_resolve_recon_n_pcs, f_gabor_flow_field, f_gabor_flow_at,
                           f_render_flow_quiver_movie, f_pick_motion_frames)

#%% Load a feature cache — pick by tag / index / filename (set the dir + selectors)
mouse_id     = 'L'
features_dir = 'F:/VR/features/' + mouse_id
dset_name    = ''          # filter the dir to one dataset (substring); '' = all files
feature_file = None        # exact/substring filename; None → use tag, then index
feature_tag  = 'gen_6_29_26'   # filename substring (e.g. GEN_TAG, shared by all datasets)
feature_idx  = 0           # which file among the (dset+tag) matches (step 0,1,2,… to browse)
out_dir      = 'F:/test_mov'
mov_tag      = ''          # appended to every saved movie filename (pairs feature+original)
save_orig    = False       # when a feature cell saves its movie, also save the original

print('available:', [p.name for p in f_list_feature_files(features_dir, dset_name)])
_path = f_resolve_feature_file(features_dir, dset_name, name=feature_file,
                               tag=feature_tag, index=feature_idx)
bundle       = f_load_features(_path)
built_blocks = bundle['built_blocks']
side_tags    = bundle['side_tags']
frame_t      = np.asarray(bundle['frame_t']).ravel()
meta         = bundle['meta']
gen_config   = bundle['gen_config']
_ds          = meta.get('dset_name', _path.stem)
# num_samp lives in gen_config: flat (cebra/encoding) or nested under 'render' (gen).
num_samp     = gen_config.get('num_samp') or gen_config.get('render', {}).get('num_samp', 101)
mon_hw       = (num_samp, num_samp)
side         = 'both' if side_tags == ['L', 'R'] else side_tags[0]
# Original input: full movie only when SAVE_MOVIE was set; a few example_frames always.
movie = bundle.get('movie')
orig_mov = movie                      # full original (L|R), or None — set by the cell below
example_frames    = bundle.get('example_frames')
example_frame_idx = bundle.get('example_frame_idx')
print(f'loaded {_path.name}: blocks={list(built_blocks)}  side={side_tags}  '
      f'T={frame_t.size}  num_samp={num_samp}  movie={"yes" if movie is not None else "no"}  '
      f'example_frames={0 if example_frames is None else len(example_frame_idx)}')


def _orig_for_side(arr):
    """Crop an L|R original-frame array to the recon's side(s) so widths match."""
    if arr is None or len(side_tags) == 2:
        return arr
    _h = arr.shape[2] // 2
    return arr[:, :, :_h] if side_tags[0] == 'L' else arr[:, :, _h:]


def _orig_frames(pidx):
    """Original frames at imaging-frame indices `pidx`, cropped to side: from the cheap
    saved example_frames when pidx == example_frame_idx, else from orig_mov, else None."""
    if pidx is None:
        return None
    if example_frames is not None and list(pidx) == list(example_frame_idx):
        return _orig_for_side(np.asarray(example_frames))
    if orig_mov is not None:
        return _orig_for_side(np.asarray(orig_mov)[list(pidx)])
    return None


def _save_with_orig(feat_mov, stem, save_idx=None, pair=True):
    """Save an already-frame-subset feature movie as <out_dir>/<stem>_<ds><_tag>.tif, and
    (when save_orig and pair) the matching ORIGINAL as <...>_ORIG.tif at the SAME frames.
    `save_idx` = the imaging-frame indices `feat_mov` was built at (for the original)."""
    _mt = f'_{mov_tag}' if mov_tag else ''
    f_save_mon_movie(feat_mov, f'{out_dir}/{stem}_{_ds}{_mt}.tif', gap_px=0)
    msg = f'saved {stem} → {out_dir}/{stem}_{_ds}{_mt}.tif'
    if pair and save_orig:
        if orig_mov is None or save_idx is None:
            msg += '  (save_orig set but no original movie — run the Original-movie cell)'
        else:
            _o = _orig_for_side(np.asarray(orig_mov)[list(save_idx)])
            f_save_mon_movie(_o, f'{out_dir}/{stem}_{_ds}{_mt}_ORIG.tif', gap_px=0)
            msg += '  (+_ORIG)'
    print(msg)


def _frame_indices(n_total, want, n):
    """Indices to display/save: `want` (explicit list) → as-is; else the saved
    example_frame_idx; else `n` evenly-spaced over n_total."""
    if want is not None:
        return [int(i) for i in want]
    if example_frames is not None:
        return list(example_frame_idx)
    return list(np.unique(np.linspace(0, n_total - 1, min(n, n_total)).round().astype(int)))


#%% Original monitor movie — the comparison baseline (render if not cached) + optional save
# orig_mov = the rendered monitor INPUT the features were extracted from. From the cache
# if SAVE_MOVIE was set, else rendered on demand (slow) when do_render=True. make_movie
# saves orig_<ds><mov_tag>.tif. If neither is available, the cheap example_frames are shown
# and still drive the inline comparisons.
if 1:
    do_render   = False         # render from the session if the full movie isn't cached
    ex_n        = 6
    ex_idx      = None
    make_movie  = False

    if movie is not None:
        orig_mov = movie
    elif do_render:
        from f_pipeline import f_load_session, f_render_session
        data_dir     = 'F:/VR/data_proc/' + mouse_id
        data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
        vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
        _n_dset = meta.get('n_dset')
        if _n_dset is None:
            raise ValueError('cache meta has no n_dset — cannot locate the session to render.')
        _rc = gen_config.get('render', gen_config)     # nested (feature_gen) or flat (cebra)
        _session = f_load_session(mouse_id, _n_dset, data_dir, data_dir_bh, vr_data_xlsx,
                                  cam_params=_rc.get('cam_params'))
        _render_cfg = {'obj_size': _rc.get('obj_size'), 'num_samp': num_samp,
                       'max_frames': _rc.get('max_frames'),
                       'render_terrain': _rc.get('render_terrain', False),
                       'terrain_stride': _rc.get('terrain_stride', 2),
                       'terrain_point_size': _rc.get('terrain_point_size', 1),
                       'terrain_clip_len': _rc.get('terrain_clip_len', None)}
        _l, _r, orig_mov = f_render_session(_session, _render_cfg)
        print(f'rendered original movie: {orig_mov.shape}')
    else:
        orig_mov = None

    if orig_mov is not None:
        f_plot_movie_frames(orig_mov, frame_idx=ex_idx, n=ex_n,
                            suptitle=f'{_ds} — original monitor (L|R)')
        if make_movie:
            _mt = f'_{mov_tag}' if mov_tag else ''
            f_save_mon_movie(orig_mov, f'{out_dir}/orig_{_ds}{_mt}.tif')
            print(f'saved original → {out_dir}/orig_{_ds}{_mt}.tif')
    elif example_frames is not None:
        f_plot_movie_frames(np.asarray(example_frames), frame_idx=None, n=ex_n,
                            suptitle=f'{_ds} — original (saved example frames only)')
        print('full original movie not available (set do_render=True to render it). The '
              'cheap example_frames are shown + drive the inline comparisons.')
    else:
        print('no original movie or example_frames in this cache — set do_render=True.')


#%% OVERVIEW — all decoding-feature reconstructions vs the original (a few frames)
# One figure: rows = original + each image-space decoding block present
# (pix/grid/flow/gabor/meng), columns = the saved example frames. Reconstructs ONLY those
# frames (fast). Prints each block's grid + feature count ("resolution"); to raise
# fidelity, increase grid / n_pca in visual_specs and REGENERATE.
if 1:
    ov_blocks  = ['pix', 'grid', 'flow', 'gabor', 'meng']
    ov_channel = {'grid': 'edge', 'flow': 'speed', 'gabor': 'auto', 'meng': 'auto'}

    _orig0 = _orig_frames(example_frame_idx) if example_frames is not None else None
    if _orig0 is None:
        print('overview needs example_frames (cheap) or a rendered original — run the '
              'Original-movie cell (do_render=True) or regenerate the cache.')
    else:
        _idx = list(example_frame_idx)
        _rows, _labels = [_orig0], ['original']
        for b in ov_blocks:
            if b not in built_blocks:
                continue
            _rec = f_reconstruct_feature_movie(built_blocks[b], b, mon_hw, side_tags,
                                               channel=ov_channel.get(b, 'auto'), frame_idx=_idx)
            _nc = built_blocks[b]['X'].shape[1]          # PCs / features kept in the block
            _g = built_blocks[b].get('params', {}).get('grid')
            _rows.append(_rec)
            _labels.append(f'{b}\n({_nc} PCs)' + (f'\ngrid {_g}' if _g else ''))
            print(f'  {b:5s}: grid={_g}  n_feat={_nc}')
        f_plot_movie_frames_compare(_rows, _idx, labels=_labels,
                                    suptitle=f'{_ds} — decoding features vs original')
        print('to increase fidelity: raise grid / n_pca in visual_specs (gen) + regenerate.')


#%% Per-block recon — deep-dive on ONE feature (sweep n_pcs / channel) vs original
#   recon_block:  'pix' (inverse-PCA) | 'grid' | 'flow' | 'gabor' | 'meng'
#   recon_channel (ignored for pix): grid 'occ'|'mean'|'edge'; flow 'speed'|'div';
#       gabor 'gabM_o{io}sf{isf}'; meng 'meng{opp|d0|d1}_o{io}sf{isf}tf{itf}'; 'auto'.
if 1:
    recon_block       = 'grid'     # 'pix' | 'grid' | 'flow' | 'gabor' | 'meng'
    recon_n_pcs       = None        # int=#PCs | float in (0,1)=variance frac | None=all
    recon_channel     = 'edge'     # see header; 'auto' = first/default channel
    recon_frame_range = (0, 3000)  # imaging frames for the saved movie
    ex_n              = 6
    ex_idx            = None        # explicit frame indices; None → example_frame_idx
    make_movie        = False       # save the full TIFF (+ original if save_orig)

    if recon_block not in built_blocks:
        raise ValueError(f'{recon_block!r} not in built_blocks: {list(built_blocks)}')
    _rb = built_blocks[recon_block]
    _pca = _rb.get('pca_pix') if recon_block == 'pix' else _rb.get('pca')
    if isinstance(_pca, dict):       # grouped PCA (gabor) — use the gabor cell for phase
        _n_total = _n_used = _rb['X'].shape[1]
        _pca = None
    else:
        _n_total = _pca.n_components_ if _pca is not None else _rb['X'].shape[1]
        _n_used = (_n_total if recon_n_pcs is None
                   else f_resolve_recon_n_pcs(recon_n_pcs, _pca))

    if _pca is not None:
        _evr = _pca.explained_variance_ratio_
        _var = (f', {np.sum(_evr[:_n_used]) / np.sum(_evr):.1%} of block variance '
                f'(block keeps {np.sum(_evr):.1%} of original in {_n_total} PCs)')
    else:
        _var = ' (raw block, no PCA)'

    # Reconstruct ONLY the frames shown inline (cheap — full-T recon would be huge).
    _T = _rb['X'].shape[0]
    _pidx = _frame_indices(_T, ex_idx, ex_n)
    recon_frames = f_reconstruct_feature_movie(
        built_blocks[recon_block], recon_block, mon_hw, side_tags,
        n_pcs=recon_n_pcs, channel=recon_channel, frame_idx=_pidx)
    print(f'reconstructed {recon_block!r} at {len(_pidx)} frames '
          f'({_n_used}/{_n_total} PCs{_var}, channel={recon_channel}, grid='
          f'{_rb.get("params", {}).get("grid")})')

    _orig = _orig_frames(_pidx)
    _rows = [_orig, recon_frames] if _orig is not None else [recon_frames]
    _labels = (['original', f'recon {recon_block}:{recon_channel}'] if _orig is not None
               else [f'recon {recon_block}:{recon_channel}'])
    f_plot_movie_frames_compare(_rows, _pidx, labels=_labels,
                                suptitle=f'{_ds} — original vs recon {recon_block}')
    if make_movie:                       # reconstruct only the save range (not full T)
        _sidx = list(range(*recon_frame_range))
        _mov = f_reconstruct_feature_movie(built_blocks[recon_block], recon_block, mon_hw,
                                           side_tags, n_pcs=recon_n_pcs, channel=recon_channel,
                                           frame_idx=_sidx)
        _save_with_orig(_mov, f'recon_{recon_block}_{recon_channel}_{_n_used}pc', _sidx)


#%% Gabor: pix scene / mag / phase (+ optional per-patch quiver overlay) at MOTION frames
# One figure, rows = pix reconstruction (scene) / gabor magnitude / gabor phase, columns =
# frames. `overlay_quiver` overlays the per-patch phase-motion arrows on `overlay_on` rows
# (0=scene, 1=mag, 2=phase). Frames are picked at high MOTION (pix change) so they're not
# stationary. Reconstructs ONLY the shown frames + a tiny window per frame for the flow —
# never the whole movie. The scene row is the pix inverse-PCA (a rank-k proxy for the
# original, labelled with the PC count) unless the true movie/example frames are available.
if 1:
    gb_io, gb_isf   = 0, 0          # gabor orientation / SF index
    gb_n_pcs        = None          # gabor inverse-PCA rank; None = all stored
    gb_phase_cmap   = 'gray'        # 'hsv' (cyclic) | 'gray' | 'twilight' (cyclic) | ...
    scene_n_pcs     = None          # pix-recon rank for the scene row; None = all stored PCs
    frame_mode      = 'motion_spread'   # 'motion_spread' | 'motion_event' | 'even' | 'saved'
    ex_n            = 6
    ex_idx          = None          # explicit frame indices (overrides frame_mode)
    ev_stride       = 3             # frame step in 'motion_event' mode (a run through 1 event)
    overlay_quiver  = True
    overlay_on      = 'all'         # 'all' | list of row indices to overlay, e.g. [0, 2]
    qv_win          = 1             # local window (frames) for the per-frame flow
    qv_sign         = -1.0          # flip (±1) if arrows point the wrong way
    qv_mag_gate     = 0.15          # hide arrows in cells with magnitude < this × reference
    qv_speed_pct    = 98.0          # speed percentile mapped to a qv_arrow_frac-cell arrow
    qv_norm         = 'movie'       # mag/speed reference window: 'movie' (global) | 'frame'
    qv_arrow_frac   = 1.5           # arrow length in cell-widths (raise for BIGGER arrows)
    qv_arrow_width  = 0.006         # shaft thickness (raise for fatter arrows)
    make_movie      = False
    gb_frame_range  = (0, 3000)     # contiguous range for the saved movies (make_movie)

    if 'gabor' not in built_blocks:
        raise ValueError(f"no 'gabor' block in cache: {list(built_blocks)}")

    # Display frames — biased to motion (pix change) unless overridden.
    if ex_idx is not None:
        _pidx = [int(i) for i in ex_idx]
    elif frame_mode == 'motion_spread':
        _pidx = f_pick_motion_frames(built_blocks, ex_n, mode='spread', block='pix', win=qv_win)
    elif frame_mode == 'motion_event':
        _pidx = f_pick_motion_frames(built_blocks, ex_n, mode='event', block='pix',
                                     stride=ev_stride, win=qv_win)
    elif frame_mode == 'saved' and example_frames is not None:
        _pidx = list(example_frame_idx)
    else:
        _T = built_blocks['pix']['X'].shape[0]
        _pidx = list(np.unique(np.linspace(0, _T - 1, ex_n).round().astype(int)))

    # Scene row: true original if the cache has it at these frames, else pix reconstruction.
    _scene = _orig_frames(_pidx)
    if _scene is not None:
        _scene_lbl = 'original'
    else:
        _scene = f_reconstruct_feature_movie(built_blocks['pix'], 'pix', mon_hw, side_tags,
                                             n_pcs=scene_n_pcs, frame_idx=_pidx)
        _npc = (built_blocks['pix']['pca_pix'].n_components_ if scene_n_pcs is None
                else f_resolve_recon_n_pcs(scene_n_pcs, built_blocks['pix']['pca_pix']))
        _scene_lbl = f'pix recon ({_npc} PCs)'
    _mag = f_reconstruct_feature_movie(built_blocks['gabor'], 'gabor', mon_hw, side_tags,
                                       n_pcs=gb_n_pcs, channel=f'gabM_o{gb_io}sf{gb_isf}',
                                       frame_idx=_pidx)
    _pha = f_reconstruct_gabor_phase_movie(built_blocks['gabor'], mon_hw, side_tags,
                                           io=gb_io, isf=gb_isf, n_pcs=gb_n_pcs,
                                           cmap=gb_phase_cmap, frame_idx=_pidx)

    # Per-frame quiver field: only at the shown frames, each from a local window (cheap).
    _field = (f_gabor_flow_at(built_blocks['gabor'], side_tags, _pidx, isf=gb_isf,
                              n_pcs=gb_n_pcs, sign=qv_sign, win=qv_win) if overlay_quiver else None)
    f_plot_movie_frames_compare(
        [_scene, _mag, _pha], _pidx, labels=[_scene_lbl, 'gabor mag', 'gabor phase'],
        cmaps=['gray', 'gray', 'gray'],
        field=_field, field_idx=list(range(len(_pidx))),
        overlay_rows=(overlay_on if overlay_quiver else None),
        mon_hw=mon_hw, side_tags=side_tags, mag_gate=qv_mag_gate, speed_pct=qv_speed_pct,
        norm=qv_norm, arrow_frac=qv_arrow_frac, arrow_width=qv_arrow_width,
        suptitle=f'{_ds} — pix / gabor mag / phase o{gb_io}sf{gb_isf}'
                 + ('  + quiver' if overlay_quiver else ''))
    print(f'frames ({frame_mode if ex_idx is None else "explicit"}): {_pidx}')

    if make_movie:                       # save mag/phase (+quiver) over gb_frame_range
        _sidx = list(range(*gb_frame_range))
        _mmag = f_reconstruct_feature_movie(built_blocks['gabor'], 'gabor', mon_hw, side_tags,
                                            n_pcs=gb_n_pcs, channel=f'gabM_o{gb_io}sf{gb_isf}',
                                            frame_idx=_sidx)
        _mpha = f_reconstruct_gabor_phase_movie(built_blocks['gabor'], mon_hw, side_tags,
                                                io=gb_io, isf=gb_isf, n_pcs=gb_n_pcs,
                                                cmap=gb_phase_cmap, frame_idx=_sidx)
        _save_with_orig(_mmag, f'gabor_mag_o{gb_io}sf{gb_isf}', _sidx)
        _save_with_orig(_mpha, f'gabor_phase_{gb_phase_cmap}_o{gb_io}sf{gb_isf}', _sidx, pair=False)
        if overlay_quiver:
            _fl = f_gabor_flow_field(built_blocks['gabor'], side_tags, isf=gb_isf,
                                     n_pcs=gb_n_pcs, sign=qv_sign, frame_range=gb_frame_range)
            _qmov = f_render_flow_quiver_movie(_fl, mon_hw, side_tags, mag_gate=qv_mag_gate,
                                               speed_pct=qv_speed_pct, norm=qv_norm,
                                               arrow_frac=qv_arrow_frac, arrow_width=qv_arrow_width)
            _save_with_orig(_qmov, f'gabor_quiver_o{gb_io}sf{gb_isf}', pair=False)
