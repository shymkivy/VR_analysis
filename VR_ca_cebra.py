# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:01:39 2026

@author: ys2605
"""

import sys
import os
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'

sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearnex import patch_sklearn
patch_sklearn()

from f_utils import f_load_caim_data, f_gauss_smooth
from f_analysis import f_hclust_firing_rates, f_circshift_rates
from f_functions import f_load_bh_data_all, f_proc_movement, f_proc_lick_rew, f_comp_FOV_adj, f_add_phase, f_get_monitor_coords, f_angles_to_movie, f_angles_to_movie_v2, f_save_mon_movie, f_add_terrain_to_monitor #, f_plot_session
from f_feature_helpers import f_resample_to_imaging, build_feature_blocks, f_detrend_col, f_resolve_detrend_sigma
from f_visual_features import build_visual_blocks
from f_cebra_helpers import make_cebra_supervision, f_run_cebra, f_blocked_cv_r2
from f_decoding import (f_imaging_fs, make_decoder, f_shuffle_neural,
                        f_pca_prefix, f_apply_block_detrend, f_insample_predict,
                        build_target_columns, build_target_matrix,
                        legacy_target_blocks,
                        f_plot_pred_scatter, f_plot_pred_traces, f_plot_decode_bars,
                        f_plot_feature_heatmap, f_plot_decode_sweep_summary,
                        f_plot_decode_real_vs_null, f_plot_sweep_lines,
                        f_plot_oof_trace_scatter, f_plot_perfeature_null_grid,
                        f_plot_grid_delta_heatmap, f_plot_grid_focus_lines,
                        f_plot_real_vs_shuffle_line, f_plot_input_raster,
                        f_plot_block_traces)

from sklearn.decomposition import PCA

import time

#%%
# ==========================================================================
# SECTION 1 · SETUP — config, load data, pick dataset, smooth + hclust-sort
# ==========================================================================
mouse_id = 'L'

data_dir = 'F:/VR/data_proc/' + mouse_id    # edit this  
data_dir_bh = 'F:/VR/Bh_data/mice_gcamp/'
# search for files to load using tags in the filename

vr_data = pd.read_excel('F:/VR/data_proc/VR_data.xlsx')
params_xls = pd.read_excel(data_dir_bh + mouse_id + '_params.xlsx')

vr_data2 = vr_data.loc[(vr_data.mouse_id == mouse_id) & (vr_data.do_proc == 1)].reset_index(drop=True)
num_dsets = len(vr_data2)

lick_col = 'red'
lick_col2 = 'lightcoral'
rew_col = 'green'
rew_col2 = 'limegreen'

obj_size = {'x':        5,
            'y':        20, # for cylinder 20, for cube 10
            'z':        5,
            'height':   2}

# cylinders radius 5, half height 10.. coordinate at center.. height is y + height

#%%
data_ca = f_load_caim_data(data_dir, vr_data2.dset_name, caiman_tag = 'results_cnmf.hdf5', cuts_tag='h5cutsinfo', frame_data_tag = 'framedata', r_values_min = 0.5, min_SNR=1.5, thresh_cnn_min=0.8)

bh_data = f_load_bh_data_all(data_dir_bh + '/' + mouse_id + '/', vr_data2.bh_dset_name, params_xls, data_ca)

#%%
n_dset = 4

num_comp = 50
n_jobs = 5

est1 = data_ca[n_dset]
# Short tag used in every figure title: mouse + dset index + session name.
dset_tag = f"{mouse_id} dset{n_dset}  {est1['dset_name']}"
     
S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]

hclust_data = f_hclust_firing_rates(S_smn, standardize=True, metric='kl', method='average', similarity_transform='auto')   # cosine, jsd, kl  # average for jsd kl

S_smn2 = S_smn[hclust_data['res_order'],:]

if 0:
    plt.close('all')
    
    plt.figure()
    plt.imshow(S_smn2, aspect='auto', vmin=0, vmax=0.5, interpolation='none')
    plt.title(hclust_data['metric'])
    
    plt.figure()
    plt.imshow(hclust_data['similarity_matrix'][hclust_data['res_order'],:][:,hclust_data['res_order']], vmin=0.1, vmax=0.8)
    plt.title(hclust_data['metric'])
    plt.colorbar()
    
    # scale
    S_smn2s = S_smn2 - np.mean(S_smn2, axis=1)[:,None]
    S_smn2s = S_smn2s/np.std(S_smn2s, axis=1)[:,None]
    # S_smn2s = scaler.fit_transform(S_smn2.T).T
    
    plt.figure()
    plt.plot(S_smn2s[0,:])
    #plt.plot(X_scaled[0,:])
    
    plt.figure()
    plt.imshow(S_smn2s, aspect='auto', interpolation='none')
    

#%% Monitor vectorization for the selected dataset
# =============================================================================
# SECTION 2 · BEHAVIOR FEATURES — monitor vectorization, feature-block registry
# =============================================================================
# Builds mov_data, vec_data_l/r, two_mon_frames — the behavior inputs the
# CEBRA cells below consume. Mirrors the monitor-vectorization block in
# VR_ca_analysis.py (~line 378+) so this script is self-contained.

cam_params = {'aspect':             16/9,           # 1920/1080
              'FOV_axis':           'vertical',     # which axis is fixed
              'FOV_deg':            65.9,           # 80
              'cam_rotation_deg':   49.2,           # was 80/2
              'clip_len':           60,
              'num_mon':            2,
              'cam_height':         0.6}            # Unity rb→camera-eye Y
                                                    # offset for this rig
                                                    # (Cam_googles -0.4 +
                                                    # Camera_eye +1 = +0.6).
cam_params = f_comp_FOV_adj(cam_params)

mov_data = f_proc_movement(bh_data[n_dset], frame_times = est1['frame_times'],
                           do_interp=1, interp_step=0.1, plot_stuff=False)

# Lick + reward event traces on the behavior clock (T_beh,). Used by the
# 'beh' feature block when build_lick_reward=True in the build cell below.
lr_data = f_proc_lick_rew(bh_data[n_dset], mov_data,
                          frame_times=est1['frame_times'], plot_stuff=False)

# Build eye-shifted mov_data so f_get_monitor_coords computes object angles
# from the camera position (not the rb anchor). Without this the vertical
# angles are off by ~arctan(cam_height / dist) ≈ 1° at typical distances,
# which propagates into agg features + PCA decoding.
mov_data_eye = dict(mov_data)
mov_data_eye['y_pos'] = mov_data['y_pos'] + cam_params['cam_height']
mov_data_eye['xyz']   = np.column_stack([mov_data['x_pos'],
                                          mov_data_eye['y_pos'],
                                          mov_data['z_pos']])

phi   = mov_data_eye['phi']
theta = mov_data_eye['theta']
mon_r_phi = f_add_phase(phi,  cam_params['cam_rotation_rad'])
mon_l_phi = f_add_phase(phi, -cam_params['cam_rotation_rad'])

vec_data_r = f_get_monitor_coords(mov_data_eye, mon_r_phi, theta, bh_data[n_dset], cam_params, remove_outside_objects=True)
vec_data_l = f_get_monitor_coords(mov_data_eye, mon_l_phi, theta, bh_data[n_dset], cam_params, remove_outside_objects=True)

num_samp = 101
# max_frames: cap on the time axis for both monitor + terrain renderers.
# Set to None to render the full session; small integer (e.g., 200, 1000) for
# fast debugging. Threaded through f_angles_to_movie_v2 and f_render_terrain.
max_frames = None
render_verbose = False        # progress prints from f_angles_to_movie_v2 +
                              # f_render_terrain. Useful first run to gauge
                              # timing; turn off once stable.
#left_mon_frames  = f_angles_to_movie(vec_data_l, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp)
#right_mon_frames = f_angles_to_movie(vec_data_r, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp)
left_mon_frames, left_mon_depth  = f_angles_to_movie_v2(vec_data_l, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp
                                        , filled = True, antialias = True, chunk_t = None, return_depth = True, max_frames = max_frames, verbose=render_verbose)
right_mon_frames, right_mon_depth = f_angles_to_movie_v2(vec_data_r, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp
                                        , filled = True, antialias = True, chunk_t = None, return_depth = True, max_frames = max_frames, verbose=render_verbose)

# Optional terrain layer behind the objects. When True, renders the heightmap
# for each monitor and Z-buffer-composites it with the object render (terrain
# in front of an object → terrain wins; object in front of terrain → object
# wins). Adds ground-plane structure to the pix-PCA input. See
# f_add_terrain_to_monitor in f_functions.py; defaults match this rig
# (chunk_pitch=122, flip_z=True, eye_height=0 since mouse_xyz is eye-shifted).
render_terrain = False
terrain_clip_len   = None     # None = use cam_params['clip_len']; override for
                              # closer/farther terrain than objects (e.g. 120).
terrain_stride     = 2      # int ≥ 1 keeps every Nth cell; float < 1 bilinear-
                              # oversamples. Memory ~ (1/stride)² when below 1.
terrain_point_size = 1        # half-size of pixel patch painted per sample.
if render_terrain:
    mouse_xyz_eye = np.stack([mov_data_eye['x_pos'], mov_data_eye['y_pos'],
                               mov_data_eye['z_pos']], axis=1)
    left_mon_frames = f_add_terrain_to_monitor(
        bh_data[n_dset]['terrainData'], mouse_xyz_eye, mon_l_phi, theta, cam_params,
        left_mon_frames, left_mon_depth,
        lat_samp=num_samp, vert_samp=num_samp,
        stride=terrain_stride, point_size=terrain_point_size,
        clip_len=terrain_clip_len, max_frames=max_frames, verbose=render_verbose,
    )
    right_mon_frames = f_add_terrain_to_monitor(
        bh_data[n_dset]['terrainData'], mouse_xyz_eye, mon_r_phi, theta, cam_params,
        right_mon_frames, right_mon_depth,
        lat_samp=num_samp, vert_samp=num_samp,
        stride=terrain_stride, point_size=terrain_point_size,
        clip_len=terrain_clip_len, max_frames=max_frames, verbose=render_verbose,
    )

two_mon_frames = np.concatenate((left_mon_frames, right_mon_frames), axis=2)

#%% Save synthetic-monitor stack to disk (for visually checking f_angles_to_movie_v2)
# Multi-page TIFF -> open in ImageJ / Fiji as a movie. Toggle on per-run.
if 0:
    mov_out_dir = 'F:/test_mov'
    dset_tag = est1['dset_name']

    # full session, both monitors side-by-side
    f_save_mon_movie(two_mon_frames,
                     os.path.join(mov_out_dir, f'two_mon_v2_{dset_tag}.tif'))

    # short clip — first 5000 s (mov_data['time'] is on the imaging clock)
    dt = float(mov_data['time'][1] - mov_data['time'][0])
    n_clip = int(round(5000.0 / dt))
    f_save_mon_movie(two_mon_frames,
                     os.path.join(mov_out_dir, f'two_mon_v2_{dset_tag}_500s.tif'),
                     frame_range=(0, n_clip))

#%% Approach 4 — CEBRA: behavior-supervised joint embedding
# Fits CEBRA twice with the same neural input but different behavior
# supervision signals:
#   (a) aggregated per-monitor scalars (8 dims): presence + nearest-obj
#       lat/vert/dist per side. Small, interpretable, fast.
#   (b) PCA-compressed pixel movie (~20 dims): top PCs of two_mon_frames
#       flattened. Captures spatial layout the aggregates miss.
#
# Same neural target lets us A/B which behavior parameterization the neural
# manifold "agrees with" better, via:
#   - visual: 2D/3D embedding scatter, colored by behavior variables.
#   - quantitative: KNN-regression decoding of behavior from embedding, with
#     time-blocked CV. The supervision signal that's better-decoded from a
#     CEBRA model trained on it is the one the neural population actually
#     represents most strongly.
#
# Time alignment: f_proc_movement samples behavior directly onto the imaging
# frames, pulse-corrected, so mov_data['time'] == est1['frame_times'] (the
# imaging clock) and mov_data['delay_corrected'] is True. Downstream resampling
# uses pulse_delay=0 (already aligned).
#
# Prereqs (all built earlier in this script):
#   bh_data[n_dset]  (with 'bh_pulse_delay', from f_load_bh_data_all)
#   mov_data         (from f_proc_movement, monitor-vectorization cell)
#   vec_data_l, vec_data_r  (from f_get_monitor_coords for each monitor)
#   two_mon_frames   (from f_angles_to_movie, concatenated L+R)
#   est1, S_smn2, n_dset, cam_params, obj_size

#%% Build behavior features aligned to imaging frame times
# Two feature sets ready to feed CEBRA. Sanity-check shapes and value ranges
# before training.
#
# Two gating knobs for distance handling (rationale: in the raw pix-PCA
# representation, far objects naturally occupy almost no pixels so the model
# discounts them automatically; the agg features need to be told this
# explicitly):
#   agg_mode         — controls how the per-side distance channel is encoded.
#                      'raw'              → [pres, lat, vert, dist]      (8 total)
#                      'angular'          → dist replaced by 2·arctan2(obj_size_x, dist)
#                                           — visual angular size, same nonlinearity
#                                           pix-PCA encodes implicitly      (8 total)
#                      'salience'         → adds salience = max(0, 1 − dist/clip_len)
#                                           alongside raw dist               (10 total)
#                      'angular_salience' → both                              (10 total)
#   near_dist_thresh — optional hard frame cut. None / 0 keeps all frames.
#                      Otherwise drops frames where no object on EITHER monitor
#                      is within near_dist_thresh. Caveat: hard cuts break
#                      temporal contiguity, which CEBRA's contrastive sampling
#                      depends on — use only if (A)/(B) don't help.

agg_mode         = 'angular'     # 'raw' | 'angular' | 'salience' | 'angular_salience'
near_dist_thresh = None      # e.g. 30 to keep only near frames; None to keep all
side             = 'R'       # 'L' | 'R' | 'both' — start simple with right monitor only
nan_absent       = True      # mask continuous agg channels to NaN when presence==0
                             # (otherwise they collapse to 0, which is ambiguous with
                             # "object at center" — see notes 2026-05-19 / task #12)
obj_collapse     = 'salience_mean'   # 'nearest' | 'salience_mean'
                             # 'nearest':       pick single nearest in-FOV obj (causes
                             #                  step discontinuities at object swaps)
                             # 'salience_mean': angular-size-weighted mean across all
                             #                  in-FOV objs (smooth, visual center of
                             #                  mass). See task #17.
build_per_obj    = False     # if True, also build X_per_obj (per-object scalar features,
                             # one slot per ORIGINAL object id, channels shared across
                             # monitors). Adds identity + multi-object info that X_agg
                             # collapses away. ~total_n_obj * 4 (or 5) * n_sides dims.
                             # Doesn't replace X_agg — separate optional feature set.
# --- Two distinct kinds of motion (see also f_motion_features /
#     f_self_motion_features in f_feature_helpers.py) ---
#   'motion'   = VISUAL/RETINAL motion of the OBJECTS on each monitor — how
#                fast/which way an object sweeps across the screen. Object-
#                relative, per monitor side, NaN when no object in FOV. Biology:
#                looming / optic-flow / motion-direction cells.
#   'self_mot' = EGOMOTION of the MOUSE itself in world coords — running speed +
#                yaw rate, from mov_data. Independent of objects, single (not
#                per side), always defined (no NaN). Biology: vestibular /
#                head-direction / locomotion.
#   They overlap (turning/running also moves objects on screen) but aren't
#   redundant: 'self_mot' isolates pure locomotion regardless of objects;
#   'motion' also carries the object's OWN world motion + looming/presence.
build_motion     = True      # if True, build 'motion' — per-side OBJECT screen
                             # velocity: d_lat, d_vert, d_dist [+d_ang_size in
                             # angular modes (looming correlate)], plus motion_mag
                             # and motion_dir_sin/cos. 6 ch/side (raw/salience) or
                             # 7 (angular). Companion to X_agg; concat downstream
                             # for "position + velocity" decoding (task #22).
motion_smooth_sigma = 2.0    # Gaussian σ (behavior samples) applied to position
                             # before differentiation. 0 disables. At 10 Hz behavior
                             # clock, σ=2 ≈ 200 ms — light low-pass.
build_self_motion = True     # if True, build 'self_mot' — MOUSE egomotion from
                             # mov_data: linear running speed (self_d_dist) + yaw
                             # rate (self_d_phi), world frame. Captures vestibular /
                             # head-direction-style signals the object-motion
                             # channels miss. 2 channels, no NaN-masking (defined
                             # everywhere). Task #23.
self_motion_smooth_sigma = 2.0   # σ for position smoothing before differentiation.
build_pix_motion = True      # if True, build 'pix_mot' block: d/dt of each
                             # pix-PC trace. Captures "rate of change of each
                             # static-scene variance mode" — pixel-side parallel
                             # to the (object-space) 'motion' block. n_pix_pca
                             # channels, no NaN-masking. Task #29.
pix_motion_smooth_sigma = 2.0    # σ for pix-PC smoothing before differentiation.
build_lick_reward = True     # if True, build 'beh' block from lr_data:
                             # 2 channels (lick, rew) per behavior frame.
                             # Sparse event counts by default. Task #35.
lick_reward_smooth_sigma = 0.0   # 0 = raw 0/1 events. Positive → Gaussian
                                 # smooth, e.g. 5 ≈ 500 ms at 10 Hz behavior
                                 # clock for a "lick rate" signal. Use raw
                                 # for sharp event decoding; smooth for
                                 # Ridge-style continuous-rate decoding.

# Visual-cortex-style monitor feature blocks (location/edge/direction-selective),
# built from the rendered monitor movie via f_visual_features.build_visual_blocks
# and merged into built_blocks. See PLAN_monitor_features.md / TODO #47.
build_visual    = True       # if True, also build the visual blocks below.
visual_specs    = {          # which visual feature types + params to build:
    # 'stats' selects which feature types to include (any subset, order kept):
    #   grid: {'occupancy','mean','edge'}   flow: {'speed','dir','div'}
    #   omit 'stats' → registry default (all). e.g. ('mean','edge') drops occupancy.
    # 'n_pca' = PCA dim on the raw cell-features (None/omit → no PCA, raw block).
    'grid': {'grid': (24, 24), 'n_pca': 100,    # Tier 1 — retinotopic pooling
             'stats': ('occupancy', 'mean', 'edge')},  # (per-cell; edit to subset).
    'flow': {'grid': (24, 24), 'n_pca': 100,    # Tier 2 — optical flow on the grid
             'method': 'lucas_kanade',       # (speed/dir/divergence per cell).
             'stats': ('speed', 'dir', 'div')},  # method 'lucas_kanade' = numpy-only
                                             # (no deps); 'farneback' needs cv2
                                             # (not in the rnn kernel env).
    # 'gabor': {'grid': (6, 6), 'n_pca': 30},  # Tier 3 — Gabor energy  (later)
}

# mov_data is already on the imaging clock when delay_corrected, so the
# build-cell resampling uses pulse_delay=0 (no second correction).
pulse_delay = 0.0 if mov_data.get('delay_corrected') else bh_data[n_dset]['bh_pulse_delay']
beh_t = mov_data['time']
frame_t = np.asarray(est1['frame_times']).ravel()
T_img = frame_t.shape[0]

# Side selection — pick which monitor(s) and which pixel movie to feed in.
if side == 'L':
    vec_data_use, side_tags, pix_frames = [vec_data_l], ['L'], left_mon_frames
elif side == 'R':
    vec_data_use, side_tags, pix_frames = [vec_data_r], ['R'], right_mon_frames
elif side == 'both':
    vec_data_use, side_tags, pix_frames = [vec_data_l, vec_data_r], ['L', 'R'], two_mon_frames
else:
    raise ValueError(f'unknown side: {side!r}')

# max_frames truncates the rendered movies (render/save inspection only); the
# analysis pipeline needs them full-length to align with beh_t / vec_data.
if pix_frames.shape[0] != T_img:
    raise ValueError(
        f'pix_frames has {pix_frames.shape[0]} frames but behavior has {T_img} '
        f'(max_frames={max_frames}). Set max_frames=None before building features '
        f'— it applies to the render/save cell only.')

# All feature blocks (agg, pix, motion, per_obj) built in one call.
# See functions/f_cebra_helpers.py:build_feature_blocks for the per-block
# logic, channel layouts, and NaN-masking conventions.
n_pix_pca = 50    # kept top-level so plot cells can reference it
built_blocks = build_feature_blocks(
    vec_data_use=vec_data_use, side_tags=side_tags,
    beh_t=beh_t, frame_t=frame_t, pulse_delay=pulse_delay,
    agg_mode=agg_mode, obj_size=obj_size,
    clip_len=cam_params['clip_len'],
    obj_collapse=obj_collapse, nan_absent=nan_absent,
    pix_frames=pix_frames, n_pix_pca=n_pix_pca,
    build_motion=build_motion, motion_smooth_sigma=motion_smooth_sigma,
    build_per_obj=build_per_obj,
    total_n_obj=(len(bh_data[n_dset]['object_data']) if build_per_obj else None),
    build_self_motion=build_self_motion,
    mov_data=mov_data,
    self_motion_smooth_sigma=self_motion_smooth_sigma,
    build_pix_motion=build_pix_motion,
    pix_motion_smooth_sigma=pix_motion_smooth_sigma,
    build_lick_reward=build_lick_reward,
    lr_data=lr_data,
    lick_reward_smooth_sigma=lick_reward_smooth_sigma,
)

# Visual-cortex-style monitor blocks merged into the SAME registry. Per-side
# movies are on the behavior clock; build_visual_blocks resamples to frame_t
# (like build_feature_blocks does for pix) and optionally PCA-compresses. Added
# before the near-distance filter below so they get the same frame subsetting.
if build_visual:
    _mon_movies = {'L': left_mon_frames, 'R': right_mon_frames}
    built_blocks.update(build_visual_blocks(
        [_mon_movies[t] for t in side_tags], side_tags,
        beh_t, frame_t, pulse_delay, visual_specs))
    for _vn in visual_specs:
        _vb = built_blocks[_vn]
        _pca_tag = f', PCA→{_vb["X"].shape[1]}' if _vb['pca'] is not None else ''
        print(f'  visual block {_vn!r}: {_vb["X"].shape}  (raw {_vb["raw_dim"]}d{_pca_tag})')

# Legacy aliases — downstream cells (raster, near-distance filter, CEBRA
# supervision, etc.) reference these names. Keep them in sync with the
# registry; the registry cell below is now redundant for the agg/pix blocks
# but still wraps things for combine_blocks consumers.
X_agg              = built_blocks['agg']['X']
X_pix              = built_blocks['pix']['X']
X_mot              = built_blocks.get('motion',  {}).get('X')
X_per_obj          = built_blocks.get('per_obj', {}).get('X')
agg_feat_names     = built_blocks['agg']['names']
mot_feat_names     = built_blocks.get('motion',  {}).get('names')
per_obj_feat_names = built_blocks.get('per_obj', {}).get('names')
n_per_side         = built_blocks['agg']['n_per_side']
pres_cols          = built_blocks['agg']['pres_cols']
side_feat_names_single = built_blocks['agg']['side_feat_names_single']
pca_pix            = built_blocks['pix']['pca_pix']
n_sides            = len(side_tags)

# Neural: same orientation as the dim-red sweep (T_img × N).
Y_neu = S_smn2.T.astype(np.float32)

# Optional near-distance frame filter (Option C). Build the "any near"
# mask on the behavior clock, resample to imaging clock, then apply to
# every feature block + Y_neu so they stay aligned.
if near_dist_thresh:
    near_per_side = [(v['obj_mon_idx'] & (v['obj_dist'] < near_dist_thresh)).any(axis=1)
                     for v in vec_data_use]
    near_beh = np.any(near_per_side, axis=0).astype(np.float32)
    near_mask_img = f_resample_to_imaging(beh_t, near_beh[:, None], frame_t, pulse_delay,
                                          kind='nearest').ravel() > 0.5
    kept = int(near_mask_img.sum())
    print(f'near-distance filter (d<{near_dist_thresh}): keeping {kept}/{T_img} frames'
          f' ({100*kept/T_img:.1f}%) — NB: hard cuts break temporal contiguity')
    for _bname in built_blocks:
        built_blocks[_bname]['X'] = built_blocks[_bname]['X'][near_mask_img]
    Y_neu = Y_neu[near_mask_img]
    # Refresh legacy aliases to point at the filtered arrays.
    X_agg = built_blocks['agg']['X']
    X_pix = built_blocks['pix']['X']
    X_mot     = built_blocks.get('motion',  {}).get('X')
    X_per_obj = built_blocks.get('per_obj', {}).get('X')
else:
    near_mask_img = np.ones(T_img, dtype=bool)

# Filtered plotting views — match the feature blocks / Y_neu length. Plot cells
# use these (NOT S_smn2 / est1['frame_times'] directly) so they stay aligned
# under near_dist_thresh. frame_t stays full so f_imaging_fs sees true spacing.
S_raster   = S_smn2[:, near_mask_img]
t_img_plot = frame_t[near_mask_img]

# Standardize each feature column for CEBRA (it does NOT internally normalize).
# CEBRA can't handle NaN — fill them with 0 in the z-scored version only.
# Decoding cells use the NaN-bearing X_agg directly.
#
# Orphaned 2026-05-21: these z-scored copies are no longer consumed —
# make_cebra_supervision (defined in the registry cell) builds the same
# matrix on demand from built_blocks. Kept commented for reference.
# X_agg_for_cebra = np.where(np.isnan(X_agg), 0.0, X_agg)
# X_agg_z = f_zscore(X_agg_for_cebra)
# X_pix_z = f_zscore(X_pix)
# X_per_obj_z = None
# if X_per_obj is not None:
#     X_per_obj_for_cebra = np.where(np.isnan(X_per_obj), 0.0, X_per_obj)
#     X_per_obj_z = f_zscore(X_per_obj_for_cebra)

# Quick shape / range check before training.
agg_nan_frac = float(np.isnan(X_agg).mean())
print(f'side={side!r}  agg_mode={agg_mode!r}  obj_collapse={obj_collapse!r}  '
      f'n_per_side={n_per_side}  agg width={X_agg.shape[1]}  NaN frac={agg_nan_frac:.2%}')
print('Y_neu     :', Y_neu.shape,   'min/max:', Y_neu.min(),  Y_neu.max())
print('X_agg     :', X_agg.shape,   'min/max:', np.nanmin(X_agg), np.nanmax(X_agg))
print('X_pix     :', X_pix.shape,   'min/max:', X_pix.min(),  X_pix.max())
_pix_n_used = pca_pix.n_components_                          # PCs kept (= n_pix_pca)
_pix_n_max  = min(pca_pix.n_samples_, pca_pix.n_features_in_)  # full pixel rank ceiling
print(f'pix PCA exp_var (cum): {np.cumsum(pca_pix.explained_variance_ratio_)[-1]:.3f}'
      f'  ({_pix_n_used}/{_pix_n_max} PCs used — {_pix_n_used} of {_pix_n_max} possible)')

# Multi-object stats per side — tells us how often the collapse matters.
for v, tag in zip(vec_data_use, side_tags):
    n_in_fov = v['obj_mon_idx'].sum(axis=1)
    any_obj  = (n_in_fov >= 1).mean()
    multi    = (n_in_fov >= 2).mean()
    max_sim  = int(n_in_fov.max()) if n_in_fov.size else 0
    print(f'  side {tag}: any-obj-in-FOV {any_obj:.1%}, multi-obj (≥2) {multi:.1%}, '
          f'max simultaneous {max_sim}')


#%% Feature registry — helpers around the built_blocks dict
# built_blocks is built by build_feature_blocks() in the build cell above;
# this cell just adds convenience helpers (combine_blocks, make_cebra_supervision)
# and the summary printout.

print(f'\n{dset_tag} — feature registry (built_blocks):')
for name, b in built_blocks.items():
    n_nan_frac = float(np.isnan(b['X']).mean())
    print(f'  {name:10s}  {b["X"].shape}  ({len(b["names"])} cols)  NaN frac={n_nan_frac:.2%}')


def combine_blocks(group_names, blocks=None):
    """Concatenate selected feature blocks horizontally.

    Parameters
    ----------
    group_names : str | list of str
        Block name(s) from the registry, e.g. ['agg', 'motion'] or 'agg'.
    blocks : dict, optional
        Registry to read from; defaults to module-global `built_blocks`.

    Returns
    -------
    X : ndarray, (T, sum_d)
        Concatenated feature matrix (same T as each block).
    names : list of str
        Column names in concatenation order.

    Raises
    ------
    ValueError if a requested block isn't in the registry (didn't get built —
    check the corresponding build_* knob in the build cell above).
    """
    if blocks is None:
        blocks = built_blocks
    if isinstance(group_names, str):
        group_names = [group_names]
    missing = [g for g in group_names if g not in blocks]
    if missing:
        avail = list(blocks.keys())
        raise ValueError(f'block(s) {missing!r} not built; available: {avail}')
    Xs = [blocks[g]['X'] for g in group_names]
    ns = []
    for g in group_names:
        ns.extend(blocks[g]['names'])
    return np.concatenate(Xs, axis=1), ns


# Usage examples (uncomment to try):
# X_test, names_test = combine_blocks(['agg'])              # just agg, equivalent to X_agg
# X_test, names_test = combine_blocks(['agg', 'motion'])    # position + velocity
# X_test, names_test = combine_blocks(['agg', 'motion', 'pix'])   # all three


# make_cebra_supervision moved to f_cebra_helpers.py on 2026-05-21.
# Now takes built_blocks as an explicit argument (no module-global lookup):
#     sup, names = make_cebra_supervision(['agg'], built_blocks)


#%% Reconstruct a monitor movie from a feature block (visualize what it keeps)
# Rank-k reconstruction of a feature representation back to image space — to SEE
# what each method encodes and how it degrades with fewer PCs. Saves a multi-page
# TIFF (open in ImageJ / Fiji to play). See f_feature_viz.py.
#
# recon_block — which block to reconstruct. Must be a key currently in
# built_blocks AND image-space (the cell raises otherwise). Available now:
#   'pix'   — true pixel reconstruction (inverse-PCA of the pix block).
#   'grid'  — paint each retinotopic cell by a stat (occ/mean/edge): blocky view.
#   'flow'  — paint each cell by a flow stat (speed/div): motion-energy field.
#   (Check what's built: list(built_blocks).)
#   Future visual tiers ('gabor', …) will use the same cell-paint path once
#   implemented in f_visual_features + added to visual_specs above (TODO #50);
#   until then selecting them raises 'not in built_blocks'.
#   NOT reconstructable: agg / motion / self_mot / pix_mot / beh / per_obj —
#   these have no monitor-image layout, so there's nothing to paint back. Use the
#   generic feature-trace viewer cell (f_plot_block_traces) to inspect those.
#
# recon_channel — which stat to paint (ignored for 'pix'):
#   grid:  'occ' | 'mean' | 'edge'        ('auto' → 'mean')
#   flow:  'speed' | 'div'                ('auto' → 'speed')
#
# recon_stack_orig — if True, the REAL monitor movie is concatenated below the
#   reconstruction (top=recon, bottom=original) into one TIFF for a synced visual
#   comparison. The original (behavior clock) is resampled to the imaging clock to
#   match the recon frames. Works for any block — bottom is always the real scene,
#   so you can see what grid/flow drop vs. the pixels.
#
# Movies are on the imaging clock (the block's clock); cap recon_frame_range for
# long sessions. (Sweep recon_n_pcs to watch detail come back as PCs are added;
# print reports used/total PCs — None hits the block's PCA ceiling, e.g. 20/20.)
if 1:
    from f_feature_viz import (f_reconstruct_feature_movie,
                               f_stack_recon_over_original,
                               f_resolve_recon_n_pcs)

    recon_block       = 'grid'      # 'pix' | 'grid' | 'flow' (built now); see header
    recon_n_pcs       = None         # int=#PCs | float in (0,1)=variance frac (0.9)
                                     #   | None=all. e.g. 0.9 → enough PCs for 90% var
    recon_channel     = 'edge'     # grid:'occ'|'mean'|'edge'; flow:'speed'|'div'; 'auto'
    recon_frame_range = (0, 3000)  # (start, stop) imaging frames to save (None = all)
    recon_save        = True
    recon_stack_orig  = True       # stack the ORIGINAL monitor movie below the recon
    recon_gap_px      = 4          # black separator rows between the two panels
    _stack_tag        = '_vsorig' if recon_stack_orig else ''

    if recon_block not in built_blocks:
        raise ValueError(f'{recon_block!r} not in built_blocks: {list(built_blocks)}')
    # resolve PCs (int / variance-fraction / None) BEFORE naming + recon so the
    # filename carries the actual PC count, not '0.9pc'.
    _rb = built_blocks[recon_block]
    _pca = _rb.get('pca_pix') if recon_block == 'pix' else _rb.get('pca')
    _n_total = _pca.n_components_ if _pca is not None else _rb['X'].shape[1]
    _n_used = (_n_total if recon_n_pcs is None
               else f_resolve_recon_n_pcs(recon_n_pcs, _pca))
    recon_out = (f'F:/test_mov/recon_{recon_block}_{recon_channel}_'
                 f'{_n_used}pc{_stack_tag}_{est1["dset_name"]}.tif')

    mov_recon = f_reconstruct_feature_movie(
        built_blocks[recon_block], recon_block, (num_samp, num_samp), side_tags,
        n_pcs=recon_n_pcs, channel=recon_channel)
    # report used/total PCs + the variance the used PCs capture. _evr_block =
    # fraction of the block's RETAINED variance (1.0 = all kept PCs); _evr_orig =
    # fraction of the ORIGINAL feature-space variance the block keeps in total.
    if _pca is not None:
        _evr = _pca.explained_variance_ratio_
        _evr_block = float(np.sum(_evr[:_n_used]) / np.sum(_evr))
        _evr_orig  = float(np.sum(_evr))
        _var_msg = (f', {_evr_block:.1%} of block variance '
                    f'(block keeps {_evr_orig:.1%} of original in {_n_total} PCs)')
    else:
        _var_msg = ' (raw block, no PCA)'
    print(f'reconstructed {recon_block!r} movie: {mov_recon.shape} '
          f'({_n_used}/{_n_total} PCs used{_var_msg}, channel={recon_channel})')

    # Stack the original (real monitor movie) below the reconstruction for a
    # synced top=recon / bottom=original comparison. Original is on the behavior
    # clock; the helper resamples it to the imaging clock to match mov_recon.
    mov_out = mov_recon
    if recon_stack_orig:
        _orig_src = {'L': left_mon_frames, 'R': right_mon_frames,
                     'both': two_mon_frames}[side]
        mov_out = f_stack_recon_over_original(
            mov_recon, _orig_src, beh_t, frame_t, pulse_delay,
            gap_px=recon_gap_px)
        print(f'stacked recon over original: {mov_out.shape} '
              f'(top=recon {recon_block!r}, bottom=original monitor)')

    if recon_save:
        f_save_mon_movie(mov_out, recon_out, frame_range=recon_frame_range, gap_px=0)


#%% Fit CEBRA — agg and pix supervision, same neural input
# =============================================================================
# SECTION 3 · CEBRA — fit, loss curves, embedding scatter
# =============================================================================
# offset10-model = standard CEBRA architecture (1D conv stack with 10-frame
# receptive field). For 60 Hz imaging that's ~167 ms of context per anchor,
# matches calcium decay time scale.
# conditional='time_delta' picks positives within a learned time window weighted
# by behavior similarity — the right choice for continuous-supervised CEBRA.
# Note CEBRA copies tensors to GPU if available; without GPU, drop max_iterations
# to 2000 or batch_size to 256.

# f_run_cebra moved to f_cebra_helpers.py on 2026-05-21. Imports the CEBRA
# package lazily inside the function — this script doesn't need to import
# cebra at top level anymore.

# Which blocks to use as supervision for each CEBRA model. Each entry is
# {model_name: [list of block names from built_blocks registry]}. Default
# matches the original setup (one CEBRA per block), but you can train CEBRA
# on combined feature sets — e.g., 'agg_motion': ['agg', 'motion'] supervises
# CEBRA jointly on position + velocity.
#
# Any block in built_blocks can be used as supervision (each must have been
# built — see the build cell's build_* knobs / visual_specs):
#   'agg'      — per-side object position (always built)
#   'pix'      — pixel-movie PCA components (always built)
#   'motion'   — per-side object screen velocity        (build_motion)
#   'self_mot' — mouse egomotion: speed + yaw rate       (build_self_motion)
#   'pix_mot'  — d/dt of each pix-PC trace               (build_pix_motion)
#   'beh'      — lick + reward events                    (build_lick_reward)
#   'per_obj'  — per-object identity slots               (build_per_obj)
#   'grid'     — retinotopic grid (visual, Tier 1)       (build_visual + visual_specs)
#                ('flow'/'gabor' = Tiers 2/3, planned — TODO #47)
# NaN cols are filled with 0 + z-scored by make_cebra_supervision (CEBRA can't
# take NaN). The model name (dict key) is free-form; it labels the embedding
# everywhere downstream and is selectable as 'cebra:<name>' in the decoder
# input-source knobs.
#
# Examples:
#   'agg': ['agg'], 'pix': ['pix']            — one CEBRA per block (default)
#   'agg_motion': ['agg', 'motion']           — position + velocity jointly
#   'grid': ['grid']                          — supervise on the retinotopic grid
#   'all_vis': ['pix', 'grid']                — combined visual representations
cebra_supervisions = {
    'agg': ['agg'],
    'pix': ['pix'],
    'grid': ['grid'],   # visual retinotopic grid (Tier 1) — needs build_visual=True
    'flow': ['flow'],   # visual optical-flow grid (Tier 2) — needs build_visual=True
}

# Old hardcoded calls kept commented:
# cebra_agg = f_run_cebra(Y_neu, X_agg_z, out_dim=3, max_iter=5000)
# cebra_pix = f_run_cebra(Y_neu, X_pix_z, out_dim=3, max_iter=5000)

cebra_models = {}
for _cname, _blocks in cebra_supervisions.items():
    _sup, _sup_names = make_cebra_supervision(_blocks, built_blocks)
    cebra_models[_cname] = f_run_cebra(Y_neu, _sup, out_dim=3, max_iter=5000)
    cebra_models[_cname]['supervision_blocks'] = list(_blocks)
    cebra_models[_cname]['supervision_names']  = _sup_names
    cebra_models[_cname]['supervision_z']      = _sup
    print(f"  cebra-{_cname:10s}  blocks={_blocks!r:25s}  d={_sup.shape[1]:>3d}  "
          f"fit {cebra_models[_cname]['duration']:.1f}s")

# (Removed the cebra_agg / cebra_pix back-compat aliases — every downstream cell
# now iterates cebra_models generically. Index it directly, e.g.
# cebra_models['agg'], if you need a specific model interactively.)

#%% CEBRA loss curves — sanity-check convergence
# CEBRA's InfoNCE loss should decrease and plateau. If it's still falling at
# max_iter, train longer. If it never moves, the behavior signal isn't
# informative w.r.t. the neural data (or hyperparameters are wrong).

if 0:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for _cname, _m in cebra_models.items():
        _blocks = _m.get('supervision_blocks', [_cname])
        _d = _m['supervision_z'].shape[1]
        ax.plot(_m['model'].state_dict_['loss'],
                label=f"{_cname} ({'+'.join(_blocks)}, {_d}d)")
    ax.set_xlabel('iteration')
    ax.set_ylabel('InfoNCE loss')
    ax.set_title(f'{dset_tag}\nCEBRA training loss')
    ax.legend(fontsize=8)
    
    plt.close('all')


#%% CEBRA embedding scatter — colored by behavior
# 3D scatter of the embedding's first 3 dims; one column per CEBRA model
# (agg-supervised, pix-supervised). Two figures share that scatter layout but
# differ in coloring source:
#   Fig 1 — each agg INPUT channel on the chosen side
#           (pres, lat, vert, dist_chan, [+salience]) — 4 or 5 rows.
#   Fig 2 — the top n_pix_plot pix-PCs — n_pix_plot rows.
# If supervision is working, points with similar coloring should cluster.

from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers projection

# Two figures: same CEBRA embeddings, different coloring sources.
#   Figure 1 — agg-supervised embedding colored by each agg INPUT channel
#              (pres, lat, vert, dist_chan, [+salience]) on the chosen side.
#   Figure 2 — pix-supervised embedding colored by the top n_pix_plot pix-PCs.
# Both arranged as grids; rows / cols knobs below.

# ── Layout knobs ─────────────────────────────────────────────────────────
agg_rows    = 2     # rows in fig 1 (n_cols derived from n_per_side)
n_pix_plot = 6     # how many top pix-PCs to color by in fig 2
pix_rows    = 2     # rows in fig 2 (cols derived from n_pix_plot)
n_grid_plot = 6     # how many top components to color by for VISUAL blocks
                    # (grid/flow/gabor — block-supervised embeddings)
grid_rows   = 2     # rows in each visual-block coloring figure
plot_2d     = False # True → scatter only the top 2 manifold dims (2D plot);
                    # False → top 3 manifold dims (3D scatter).
# ─────────────────────────────────────────────────────────────────────────

# Pick which side to color by — prefer 'R' if it's in use, otherwise whatever
# is. Each side's block starts at side_idx * n_per_side.
color_tag = 'R' if 'R' in side_tags else side_tags[0]
side_idx  = side_tags.index(color_tag)
base      = side_idx * n_per_side

# Per-channel cmap (length matches side_feat_names_single, 4 or 5).
chan_cmaps = ['binary', 'twilight', 'twilight', 'viridis']  # pres, lat, vert, dist/ang_size
if n_per_side == 5:
    chan_cmaps.append('plasma')                              # salience

_agg_X = built_blocks['agg']['X']
agg_color_specs = [
    (f'{nm}_{color_tag}', _agg_X[:, base + ci], chan_cmaps[ci])
    for ci, nm in enumerate(side_feat_names_single)
]

_pix_X = built_blocks['pix']['X']
n_pix_plot = min(n_pix_plot, _pix_X.shape[1])
pix_color_specs = [(f'pix_PC{i+1}', _pix_X[:, i], 'viridis') for i in range(n_pix_plot)]

# Visual-block coloring — for any visual feature block (grid/flow/gabor/...),
# color a block-supervised embedding by its top components (cols are '<blk>_PC*'
# when PCA'd, else raw channels). One color-spec list per built visual block;
# the scatter loop below plots whichever block the model was supervised on.
visual_color_blocks = [b for b in ('grid', 'flow', 'gabor') if b in built_blocks]
visual_color_specs = {}
for _vb in visual_color_blocks:
    _vx = built_blocks[_vb]['X']; _vn = built_blocks[_vb]['names']
    _k = min(n_grid_plot, _vx.shape[1])
    visual_color_specs[_vb] = [(_vn[i], _vx[:, i], 'viridis') for i in range(_k)]


def _plot_emb_scatter(model, color_specs, n_rows, suptitle):
    # Grid scatter: one CEBRA model, n_rows × n_cols panels (one per coloring).
    # n_cols is derived from len(color_specs) so the grid fits.
    # Uses 2D (top 2 dims) or 3D (top 3 dims) scatter per `plot_2d` knob above.
    n_panels = len(color_specs)
    n_cols = (n_panels + n_rows - 1) // n_rows
    fig = plt.figure(figsize=(4.0*n_cols, 3.2*n_rows))
    emb = model['embedding']
    stride = max(1, emb.shape[0] // 10000)   # 60 Hz scatter is too dense raw
    for i, (cname, cvals, cmap) in enumerate(color_specs):
        if plot_2d:
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            sc = ax.scatter(emb[::stride, 0], emb[::stride, 1],
                            c=cvals[::stride], cmap=cmap, s=2, alpha=0.5)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.set_aspect('equal', adjustable='datalim')
        else:
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
            sc = ax.scatter(emb[::stride, 0], emb[::stride, 1], emb[::stride, 2],
                            c=cvals[::stride], cmap=cmap, s=2, alpha=0.5)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(f'color: {cname}', fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f'{dset_tag}\n{suptitle}', fontsize=11)
    fig.tight_layout()


# Iterate over cebra_models — for each, plot one figure per coloring scheme
# matching that model's supervision (agg-supervised → agg-channel coloring;
# pix-supervised → pix-PC coloring; models supervised by both get both
# figures). Replaces the hardcoded two-call setup.
#
# Old hardcoded calls kept commented:
# _plot_emb_scatter(cebra_agg, agg_color_specs, agg_rows,
#                   f'CEBRA-agg embedding  |  color: agg inputs  (side={color_tag!r})')
# _plot_emb_scatter(cebra_pix, pix_color_specs, pix_rows,
#                   f'CEBRA-pix embedding  |  color: top {n_pix_plot} pix-PCs')

for _cname, _m in cebra_models.items():
    _blocks = _m.get('supervision_blocks', [])
    if 'agg' in _blocks:
        _plot_emb_scatter(_m, agg_color_specs, agg_rows,
                          f'CEBRA-{_cname} embedding  |  color: agg inputs  '
                          f'(supervised by {"+".join(_blocks)}, side={color_tag!r})')
    if 'pix' in _blocks:
        _plot_emb_scatter(_m, pix_color_specs, pix_rows,
                          f'CEBRA-{_cname} embedding  |  color: top {n_pix_plot} pix-PCs  '
                          f'(supervised by {"+".join(_blocks)})')
    for _vb in visual_color_blocks:
        if _vb in _blocks and visual_color_specs[_vb]:
            _plot_emb_scatter(_m, visual_color_specs[_vb], grid_rows,
                              f'CEBRA-{_cname} embedding  |  color: top '
                              f'{len(visual_color_specs[_vb])} {_vb} comps  '
                              f'(supervised by {"+".join(_blocks)})')


#%% Decode behavior from CEBRA embedding — time-blocked KNN cross-validation
# =============================================================================
# SECTION 4 · DECODING · core — bar A/B, PCA sweep (R²+AUC), shuffled null
# =============================================================================
# The right A/B test: for each behavior feature, train a KNN regressor on the
# embedding and predict the feature on held-out time blocks. Higher R² ⇒ the
# CEBRA embedding (and thus the neural population, given CEBRA's supervision)
# represents that feature more.
#
# Time-blocked CV (contiguous folds) avoids the temporal-autocorrelation
# leakage that random-split CV produces — adjacent imaging frames are highly
# correlated, so random-split scores are inflated.
#
# Compare three columns:
#   (1) PCA-on-neural baseline (out_dim=3) — unsupervised reference.
#   (2) CEBRA-agg embedding.
#   (3) CEBRA-pix embedding.
# Feature being decoded is the SAME across columns. If CEBRA-agg wins on
# decoding agg features but loses on pix-PCA features, that's the expected
# pattern. If both CEBRA models lose to PCA baseline, supervision isn't
# helping (signal too weak or hyperparameters off).
#
# CAVEAT — in-sample embeddings: the CEBRA embeddings are fit on ALL frames
# WITH the behavior labels (build cell), so the blocked CV protects only the
# decoder, not the embedding. CEBRA columns are therefore optimistically biased
# and NOT directly comparable to the unsupervised PCA-3 baseline (which never
# saw labels). Read CEBRA-vs-CEBRA differences, not CEBRA-vs-PCA magnitude. A
# leak-free version needs per-fold CEBRA refit (expensive; not implemented).

# Decoding R² now comes from the unified f_blocked_cv_r2 (imported from
# f_cebra_helpers, 2026-06-03) — same implementation as the PCA sweep + the
# decoding-vs-n_pcs cell, so the three can't drift apart. The old inline KNN-
# only version (no embargo / no standardize) is gone; the knobs below expose
# the new options for this cell.
#
# Old inline signature (for reference): f_blocked_cv_r2(emb, target, n_folds=5,
#   k_nn=15, multioutput='uniform_average') -> (mean, std), KNN only.

# ── Decoder / CV knobs ──────────────────────────────────────────────────────
bar_decoder        = 'ridge'   # 'knn' | 'ridge' | 'ridgecv' — KNN matches the
                             # original A/B comparison (local nonlinearity).
bar_k_nn           = 15
bar_standardize    = True    # z-score embedding cols (train-fold stats) — stops
                             # one embedding axis from dominating KNN distance.
                             # (New default; the old cell did not standardize.)
bar_embargo_sec    = 0.0     # purge gap (s) each side of the test block.
bar_detrend_sigma_sec = None # GLOBAL high-pass σ (s) for every target column;
                             # None = off. Use LARGE σ (~20–60 s) — behavior is
                             # slow. See #8/#10. Used only when bar_detrend_blocks
                             # is None.
bar_detrend_blocks = None    # block-selective high-pass; overrides the global σ
                             # above when set. How to write it:
                             #   None          → global σ for ALL blocks (or off)
                             #   {'pix': 30.0} → pix at σ=30 s, agg left raw
                             #   {'agg': 60.0, 'pix': 30.0} → per-block σ
                             #   ['pix']       → pix at bar_detrend_sigma_sec, agg raw
# ─────────────────────────────────────────────────────────────────────────────

# Imaging fs for the sec→frame conversions (frame_t set in the build cell).
_fs_bar = f_imaging_fs(frame_t)
bar_embargo_fr     = int(round(bar_embargo_sec * _fs_bar))

# PCA-3 baseline on neural data — same out_dim as CEBRA.
pca_neu = f_pca_prefix(Y_neu, 3)

# Which feature blocks (from built_blocks registry) to decode. Each entry
# in target_blocks_bar gets its own bar group, with bars colored by embedding.
# Examples:
#   ['agg', 'pix']                — original A/B comparison
#   ['agg', 'motion']             — position vs velocity from same embeddings
#   ['agg', 'motion', 'pix']      — three target groups, three bar clusters
target_blocks_bar = ['agg', 'pix', 'grid', 'flow']

# How to aggregate per-column R² when a block has multiple output dims.
# 'uniform_average'   — current default; each column counts equally. Drags
#                       low when binary / low-variance columns fail to decode.
# 'variance_weighted' — joint R² (task #26); weights each column by its
#                       variance. More honest when high- and low-variance
#                       targets are mixed (e.g. lat/vert vs presence).
bar_multioutput = 'uniform_average'
bar_skip_presence = True       # drop binary 'pres_*' columns (degenerate under
                               # the R² metric); score presence via the AUC sweep.

# Old hardcoded list kept commented for reference:
# feats_to_decode = [
#     ('agg (8d)',     X_agg),
#     ('pix-PCA (%dd)' % n_pix_pca, X_pix),
# ]

feats_to_decode = []
for grp in target_blocks_bar:
    if grp not in built_blocks:
        raise ValueError(f'block {grp!r} not built; available: {list(built_blocks)}')
    b = built_blocks[grp]
    Xb = b['X']
    if bar_skip_presence:
        keep = [i for i, nm in enumerate(b['names']) if not nm.startswith('pres')]
        Xb = Xb[:, keep]
    Xb = f_apply_block_detrend(Xb, grp, bar_detrend_blocks,   # #8/#10 high-pass
                               bar_detrend_sigma_sec, _fs_bar)
    feats_to_decode.append((f'{grp} ({Xb.shape[1]}d)', Xb))

# Embeddings to compare — PCA-3 baseline + every model in cebra_models.
# Old hardcoded list kept commented:
# embeddings = [
#     ('PCA-3 (unsup)', pca_neu),
#     ('CEBRA-agg',     cebra_agg['embedding']),
#     ('CEBRA-pix',     cebra_pix['embedding']),
# ]
embeddings = [('PCA-3 (unsup)', pca_neu)]
for _cname, _m in cebra_models.items():
    embeddings.append((f'CEBRA-{_cname}', _m['embedding']))

r2_table = np.zeros((len(feats_to_decode), len(embeddings)))
r2_std   = np.zeros_like(r2_table)
for i, (fname, target) in enumerate(feats_to_decode):
    for j, (ename, emb) in enumerate(embeddings):
        m, s = f_blocked_cv_r2(emb, target, n_folds=5, decoder=bar_decoder,
                               k_nn=bar_k_nn, embargo=bar_embargo_fr,
                               standardize=bar_standardize,
                               multioutput=bar_multioutput, return_std=True)
        r2_table[i, j] = m
        r2_std[i, j]   = s
        print(f'  decode {fname:18s}  from  {ename:18s}  R² = {m:+.3f} ± {s:.3f}')

# Grouped bar plot — feature groups × embeddings.
f_plot_decode_bars(
    [n for n, _ in feats_to_decode], [n for n, _ in embeddings], r2_table, r2_std,
    title=f'{dset_tag}\ndecoding behavior from 3D embeddings '
          f'({bar_decoder}, k={bar_k_nn}, standardize={bar_standardize})')


#%% PCA decoding sweep — per-feature R² as a function of n_neural_PCs
# Unsupervised baseline only (no CEBRA). The previous bar plot mashed 8/20
# target columns into a single uniform_average R² per (algo, target-group),
# which goes negative whenever a few low-variance / nonstationary columns
# fail to decode — regardless of whether the rest worked. This sweep
# decomposes that:
#   - vary the number of neural PCs used as the KNN input embedding (1..50)
#   - decode each behavior feature SEPARATELY (per-column R²)
#   - same blocked 5-fold CV harness, KNN(k=15, weights='distance')
#
# Reads as:
#   - curve rising and plateauing → feature is decodable from neural PCs,
#     plateau height = ceiling for that feature, plateau onset = #dims
#     it actually needs.
#   - curve flat at ~0 / negative → feature isn't represented in neural
#     variance (or is, but only at scales the embedding can't capture).
#   - peak then dropping → curse of dimensionality at high n_pcs with k=15
#     too few neighbors; tune k if it keeps happening.
#
# Targets covered: each agg feature column (n_per_side*2 = 8 or 10) and
# each pix-PCA component (n_pix_pca = 20). Plotted as (feature × n_pcs)
# heatmap plus per-group line plots.

if 1:
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    
    # Decoder choice. KNN = local nonlinearity, what CEBRA paper uses.
    # Ridge = strict linear ceiling — tests whether the feature is a linear
    # function of the embedding axes. More decoders to come.
    decoder     = 'knn'   # 'knn' | 'ridge' | 'ridgecv'
    k_nn        = 15        # used when decoder == 'knn'
    ridge_alpha = 1.0       # used when decoder == 'ridge'
    ridgecv_alphas = None   # used when decoder == 'ridgecv'; None → logspace(-2,4,13).
                            # RidgeCV tunes α by leave-one-out CV inside each train
                            # fold — a fairer linear ceiling than a fixed α.

    # ── Regression-quality knobs (added 2026-06-03) ─────────────────────────
    # All default to prior behavior EXCEPT standardize_emb (now on — it makes
    # the n_pcs sweep meaningful for both ridge and knn). Toggle the rest to
    # attack #8 (adversarial nonstationarity) and embedding-scale problems.
    # These are read by f_cv_r2_single below AND by the shuffle-null cell, so
    # the real and null paths stay in lock-step.
    standardize_emb   = True    # z-score embedding columns (TRAIN stats only)
                                # before decoding. Stops high-variance PC1 from
                                # dominating the KNN distance / Ridge penalty;
                                # makes "R² vs n_pcs" reflect info, not scale.
    embargo_sec       = 0.0     # gap (s) purged on each side of the test block
                                # (removed from train) so the decoder can't learn
                                # from frames autocorrelated with the test fold.
                                # >0 cuts the leakage channel #8 rides on.
    detrend_sigma_sec = None    # GLOBAL high-pass σ (s): subtract a NaN-aware
                                # Gaussian-smoothed copy of each TARGET before CV
                                # to remove session-spanning drift. Task #10.
                                # None = off. NB behavior here is SLOW (objects
                                # approach over seconds), so σ must be LARGE
                                # (~20–60 s) to strip only drift, not real signal.
                                # Used only when detrend_blocks is None.
    detrend_blocks    = None   # BLOCK-SELECTIVE detrend (overrides the global
                                # σ above when set). Slow agg/position targets
                                # carry real signal in their slow component, so
                                # leave them raw; high-pass only the drift-prone
                                # pix block. Blocks absent from a dict are raw.
                                # How to write it:
                                #   None                  → use detrend_sigma_sec
                                #                           for ALL blocks (or off)
                                #   {'pix': 30.0}         → pix at σ=30 s, agg raw
                                #   {'agg': 60.0,
                                #    'pix': 30.0}          → different σ per block
                                #   {'pix': None}         → pix off (same as raw)
                                #   ['pix']               → pix at the global
                                #                           detrend_sigma_sec, agg raw
    detrend_embedding = False   # also high-pass the neural embedding columns
                                # (#11). Needs detrend_sigma_sec set; only applies
                                # to the precomputed-PCA path (ignored when
                                # pca_within_fold=True).
    pca_within_fold   = False   # fit PCA on TRAIN rows only inside each fold (no
                                # test-frame leakage into the embedding axes).
                                # Slower (refits per fold × n_pcs × feature).
    # ─────────────────────────────────────────────────────────────────────────

    # Imaging sample rate from frame_times, for the sec→frame conversions above.
    fs_img = f_imaging_fs(frame_t)
    embargo_fr       = int(round(embargo_sec * fs_img))
    detrend_sigma_fr = (detrend_sigma_sec * fs_img) if detrend_sigma_sec else None
    # f_detrend_col is imported from f_feature_helpers (unified 2026-06-03;
    # was an inline def here).

    n_pcs_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    n_folds = 5
    sweep_target_blocks = ['agg', 'pix', 'motion', 'pix_mot', 'beh', 'self_mot', 'grid', 'flow']    # which built_blocks groups to sweep over
                                            # All possible block names (must have been
                                            # built in the build cell — see its build_* knobs):
                                            #   'agg'      — per-side object POSITION
                                            #                (pres/lat/vert/dist|ang_size). Always built.
                                            #   'pix'      — pixel-movie PCA components. Always built.
                                            #   'motion'   — per-side OBJECT screen velocity
                                            #                (d_lat/d_vert/d_dist[/d_ang_size],
                                            #                motion_mag, dir sin/cos). build_motion=True.
                                            #   'self_mot' — MOUSE egomotion: running speed + yaw rate.
                                            #                build_self_motion=True.
                                            #   'pix_mot'  — d/dt of each pix-PC trace (pixel-side
                                            #                "pca motion"). build_pix_motion=True.
                                            #   'beh'      — lick + reward events. build_lick_reward=True.
                                            #   'per_obj'  — per-object identity slots. build_per_obj=True.
                                            #   'grid'     — visual retinotopic grid pooling (Tier 1):
                                            #                per-cell occupancy/mean/edge. build_visual.
                                            #   'flow'     — visual optical-flow grid (Tier 2): per-cell
                                            #                speed/direction/divergence. build_visual.
                                            #                ('gabor' = Tier 3, planned — TODO #47 /
                                            #                PLAN_monitor_features.md.)
                                            # (presence is auto-split off to the ROC-AUC path.)
                                            # examples: ['agg'], ['motion'], ['self_mot'],
                                            # ['agg', 'pix'], ['agg', 'motion', 'self_mot', 'pix_mot']

    # Targets: one entry per (block, column) for each block in sweep_target_blocks.
    # Uses the built_blocks registry so adding a new feature block (motion,
    # per_obj, pix_mot, ...) just means appending its name to sweep_target_blocks.
    #
    # Old per-block construction kept commented:
    # pix_feat_names = [f'pix_PC{i+1}' for i in range(n_pix_pca)]
    # targets_all = []
    # for i, name in enumerate(agg_feat_names):
    #     targets_all.append(('agg', name, X_agg[:, i]))
    # for i, name in enumerate(pix_feat_names):
    #     targets_all.append(('pix', name, X_pix[:, i]))

    # Binary (0/1) targets — e.g. presence_L/R — are split off into targets_bin
    # and scored with ROC-AUC (chance 0.5) instead of R². R² is misleading for a
    # base-rate-skewed binary target: SS_tot = n·p(1-p) is tiny, so even a good
    # predictor lands near 0 / deeply negative under blocked-CV mean drift. AUC
    # is base-rate robust and has a clean null. Binary targets are NOT detrended
    # (high-passing a 0/1 signal is meaningless). build_target_columns does the
    # split + block-selective high-pass. See notes 2026-06-04.
    targets_all, targets_bin = build_target_columns(
        built_blocks, sweep_target_blocks, detrend_blocks=detrend_blocks,
        detrend_sigma_sec=detrend_sigma_sec, fs=fs_img, split_binary=True)

    # Fit PCA once at max k; prefixes give exact lower-k embeddings.
    # (When pca_within_fold=True the scoring refits PCA per fold; this full-data
    # fit is then only used by the heatmap / legacy consumers.)
    max_pcs = max(n_pcs_sweep)
    pca_neu_full = f_pca_prefix(Y_neu, max_pcs)
    if detrend_embedding and detrend_sigma_fr and not pca_within_fold:
        pca_neu_full = np.column_stack([
            f_detrend_col(pca_neu_full[:, j], detrend_sigma_fr)
            for j in range(pca_neu_full.shape[1])])   # task #11
    
    # f_cv_r2_single is now a thin wrapper over the unified f_blocked_cv_r2
    # (f_cebra_helpers) so the sweep, the shuffle-null cell, and the detrend
    # grid all share ONE blocked-CV implementation (no drift). The shuffle +
    # grid cells call f_cv_r2_single unchanged. (Inline blocked-CV loop removed
    # 2026-06-03; logic preserved verbatim inside f_blocked_cv_r2.)
    def f_cv_r2_single(emb, y, n_folds=5, embargo=0, standardize=True,
                       Y_raw=None, n_pcs=None, task='regression'):
        return f_blocked_cv_r2(emb, y, n_folds=n_folds, embargo=embargo,
                               standardize=standardize, decoder=decoder,
                               ridge_alpha=ridge_alpha, k_nn=k_nn,
                               ridgecv_alphas=ridgecv_alphas,
                               Y_raw=Y_raw, n_pcs=n_pcs, task=task)

    r2_mat = np.full((len(targets_all), len(n_pcs_sweep)), np.nan)
    t0 = time.perf_counter()
    for fi, (grp, name, y) in enumerate(targets_all):
        for ki, n_pcs in enumerate(n_pcs_sweep):
            if pca_within_fold:
                r2_mat[fi, ki] = f_cv_r2_single(
                    None, y, n_folds=n_folds, embargo=embargo_fr,
                    standardize=standardize_emb, Y_raw=Y_neu, n_pcs=n_pcs)
            else:
                r2_mat[fi, ki] = f_cv_r2_single(
                    pca_neu_full[:, :n_pcs], y, n_folds=n_folds,
                    embargo=embargo_fr, standardize=standardize_emb)
        print(f'  {grp:3s}  {name:14s}  '
              + '  '.join(f'k={k:2d}:{r:+.2f}' for k, r in zip(n_pcs_sweep, r2_mat[fi])))
    print(f'PCA decoding sweep ({decoder}): {time.perf_counter()-t0:.1f}s')

    # Binary targets — same blocked-CV sweep but classification (ROC-AUC, chance
    # 0.5). auc_mat mirrors r2_mat; the shuffle-null cell reads both.
    auc_mat = np.full((len(targets_bin), len(n_pcs_sweep)), np.nan)
    if targets_bin:
        t0b = time.perf_counter()
        for fi, (grp, name, y) in enumerate(targets_bin):
            for ki, n_pcs in enumerate(n_pcs_sweep):
                if pca_within_fold:
                    auc_mat[fi, ki] = f_cv_r2_single(
                        None, y, n_folds=n_folds, embargo=embargo_fr,
                        standardize=standardize_emb, Y_raw=Y_neu, n_pcs=n_pcs,
                        task='classification')
                else:
                    auc_mat[fi, ki] = f_cv_r2_single(
                        pca_neu_full[:, :n_pcs], y, n_folds=n_folds,
                        embargo=embargo_fr, standardize=standardize_emb,
                        task='classification')
            print(f'  {grp:3s}  {name:14s}  [AUC]  '
                  + '  '.join(f'k={k:2d}:{a:.2f}' for k, a in zip(n_pcs_sweep, auc_mat[fi])))
        print(f'binary AUC sweep ({decoder}): {time.perf_counter()-t0b:.1f}s')

    # Group → row indices mapping (preserves first-appearance order from targets_all).
    # Replaces the old hardcoded n_agg / agg-vs-pix split — now works for any
    # combination of blocks in sweep_target_blocks.
    group_rows = {}
    for ri, (grp, _, _) in enumerate(targets_all):
        group_rows.setdefault(grp, []).append(ri)
    sweep_groups_present = list(group_rows.keys())

    # Legacy variables (used by the shuffle cell). Present only when 'agg'/'pix'
    # are in the sweep; otherwise NaN-sized arrays.
    n_agg = len(group_rows.get('agg', []))
    r2_agg = r2_mat[group_rows.get('agg', [])]
    r2_pix = r2_mat[group_rows.get('pix', [])]

    # Heatmap — single overview (clipped ±0.5; dividers between feature groups).
    f_plot_feature_heatmap(
        r2_mat, n_pcs_sweep, [f'{g}:{n}' for g, n, _ in targets_all],
        group_sizes=[len(group_rows[g]) for g in sweep_groups_present],
        title=f'{dset_tag}\nper-feature blocked-CV R² (PCA → {decoder} decoding)',
        cbar_label='R²  (clipped at ±0.5)')

    # Per-feature R²-vs-n_pcs line plots, one panel per group.
    f_plot_sweep_lines(n_pcs_sweep, r2_mat, targets_all, group_rows,
                       sweep_groups_present, dset_tag=dset_tag)

    # Aggregate decoding-performance-vs-n_pcs summary (2-axis: analog R² left,
    # binary AUC right) — see f_decoding.f_plot_decode_sweep_summary.
    f_plot_decode_sweep_summary(n_pcs_sweep, r2_mat, group_rows,
                                sweep_groups_present, targets_bin, auc_mat,
                                dset_tag=dset_tag, decoder=decoder)


#%% Shuffled-control null distribution for the PCA decoding sweep
# Reruns the same decoding sweep N_shuffles times on data where the
# neural↔behavior relationship is broken. Two ways to break it:
#
#   shuffle_mode = 'circshift_target'
#       Shift every behavior target (agg + pix-PCA) by one random offset
#       per shuffle. Preserves ALL neural structure (cross-cell correlations,
#       autocorrelation) and ALL behavior structure — only the time
#       alignment between them is destroyed. Right null for "does the
#       embedding contain behavior information?". Fast: PCA on neural is
#       reused; only the target arrays change.
#
#   shuffle_mode = 'circshift_neural_percell'
#       Per-cell independent circular shift of the neural matrix
#       (f_circshift_rates). Breaks BOTH cross-cell timing and behavior
#       alignment — a much stronger null. Tests whether decoding works
#       beyond what per-cell rate / autocorrelation alone could give.
#       Slower: PCA must be re-fit each shuffle.
#
# Interpretation:
#   real  > null + null 95% band  →  embedding really contains behavior info
#   real ≈ null                   →  no behavioral signal in this embedding
#   null itself > 0               →  blocked-CV bias from autocorrelation
#                                    structure that survives the shuffle
#
# Prereqs from the previous cell: r2_mat, pca_neu_full, max_pcs, targets_all,
# group_rows, sweep_groups_present, n_pcs_sweep, n_folds, Y_neu,
# f_cv_r2_single, decoder. (Legacy n_agg/r2_agg/r2_pix/
# agg_feat_names/pix_feat_names also published by the previous cell for
# backward compat but not used by this cell anymore.)

n_shuffles       = 2
shuffle_mode     = 'circshift_neural_percell'   # 'circshift_target' | 'circshift_neural_percell'
min_shift_frames = 0                    # 0 = uniform null (unbiased); nonzero biases it
shuffle_seed     = 42
independent_target_shifts = False       # circshift_target only: independent
                                        # shift per target column (decorrelates
                                        # the per-feature null) instead of one
                                        # shared shift across all targets.

# NB: the CV preprocessing (embargo_fr, standardize_emb, pca_within_fold)
# is inherited from the sweep cell so the null runs through the IDENTICAL
# decoding pipeline as the real data — the only thing broken is the
# neural↔behavior alignment.

T_img = Y_neu.shape[0]
rng = np.random.default_rng(shuffle_seed)
r2_mat_null  = np.full((n_shuffles, len(targets_all), len(n_pcs_sweep)), np.nan)
auc_mat_null = np.full((n_shuffles, len(targets_bin), len(n_pcs_sweep)), np.nan)

print(f'shuffled-control sweep ({shuffle_mode}, n={n_shuffles}, decoder={decoder})...')
t_null = time.perf_counter()
for ns in range(n_shuffles):
    Y_for_null = Y_neu
    if shuffle_mode == 'circshift_target':
        if independent_target_shifts:
            targets_shuf = [
                (g, name, np.roll(y, int(rng.integers(min_shift_frames,
                                                      T_img - min_shift_frames))))
                for g, name, y in targets_all]
            targets_bin_shuf = [
                (g, name, np.roll(y, int(rng.integers(min_shift_frames,
                                                      T_img - min_shift_frames))))
                for g, name, y in targets_bin]
        else:
            shift = int(rng.integers(min_shift_frames, T_img - min_shift_frames))
            targets_shuf = [(g, name, np.roll(y, shift)) for g, name, y in targets_all]
            targets_bin_shuf = [(g, name, np.roll(y, shift)) for g, name, y in targets_bin]
        pca_for_null = None if pca_within_fold else pca_neu_full
    elif shuffle_mode == 'circshift_neural_percell':
        # f_circshift_rates expects (n_cells, T); Y_neu is (T, n_cells).
        # Seeded rng (was the global RNG, ignoring shuffle_seed); uniform null
        # (min_shift=0 — a nonzero min biases the null toward significance).
        Y_shuf = f_shuffle_neural(Y_neu, rng, min_shift=min_shift_frames)
        Y_for_null = Y_shuf
        pca_for_null = None if pca_within_fold else f_pca_prefix(Y_shuf, max_pcs)
        # Match the real path's embedding high-pass (#11) so null and real run
        # the SAME preprocessing — else the null band is invalid.
        if detrend_embedding and detrend_sigma_fr and not pca_within_fold:
            pca_for_null = np.column_stack([
                f_detrend_col(pca_for_null[:, j], detrend_sigma_fr)
                for j in range(pca_for_null.shape[1])])
        targets_shuf = targets_all
        targets_bin_shuf = targets_bin
    else:
        raise ValueError(f'unknown shuffle_mode: {shuffle_mode!r}')

    for fi, (grp, name, y) in enumerate(targets_shuf):
        for ki, n_pcs in enumerate(n_pcs_sweep):
            if pca_within_fold:
                r2_mat_null[ns, fi, ki] = f_cv_r2_single(
                    None, y, n_folds=n_folds, embargo=embargo_fr,
                    standardize=standardize_emb, Y_raw=Y_for_null, n_pcs=n_pcs)
            else:
                r2_mat_null[ns, fi, ki] = f_cv_r2_single(
                    pca_for_null[:, :n_pcs], y, n_folds=n_folds,
                    embargo=embargo_fr, standardize=standardize_emb)
    # Binary targets — same null, classification (ROC-AUC). Null AUC ~ 0.5.
    for fi, (grp, name, y) in enumerate(targets_bin_shuf):
        for ki, n_pcs in enumerate(n_pcs_sweep):
            if pca_within_fold:
                auc_mat_null[ns, fi, ki] = f_cv_r2_single(
                    None, y, n_folds=n_folds, embargo=embargo_fr,
                    standardize=standardize_emb, Y_raw=Y_for_null, n_pcs=n_pcs,
                    task='classification')
            else:
                auc_mat_null[ns, fi, ki] = f_cv_r2_single(
                    pca_for_null[:, :n_pcs], y, n_folds=n_folds,
                    embargo=embargo_fr, standardize=standardize_emb,
                    task='classification')
    print(f'  shuffle {ns+1:2d}/{n_shuffles}  cumulative {time.perf_counter()-t_null:.1f}s')
print(f'shuffled-control total: {time.perf_counter()-t_null:.1f}s')

# Real vs null — group-mean overlay. One color per group; iterates over
# sweep_groups_present so adding new feature groups doesn't require
# touching this plot.
# Solid colored line = real mean across features in group.
# Dotted same color  = null mean across features (averaged over shuffles).
# Shaded band        = 2.5–97.5th percentile across shuffles of the
#                      per-shuffle group-mean R² (i.e. 95% empirical CI
#                      for the null group mean).

# Combined 2-axis real-vs-null figure (analog R² left, binary AUC right) —
# see f_decoding.f_plot_decode_real_vs_null.
f_plot_decode_real_vs_null(n_pcs_sweep, r2_mat, r2_mat_null, group_rows,
                           sweep_groups_present, targets_bin, auc_mat, auc_mat_null,
                           dset_tag=dset_tag, decoder=decoder,
                           shuffle_mode=shuffle_mode, n_shuffles=n_shuffles)

# Per-feature signal-above-chance heatmap: real − null_mean.
# Reads as "how much R² each feature gains over chance, at each n_pcs."
null_mean_per_feat = r2_mat_null.mean(axis=0)        # (n_features, n_pcs)
delta = r2_mat - null_mean_per_feat
f_plot_feature_heatmap(
    delta, n_pcs_sweep, [f'{g}:{n}' for g, n, _ in targets_all],
    group_sizes=[len(group_rows[g]) for g in sweep_groups_present],
    title=f'{dset_tag}\nR²(real) − R²(null mean)  [{shuffle_mode}, {decoder}]',
    cbar_label='Δ R²  (clipped at ±0.5)')

# Per-feature line plots with shuffled-control overlay.
# Two variants:
#   Top row    — raw R² lines per feature (as before), with a grey shaded
#                band showing the pooled null 95% interval across all
#                features × shuffles in that group. A real feature whose
#                line sits above the band is above chance for the
#                "random feature under null" distribution.
#   Bottom row — Δ R² (real − null_mean) per feature. Zero = chance after
#                removing blocked-CV / autocorr bias that survives the
#                shuffle. The cleanest "is this feature really decodable"
#                view.

# Per-feature 2×N grid (raw R² + null band; ΔR²) — one column per group.
f_plot_perfeature_null_grid(n_pcs_sweep, r2_mat, r2_mat_null, group_rows,
                            sweep_groups_present, targets_all, dset_tag=dset_tag,
                            decoder=decoder, shuffle_mode=shuffle_mode,
                            n_shuffles=n_shuffles)


#%% Detrend × embargo sweep — does removing slow drift rescue pix-PC1/2? (#8/#10)
# =============================================================================
# SECTION 5 · DECODING · diagnostics — detrend×embargo grid, per-fold, OOF, raster
# =============================================================================
# Reruns the per-feature blocked-CV decoding across a GRID of high-pass
# strengths (detrend σ, including OFF) × embargo gaps, with a fast
# circshift_target null at each grid point. This is the direct test of the
# adversarial-nonstationarity hypothesis (#8): slow targets like pix-PC1/PC2
# start with real << null (decoder actively misled by session-spanning drift)
# and should climb toward / above the null band as the high-pass strengthens.
#
# Detrend is a swept OPTION here, not forced on — detrend_grid includes None
# (off) so the leftmost point of every curve is the current default behavior.
#
# Prereqs: run the "PCA decoding sweep" cell first. This cell reuses
# f_cv_r2_single, f_detrend_col, fs_img, pca_neu_full, Y_neu,
# built_blocks, n_folds, standardize_emb and the `decoder` choice from it.

if 1:
    # ── Grid knobs ──────────────────────────────────────────────────────────
    # Behavior here is SLOW (objects approach over seconds), so the σ values are
    # LARGE — a small σ (~4 s) would high-pass real position signal out along
    # with the drift. Pick σ ≳ the timescale of real behavioral variation so
    # only minute-scale drift is removed.
    detrend_grid     = [None, 15.0, 30.0, 60.0] # high-pass σ in seconds (None = off)
    embargo_grid     = [0.0, 10.0]             # purge gap (s) each side of test block
    grid_blocks      = ['agg', 'pix']          # which feature blocks to score
    grid_detrend_blocks = ['pix']              # which of grid_blocks the swept σ is
                                                # APPLIED to (a LIST of block names,
                                                # NOT a dict — the σ comes from
                                                # detrend_grid). Others scored raw at
                                                # every σ. How to write it:
                                                #   ['pix']        → only pix swept,
                                                #                    agg flat reference
                                                #   ['agg', 'pix'] → both swept
                                                #   None           → detrend all blocks
    n_pcs_grid       = 16                       # # neural PCs used as the embedding
    grid_n_shuf      = 10                       # circshift_target null draws / cell
    grid_min_shift   = 0                        # 0 = uniform null (unbiased)
    grid_seed        = 42
    focus_feats      = ['pix_PC1', 'pix_PC2']   # features that get the real-vs-null
                                                # line plot (all go in the heatmap)
    grid_within_fold = False                    # within-fold PCA in the grid (slow)
    # ─────────────────────────────────────────────────────────────────────────

    # Raw (undetrended) target columns for the chosen blocks — detrending is
    # applied per grid cell below so each σ gets its own high-pass. (fs omitted →
    # build_target_columns returns raw columns; presence included, like before.)
    grid_targets_raw = build_target_columns(built_blocks, grid_blocks)
    grid_names       = [n for _, n, _ in grid_targets_raw]
    grid_feat_labels = [f'{g}:{n}' for g, n, _ in grid_targets_raw]
    n_gf             = len(grid_targets_raw)

    combos       = [(ds, eb) for ds in detrend_grid for eb in embargo_grid]
    combo_labels = [f'σ={"off" if ds is None else ds:>4}|emb={eb:g}s' for ds, eb in combos]

    grid_rng = np.random.default_rng(grid_seed)
    T_grid   = Y_neu.shape[0]

    grid_real = np.full((n_gf, len(combos)), np.nan)   # real R²
    grid_null = np.full((n_gf, len(combos)), np.nan)   # null mean R²
    grid_p    = np.full((n_gf, len(combos)), np.nan)   # one-sided empirical p
    # focus-feature per-shuffle null draws, for the line-plot band:
    focus_null_all = {nm: np.full((len(combos), grid_n_shuf), np.nan) for nm in focus_feats}

    def _grid_r2(y, eb_fr):
        # one blocked-CV R² at n_pcs_grid, honoring within-fold + standardize.
        if grid_within_fold:
            return f_cv_r2_single(None, y, n_folds=n_folds, embargo=eb_fr,
                                  standardize=standardize_emb, Y_raw=Y_neu, n_pcs=n_pcs_grid)
        return f_cv_r2_single(pca_neu_full[:, :n_pcs_grid], y, n_folds=n_folds,
                              embargo=eb_fr, standardize=standardize_emb)

    _t0 = time.perf_counter()
    for cj, (ds, eb) in enumerate(combos):
        ds_fr = (ds * fs_img) if ds else None
        eb_fr = int(round(eb * fs_img))
        # Apply the swept σ only to blocks in grid_detrend_blocks (None = all);
        # other blocks stay raw at every σ as a fixed reference.
        ys = []
        for grp, _, yraw in grid_targets_raw:
            _apply = (ds_fr is not None) and (grid_detrend_blocks is None
                                              or grp in grid_detrend_blocks)
            ys.append(f_detrend_col(yraw, ds_fr) if _apply else yraw)

        # real
        for fi, y in enumerate(ys):
            grid_real[fi, cj] = _grid_r2(y, eb_fr)

        # null — circshift_target (PCA reused; only targets rolled). One shift
        # per draw, shared across features so cross-target structure is kept.
        null_draws = np.full((grid_n_shuf, n_gf), np.nan)
        for ns in range(grid_n_shuf):
            shift = int(grid_rng.integers(grid_min_shift, T_grid - grid_min_shift))
            for fi, y in enumerate(ys):
                null_draws[ns, fi] = _grid_r2(np.roll(y, shift), eb_fr)
        grid_null[:, cj] = np.nanmean(null_draws, axis=0)

        # one-sided empirical p: P(null >= real), add-one smoothed.
        for fi in range(n_gf):
            nd = null_draws[:, fi][~np.isnan(null_draws[:, fi])]
            if nd.size and not np.isnan(grid_real[fi, cj]):
                grid_p[fi, cj] = (1 + np.sum(nd >= grid_real[fi, cj])) / (nd.size + 1)
        for nm in focus_feats:
            if nm in grid_names:
                focus_null_all[nm][cj] = null_draws[:, grid_names.index(nm)]

        print(f'  combo {cj+1:2d}/{len(combos)}  {combo_labels[cj]}  '
              f'cumulative {time.perf_counter()-_t0:.1f}s')
    print(f'detrend×embargo grid ({decoder}, n_pcs={n_pcs_grid}): '
          f'{time.perf_counter()-_t0:.1f}s')

    grid_delta = grid_real - grid_null

    # Δ R² heatmap over the grid (with p<0.05 stars) + focus-feature rescue
    # line plots — see f_decoding.f_plot_grid_delta_heatmap / _focus_lines.
    f_plot_grid_delta_heatmap(grid_delta, grid_p, combo_labels, grid_feat_labels,
                              dset_tag=dset_tag, n_pcs_grid=n_pcs_grid, decoder=decoder)
    f_plot_grid_focus_lines(detrend_grid, embargo_grid, combos, grid_real,
                            focus_null_all, focus_feats, grid_names,
                            dset_tag=dset_tag, n_pcs_grid=n_pcs_grid)


#%% Per-fold decoding diagnostic for ONE feature (#9 — why R² goes negative)
# Pins down WHY a feature (e.g. ang_size) decodes negative / worsens with n_pcs.
# For one chosen target column it reports, per CV fold: R², n valid train/test
# frames, the out-of-fold predicted-vs-real trace, and the target distribution.
#
# Reading guide (which mechanism is it?):
#   - ONE fold hugely negative, others ~0      → nonstationarity: a rare spike /
#                                                drift regime sits in that test
#                                                block. (Slow/heavy-tail story.)
#   - ALL folds uniformly more negative as k↑  → overfitting capacity — cap n_pcs
#                                                at the sweep plateau.
#   - few valid frames in some folds / valid
#     mask clustered in time                   → presence-masking × blocked-CV
#                                                instability (NaN-drop issue #12).
#   - histogram dominated by a rare tail AND
#     Spearman ≫ R²                            → R² is being eaten by extremes;
#                                                transform the target or report a
#                                                rank metric for this channel.
#
# Prereqs: run the "PCA decoding sweep" cell first (uses pca_neu_full,
# built_blocks, Y_neu, n_folds, embargo_fr, standardize_emb, decoder,
# ridge_alpha, k_nn, est1, dset_tag).

if 1:
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr, skew

    diag_feat_match  = 'pix_PC1'        # substring; first matching column used
                                         # (e.g. 'ang_size', 'dist', 'lat', 'pix_PC1')
    diag_n_pcs_list  = [3, 8, 16, 32, 50]  # per-fold R² shown for these k
    diag_n_pcs_trace = 16                # n_pcs for the OOF prediction trace
    diag_insample_n_pcs = 50            # n_pcs for the in-sample (train) recon figure
    diag_n_folds     = n_folds

    # Locate the target column in built_blocks (first name containing the match).
    _hit = None
    for _g, _b in built_blocks.items():
        for _i, _nm in enumerate(_b['names']):
            if diag_feat_match in _nm:
                _hit = (_g, _nm, _b['X'][:, _i]); break
        if _hit:
            break
    if _hit is None:
        raise ValueError(f'no built_blocks column matches {diag_feat_match!r}')
    diag_grp, diag_name, y_diag = _hit

    valid = ~np.isnan(y_diag)
    T = pca_neu_full.shape[0]
    fold_size = T // diag_n_folds
    fold_bounds = [(f*fold_size, (f+1)*fold_size if f < diag_n_folds-1 else T)
                   for f in range(diag_n_folds)]

    # Per-fold R² across n_pcs + the OOF prediction trace — both via the unified
    # f_blocked_cv_r2(return_predictions=True), so the fold / embargo / standardize
    # logic matches the sweep exactly (the re-rolled _fit_fold loop is gone).
    def _diag_cv(npc):
        return f_blocked_cv_r2(pca_neu_full[:, :npc], y_diag, n_folds=diag_n_folds,
                               embargo=embargo_fr, standardize=standardize_emb,
                               decoder=decoder, ridge_alpha=ridge_alpha, k_nn=k_nn,
                               return_predictions=True)

    perfold       = np.full((len(diag_n_pcs_list), diag_n_folds), np.nan)
    perfold_train = np.full((len(diag_n_pcs_list), diag_n_folds), np.nan)
    nval_te = np.zeros(diag_n_folds, int)
    for ci, npc in enumerate(diag_n_pcs_list):
        res = _diag_cv(npc)
        perfold[ci]       = res['r2_folds']            # per-fold TEST R²
        perfold_train[ci] = res['r2_folds_train']      # per-fold TRAIN R² (in-sample)
        if res['fold'].size:                           # valid test points per fold
            nval_te = np.bincount(res['fold'], minlength=diag_n_folds)

    # OOF predictions at diag_n_pcs_trace (pooled across folds for the trace).
    res_tr = _diag_cv(diag_n_pcs_trace)
    oof = np.full(T, np.nan)
    oof[res_tr['idx']] = res_tr['y_pred']
    m = valid & ~np.isnan(oof)
    rho = spearmanr(y_diag[m], oof[m]).correlation if m.sum() > 5 else np.nan
    # Pooled OOF R² across all folds' predictions (≠ mean of per-fold R²).
    r2_all = r2_score(y_diag[m], oof[m]) if m.sum() > 5 else np.nan

    print(f'diagnosing {diag_grp}:{diag_name}')
    print(f'  valid frames: {valid.sum()}/{T} ({100*valid.mean():.1f}%)')
    print(f'  per-fold n valid (test): {nval_te.tolist()}')
    print(f'  target skew: {skew(y_diag[valid]):+.2f}   '
          f'max/std: {y_diag[valid].max()/y_diag[valid].std():.1f}')
    print(f'  @k={diag_n_pcs_trace}:  R²(OOF)={r2_all:+.3f}   Spearman={rho:+.3f}  '
          f'(Spearman ≫ R² ⇒ tail/extremes eating R²)')
    # Train vs test by k — the overfitting / nonstationarity readout. A big
    # train-test gap that WIDENS with k (e.g. the ang_size cliff at ~PC10) =
    # the embedding fits in-sample but the relationship doesn't generalize
    # across blocks (drift / #8). Low train too = feature not in the PCs.
    print('  TRAIN R² (fold mean) by k: '
          + ', '.join(f'k={k}:{np.nanmean(perfold_train[ci]):+.2f}'
                      for ci, k in enumerate(diag_n_pcs_list)))
    print('  TEST  R² (fold mean) by k: '
          + ', '.join(f'k={k}:{np.nanmean(perfold[ci]):+.2f}'
                      for ci, k in enumerate(diag_n_pcs_list)))

    t_img = t_img_plot
    fig, ax = plt.subplots(4, 1, figsize=(12, 10.5),
                           gridspec_kw={'height_ratios': [2, 2, 2, 1.4]})

    def _draw_perfold_hm(axh, mat, title, cmap, vmin, vmax, seq=False):
        im = axh.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='none')
        axh.set_yticks(range(len(diag_n_pcs_list)))
        axh.set_yticklabels([f'k={k}' for k in diag_n_pcs_list])
        axh.set_xticks(range(diag_n_folds))
        axh.set_xticklabels([f'fold{f}\nn={nval_te[f]}' for f in range(diag_n_folds)])
        for ci in range(len(diag_n_pcs_list)):
            for f in range(diag_n_folds):
                v = mat[ci, f]
                if not np.isnan(v):
                    light = (v < 0.5) if seq else (abs(v) > 0.5)
                    axh.text(f, ci, f'{v:.2f}', ha='center', va='center', fontsize=7,
                             color='white' if light else 'black')
        axh.set_title(title)
        fig.colorbar(im, ax=axh, fraction=0.03)

    # (1) per-fold TEST R² — held-out score (trained on the other folds).
    _draw_perfold_hm(ax[0], perfold,
                     f'{dset_tag}: {diag_grp}:{diag_name} — per-fold TEST R² '
                     f'(RdBu ±1, {decoder})', 'RdBu_r', -1, 1)
    # (2) per-fold TRAIN R² — in-sample fit on the training rows. High train +
    #     negative test above ⇒ overfitting / nonstationarity; low train too ⇒
    #     the feature isn't (linearly) in these PCs.
    _draw_perfold_hm(ax[1], perfold_train,
                     'per-fold TRAIN R² (in-sample; viridis 0–1)',
                     'viridis', 0, 1, seq=True)
    # (3) real vs OOF prediction trace
    ax[2].plot(t_img, y_diag, 'C0', lw=0.7, label='real')
    ax[2].plot(t_img, oof,   'C3', lw=0.7, alpha=0.8, label=f'OOF pred (k={diag_n_pcs_trace})')
    for lo, hi in fold_bounds:
        ax[2].axvline(t_img[min(hi, T-1)], color='gray', ls=':', alpha=0.5)
    ax[2].set_ylabel(diag_name); ax[2].set_xlabel('imaging time (s)')
    ax[2].legend(fontsize=8, loc='upper right')
    ax[2].set_title('real vs out-of-fold prediction (fold boundaries dotted)')
    # (4) target distribution
    ax[3].hist(y_diag[valid], bins=80, color='C0')
    ax[3].set_yscale('log'); ax[3].set_xlabel(diag_name); ax[3].set_ylabel('count (log)')
    ax[3].set_title('target distribution on valid frames (log y)')
    fig.tight_layout()

    # In-sample (TRAIN) reconstruction at diag_insample_n_pcs — fit on ALL valid
    # rows, predict in-sample (no CV). This is the reconstruction CEILING. With
    # train R² low (e.g. ~0.3 even at k=50) the feature is only weakly/linearly
    # in the neural PCs at all; the OOF (test) being negative on top of that is
    # then weak-signal + blocked-CV pessimism rather than pure overfitting.
    ins = f_insample_predict(pca_neu_full[:, :diag_insample_n_pcs], y_diag,
                             decoder=decoder, ridge_alpha=ridge_alpha, k_nn=k_nn,
                             standardize=standardize_emb)
    print(f'  in-sample TRAIN R² @k={diag_insample_n_pcs}: {ins["r2"]:+.3f}')
    figi, axi = plt.subplots(1, 2, figsize=(13, 3.2),
                             gridspec_kw={'width_ratios': [2.4, 1]})
    axi[0].plot(t_img, y_diag, 'C0', lw=0.7, label='real')
    axi[0].plot(t_img[ins['idx']], ins['y_pred'], 'C2', lw=0.7, alpha=0.8,
                label=f'in-sample fit (k={diag_insample_n_pcs})')
    axi[0].set_xlabel('imaging time (s)'); axi[0].set_ylabel(diag_name)
    axi[0].legend(fontsize=8, loc='upper right')
    axi[0].set_title(f'{dset_tag}: {diag_grp}:{diag_name} — in-sample (train) '
                     f'reconstruction ({decoder})')
    yt_i, yp_i = ins['y_true'], ins['y_pred']
    axi[1].scatter(yt_i, yp_i, s=4, alpha=0.3)
    _lo = float(min(yt_i.min(), yp_i.min())); _hi = float(max(yt_i.max(), yp_i.max()))
    axi[1].plot([_lo, _hi], [_lo, _hi], 'k--', lw=0.8, alpha=0.7)
    axi[1].set_xlabel('actual'); axi[1].set_ylabel('in-sample pred')
    axi[1].set_title(f'train R²={ins["r2"]:+.3f}  (k={diag_insample_n_pcs})')
    figi.tight_layout()


#%% OOF trace + scatter small-multiples — diagnose negative R² per feature (#9)
# Builds on the unified f_blocked_cv_r2(..., return_predictions=True): for one
# chosen embedding source and one feature block, fits the SAME blocked-CV
# decoder used by the sweep / bar cells and plots, per feature column:
#   left  — real target vs out-of-fold prediction over time (fold boundaries
#           dotted). Shows WHERE the decoder fails.
#   right — predicted-vs-actual scatter, points colored by held-out fold, with
#           the y=x identity line + global OLS fit line (slope).
#
# Reading the scatter (this is the plot that disambiguates negative R²):
#   - cloud collapsed to a horizontal BAND / OLS slope ~0  → decoder predicts
#     ~train mean; negative R² is just blocked-CV pessimism / low variance
#     (Mechanism 1 — not a real failure).
#   - cloud tilted OFF y=x (wrong slope, even negative) → decoder actively
#     misled by nonstationary drift (#8). Detrend that block.
#   - cloud hugging y=x (slope ~1) → genuine decoding (R² > 0).
#
# Works for the PCA baseline AND any trained CEBRA model, via pred_input_source.
# Inherits CV knobs (embargo_fr, standardize_emb, decoder, ridge_alpha, k_nn,
# detrend_blocks / detrend_sigma_sec, fs_img, n_folds) from the PCA decoding
# sweep cell so the numbers match the sweep / null exactly.
#
# Prereqs: run the "PCA decoding sweep" cell (pca_neu_full + CV knobs); for the
# 'cebra:<name>' source, also run the CEBRA fit cell (cebra_models).

if 1:
    pred_input_source = 'pca'      # 'pca' | 'cebra:<name>' | 'block:<name>'
                                   #   'pca'          — top pred_n_pcs of pca_neu_full
                                   #   'cebra:<name>' — a trained CEBRA embedding
                                   #   'block:<name>' — any built_blocks block's X as the
                                   #                    decoder input (e.g. 'block:grid')
    pred_n_pcs        = 16         # # cols used for 'pca' / 'block:' sources
    pred_block        = 'agg'      # which built_blocks group to diagnose (TARGET)
    pred_max_feats    = 8          # cap # feature columns plotted (rows)

    # ── pick the embedding ──────────────────────────────────────────────────
    if pred_input_source == 'pca':
        emb_diag = pca_neu_full[:, :pred_n_pcs]
        emb_tag  = f'PCA-{pred_n_pcs}'
    elif pred_input_source.startswith('cebra:'):
        _cname = pred_input_source.split(':', 1)[1]
        if _cname not in cebra_models:
            raise ValueError(f'{_cname!r} not in cebra_models: {list(cebra_models)}')
        emb_diag = cebra_models[_cname]['embedding']
        emb_tag  = f'CEBRA-{_cname}'
    elif pred_input_source.startswith('block:'):
        _bn = pred_input_source.split(':', 1)[1]
        if _bn not in built_blocks:
            raise ValueError(f'block {_bn!r} not in built_blocks: {list(built_blocks)}')
        emb_diag = built_blocks[_bn]['X'][:, :pred_n_pcs]
        emb_tag  = f'{_bn}-block'
    else:
        raise ValueError(f'bad pred_input_source: {pred_input_source!r}')

    # Prepare target columns (block-selective detrend matches the sweep).
    if pred_block not in built_blocks:
        raise ValueError(f'block {pred_block!r} not built; have {list(built_blocks)}')
    _b = built_blocks[pred_block]
    _sig_fr = f_resolve_detrend_sigma(pred_block, detrend_blocks,
                                      detrend_sigma_sec, fs_img)
    _oof_targets = []
    for ci in range(min(pred_max_feats, _b['X'].shape[1])):
        y = _b['X'][:, ci]
        if _sig_fr:
            y = f_detrend_col(y, _sig_fr)
        _oof_targets.append((_b['names'][ci], y))

    f_plot_oof_trace_scatter(
        emb_diag, _oof_targets, t_img_plot,
        dset_tag=dset_tag, emb_tag=emb_tag, block_label=pred_block,
        n_folds=n_folds, embargo=embargo_fr, standardize=standardize_emb,
        decoder=decoder, ridge_alpha=ridge_alpha, k_nn=k_nn)


#%% Diagnostic — neural raster + decoder input features (linked x-axis)
# Visualizes what the decoder is actually seeing. All panels share the
# imaging-time axis; any misalignment between behavior and neural should
# show up as obvious lag/lead. Also lets us eyeball nonstationarity of
# the top pix-PCs (task #9 — diagnose adversarial nonstationarity).
#
# Stack:
#   1. Neural raster (S_smn2, hclust-sorted)
#   2. Presence channels (binary, both monitors)
#   3. Lateral angle of nearest object (both monitors)
#   4. Distance channel of nearest object (raw dist or angular size,
#      depending on agg_mode)
#   5. Top n_pix_plot pix-PCs as line traces
#
# Reading guide for nonstationarity:
#   - If pix_PC1 or PC2 shows a slow, monotonic-looking trend across the
#     whole session, that's the smoking gun for the adversarial-CV pattern.
#   - If they fluctuate fast around zero with no global trend, nonstationarity
#     isn't the cause and we need a different diagnosis.

if 1:
    dist_name = side_feat_names_single[3]   # 'dist' or 'ang_size'
    f_plot_input_raster(
        S_raster, t_img_plot,
        built_blocks['agg']['X'], built_blocks['pix']['X'],
        side_tags, n_per_side, dist_name=dist_name, n_pix_plot=3,
        dset_tag=dset_tag, side=side, agg_mode=agg_mode)


#%% Diagnostic — generic feature-block viewer (plot ANY block's channels)
# Block-agnostic companion to the agg+pix raster above. Pick any block from
# built_blocks and see its channels over imaging time the same way — agg, pix,
# grid, flow, motion, self_mot, pix_mot, beh, per_obj. Few channels render as
# line traces; many (pix PCs, grid cells) render as a channel×time heatmap.
# Set plot_block_raster to stack the neural raster on top (shared x-axis) for
# behavior↔neural alignment / nonstationarity eyeballing (#9).
#   plot_block_zscore  — z-score each channel so unlike scales share an axis.
#   plot_block_heatmap — None=auto by max_lines; True/False to force the mode.
if 1:
    plot_block_name    = 'grid'    # any key in built_blocks
    plot_block_raster  = True      # stack neural raster (S_smn2) on top
    plot_block_zscore  = True      # per-channel z-score for display
    plot_block_heatmap = None      # None=auto | True=heatmap | False=lines
    plot_block_maxlines = 14       # d <= this → line traces, else heatmap

    if plot_block_name not in built_blocks:
        raise ValueError(f'{plot_block_name!r} not in built_blocks: '
                         f'{list(built_blocks)}')
    _pb = built_blocks[plot_block_name]
    f_plot_block_traces(
        _pb['X'], _pb['names'], t_img_plot,
        S_sorted=(S_raster if plot_block_raster else None),
        title=f'{dset_tag}: block {plot_block_name!r} '
              f'({_pb["X"].shape[1]} channels)',
        max_lines=plot_block_maxlines, zscore=plot_block_zscore,
        heatmap=plot_block_heatmap)


#%% Unsupervised: PCA reconstruction R² vs n_neural_PCs (real vs per-cell shuffle)
# =============================================================================
# SECTION 6 · DECODING · dimensionality — unsup PCA R², supervised R², recon
# =============================================================================
# Dimensionality / noise-floor check. For each k = 1…max_k, plot the cumulative
# explained-variance ratio of PCA fit on Y_neu — i.e. the R² of reconstructing
# the neural data from its top-k principal components.
#
#   Real line:     PCA on Y_neu directly.
#   Shuffled line: per-cell independent circular shift of Y_neu (using
#                  f_circshift_rates). Breaks cross-cell timing but preserves
#                  per-cell rate, autocorrelation, and burstiness — so any R²
#                  the shuffled curve achieves is what "no real co-firing
#                  structure" looks like.
#
# How to read:
#   - Real curve rising steeply above shuffled → cross-cell co-activity at
#     low k. The shuffled curve at k* gives the noise-floor R²; real − shuffled
#     at k* is the signal above noise.
#   - Curves overlap → no structure beyond per-cell baseline.
#   - Knee in the real curve where it pulls ahead of shuffled = effective
#     dimensionality of the cross-cell signal.

n_shuffles = 5                              # independent shuffles for null band
max_k = min(50, Y_neu.shape[1])

# Real
pca_real = PCA(n_components=max_k, random_state=42).fit(Y_neu)
cum_real = pca_real.explained_variance_ratio_.cumsum()

# Shuffled (per-cell circshift; f_circshift_rates expects (cells, T))
# uniform circshift null (seeded rng for reproducibility).
cum_shuf_runs = np.zeros((n_shuffles, max_k))
_rng_pca = np.random.default_rng(42)
_min_shift_pca = 0
for s in range(n_shuffles):
    Y_shuf = f_shuffle_neural(Y_neu, _rng_pca, min_shift=_min_shift_pca)
    cum_shuf_runs[s] = PCA(n_components=max_k, random_state=42).fit(Y_shuf) \
                          .explained_variance_ratio_.cumsum()
ks = np.arange(1, max_k + 1)
f_plot_real_vs_shuffle_line(
    ks, cum_real, cum_shuf_runs, xlabel='# neural PCs',
    ylabel='R² of reconstruction (cumulative explained variance)',
    title=f'{dset_tag}\nNeural PCA reconstruction R²  '
          f'(n_cells={Y_neu.shape[1]}, T={Y_neu.shape[0]})',
    hline=1.0, legend_loc='lower right')


#%% Supervised: monitor-decoding R² vs n_neural_PCs (real vs per-cell shuffle)
# What you actually asked for. Per k = 1…max(n_pcs_sweep):
#   - fit PCA on Y_neu, take top-k components as the regressor input
#   - fit Ridge on top-k → multi-output target (X_agg or X_pix)
#   - blocked 5-fold CV, NaN-aware
#   - report mean R² across folds, averaged across target columns
# Repeat with per-cell circular-shifted neural for the shuffle line.
# Real − shuffled at any k = behavior info above per-cell-baseline neural noise.

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ── Knobs ────────────────────────────────────────────────────────────────
target_blocks_dec = ['agg']    # list of block names from built_blocks registry
                               # Examples: ['agg'], ['pix'], ['grid'], ['agg', 'pix'],
                               # ['agg', 'motion'], ['agg', 'grid', 'motion', 'pix'].
                               # Any built block works (incl. visual 'grid'; see
                               # the sweep cell's comment for the full list).
                               # Presence channels dropped when skip_presence_dec
                               # (binary; awkward for a regression R² metric).
target_choice = None            # legacy: 'X_agg' | 'X_pix' | 'both' (None = use target_blocks_dec)
n_pcs_sweep_dec = [1, 2, 3, 5, 8, 16, 32, 50]
n_folds       = 5
ridge_alpha   = 1.0
n_shuf_dec    = 5
# Propagated regression-quality knobs (unified harness, 2026-06-03):
decoder_dec           = 'ridge'  # 'ridge' | 'ridgecv' | 'knn'
k_nn_dec              = 15        # used when decoder_dec == 'knn'
standardize_dec       = True      # z-score PCs (train-fold stats) before decode
embargo_sec_dec       = 0.0       # purge gap (s) each side of the test block
detrend_sigma_sec_dec = None      # GLOBAL high-pass σ (s) for all target cols;
                                  # None=off. Use LARGE σ (~20–60 s) — behavior
                                  # is slow. Used only when detrend_blocks_dec None.
detrend_blocks_dec    = None      # block-selective high-pass; overrides the global
                                  # σ above when set. How to write it:
                                  #   None          → global σ for ALL blocks (or off)
                                  #   {'pix': 30.0} → pix at σ=30 s, agg left raw
                                  #   {'agg': 60.0, 'pix': 30.0} → per-block σ
                                  #   ['pix']       → pix at detrend_sigma_sec_dec
# How to aggregate per-column R² into one number when target has multiple
# output dims. Task #26.
#   'uniform_average'   — mean of per-column R²; columns count equally. Low-
#                         variance columns that fail to decode (e.g. presence)
#                         drag the mean negative even when high-variance dims
#                         are fine.
#   'variance_weighted' — joint R² across all output dims and time. Per-column
#                         R² weighted by that column's variance; high-variance
#                         dims dominate. The "what fraction of total target
#                         variance does the decoder explain" reading.
multioutput_dec = 'uniform_average'
skip_presence_dec = True       # drop binary 'pres_*' columns from the target.
                               # They're degenerate under a regression R² metric
                               # (constant after NaN-row drop with nan_absent) and
                               # drag uniform_average. Score presence via the AUC
                               # sweep cell instead. #12/#26.
# ─────────────────────────────────────────────────────────────────────────

# Build target via combine_blocks. Old per-string dispatch kept commented:
# if target_choice == 'X_agg':     target_dec = X_agg
# elif target_choice == 'X_pix':   target_dec = X_pix[:, :n_pix_pca]
# elif target_choice == 'both':    target_dec = np.concatenate([X_agg, X_pix], axis=1)

if target_choice is not None:
    target_blocks_dec = legacy_target_blocks(target_choice)
    print(f'  legacy target_choice={target_choice!r} → target_blocks_dec={target_blocks_dec!r}')

# Imaging fs for sec→frame conversions.
_fs_dec = f_imaging_fs(frame_t)
embargo_fr_dec = int(round(embargo_sec_dec * _fs_dec))

# Concatenated multi-output target, block-selectively high-passed (#8/#10).
# Old single-call form: target_dec, _ = combine_blocks(target_blocks_dec)
target_dec, target_dec_names = build_target_matrix(
    built_blocks, target_blocks_dec, detrend_blocks=detrend_blocks_dec,
    detrend_sigma_sec=detrend_sigma_sec_dec, fs=_fs_dec,
    skip_presence=skip_presence_dec)
target_label_dec = '+'.join(target_blocks_dec)


# _cv_r2_multi → thin wrapper over the unified f_blocked_cv_r2 (multi-output)
# so this cell shares the same blocked-CV implementation as the sweep + bar
# cells. (Inline Ridge loop removed 2026-06-03.) Same positional signature
# (emb, target, n_folds, alpha, multioutput) plus the new embargo/standardize/
# within-fold-PCA options.
def _cv_r2_multi(emb, target, n_folds=5, alpha=1.0, multioutput='uniform_average',
                 embargo=0, standardize=True, Y_raw=None, n_pcs=None):
    return f_blocked_cv_r2(emb, target, n_folds=n_folds, embargo=embargo,
                           standardize=standardize, decoder=decoder_dec,
                           ridge_alpha=alpha, k_nn=k_nn_dec,
                           multioutput=multioutput, Y_raw=Y_raw, n_pcs=n_pcs)


max_pcs_dec = max(n_pcs_sweep_dec)

# Real
pca_real_full = f_pca_prefix(Y_neu, max_pcs_dec)
r2_real_dec = np.array([_cv_r2_multi(pca_real_full[:, :k], target_dec, n_folds, ridge_alpha,
                                      multioutput=multioutput_dec,
                                      embargo=embargo_fr_dec, standardize=standardize_dec)
                        for k in n_pcs_sweep_dec])

# Shuffled — per-cell circshift; refit PCA each shuffle.
# uniform circshift null (seeded rng for reproducibility).
r2_shuf_runs_dec = np.zeros((n_shuf_dec, len(n_pcs_sweep_dec)))
t0 = time.perf_counter()
_rng_dec = np.random.default_rng(42)
_min_shift_dec = 0
for s in range(n_shuf_dec):
    Y_shuf = f_shuffle_neural(Y_neu, _rng_dec, min_shift=_min_shift_dec)
    pca_shuf_full = f_pca_prefix(Y_shuf, max_pcs_dec)
    for ki, k in enumerate(n_pcs_sweep_dec):
        r2_shuf_runs_dec[s, ki] = _cv_r2_multi(pca_shuf_full[:, :k], target_dec, n_folds, ridge_alpha,
                                                multioutput=multioutput_dec,
                                                embargo=embargo_fr_dec, standardize=standardize_dec)
    print(f'  shuffle {s+1}/{n_shuf_dec}  cumulative {time.perf_counter()-t0:.1f}s')
_valid_dec = int((~np.isnan(target_dec).any(axis=1)).sum())
f_plot_real_vs_shuffle_line(
    n_pcs_sweep_dec, r2_real_dec, r2_shuf_runs_dec, xlabel='# neural PCs',
    ylabel=f'blocked-CV R²  ({decoder_dec} α={ridge_alpha}, '
           f'multioutput={multioutput_dec}, standardize={standardize_dec})',
    title=f'{dset_tag}\nDecoding {target_label_dec} from neural PCs  '
          f'(target dim={target_dec.shape[1]}, '
          f'valid frames={_valid_dec}/{target_dec.shape[0]})',
    hline=0.0)


#%% Real vs reconstructed monitor inputs (data or shuffled control)
# Trains the chosen decoder from top-k neural PCs to the multi-dim target;
# collects OUT-OF-FOLD predictions (each frame's prediction comes from a model
# that never saw that frame in training). Then makes two figures:
#   (1) scatter: real vs predicted, one panel per target column (+ per-feat R²)
#   (2) temporal traces: real (blue) and predicted (red) overlaid per column
#
# Decoder choice (decoder_rec knob):
#   'ridge' — Ridge multi-output. Mathematically = independent Ridge per output
#             (no joint structure exploited; same per-column R² as fitting
#             each target alone). Linear; misses conjunctive/multiplicative
#             codes.
#   'knn'   — KNeighborsRegressor multi-output. For each test point, averages
#             the FULL target VECTORS of k nearest training neighbors.
#             Genuinely joint: prediction inherits training-data correlations
#             across outputs. Nonlinear; reads conjunctive codes Ridge misses.
#   'pls'   — PLSRegression. Finds n_components_pls pairs of latent factors
#             (u_i = neural projection, v_i = target projection) maximizing
#             cov(u_i, v_i). Predicts all outputs from those shared latents.
#             Linear but JOINT (rank-r constraint on weight matrix). Wins
#             over Ridge when output dims share underlying neural directions
#             (typically true for lat / vert / ang_size together).
#   'cca'   — Canonical Correlation Analysis. Like PLS but maximizes
#             corr(u_i, v_i) instead of cov. Scale-invariant — can find
#             small-variance high-correlation directions PLS would skip.
#             More an interpretive tool (canonical correlations ρ_i ∈ [0,1]
#             give a "shared dimensionality" readout) than a robust predictor;
#             can overfit when input is high-dim relative to T.
#
# Toggle `use_shuffle = True` to repeat on per-cell circshifted neural — the
# scatter should collapse to a horizontal stripe and traces should look like
# the model just predicts the mean.

# KNN is nonlinear and joint (training-distribution grounded). PLS is linear and joint (rank-constrained). Where they
#   differ tells you something:

#   - KNN >> PLS → joint structure is nonlinear (conjunctive code)
#   - KNN ≈ PLS → joint structure is linear, just rank-constrained
#   - PLS > Ridge → outputs share latent neural directions
#   - Ridge ≈ PLS at high n_components_pls → expected (rank constraint vacuous)


# ── Knobs ────────────────────────────────────────────────────────────────
input_source       = 'pca_neural'  # 'pca_neural' | 'cebra:<name>' | 'block:<name>' | 'pix_pca'
                                # What the decoder reads from:
                                #   'pca_neural'   — top-k PCs of Y_neu (default)
                                #   'cebra:<name>' — cebra_models[name]['embedding']
                                #                    (needs CEBRA fit cell; legacy 'cebra_<name>' ok)
                                #   'block:<name>' — top-k columns of ANY built_blocks block's X
                                #                    (e.g. 'block:grid', 'block:pix', 'block:motion')
                                #                    — sanity test: reconstruct a monitor target
                                #                    from a DIFFERENT monitor representation.
                                #   'pix_pca'      — back-compat alias for 'block:pix'.
target_blocks      = ['agg']    # list of block names from built_blocks registry
                                # Examples:
                                #   ['agg']                — position channels
                                #   ['motion']             — velocity channels
                                #   ['pix']                — pix-PCA components
                                #   ['grid']               — retinotopic grid (visual, Tier 1)
                                #   ['agg', 'motion']      — position + velocity
                                #   ['agg', 'pix']         — position + pixel scene
                                #   ['agg', 'grid', 'motion', 'pix'] — everything
                                # Available block names depend on the build cell:
                                #   'agg', 'pix' are always present.
                                #   'motion' present if build_motion=True.
                                #   'per_obj' present if build_per_obj=True.
# Legacy alias — auto-translated to target_blocks below. Set to None to disable.
target_choice_rec  = None       # legacy: 'X_agg' | 'X_pix' | 'X_mot' | 'X_agg+X_mot' | 'both'
decoder_rec        = 'cca'      # 'ridge' | 'knn' | 'pls' | 'cca'
n_pcs_rec          = 50         # number of neural PCs as decoder input.
                                # Used by 'pca_neural' and 'pix_pca' (truncation).
                                # Ignored by CEBRA modes — they use the trained
                                # embedding's full output_dim (typically 3).
ridge_alpha_rec    = 1.0        # used when decoder_rec == 'ridge'
k_nn_rec           = 15         # used when decoder_rec == 'knn'
n_components_pls   = 1          # used when decoder_rec == 'pls'.
                                # Rank of the joint latent space. Auto-clamped
                                # to min(n_components_pls, in_dim, n_target).
                                # Sweep 1..min(in_dim, p) to find where joint
                                # R² plateaus.
n_components_cca   = 1          # used when decoder_rec == 'cca'. Same auto-
                                # clamp rule as PLS. CCA maximizes corr (vs
                                # PLS's cov), so it's scale-invariant — useful
                                # diagnostic for shared-latent dim, can overfit
                                # on small high-correlation directions.
n_folds_rec        = 5
standardize_rec    = True       # z-score decoder-input columns (train-fold stats)
                                # before fit/predict, matching the rest of the
                                # harness. Stops PC1 from dominating KNN distance /
                                # Ridge penalty. PLS/CCA already scale internally.
use_shuffle        = False      # True → use per-cell circshifted neural
                                # (only takes effect when input_source='pca_neural';
                                # other inputs are pre-computed and not re-derivable
                                # from a shuffled neural recording here).
shuf_seed          = 42
max_targets_plot   = 8          # cap on # of target columns shown (top N)
skip_presence      = True       # drop 'pres_*' channels from target (binary,
                                # R² on regression is awkward; usually not
                                # interpretable in scatter/traces).
mask_pred_to_valid = False       # mask predictions to frames where target is
                                # non-NaN. With nan_absent=True, frames with
                                # no visible object become NaN in target;
                                # this propagates the NaN to predictions so
                                # plots show aligned gaps instead of the
                                # model "filling in" extrapolated values
                                # during empty periods. False → preds shown
                                # for all frames (diagnostic).
# ─────────────────────────────────────────────────────────────────────────

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression, CCA

# Pick target columns + names via the feature registry (built_blocks).
# Legacy `target_choice_rec` strings are auto-translated to target_blocks.
#
# Old per-string if-chain kept commented below for reference:
# def _require_mot():
#     if X_mot is None or mot_feat_names is None:
#         raise ValueError("target_choice_rec needs X_mot but build_motion=False; "
#                          "set build_motion=True in the build cell and rerun.")
# if target_choice_rec == 'X_agg':       target_rec = X_agg.copy();       target_names_rec = list(agg_feat_names)
# elif target_choice_rec == 'X_pix':     target_rec = X_pix.copy();       target_names_rec = [f'pix_PC{i+1}' for i in range(X_pix.shape[1])]
# elif target_choice_rec == 'X_mot':     _require_mot(); target_rec = X_mot.copy();  target_names_rec = list(mot_feat_names)
# elif target_choice_rec == 'X_agg+X_mot': _require_mot(); target_rec = np.concatenate([X_agg, X_mot], axis=1); target_names_rec = list(agg_feat_names) + list(mot_feat_names)
# elif target_choice_rec == 'both':      target_rec = np.concatenate([X_agg, X_pix], axis=1); target_names_rec = list(agg_feat_names) + [f'pix_PC{i+1}' for i in range(X_pix.shape[1])]

if target_choice_rec is not None:
    target_blocks = legacy_target_blocks(target_choice_rec)
    print(f'  legacy target_choice_rec={target_choice_rec!r} → target_blocks={target_blocks!r}')

# Concatenated target (fresh array — no view mutation), optionally dropping the
# binary presence channels. No detrend here (fs omitted). skip_presence affects
# both the decoder fit and the plots.
target_rec, target_names_rec = build_target_matrix(
    built_blocks, target_blocks, skip_presence=skip_presence)

# Build decoder input based on input_source.
if input_source == 'pca_neural':
    if use_shuffle:
        # uniform circshift null (seeded rng).
        _rng_rec = np.random.default_rng(shuf_seed)
        Y_use = f_shuffle_neural(Y_neu, _rng_rec, min_shift=0)
        src_tag = 'SHUFFLED neural (per-cell circshift)  ←  PCA'
    else:
        Y_use = Y_neu
        src_tag = 'real neural  ←  PCA'
    pca_rec_full = f_pca_prefix(Y_use, n_pcs_rec)
elif input_source.startswith('cebra_') or input_source.startswith('cebra:'):
    # Accepts either 'cebra:<name>' or legacy 'cebra_<name>'. Looks up the
    # corresponding entry in cebra_models. Any model name configured in
    # cebra_supervisions is valid (e.g., 'cebra:agg', 'cebra:pix',
    # 'cebra:agg_motion').
    _cname = input_source.split(':', 1)[1] if ':' in input_source else input_source[len('cebra_'):]
    if _cname not in cebra_models:
        raise ValueError(f"cebra model {_cname!r} not in cebra_models; "
                         f"available: {list(cebra_models)} "
                         f"(set cebra_supervisions in the fit cell to add more).")
    pca_rec_full = np.asarray(cebra_models[_cname]['embedding'])
    src_tag = f"CEBRA-{_cname} embedding (dim={pca_rec_full.shape[1]})"
    if use_shuffle:
        print(f'  NOTE: use_shuffle is ignored for input_source={input_source!r} '
              '(embedding is pre-trained on real neural).')
elif input_source == 'pix_pca' or input_source.startswith('block:'):
    # Use ANY built_blocks block's X directly as the decoder input (top-k cols).
    # 'pix_pca' is the back-compat alias for 'block:pix'. Lets you reconstruct a
    # monitor target from a DIFFERENT monitor representation (e.g. 'block:grid').
    _bn = 'pix' if input_source == 'pix_pca' else input_source.split(':', 1)[1]
    if _bn not in built_blocks:
        raise ValueError(f'block {_bn!r} not in built_blocks; '
                         f'available: {list(built_blocks)}')
    _blk = built_blocks[_bn]['X']
    n_use = min(n_pcs_rec, _blk.shape[1])
    pca_rec_full = _blk[:, :n_use].astype(np.float32, copy=False)
    src_tag = f'{_bn} block (top {n_use} of {_blk.shape[1]} cols)'
    if use_shuffle:
        print(f'  NOTE: use_shuffle is ignored for input_source={input_source!r} '
              '(precomputed feature block).')
else:
    raise ValueError(f'unknown input_source: {input_source!r}')

print(f'decoder input: {src_tag}  shape={pca_rec_full.shape}')

# Out-of-fold predictions — every frame predicted by a model trained on the
# other folds. NaN-aware: rows with any NaN target column are excluded from
# training, but (with mask_pred_to_valid=False) ALL test frames get a
# prediction — incl. NaN-target empty frames, for the "does it extrapolate?"
# diagnostic. This is deliberately NOT routed through f_blocked_cv_r2: that
# helper only scores/predicts valid test rows, so it can't reproduce the
# predict-empty-frames behavior this cell relies on. PLS/CCA also live here.
T = pca_rec_full.shape[0]
fold = T // n_folds_rec
n_target = target_rec.shape[1]
preds_rec = np.full(target_rec.shape, np.nan, dtype=float)
valid_rec = ~np.isnan(target_rec).any(axis=1)
for f in range(n_folds_rec):
    lo, hi = f*fold, ((f+1)*fold if f < n_folds_rec-1 else T)
    te_idx = np.zeros(T, dtype=bool); te_idx[lo:hi] = True
    tr = (~te_idx) & valid_rec
    if tr.sum() < 2:
        continue
    if decoder_rec == 'ridge':
        reg = Ridge(alpha=ridge_alpha_rec)
    elif decoder_rec == 'knn':
        reg = KNeighborsRegressor(n_neighbors=k_nn_rec, weights='distance')
    elif decoder_rec == 'pls':
        # n_components must be ≤ min(n_features_input, n_features_output)
        n_c = min(n_components_pls, pca_rec_full.shape[1], target_rec.shape[1])
        if n_c < 1:
            raise ValueError('PLS needs n_components >= 1 (check target & input dims)')
        reg = PLSRegression(n_components=n_c, scale=True)
    elif decoder_rec == 'cca':
        # Same dim rule as PLS. scale=True puts both sides on comparable numerics.
        n_c = min(n_components_cca, pca_rec_full.shape[1], target_rec.shape[1])
        if n_c < 1:
            raise ValueError('CCA needs n_components >= 1 (check target & input dims)')
        reg = CCA(n_components=n_c, scale=True)
    else:
        raise ValueError(f'unknown decoder_rec: {decoder_rec!r}')
    Xin = pca_rec_full
    if standardize_rec:
        _mu = pca_rec_full[tr].mean(axis=0)
        _sd = pca_rec_full[tr].std(axis=0); _sd[_sd == 0] = 1.0
        Xin = (pca_rec_full - _mu) / _sd
    reg.fit(Xin[tr], target_rec[tr])
    if mask_pred_to_valid:
        # Predict only for test frames whose target is non-NaN; rest stay NaN.
        te_pred = te_idx & valid_rec
        if te_pred.sum() > 0:
            preds_rec[te_pred] = reg.predict(Xin[te_pred])
    else:
        preds_rec[te_idx] = reg.predict(Xin[te_idx])

# Cap target columns shown.
# Figures via f_decoding plot helpers (scatter grid + temporal traces).
_rec_tag = (f'target={"+".join(target_blocks)}  |  decoder={decoder_rec}  '
            f'|  input={src_tag}  |  in_dim={pca_rec_full.shape[1]}')
f_plot_pred_scatter(target_rec, preds_rec, target_names_rec,
                    suptitle=f'{dset_tag}\nreal vs predicted  |  {_rec_tag}',
                    max_panels=max_targets_plot)
t_img = t_img_plot
f_plot_pred_traces(target_rec, preds_rec, target_names_rec, t_img,
                   suptitle=f'{dset_tag}\ntemporal traces — real vs predicted  |  {_rec_tag}',
                   max_panels=max_targets_plot)


#%% Procrustes — manifold similarity between embeddings
# =============================================================================
# SECTION 7 · EMBEDDING COMPARISON — Procrustes manifold similarity
# =============================================================================
# Procrustes finds the optimal rotation/reflection/scaling/translation that
# aligns one point cloud to another, then reports the residual distance.
# Strips off basis ambiguity of dim-red outputs (PCA, CEBRA, etc.) so two
# embeddings can be compared on shape alone.
#
# Disparity range: 0 (identical geometry up to rigid+scale transform)
# to 1 (totally different). Typical "very similar manifold" ≈ 0.05-0.2;
# >0.5 is qualitatively different.
#
# Use cases:
#   - CEBRA-agg vs CEBRA-pix → same neural manifold, different supervision,
#     just rotated? Or genuinely different latent geometries?
#   - PCA-3 (unsupervised) vs CEBRA-3 → does contrastive training change
#     the embedding geometry, or is it ~equivalent to top neural PCs?
#   - Train-fold vs test-fold embeddings → stability check (not implemented
#     here, but easy add).

from scipy.spatial import procrustes

# Build the list of embeddings to compare: PCA-3 baseline + every model in
# cebra_models. Adding a new CEBRA model to cebra_supervisions automatically
# enlarges this comparison.
#
# Old hardcoded list kept commented:
# embeddings_to_compare = [
#     ('PCA-3 (unsup)',  pca_neu3),
#     ('CEBRA-agg',      np.asarray(cebra_agg['embedding'])),
#     ('CEBRA-pix',      np.asarray(cebra_pix['embedding'])),
# ]
pca_neu3 = f_pca_prefix(Y_neu, 3)
embeddings_to_compare = [('PCA-3 (unsup)', pca_neu3)]
for _cname, _m in cebra_models.items():
    embeddings_to_compare.append((f'CEBRA-{_cname}', np.asarray(_m['embedding'])))

# Pairwise disparity matrix.
n_emb = len(embeddings_to_compare)
disp_mat = np.zeros((n_emb, n_emb))
print(f'\n{dset_tag} — Procrustes disparity '
      f'(0 = identical shape, 1 = totally different):')
for i, (name_i, e_i) in enumerate(embeddings_to_compare):
    for j, (name_j, e_j) in enumerate(embeddings_to_compare):
        if i == j:
            continue
        if i < j:
            _, _, disparity = procrustes(e_i, e_j)
            disp_mat[i, j] = disparity
            disp_mat[j, i] = disparity
            print(f'  {name_i:18s} vs {name_j:18s}  →  {disparity:.4f}')

# Heatmap of disparities.
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
im = ax.imshow(disp_mat, vmin=0, vmax=0.5, cmap='Reds', interpolation='none')
ax.set_xticks(range(n_emb)); ax.set_xticklabels([n for n, _ in embeddings_to_compare],
                                                rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(n_emb)); ax.set_yticklabels([n for n, _ in embeddings_to_compare], fontsize=9)
ax.set_title(f'{dset_tag}\nProcrustes disparity (low = same manifold geometry)', fontsize=10)
for i in range(n_emb):
    for j in range(n_emb):
        ax.text(j, i, f'{disp_mat[i,j]:.3f}', ha='center', va='center',
                color='white' if disp_mat[i,j] > 0.25 else 'black', fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.04, label='disparity (clipped at 0.5)')
fig.tight_layout()

# Aligned-embedding overlay for one pair of CEBRA models. Choose which two
# to align via proc_align_pair below. The two CEBRA names must be in
# cebra_models (i.e., trained by the CEBRA fit cell above).
proc_align_pair = ('agg', 'pix')   # ('<name_a>', '<name_b>')

if (proc_align_pair[0] in cebra_models) and (proc_align_pair[1] in cebra_models):
    _a_name, _b_name = proc_align_pair
    _a_emb = np.asarray(cebra_models[_a_name]['embedding'])
    _b_emb = np.asarray(cebra_models[_b_name]['embedding'])
    _a_std, _b_aligned, disparity = procrustes(_a_emb, _b_emb)

    stride_proc = max(1, _a_std.shape[0] // 5000)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(_a_std    [::stride_proc, 0], _a_std    [::stride_proc, 1], _a_std    [::stride_proc, 2],
               c='C0', s=2, alpha=0.45, label=f'CEBRA-{_a_name} (standardized)')
    ax.scatter(_b_aligned[::stride_proc, 0], _b_aligned[::stride_proc, 1], _b_aligned[::stride_proc, 2],
               c='C3', s=2, alpha=0.45, label=f'CEBRA-{_b_name} (aligned to {_a_name})')
    ax.set_title(f'{dset_tag}\nCEBRA-{_a_name} vs CEBRA-{_b_name} aligned via Procrustes  '
                 f'(disparity = {disparity:.3f})', fontsize=10)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.legend(fontsize=8)
    fig.tight_layout()
else:
    print(f'  proc_align_pair={proc_align_pair!r}: one or both not in cebra_models '
          f'({list(cebra_models)}); skipping aligned-overlay scatter.')


#%% Ensemble extraction on CEBRA latents — behavior-clamped branch (a)
# =============================================================================
# SECTION 8 · ENSEMBLES — ensemble extraction on CEBRA latents (branch a)
# =============================================================================
# Treat CEBRA latent dims as the "scores" (k_emb, T) and fit a per-cell
# ridge to recover "coeffs" (n_cells, k_emb). Then run the same threshold
# extraction as for NMF/PCA. Output is a MATLAB-style ens_out: cells
# grouped per latent dim, active frames per dim, sorted-by-coeff raster.

if 1:
    from f_ensembles import f_ens_get_thresh, f_apply_thresh
    from f_ensemble_plots import f_plot_ens_overview, f_plot_comp_scatter
    from sklearn.linear_model import Ridge

    # Pick which CEBRA model to dissect. Use the agg-supervised one by
    # default; swap to 'pix' / 'grid' / others as needed.
    cebra_key = 'agg' if 'agg' in cebra_models else list(cebra_models)[0]
    cm = cebra_models[cebra_key]
    emb = cm['embedding']                         # (T, d_emb)
    scores_c = emb.T                              # (d_emb, T) — k × T
    S = est1['S']                                  # (n_cells, T)

    # ridge fit: S.T (T, n_cells) on emb (T, d_emb)
    W = Ridge(alpha=1.0).fit(emb, S.T).coef_      # (n_cells, d_emb)
    coeffs_c = W

    thresh_c, thresh_s = f_ens_get_thresh(
        S, coeffs_c, scores_c, mode='signal_z',
        signal_z_thresh=2.5, dred_method='nmf',   # one-sided shuff doesn't apply
    )
    ens_out_cebra = f_apply_thresh(coeffs_c, scores_c, thresh_c, thresh_s)

    # Decorate the dict so f_plot_ens_overview sees the same shape as a
    # regular f_ensemble_extract result.
    ens_out_cebra.update({
        'coeffs':          coeffs_c,
        'scores':          scores_c,
        'ord_cell':        np.argsort(-np.max(coeffs_c, axis=1)),
        'num_comps':       coeffs_c.shape[1],
        'extraction_method': 'thresh',
        'ensemble_method': f'CEBRA-{cebra_key}',
        'active_cells_mask': np.ones(S.shape[0], dtype=bool),
    })
    for i, ci in enumerate(ens_out_cebra['cells']['ens_list']):
        print(f"  latent {i+1}: {len(ci)} cells, "
              f"{len(ens_out_cebra['trials']['ens_list'][i])} active frames")

    f_plot_ens_overview(ens_out_cebra, S, mouse_dset_tag=dset_tag)
    f_plot_comp_scatter(coeffs_c, ens_out_cebra['cells']['clust_ident'],
                        dim=3, title=f'{dset_tag} — CEBRA-{cebra_key} coeff scatter')
