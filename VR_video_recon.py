# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:28:47 2026

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
from f_analysis import f_hclust_firing_rates
from f_functions import f_load_bh_data_all, f_proc_movement, f_comp_FOV_adj, f_add_phase, f_get_monitor_coords, f_angles_to_movie_v2, f_render_terrain, f_save_mon_movie, f_composite_with_depth, f_terrain_world_coords, f_add_terrain_to_monitor #, f_plot_session
from f_render_diagnostics import f_plot_terrain_layout, f_plot_terrain_mouse_alignment, f_plot_terrain_object_alignment, f_terrain_coord_diagnostic, f_add_orientation_stripes, f_plot_monitor_frame, f_plot_obj_terrain_heights, f_diag_chunk0_mouse_height, f_diag_chunk0_mouse_height_bilinear, f_diag_chunk0_offset_sweep, f_diag_chunk0_offset_sweep_2d, f_diag_chunk0_offset_sweep_2d_bilinear, f_plot_mouse_vs_terrain, f_diag_chunk_overlap, f_check_obj_coord_system, f_run_terrain_diagnostics

#%%
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
n_dset = 3

num_comp = 50
n_jobs = 5

est1 = data_ca[n_dset]
     
S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]

hclust_data = f_hclust_firing_rates(S_smn, standardize=True, metric='kl', method='average', similarity_transform='auto')   # cosine, jsd, kl  # average for jsd kl

S_smn2 = S_smn[hclust_data['res_order'],:]

if 0:
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
# Builds mov_data, vec_data_l/r, two_mon_frames — the behavior inputs the
# CEBRA cells below consume. Mirrors the monitor-vectorization block in
# VR_ca_analysis.py (~line 378+) so this script is self-contained.

cam_params = {'aspect':             16/9,           # 1920/1080
              'FOV_axis':           'vertical',     # which axis is fixed
              'FOV_deg':            65.9,           # 80
              'cam_rotation_deg':   49.2,           # was 80/2
              'clip_len':           60,
              'num_mon':            2,
              'cam_height':         0.6}            # camera Y offset above
                                                    # mov_data['y_pos'] (= rb
                                                    # anchor). In Unity scene
                                                    # this is the net y of
                                                    # Cam_googles (−0.4) +
                                                    # Camera_eye (+1) = +0.6.
                                                    # Camera world y = mouse_y
                                                    # + cam_height.
cam_params = f_comp_FOV_adj(cam_params)

mov_data = f_proc_movement(bh_data[n_dset], frame_times = est1['frame_times'],
                           do_interp=1, interp_step=0.1, plot_stuff=False)

# Build an eye-shifted copy of mov_data so both the object angle computation
# (f_get_monitor_coords) and the terrain renderer use the SAME camera
# position. Shallow copy with y_pos and the xyz column shifted up by
# cam_height; everything else (time, phi, theta, ...) is shared.
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
max_frames = 500
# left_mon_frames  = f_angles_to_movie(vec_data_l, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp)
# right_mon_frames = f_angles_to_movie(vec_data_r, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp)
left_mon_frames, left_mon_depth  = f_angles_to_movie_v2(vec_data_l, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp
                                        , filled = True, antialias = True, chunk_t = None, return_depth = True, max_frames = max_frames)
right_mon_frames, right_mon_depth = f_angles_to_movie_v2(vec_data_r, mov_data['time'], cam_params, obj_size, lat_samp=num_samp, vert_samp=num_samp
                                        , filled = True, antialias = True, chunk_t = None, return_depth = True, max_frames = max_frames)

# Terrain coord-system sanity checks. Each block is toggled by `if 0:` / `if 1:`.
# All four use chunk_pitch=122, chunk_centered=True, flip_z=True (the orientation
# confirmed for this rig) — adjust the kwargs if you ever swap rigs or terrains.
_terr_kw = dict(chunk_pitch=122.0, chunk_centered=True, cell_size=1.0,
                flip_x=False, flip_z=True, swap_xz=False)


if 0:
    # 1. Layout: top-down + side view of terrain heightmap with mouse trajectory
    #    and object positions overlaid. Confirms world-coord overlap.
    f_plot_terrain_layout(bh_data[n_dset]['terrainData'], mov_data,
                          bh_data[n_dset]['object_data'], **_terr_kw)

    # 2. Mouse vertical alignment: mouse_y vs terrain_y at each frame's (x, z).
    #    Median of `delta` is a candidate eye_height.:
    _mouse_align = f_plot_terrain_mouse_alignment(
        bh_data[n_dset]['terrainData'], mov_data, **_terr_kw)

    # 3. Object vertical alignment: obj_y - terrain_y at (ObjLocX, ObjLocZ) should
    #    sit ~obj_size['height'] for every object. Pure coord-system check.
    _obj_align = f_plot_terrain_object_alignment(
        bh_data[n_dset]['terrainData'], bh_data[n_dset]['object_data'],
        obj_size, **_terr_kw)

    # 4. Brute-force orientation diagnostic. Tries all 8 (flip_x, flip_z, swap_xz)
    #    combos and reports per-combo delta stats. Use when adapting to a new rig.
    
    _best = f_terrain_coord_diagnostic(
        bh_data[n_dset]['terrainData'], bh_data[n_dset]['object_data'], obj_size,
        chunk_pitch=122.0, chunk_centered=True, cell_size=1.0)

# Optional terrain rendering. Z-buffer composited against the object silhouettes:
# at each pixel the nearer surface wins, so objects behind a ridge are correctly
# hidden by it (and a ridge in front of an object gets occluded by the object).
# ChunkPosX/Z in the terrain data are world-coord chunk CENTERS; chunks are
# spaced 122 world units apart with 125 cells each (1-cell overlap), so the
# renderer uses cell_size = chunk_pitch / cell_max ≈ 0.984 internally.
# Adjust eye_height if the mouse should be raised above terrain.
render_terrain = True
terrain_clip_len   = None   # None = use cam_params['clip_len']. Override to render
                             # terrain farther/closer than objects (e.g., 120).
terrain_stride     = 0.1      # integer ≥ 1 keeps every Nth cell; float < 1
                             # (e.g., 0.5) oversamples via bilinear interp.
                             # Memory ~ (1/stride)² when below 1.
terrain_point_size = 1      # pixel half-size of patch painted per sample.
                             # 1 = single pixel (gappy near-field); 2-3 fills
                             # gaps cheaply without increasing sample count.
if render_terrain:
    # Use the eye-shifted xyz (mouse_y + cam_height) so the renderer's camera
    # matches f_get_monitor_coords' camera. eye_height=0 because the shift
    # is already in mouse_xyz_eye.
    mouse_xyz_eye = np.stack([mov_data_eye['x_pos'], mov_data_eye['y_pos'],
                               mov_data_eye['z_pos']], axis=1)
    left_mon_frames = f_add_terrain_to_monitor(
        bh_data[n_dset]['terrainData'], mouse_xyz_eye, mon_l_phi, theta, cam_params,
        left_mon_frames, left_mon_depth,
        lat_samp=num_samp, vert_samp=num_samp,
        stride=terrain_stride, point_size=terrain_point_size,
        clip_len=terrain_clip_len, max_frames=max_frames,
    )
    right_mon_frames = f_add_terrain_to_monitor(
        bh_data[n_dset]['terrainData'], mouse_xyz_eye, mon_r_phi, theta, cam_params,
        right_mon_frames, right_mon_depth,
        lat_samp=num_samp, vert_samp=num_samp,
        stride=terrain_stride, point_size=terrain_point_size,
        clip_len=terrain_clip_len, max_frames=max_frames,
    )

two_mon_frames = np.concatenate((left_mon_frames, right_mon_frames), axis=2)

# Orientation debug overlay. In default ImageJ / matplotlib display (row 0
# at top of screen) the bright stripe should sit at the TOP, the gray
# stripe at the BOTTOM. If they swap, the displayed image is flipped vs.
# the array. See f_add_orientation_stripes for details.
if 1:
    two_mon_frames = f_add_orientation_stripes(two_mon_frames, in_place=True)


#%% Quick-look: plot frame 0 to sanity-check orientation in a single still
f_plot_monitor_frame(two_mon_frames, num_samp, n_fr=50)
plt.show()


#%% Terrain coord-system diagnostics — one-shot health check
# Runs the canonical sanity suite (object-coord-system range check,
# chunk-overlap RMS-diff, object alignment, mouse-vs-terrain time series
# + histogram) for the current dataset. Pass mov_data_eye too if you've
# built the eye-shifted variant, to see both rb_y and cam_y plots.
if 0:
    f_run_terrain_diagnostics(bh_data[n_dset]['terrainData'],
                                bh_data[n_dset]['object_data'],
                                mov_data,
                                mov_data_eye=mov_data_eye,
                                obj_size=obj_size,
                                terr_kw=_terr_kw,
                                expected_delta=0.4)


#%% Individual diagnostics — call separately when you want only one
if 0:
    # camera-vs-terrain time series + delta histogram (pass mov_data_eye
    # for camera; mov_data for rb anchor).
    f_plot_mouse_vs_terrain(bh_data[n_dset]['terrainData'], mov_data_eye,
                              terr_kw=_terr_kw, expected_delta=0.4,
                              label='cam_y', print_first_n=100)
if 0:
    # adjacent-chunk boundary comparison (verify overlap convention)
    f_diag_chunk_overlap(bh_data[n_dset]['terrainData'], n_compare=3)
if 0:
    # heuristic: ObjLocX/Z are world coords or chunk-relative?
    f_check_obj_coord_system(bh_data[n_dset]['terrainData'],
                                bh_data[n_dset]['object_data'],
                                mov_data=mov_data)


#%% Per-object terrain-height labels on a single frame
# For each visible object, overlays the terrain height at its
# (ObjLocX, ObjLocZ) on the rendered frame. Labels should land on the
# painted cylinder silhouettes; mismatch indicates projection or
# heightmap-lookup drift.
f_plot_obj_terrain_heights(two_mon_frames,
                            vec_data_l, vec_data_r,
                            bh_data[n_dset]['object_data'],
                            bh_data[n_dset]['terrainData'],
                            cam_params, num_samp,
                            n_fr=50, terr_kw=_terr_kw)
plt.show()

#%%

res = f_diag_chunk0_mouse_height(bh_data[n_dset]['terrainData'], mov_data,
                                     max_frames=5000)

sweep = f_diag_chunk0_offset_sweep(bh_data[n_dset]['terrainData'], mov_data,
                                       max_frames=5000)


sweep2d = f_diag_chunk0_offset_sweep_2d(bh_data[n_dset]['terrainData'], mov_data, center=62, half_range=5)


# Compare canonical (0,0) to a few off-diagonal candidates
sweep2d = f_diag_chunk0_offset_sweep_2d(
    bh_data[n_dset]['terrainData'], mov_data, max_frames=5000,
    plot_hist_offsets=[(0, 0), (1, 1), (-1, -1), (2, 0)]
)

# Or look at the optimum found by the heatmap
sweep2d = f_diag_chunk0_offset_sweep_2d(
    bh_data[n_dset]['terrainData'], mov_data, max_frames=5000,
)
# then call again with the recommended offsets:
f_diag_chunk0_offset_sweep_2d(
    bh_data[n_dset]['terrainData'], mov_data, max_frames=5000,
    plot_hist_offsets=[(sweep2d['best_ox_std'], sweep2d['best_oz_std']),
                       (sweep2d['best_ox_peak'], sweep2d['best_oz_peak']),
                       (0, 0)]
)

res = f_diag_chunk0_mouse_height_bilinear(bh_data[n_dset]['terrainData'], mov_data,
                                              max_frames=2000)


#%% Save synthetic-monitor stack to disk (for visually checking f_angles_to_movie_v2)
# Multi-page TIFF -> open in ImageJ / Fiji as a movie. Toggle on per-run.
if 0:
    mov_out_dir = 'F:/test_mov'
    dset_tag = est1['dset_name']

    # full session, both monitors side-by-side
    f_save_mon_movie(two_mon_frames,
                     os.path.join(mov_out_dir, f'two_mon_v2_{dset_tag}.tif'))

    # short clip — first 500 s (mov_data['time'] is on the imaging clock)
    dt = float(mov_data['time'][1] - mov_data['time'][0])
    n_clip = int(round(500.0 / dt))
    f_save_mon_movie(two_mon_frames,
                     os.path.join(mov_out_dir, f'two_mon_v2_{dset_tag}_500s.tif'),
                     frame_range=(0, n_clip))

