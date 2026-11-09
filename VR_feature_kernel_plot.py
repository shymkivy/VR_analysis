# -*- coding: utf-8 -*-
"""
VR_feature_kernel_plot.py — overlay gabor/meng feature KERNELS to scale on a monitor frame.

Verification / troubleshooting tool for feature SIZES + SAMPLING: builds the SAME feature spec list as the
search (f_feature_source.f_pyramid_specs / _normalize_spec) and draws each filter's FULL kernel — built by
the SAME kernel builders the extractor uses (_gabor_pair / _gauss_env) — to scale at every sample location
on a rendered monitor frame. Change the kernels → this plot changes. ONE figure per individual filter
(orientation × SF). Cells: load+render → kernel gallery → per-filter overlay + size/sampling readout →
single-filter deep-dive. Created 2026-07-22.
"""

import sys
from pathlib import Path
path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0   # per-filter loop opens many figures

from f_pipeline import f_load_session, f_render_session
from f_feature_source import f_pyramid_specs, _normalize_spec
from f_feature_viz import f_plot_filter_kernel, f_kernel_gallery, f_kernel_geom

#%% Config — mouse, dataset, render size, filter set
mouse_id     = 'L'
data_dir     = 'F:/VR/data_proc/' + mouse_id
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
n_dset       = 4
obj_size     = {'x': 5, 'y': 20, 'z': 5, 'height': 2}
lat_samp     = 150            # WIDTH  (angularly-square with vert 101 → faithful scalar deg/px)
vert_samp    = 101            # HEIGHT
max_frames   = 300            # cap the render (we only need one frame)
side         = 'L'            # which monitor to overlay ('L' | 'R'); extraction is per-monitor
frame_i      = 200            # frame index to display
draw         = 'even'         # kernel component: 'even' (cos) | 'odd' (sin) | 'env' (luminance/Gaussian)
samp_factor  = 1.0            # sampling vs Nyquist: 1=Nyquist (λ/2) | 2=2× oversample | <1=undersample
kernel_sigma = 3.0            # drawn gabor extent in σ (3 = full ±3σ kernel support)
bg_color     = 'black'        # kernel-plot background: 'black' | 'white' | 'gray' | 128 (0-255) | None=movie frame

# Feature spec list — IDENTICAL code to the search (f_pyramid_specs). Edit to match the config under test.
# NB per-filter → one figure per orientation×SF; keep n_orient / cyc_fov / odegs small to limit figures.
specs = f_pyramid_specs('gabor', cyc_fov=(1, 2, 4, 8, 16), n_orient=2, samp_factor=samp_factor,
                        stats=('magnitude', 'luminance'))
# specs = f_pyramid_specs('gabor', sf_range=(0.05, 0.4), n_sf=5, n_orient=4)
# specs = [{'kind': 'gabor', 'sf': 0.13, 'ori': (0, 90)}, {'kind': 'meng', 'sf': 0.13, 'tf': 1.0}]

#%% Load + render one monitor frame
session = f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx)
cam_params = session['cam_params']                          # carries hFOV_rad/vFOV_rad (f_comp_FOV_adj)
_l, _r, two = f_render_session(session, {'obj_size': obj_size, 'num_samp': lat_samp,
                                         'lat_samp': lat_samp, 'vert_samp': vert_samp,
                                         'max_frames': max_frames})
_W = two.shape[2] // 2                                      # per-monitor width
frame = two[frame_i, :, :_W] if side == 'L' else two[frame_i, :, _W:]
print(f'frame {frame_i}: {frame.shape} (one {side} monitor)  render {vert_samp}×{lat_samp}')

#%% Kernel gallery — raw bank kernels (orientation × SF), pixels
if 1:
    _sfs = sorted({float(np.atleast_1d(s['sf_cyc_deg'])[0]) for s in specs})
    _wl_px = [f_kernel_geom(_normalize_spec({'kind': 'gabor', 'sf_cyc_deg': sf}),
                            cam_params, lat_samp, vert_samp)['wl_px'] for sf in _sfs]
    f_kernel_gallery(n_orient=4, wavelengths=tuple(_wl_px), which=draw if draw != 'env' else 'even',
                     suptitle=f'{mouse_id}/{n_dset} — gabor kernels (λ px per SF)')

#%% Per-filter overlay — ONE figure per (orientation × SF); + size/sampling readout table
rows = []
for si, spec in enumerate(specs):
    inp = _normalize_spec(spec)
    n_ori = inp['n_orient']
    for io in range(n_ori):
        fig, ax = plt.subplots(figsize=(6.5, 6.5 * frame.shape[0] / frame.shape[1] + 0.5))
        g = f_plot_filter_kernel(frame, inp, cam_params, lat_samp, vert_samp, draw=draw, io=io,
                                 sigma_extent=kernel_sigma, show_points=True, bg=bg_color, ax=ax)
        fig.tight_layout()
        rows.append({'sf_cyc_deg': g['sf_cyc_deg'], 'theta_deg': g['theta_deg'], 'wl_px': g['wl_px'],
                     'sigma_px': g['sigma_px'], 'support_px': g['support_px'], 'ds': g['ds'],
                     'n_samples': g['n_samples'], 'n_tiles': g['n_tiles']})
readout = pd.DataFrame(rows)
print(readout.drop_duplicates('sf_cyc_deg').to_string(index=False))   # size/sampling per SF

#%% Single-filter deep-dive — one filter larger (points + to-scale gabor at center sample)
if 0:
    inp = _normalize_spec(specs[-1])         # finest SF
    fig, ax = plt.subplots(figsize=(9, 9 * frame.shape[0] / frame.shape[1] + 0.5))
    f_plot_filter_kernel(frame, inp, cam_params, lat_samp, vert_samp, draw=draw, io=0,
                         sigma_extent=kernel_sigma, show_points=True, show_grid=True, bg=bg_color, ax=ax)
    fig.tight_layout()
