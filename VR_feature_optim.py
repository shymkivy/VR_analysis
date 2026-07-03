# -*- coding: utf-8 -*-
"""
VR_feature_optim.py — parameter sweep to optimize the GABOR video decoder (Chain-1),
per dataset (TODO #47/#66/#68).

Optimizes gabor feature→movie reconstruction R² (pooled) by sweeping parameters on ONE
dataset. Efficient: render + extract a gabor bank ONCE, then sweep cheaply —
  - n feature PCs : nested PCA truncation (free)
  - decoder lags  : re-score (cheap)
  - decoder type / alpha : re-score (cheap)
  - [later] grid / #orientations / spatial-frequencies : conv-once/pool-many + column subset
Marginal by default (one axis at a time); a JOINT cell crosses the cheap axes. The
engine (f_sweep_recon / f_sweep_marginal) is feature-agnostic — the meng sweep reuses it.
Compute in functions/f_feature_optim.py. NO neural data. Spyder cell-based.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import numpy as np

from f_pipeline import f_load_session, f_render_session
from f_decoding import f_imaging_fs
from f_feature_recon import f_truncate_session, f_acuity_sigma_px, f_make_target
from f_feature_optim import (f_build_gabor_banks, f_gabor_configs, f_build_meng_banks,
                             f_meng_configs, f_sweep_marginal, f_sweep_recon,
                             f_plot_sweep_marginal)

#%% Config — single dataset, render + smoothing, gabor MASTER bank, sweep axes
mouse_id     = 'L'
n_dset       = 4                 # per-dataset optimization first (cross-dataset later)
data_dir     = 'F:/VR/data_proc/' + mouse_id
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
obj_size     = {'x': 5, 'y': 20, 'z': 5, 'height': 2}
n_frames     = 20000    # subset is fine for PARAM SEARCH (relative ranking is stable)
num_samp     = 101

# target-movie smoothing — HELD FIXED across the sweep (moving it games the score).
# Keep these in sync with VR_feature_recon.py so the winning params transfer over.
target_mode  = 'lowpass'
cutoff_cpd   = 0.1
n_target_pcs = 120

#%% Load + render + build target (shared by the gabor + meng sweeps)
session = f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx)
dset_tag = f"{mouse_id} dset{n_dset}  {session['est1']['dset_name']}"
fs_img  = f_imaging_fs(session['frame_t'])
sigma_px = f_acuity_sigma_px(session['cam_params'], num_samp, cutoff_cpd)
_sub = f_truncate_session(session, n_frames)

_rcfg = {'obj_size': obj_size, 'num_samp': num_samp, 'render_terrain': False}
left, right, two = f_render_session(_sub, _rcfg)
target = f_make_target(two, mode=target_mode, sigma_px=sigma_px, n_target_pcs=n_target_pcs, n_mon=2)
# motion for the hi/lo split = ‖Δ target movie-PCs‖ per frame (on-screen change)
motion = np.zeros(target['scores'].shape[0])
motion[1:] = np.linalg.norm(np.diff(target['scores'], axis=0), axis=1)
mon_movies, side_tags = [left, right], ['L', 'R']
del two
print(f'{dset_tag}: {_sub["T_img"]} frames  fs≈{fs_img:.1f} Hz  '
      f'target {target_mode} keeps {target["pca"].explained_variance_ratio_.sum():.1%} of movie var')

# shared CV config (single dataset → time-blocked; interleaved is robust to flat epochs)
cv_kw = dict(n_folds=5, embargo=int(round(0.5 * fs_img)), standardize=True,
             fold_scheme='interleaved', n_blocks=None)

#%% GABOR MAGNITUDE optimization — build a MASTER bank once, marginal over grid/#orient/SF/n_pca/decoder
# The bank is built at ALL gb_grids (multigrid: conv ONCE → pool every grid) × master n_orient ×
# master wavelengths, magnitude only. The extraction axes then SUBSET this bank for FREE:
#   grid     → pooling density (bank keys)                    [the #1 lever for mag form]
#   n_orient → must DIVIDE gb_n_orient (orientation subset)
#   wl       → any subset of gb_wavelengths (spatial freqs)
#   n_pca    → nested PCA truncation (free);  decoder → re-score
# lags fixed [0] (mag = static form; lags only overfit). Building 4 grids × 8 orient × 5 SF is
# the one expensive step; everything after is column-subset + PCA + CV.
if 1:
    # master bank
    gb_grids       = [(6, 6), (8, 8), (12, 12), (16, 16)]     # grid axis
    gb_n_orient    = 8                                         # master; n_orient axis divides it
    gb_wavelengths = (2, 4, 6, 9, 13, 19)                         # master SF ladder; wl axis subsets it

    banks = f_build_gabor_banks(mon_movies, side_tags, _sub['beh_t'], _sub['frame_t'],
                                _sub['pulse_delay'], gb_grids, n_orient=gb_n_orient,
                                wavelengths=gb_wavelengths, stats=('magnitude',))
    print(f'gabor bank grids={list(banks)}  raw dims {[banks[g]["X"].shape[1] for g in banks]}')

    # baseline = fixed point for the OTHER axes while one is swept ('grid' held as a 1-elem list)
    baseline = {'grid': [(8, 8)], 'n_orient': 4, 'wl': (6, 13), 'n_pca': 40,
                'lags': [0], 'decoder': ('ridge', 1.0)}
    axes = {'grid':     [(6, 6), (8, 8), (12, 12), (16, 16)],
            'n_orient': [2, 4],
            'wl':       [(2,), (4,), (6,), (9,), (13,), (19,)], #, (6, 13), (4, 9, 19), (4, 6, 9, 13, 19)],
            'n_pca':    [5, 10, 20, 40, 80, 120],
            'decoder':  [('ridge', 1.0), ('ridge', 10.0), ('ridgecv', None)]}

    config_fn = lambda **kw: f_gabor_configs(banks, **kw)
    rows = f_sweep_marginal(config_fn, target, cv_kw, fs_img, baseline, axes,
                            stats_keep=('M',), motion=motion)   # 'M' = magnitude token
    f_plot_sweep_marginal(rows, suptitle=f'{dset_tag} — gabor MAG recon R² marginal '
                          f'({target_mode} {cutoff_cpd}cpd, target {n_target_pcs}pc)')

    # best value per axis
    _val = {'grid': 'grid', 'n_orient': 'n_orient', 'wl': 'wavelengths',
            'n_pca': 'n_pca', 'decoder': 'decoder_lbl', 'lags': 'lags_lbl'}
    for a in axes:
        sel = [r for r in rows if r['axis'] == a]
        b = max(sel, key=lambda r: (r['r2'] if np.isfinite(r['r2']) else -np.inf))
        print(f'  best {a:9s}: {str(b[_val[a]]):>18s}   R²={b["r2"]:+.3f}')

#%% JOINT — grid × n_pca interaction (a finer grid may want MORE PCs), at the best decoder
if 0:
    cfgs = list(f_gabor_configs(banks, grids=[(8, 8), (12, 12), (16, 16)], n_orient_list=[4],
                                wl_subsets=[(4, 6, 9, 13, 19)], stats_keep=('M',)))
    joint = f_sweep_recon(cfgs, target, cv_kw, fs_img, n_pca_list=[20, 40, 80, 120],
                          lags_list=[[0]], decoder_list=[('ridgecv', None)], motion=motion)
    best = max(joint, key=lambda r: (r['r2'] if np.isfinite(r['r2']) else -np.inf))
    print('best joint:', {k: best[k] for k in ('grid', 'n_pca', 'decoder_lbl', 'r2')})

#%% MENG sweep — SAME engine; judged on HIGH-motion R² (meng can't rebuild static form)
# Extra axis 'tf' = temporal frequencies (speeds). To sweep directions raise mg_n_orient (+ add
# 'n_orient'), spatial SF widen mg_wavelengths (+ 'wl'), grid gb→mg_grids (+ 'grid').
if 1:
    mg_grids          = [(6, 6)]
    mg_n_orient       = 4                       # orientations → 2·n_orient directions (opponent)
    mg_wavelengths    = (8,)
    mg_temporal_freqs = (0.1, 0.15, 0.25)       # speeds ≈ tf·λ px/frame
    mg_support        = 11

    mbanks = f_build_meng_banks(mon_movies, side_tags, _sub['beh_t'], _sub['frame_t'],
                                _sub['pulse_delay'], mg_grids, n_orient=mg_n_orient,
                                wavelengths=mg_wavelengths, temporal_freqs=mg_temporal_freqs,
                                temporal_support=mg_support, output='opponent')
    print(f'meng bank {list(mbanks)} raw dims {[mbanks[g]["X"].shape[1] for g in mbanks]}')

    mbaseline = {'grid': mg_grids, 'n_orient': mg_n_orient, 'wl': mg_wavelengths,
                 'tf': mg_temporal_freqs, 'n_pca': 20, 'lags': [0, 100, 200],
                 'decoder': ('ridge', 1.0)}
    maxes = {'n_pca':   [2, 5, 10, 20, 40],
             'tf':      [(0.1,), (0.15,), (0.25,), (0.1, 0.15, 0.25)],
             'lags':    [[0], [0, 100], [0, 100, 200], [-200, -100, 0, 100, 200]],
             'decoder': [('ridge', 1.0), ('ridgecv', None)]}

    mconfig_fn = lambda **kw: f_meng_configs(mbanks, **kw)
    mrows = f_sweep_marginal(mconfig_fn, target, cv_kw, fs_img, mbaseline, maxes,
                             stats_keep=None, motion=motion)     # None = keep the opponent channel
    # meng's value is MOTION → score on HIGH-motion R² (y='r2_hi'), not full-scene pooled R²
    f_plot_sweep_marginal(mrows, y='r2_hi',
                          suptitle=f'{dset_tag} — meng recon R² (HIGH-motion) marginal sweep')

    _mval = {'n_pca': 'n_pca', 'lags': 'lags_lbl', 'decoder': 'decoder_lbl', 'tf': 'temporal_freqs'}
    for a in maxes:
        sel = [r for r in mrows if r['axis'] == a]
        b = max(sel, key=lambda r: (r['r2_hi'] if np.isfinite(r['r2_hi']) else -np.inf))
        print(f'  best {a:8s}: {str(b[_mval[a]]):>18s}   R²(hi-motion)={b["r2_hi"]:+.3f}')
