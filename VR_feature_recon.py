# -*- coding: utf-8 -*-
"""
VR_feature_recon.py — Chain-1 groundwork: reconstruct the monitor MOVIE from feature
blocks, neuron-free (TODO #47/#66). "How much of what the mouse sees does each feature
set retain?" — the offline fidelity ceiling that grounds the neural decode/encode arms.

Pipeline (Spyder cell-based): load session → truncate to a frame SUBSET → render +
extract (gabor magnitude+phase / meng / pix) → build a perceptually SMOOTHED target
movie (mouse-acuity lowpass) → blocked-CV ridge from each feature set to the target's
PCs → out-of-fold R² (overall + high/low on-screen motion) → example reconstructed
frames vs the target at high-motion timepoints.

Compares: gabor MAGNITUDE only / meng only / mag+meng / pix (near-ceiling). The target
is lowpass (not rank-reduced) so pix stays a FAIR competitor. Compute in
functions/f_feature_recon.py. NO neural data.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import gc

import numpy as np
import matplotlib.pyplot as plt

from f_pipeline import f_load_session, f_build_features, f_render_session
from f_decoding import f_imaging_fs, f_lags_from_ms
from f_feature_viz import f_plot_movie_frames_compare, f_pick_motion_frames
from f_functions import f_save_mon_movie
from f_feature_recon import (f_truncate_session, f_acuity_sigma_px, f_deg_per_sample,
                             f_make_target, f_motion_score, f_recon_compare,
                             f_recon_frames_at, f_target_lowrank_at, f_plot_recon_bars,
                             f_target_evr_curve, f_plot_target_evr,
                             f_pc_sweep, f_plot_pc_sweep,
                             f_plot_motion_traces, f_plot_pred_overlay, f_recon_diagnose,
                             f_accumulate_session, f_pool_sessions)

#%% Config — mouse, dataset(s), frame subset, feature specs
mouse_id     = 'L'
data_dir     = 'F:/VR/data_proc/' + mouse_id
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'
obj_size     = {'x': 5, 'y': 20, 'z': 5, 'height': 2}

# dsets = which do_proc datasets go into the ONE video decoder. A single-element list
# reproduces the original per-session run; a multi-element list POOLS sessions (shared
# feature + movie PCA, leave-session-out CV). Entries = n_dset (this mouse) or (mouse, n_dset).
dsets    = [4]           # e.g. [4] single | [4, 5, 6] pooled | [('L',4), ('M',2)] cross-mouse
n_frames = 20000          # frame SUBSET PER session (contiguous, keeps meng temporal continuity)
num_samp = 101

# Feature source (SINGLE-session path only): 'build' = render + extract (unchanged) |
# 'conv_repo' = still render the movie for the TARGET, but LOAD the feature blocks from a
# conv_repo (skip re-extraction) — blocks named gaborS_ds*/mengS_ds* (point feature_sets at them).
feat_source  = 'build'
conv_tag     = 'gen_7_11_26'   # must match the conv_repo you built (folder <dset>_<conv_tag>)
conv_ds_list = [8]        # sampling stride(s) loaded from the repo
conv_n_pca   = 120

# Target-movie smoothing (used by BOTH single + pooled paths).
target_mode  = 'lowpass'   # 'lowpass' (acuity Gaussian) | 'rank' (drop pix) | 'raw'
cutoff_cpd   = 0.1         # mouse spatial-acuity cutoff (cyc/deg); sweep it (~0.5 = literature)
target_rank  = 50          # 'rank' mode only
n_target_pcs = 120          # #target-movie PCs = the regression target       53 - 95% 119 - 99%

# gabor: magnitude + phase (grouped PCA so phase's small scale isn't washed out).
# meng: opponent motion energy (direction+speed). pix built from n_pix_pca.
# n_pca here = the BUILD MAX; the n_pcs sweep truncates DOWN from it (nested, free).
visual_specs = {
    'gabor': {'grid': (8, 8), 'n_orient': 4, 'wavelengths': (9),      # 'wavelengths':(6, 12)  'grid': (12, 12)
              'stats': ('magnitude', 'phase'), 'n_pca': {'gabM': 120, 'gabP': 120}},
    'meng':  {'grid': (8, 8), 'n_orient': 4, 'wavelengths': (9),      # 'wavelengths':(6, 12)  'grid': (12, 12)
              'temporal_freqs': (0.15,), 'output': 'opponent', 'n_pca': 120},
}

#%% Build the video-decoder training set — ONE dataset (as before) OR pooled across dsets
# len(dsets)==1 → single-session build (f_build_features, unchanged; built_blocks incl. pix).
# len>1 → per-session render + RAW extract, then f_pool_sessions fits ONE shared feature PCA
# + ONE shared movie-target PCA (pooled built_blocks has the visual blocks only, no pix) and
# tags each frame with a session id for leave-session-out CV below.
cfg = {
    'side': 'both', 'agg_mode': 'angular', 'obj_collapse': 'salience_mean',
    'nan_absent': True, 'n_pix_pca': 120,
    'build_visual': True, 'visual_specs': visual_specs,
    'obj_size': obj_size, 'num_samp': num_samp,
}

def _mouse_dset(entry):
    return entry if isinstance(entry, tuple) else (mouse_id, entry)

if len(dsets) == 1:
    _m, _d = _mouse_dset(dsets[0])
    session = f_load_session(_m, _d, 'F:/VR/data_proc/' + _m, data_dir_bh, vr_data_xlsx)
    fs_img = f_imaging_fs(session['frame_t'])
    sigma_px = f_acuity_sigma_px(session['cam_params'], num_samp, cutoff_cpd)
    session_sub = f_truncate_session(session, n_frames)
    if feat_source == 'build':
        built_blocks, frame_t, side_tags, two_mon_frames = f_build_features(session_sub, cfg)
    else:                                      # conv_repo: render for the target, LOAD features
        from f_conv_repo import f_conv_repo_dir, f_load_conv_blocks
        _l, _r, two_mon_frames = f_render_session(session_sub, cfg)
        frame_t, side_tags = session_sub['frame_t'], ['L', 'R']
        _repo = f_conv_repo_dir('F:/VR/features/' + _m, session['est1']['dset_name'], conv_tag)
        built_blocks, _ = f_load_conv_blocks(_repo, conv_ds_list, n_pca=conv_n_pca)
        _n = two_mon_frames.shape[0]           # repo maps are FULL-length → truncate to the subset
        for _b in built_blocks.values():
            _b['X'] = _b['X'][:_n]
    target = f_make_target(two_mon_frames, mode=target_mode, sigma_px=sigma_px,
                           rank=target_rank, n_target_pcs=n_target_pcs, n_mon=len(side_tags))
    _mblk = 'pix' if 'pix' in built_blocks else next(iter(built_blocks))
    motion = f_motion_score(built_blocks, block=_mblk)
    session_ids = None
    dset_tag = f"{_m} dset{_d}  {session['est1']['dset_name']}"
    print(f'{dset_tag}: {session_sub["T_img"]} frames  movie {two_mon_frames.shape}  '
          f'feat_source={feat_source}')
else:
    accs, sigma_px, fs_img, _names = [], None, None, []
    for _entry in dsets:
        _m, _d = _mouse_dset(_entry)
        _sess = f_load_session(_m, _d, 'F:/VR/data_proc/' + _m, data_dir_bh, vr_data_xlsx)
        if sigma_px is None:                       # rig geometry / fs shared across sessions
            sigma_px = f_acuity_sigma_px(_sess['cam_params'], num_samp, cutoff_cpd)
            fs_img = f_imaging_fs(_sess['frame_t'])
        accs.append(f_accumulate_session(_sess, cfg, n_frames, visual_specs))
        _names.append(f'{_m}d{_d}')
        del _sess; gc.collect()
    _tcfg = dict(mode=target_mode, sigma_px=sigma_px, rank=target_rank, n_target_pcs=n_target_pcs)
    pooled = f_pool_sessions(accs, visual_specs, _tcfg, fs=fs_img)
    built_blocks, target, session_ids = pooled['built_blocks'], pooled['target'], pooled['session_ids']
    motion, frame_t, side_tags = pooled['motion'], pooled['frame_t'], pooled['side_tags']
    two_mon_frames = None
    dset_tag = f"{mouse_id} POOLED {_names} ({len(dsets)} sess, {sum(pooled['ns'])} fr)"
    del accs; gc.collect()

_evr = target['pca'].explained_variance_ratio_.sum()
print('blocks:', {k: built_blocks[k]['X'].shape for k in built_blocks},
      f' fs≈{fs_img:.1f} Hz  target {target_mode} keeps {_evr:.1%} of movie var')

# smoothing preview (single-session only — needs the raw movie in memory)
if two_mon_frames is not None:
    _pf = f_pick_motion_frames(built_blocks, 5, mode='spread', block='pix')
    if 0:
        f_plot_movie_frames_compare([two_mon_frames[_pf], target['movie'][_pf]], _pf,
                                    labels=['raw render', f'target ({target_mode})'],
                                    suptitle=f'{dset_tag} — smoothing at {cutoff_cpd} cpd')

#%% How many target PCs? — cumulative movie-info captured vs #PCs (pick n_target_pcs)
# The smoothed target's PCA explained-variance curve = the low-rank reconstruction ceiling at
# each rank. Fit ABOVE n_target_pcs to see where it plateaus. If 99% is reached well before
# n_target_pcs (as with pix ≈ .99), you can lower n_target_pcs; if still climbing, raise it.
if 0:
    evr_max_pcs = 150       # fit the target PCA this high to see the tail (> n_target_pcs)
    _k, _cum = f_target_evr_curve(target['movie'], max_pcs=evr_max_pcs)
    f_plot_target_evr(_k, _cum, n_target_pcs=n_target_pcs,
                      suptitle=f'{dset_tag} — target movie: variance captured vs #PCs')
    for _m in (0.90, 0.95, 0.99):
        print(f'  {_m:.0%} of smoothed-movie variance @ {int(np.searchsorted(_cum, _m)) + 1} PCs '
              f'(n_target_pcs={n_target_pcs} → {_cum[min(n_target_pcs, _cum.size) - 1]:.1%})')

#%% Decode the movie from each feature set — blocked-CV ridge, OOF R² (+ motion split)
# Feature SETS: (block, name_prefix|None). 'gabor_gabM' = gabor MAGNITUDE PCs,
# 'gabor_gabP' = phase PCs (grouped-PCA names). Lags let motion features contribute
# (and make phase meaningful via Δφ). If target_mode='rank', REMOVE 'pix' (ceiling=1).
if 1:
    recon_lags_ms = [0]      # , 100, 200 causal window; features(t..t-Δ) → movie(t).
                             # STATIC form (mag) wants [0]; MOTION (meng/phase) wants a window.
    recon_folds   = 5
    recon_embargo_sec = 0.5
    recon_decoder = 'ridgecv'  # 'ridge' (fixed α) | 'ridgecv' (tunes α per fold — helps overfit)
    recon_alpha   = 1.0      # ridge α (ignored for ridgecv)
    # fold_scheme: 'contiguous' (each fold = one time-block, honest whole-epoch test) or
    # 'interleaved' (chop into n_blocks small blocks round-robin'd to folds → every fold
    # spans the session, so a flat end epoch can't tank a fold). n_blocks None = n_folds*10.
    recon_fold_scheme = 'interleaved'   # flip to 'interleaved' to chop
    recon_n_blocks    = None

    feature_sets = {
        'agg':      [('agg', None)],        # object-model (near-GENERATIVE: from vec_data the
                                            # render is drawn from) — object-position ceiling, low-D
        'mag':      [('gabor', 'gabor_gabM')],
        'phase':    [('gabor', 'gabor_gabP')],
        'meng':     [('meng', None)],
        'mag+meng': [('gabor', 'gabor_gabM'), ('meng', None)],
        'mag+phase+meng': [('gabor', None), ('meng', None)],
        'pix':      [('pix', None)],
    }
    # drop feature sets whose blocks aren't present (pooled built_blocks has no pix)
    feature_sets = {k: v for k, v in feature_sets.items()
                    if all(b in built_blocks for b, *_ in v)}
    lags = f_lags_from_ms(recon_lags_ms, fs_img)
    cv_kw = dict(n_folds=recon_folds, embargo=int(round(recon_embargo_sec * fs_img)),
                 decoder=recon_decoder, ridge_alpha=recon_alpha, standardize=True)
    if session_ids is not None:            # POOLED → leave-session-out CV
        cv_kw['fold_ids'] = session_ids
        print(f'leave-session-out CV across {len(np.unique(session_ids))} sessions')
    else:                                  # single dataset → time-blocked CV
        cv_kw.update(fold_scheme=recon_fold_scheme, n_blocks=recon_n_blocks)
    print(f'lags(ms)={recon_lags_ms}→{lags}  embargo={cv_kw["embargo"]}fr  target={target_mode}')

    results = f_recon_compare(built_blocks, feature_sets, target, cv_kw,
                              lags=lags, motion=motion)
    if session_ids is not None:            # per-session held-out R² (generalization)
        for label, r in results.items():
            print(f"  {label:14s} per-session held-out R²: "
                  + ' '.join(f'{v:+.2f}' for v in r['r2_per_fold'].values()))
    f_plot_recon_bars(results, motion=True,
                      suptitle=f'{dset_tag} — movie reconstruction R² ({target_mode}, '
                               f'{cutoff_cpd} cpd, lags {recon_lags_ms} ms)')

#%% Recon R² vs #PCs — sweep ONE feature group (MAGNITUDE first)
# Cheap: truncates the built block's PCs (nested), NO re-extraction. Build the group at a
# high max in visual_specs['gabor']['n_pca'] so the sweep has range; k > built #PCs are
# skipped. For other groups: ('gabor','gabor_gabP') phase, or ('meng', None) whole block.
# Uses cv_kw / lags / motion / target from the decode cell above.
if 1:
    sweep_block  = 'meng'
    sweep_prefix = 'meng_PC2'        # 'gabor_gabM' mag | 'gabor_gabP' phase | meng→None
    k_list       = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    pc_series    = ('pooled', 'high', 'low')        # ('pooled','per_fold','high','low')

    pc_rows = f_pc_sweep(built_blocks, sweep_block, sweep_prefix, target, cv_kw,
                         k_list, lags=lags, motion=motion)
    f_plot_pc_sweep(pc_rows, series=pc_series,
                    suptitle=f'{dset_tag} — recon R² vs #PCs ({sweep_prefix or sweep_block}, '
                             f'{target_mode} {cutoff_cpd}cpd, lags {recon_lags_ms}ms)')

#%% Example reconstructed frames — each set vs the target, at motion frames
# Reconstructs from the OOF predictions (inverse target-PCA), so only frames that were held
# out can be shown. Rows = target + each set in show_sets. 'lowrank' = the n_target_pcs
# inverse-PCA of the target itself (the decode CEILING); the rest are decoder OOF outputs.
if 1:
    show_sets = ['lowrank', 'pix', 'agg', 'mag', 'meng', 'mag+meng']
    show_sets = [s for s in show_sets if s == 'lowrank' or s in results]
    _mblk = 'pix' if 'pix' in built_blocks else list(built_blocks)[0]   # pooled has no pix
    want = f_pick_motion_frames(built_blocks, 6, mode='spread', block=_mblk)

    # align to frames that are OOF in a decoded set (same CV split → same OOF idx for all)
    _probe = results[next(s for s in show_sets if s != 'lowrank')]['res']
    _oof = set(int(i) for i in _probe['idx'])
    pidx = [f for f in want if f in _oof] or sorted(_oof)[:6]

    rows = [target['movie'][pidx]]
    labels = [f'target ({target_mode})']
    for s in show_sets:
        if s == 'lowrank':
            rec = f_target_lowrank_at(target, pidx)
            lbl = f'target low-rank\n({n_target_pcs} PCs)'
        else:
            rec, _found = f_recon_frames_at(results[s]['res'], target['pca'], target['hw'], pidx)
            lbl = f"{s}\nR²={results[s]['r2']:+.2f}"
        rows.append(rec)
        labels.append(lbl)
    f_plot_movie_frames_compare(rows, pidx, labels=labels, cmaps=['gray'] * len(rows),
                                suptitle=f'{dset_tag} — recon vs target at motion frames')
    print('frames:', pidx)

#%% Diagnostics — pix PCs with the motion split shaded + target-vs-prediction traces
# (a) pix PC1/PC2 over time with HIGH-motion frames shaded (the median split the R²-hi/lo
#     uses) → see what "high/low motion" actually labels. (b) true target PCs (blue) vs the
#     OOF prediction (red) for `diag_set`, high-motion shaded → see where the fit tracks.
if 1:
    diag_set = 'mag'            # which feature set's OOF predictions to overlay
    diag_pcs = (0, 1, 2)        # target movie-PCs to show true vs pred

    # f_plot_motion_traces([built_blocks['pix']['X'][:, 0], built_blocks['pix']['X'][:, 1]],
    #                      motion, labels=['pix PC1', 'pix PC2'],
    #                      suptitle=f'{dset_tag} — pix PCs + high-motion shading (median split)')

    f_plot_pred_overlay(results[diag_set]['res'], cols=diag_pcs, motion=motion,
                        suptitle=f'{dset_tag} — target PCs: true (blue) vs {diag_set} pred (red)')

#%% Diagnose R² — why the headline goes negative while the trace looks good
# Per-fold vs POOLED R², per-fold train-vs-test (overfit), and per-target-PC pooled R².
# Expect: pooled ≫ per-fold-mean, and one/few folds catastrophically negative because slow
# high-variance PCs barely vary within a held-out block (tiny denominator).
if 1:
    f_recon_diagnose(results['meng']['res'], n_show=100,
                     title=f"{dset_tag} — 'mag' R² breakdown")

#%% Save movies to inspect in ImageJ — LOW-RANK target (+ smoothed target, + raw render)
# Writes multi-page TIFFs. Low-rank + smoothed are scaled by the SAME range (fair compare);
# raw is already 0-255. Bump n_target_pcs / cutoff_cpd above and rebuild to see the ceiling
# change. Set save_range to bound file size (full T can be large).
if 1:
    out_dir    = 'F:/test_mov'
    save_range = (0, 3000)
    _idx = list(range(*save_range))
    _ftag = '_'.join(dset_tag.split())

    _lr = f_target_lowrank_at(target, _idx)          # rank-n_target_pcs reconstruction
    _sm = np.asarray(target['movie'])[_idx]          # full smoothed target
    _lo, _hi = float(np.nanmin(_sm)), float(np.nanmax(_sm))
    _u8 = lambda m: np.clip((np.asarray(m, float) - _lo) / (_hi - _lo + 1e-9) * 255, 0, 255).astype('uint8')

    f_save_mon_movie(_u8(_lr), f'{out_dir}/lowrank_{n_target_pcs}pc_{_ftag}.tif', gap_px=0)
    f_save_mon_movie(_u8(_sm), f'{out_dir}/target_smoothed_{_ftag}.tif', gap_px=0)
    if two_mon_frames is not None:
        f_save_mon_movie(np.clip(np.asarray(two_mon_frames)[_idx], 0, 255).astype('uint8'),
                         f'{out_dir}/raw_render_{_ftag}.tif', gap_px=0)
    print(f'saved lowrank/smoothed{"/raw" if two_mon_frames is not None else ""} '
          f'({len(_idx)} frames) → {out_dir}')
