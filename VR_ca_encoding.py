# -*- coding: utf-8 -*-
"""
Encoding arm (TODO #57) — predict each neuron's deconvolved firing-rate proxy
from stimulus + self-motion features. v1 = minimal validation pass: per-neuron
blocked-CV ridge with a short causal FIR, vs a per-cell circular-shift null.

Run on a DARK-condition session (objects on black → objects-only render ≈ true
stimulus). Compute helpers in functions/f_encoding.py. See PROJECT_GOAL.md /
REVIEW_2026-06 P1-1.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import os
# cap BLAS/MKL threads so the non-parallel fits (encode, importance) don't peg
# every core; MUST be set before numpy imports. None = default (all cores).
_thread_cap = 6
if _thread_cap:
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS'):
        os.environ[_v] = str(_thread_cap)

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn
patch_sklearn()
# sklearnex's fast Euclidean KNN ignores the redundant p=2 Minkowski kwarg and
# warns every call — mute just that one so other warnings stay visible.
warnings.filterwarnings('ignore', message=r'Some metric_kwargs have been passed',
                        category=UserWarning)

from f_utils import f_load_caim_data, f_gauss_smooth
from f_functions import (f_load_bh_data_all, f_proc_movement, f_proc_lick_rew,
                         f_comp_FOV_adj, f_add_phase, f_get_monitor_coords,
                         f_angles_to_movie_v2)
from f_pipeline import f_load_session, f_acquire_features, f_build_features
from f_feature_helpers import build_feature_blocks
from f_visual_features import build_visual_blocks, f_gabor_bank, f_gabor_features
from f_decoding import (f_imaging_fs, f_plot_pred_scatter, f_plot_pred_traces,
                        f_shuffle_neural)
from f_encoding import (f_lags_from_ms, f_encode_per_neuron, f_encode_null,
                        f_encode_summary, f_encode_block_importance,
                        f_encode_latency_scan, f_encode_latency_stats,
                        f_encode_null_dist, f_encode_signif)

#%% Config — mouse, paths, object size
mouse_id = 'L'
n_dset   = 4        # ← pick a DARK-condition session (index into do_proc==1 rows)
data_dir = 'F:/VR/data_proc/' + mouse_id
data_dir_bh = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'   # dataset table (f_load_session reads it)

obj_size = {'x': 5, 'y': 20, 'z': 5, 'height': 2}

# --- Feature source: BUILD (render + extract) or LOAD a cached features file ---
# Gates the render + build below (f_acquire_features). Neural (Y_enc) always from
# data_ca. mov_data / vec_data stay live (f_load_session) either way.
load_cached_features = False     # True = load a cache instead of render + build
features_dir = 'F:/VR/features/' + mouse_id
feature_file = None     # exact/substring cache filename; None → fall through
feature_tag  = None     # filename containing this unique tag string; None → index
feature_idx  = 0        # position in the dataset's sorted file list
save_features = False    # write the built features to a cache
save_movie    = False    # also store raw two_mon_frames (large; for the viz cells)
save_tag      = 'v1'     # unique tag baked into the saved filename

#%% Load calcium + behavior + cheap vectorization (f_load_session)
# Shared prelude: calcium + behavior, est1, mov_data/lr_data/vec_data_l/r, clock.
# Neural target (Y_enc) + render stay below.
session = f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx)
data_ca      = session['data_ca']
bh_data      = session['bh_data']
est1         = session['est1']
cam_params   = session['cam_params']
mov_data     = session['mov_data']
lr_data      = session['lr_data']
mov_data_eye = session['mov_data_eye']
theta        = session['theta']
mon_l_phi    = session['mon_l_phi']
mon_r_phi    = session['mon_r_phi']
vec_data_l   = session['vec_data_l']
vec_data_r   = session['vec_data_r']
pulse_delay  = session['pulse_delay']
beh_t        = session['beh_t']
frame_t      = session['frame_t']

#%% Neural target (deconvolved firing-rate proxy, native cell order)
dset_tag = f"{mouse_id} dset{n_dset}  {est1['dset_name']}"

S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)   # ~100 ms smooth (the rate proxy)
_rowmax = np.max(S_sm, axis=1)[:, None]
S_smn = S_sm / np.where(_rowmax == 0, 1.0, _rowmax)  # per-cell peak=1 (zero-row safe)
Y_enc_full = S_smn.T.astype(np.float32)              # (T, n_cells); NO hclust sort

# cell subset: None=all (ridge multi-output, per-cell r is free), int N=N best-active, or list
enc_cell_subset = None
if enc_cell_subset is None:
    enc_cell_ids = np.arange(Y_enc_full.shape[1])
elif np.isscalar(enc_cell_subset):
    enc_cell_ids = np.argsort(S_sm.std(axis=1))[::-1][:int(enc_cell_subset)]
    enc_cell_ids = np.sort(enc_cell_ids)
else:
    enc_cell_ids = np.asarray(enc_cell_subset)
Y_enc = Y_enc_full[:, enc_cell_ids]                  # columns = enc_cell_ids (orig ids)
print(f'{dset_tag}: Y_enc {Y_enc.shape}  (T, n_cells)'
      + ('' if enc_cell_subset is None else f'   [SUBSET of {Y_enc_full.shape[1]}]'))

#%% Build feature blocks (agg + self_mot + beh + motion + pix) — or load a cache
# The render (f_angles_to_movie_v2) now lives in f_pipeline.f_build_features, called
# from the junction below; these knobs feed it via cfg.
num_samp         = 101
side             = 'both'        # 'L' | 'R' | 'both'
agg_mode         = 'angular'
obj_collapse     = 'salience_mean'
nan_absent       = True
n_pix_pca        = 50
motion_smooth_sigma      = 2.0
self_motion_smooth_sigma = 2.0
lick_reward_smooth_sigma = 0.0
# visual-cortex feature banks (Tier 1 grid / 2 flow / 3 gabor / 4a meng); merged in.
# gabor stats: magnitude + phase (cos/sin φ). n_pca as a DICT → SEPARATE PCA per
# channel group (so phase's small [-1,1] scale isn't washed out by magnitude):
# prefix 'gabM' = magnitude, 'gabP' = both phase channels. Each value = int #comps
# OR float in (0,1) = variance fraction (e.g. 'gabP': 0.9).
# meng = direction+speed motion energy (opponent). Add 'grid'/'flow' entries to taste.
visual_specs = {
    'gabor': {'grid': (6, 6), 'n_orient': 4, 'wavelengths': (6, 12),
              'stats': ('magnitude', 'phase'),
              'n_pca': {'gabM': 20, 'gabP': 10}},
    'meng':  {'grid': (6, 6), 'n_orient': 4, 'wavelengths': (8,),
              'temporal_freqs': (0.15,), 'output': 'opponent', 'n_pca': 20},
}

# cfg = render+build config (gen_config provenance AND f_build_features input).
cfg = {
    'side': side, 'agg_mode': agg_mode, 'obj_collapse': obj_collapse,
    'nan_absent': nan_absent, 'n_pix_pca': n_pix_pca,
    'build_motion': True, 'motion_smooth_sigma': motion_smooth_sigma,
    'build_self_motion': True, 'self_motion_smooth_sigma': self_motion_smooth_sigma,
    'build_lick_reward': True, 'lick_reward_smooth_sigma': lick_reward_smooth_sigma,
    'build_visual': True, 'visual_specs': visual_specs,
    'cam_params': cam_params, 'obj_size': obj_size, 'num_samp': num_samp,
}
_meta = {'n_dset': n_dset, 'dset_name': est1['dset_name'],
         'n_cells': int(Y_enc_full.shape[1]), 'n_frames': int(frame_t.size)}

# The junction: build (render+extract via f_build_features, +optional save) OR load
# a cache by file/tag/index.
built_blocks, frame_t, side_tags, _movie = f_acquire_features(
    'cache' if load_cached_features else 'build', est1['dset_name'],
    build_fn=lambda: f_build_features(session, cfg),
    features_dir=features_dir, mouse_id=mouse_id,
    cache_name=feature_file, cache_tag=feature_tag, cache_idx=feature_idx,
    save=save_features, save_tag=save_tag, save_movie=save_movie,
    gen_config=cfg, meta=_meta)

fs_img = f_imaging_fs(frame_t)
print(f'imaging fs ≈ {fs_img:.1f} Hz')

# Reconstruct the per-side movies for the viz cells (None in load mode w/o save_movie).
mon_movies = None
if _movie is not None:
    two_mon_frames = _movie
    _W = two_mon_frames.shape[2] // 2
    left_mon_frames, right_mon_frames = two_mon_frames[:, :, :_W], two_mon_frames[:, :, _W:]
    mon_movies = [{'L': left_mon_frames, 'R': right_mon_frames}[t] for t in side_tags]

print('blocks:', {k: built_blocks[k]['X'].shape for k in built_blocks})

#%% Gabor bank sanity — kernels + one frame's pooled magnitude map
if 1:
    gb_grid, gb_nori, gb_wav = (6, 6), 4, (6, 12)
    bank = f_gabor_bank(n_orient=gb_nori, wavelengths=gb_wav)
    fig, axes = plt.subplots(gb_nori, len(gb_wav), figsize=(2.2*len(gb_wav), 2.2*gb_nori),
                             squeeze=False)
    for k in bank:
        ax = axes[k['io']][k['isf']]
        ax.imshow(k['even'], cmap='gray'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"θ={np.degrees(k['theta']):.0f}°  λ={k['wavelength']:.0f}px", fontsize=8)
    fig.suptitle('Gabor bank (even/cos kernels)'); fig.tight_layout()

    # raw frame vs per-orientation pooled magnitude at one time
    if mon_movies is None:
        raise RuntimeError('raw monitor movie not in memory — this sanity cell needs '
                           'the rendered frames. Run in build mode, or load a cache '
                           'written with save_movie=True.')
    mv = mon_movies[0]
    t0 = mv.shape[0] // 2
    Xg, ng = f_gabor_features(mv[max(0, t0-1):t0+1], grid=gb_grid, n_orient=gb_nori,
                              wavelengths=gb_wav)
    eng = Xg[-1].reshape(gb_grid[0], gb_grid[1], gb_nori, len(gb_wav))
    fig, ax = plt.subplots(1, gb_nori + 1, figsize=(2.6*(gb_nori+1), 2.8))
    ax[0].imshow(mv[t0], cmap='gray'); ax[0].set_title(f'{side_tags[0]} frame t={t0}')
    ax[0].set_xticks([]); ax[0].set_yticks([])
    for io in range(gb_nori):
        ax[io+1].imshow(eng[:, :, io, 0], cmap='magma')   # SF index 0
        ax[io+1].set_title(f'θ={180*io/gb_nori:.0f}° magnitude', fontsize=8)
        ax[io+1].set_xticks([]); ax[io+1].set_yticks([])
    fig.suptitle(f'{dset_tag}  —  Gabor magnitude (SF {gb_wav[0]}px) pooled {gb_grid}')
    fig.tight_layout()

#%% Encode — per-neuron CV ridge (real) vs per-cell circshift null
# Short causal FIR: features over a small lag window predict the rate proxy.
enc_blocks      = ['self_mot', 'beh', 'agg', 'pix', 'gabor', 'meng']
# 'gabor' = V1 oriented-edge energy; 'meng' = motion energy (direction+speed).
# Both built in the build cell's visual_specs (add 'phase' to gabor stats for
# velocity-via-FIR; add 'grid'/'flow' specs for the cheaper tiers).
# available now: 'agg' 'pix' 'motion' 'self_mot' 'beh' + visual 'gabor' 'meng'
# need build flag: 'pix_mot' 'per_obj' (enable in build_feature_blocks above)
enc_lags_ms     = [0, 100, 200, 300]           # → frames via fs (rise→peak window)
enc_embargo_sec = 0.5
enc_decoder     = 'ridge'      # 'ridge' | 'ridgecv' | 'knn' | 'pls' | 'cca'
enc_alpha       = 1.0          # ridge penalty (decoder='ridge')
enc_n_components = 20          # pls/cca joint latent rank (auto-clamped per fold)
enc_n_folds     = 5
enc_shuffle_seed = 42
enc_n_jobs      = -2           # scan-lag cores: -1 all, -2 all-but-one, N = N cores, 1 = serial

lags = f_lags_from_ms(enc_lags_ms, fs_img)
embargo_fr = int(round(enc_embargo_sec * fs_img))
_mkw = dict(embargo=embargo_fr, n_folds=enc_n_folds, decoder=enc_decoder,
            ridge_alpha=enc_alpha, n_components=enc_n_components)
print(f'lags(ms)={enc_lags_ms} → frames={lags};  embargo={embargo_fr} fr;  decoder={enc_decoder}')

real = f_encode_per_neuron(built_blocks, enc_blocks, Y_enc, lags, **_mkw)
null = f_encode_null(built_blocks, enc_blocks, Y_enc, lags,
                     np.random.default_rng(enc_shuffle_seed), **_mkw)
summ = f_encode_summary(real, null)
print(f"blocks={enc_blocks}  design={real['n_design']}d  "
      f"median r real={summ['median_r_real']:+.3f} null={summ['median_r_null']:+.3f}  "
      f"frac>null95={summ['frac_above_null']:.1%}")

#%% Speed profile — time the unit fits, project each downstream cell's cost
# Ridge fits all cells in ONE multi-output solve, so cost ≈ n_folds * n_lags *
# n_blocks * (1+n_shuf), NOT cell count. Times one full-design and one single-lag
# fit, then projects the latency-scan cells from those unit costs.
if 0:
    import time
    def _t(fn, *a, **k):
        t0 = time.perf_counter(); fn(*a, **k); return time.perf_counter() - t0

    t_full = _t(f_encode_per_neuron, built_blocks, enc_blocks, Y_enc, lags, **_mkw)
    t_1lag = _t(f_encode_per_neuron, built_blocks, enc_blocks, Y_enc, [0], **_mkw)
    t_blk  = {b: _t(f_encode_per_neuron, built_blocks, [b], Y_enc, [0], **_mkw)
              for b in enc_blocks}
    nL_scan = len(range(-4000, 4001, 200))            # match the scan cell
    nL_pb   = len(range(-300, 401, 50))               # match the per-block cell
    proj = {
        'encode real+null (multi-lag x2)':      2 * t_full,
        'latency scan (combined)':              nL_scan * t_1lag,
        'individual-cell (per-block,real+2null)': sum(t_blk.values()) * 3 * nL_scan,
        'per-block latency (real+5null)':       sum(t_blk.values()) * 6 * nL_pb,
        'block importance (1+2*n_blk fits)':    (1 + 2 * len(enc_blocks)) * t_full,
    }
    print(f'\nunit fit times (s):  full-design={t_full:.3f}  single-lag={t_1lag:.3f}')
    print('  per-block single-lag:', {b: round(v, 3) for b, v in t_blk.items()})
    print('projected cell cost (s), slowest first:')
    for name, sec in sorted(proj.items(), key=lambda kv: -kv[1]):
        print(f'  {sec:8.1f}   {name}')

#%% Diagnostics — per-neuron r distribution (real vs null) + sanity gates
if 1:
    r, rn = real['r_per_cell'], null['r_per_cell']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-0.3, max(0.6, np.nanmax(r)), 50)
    ax[0].hist(rn, bins=bins, alpha=0.5, label='null', color='gray', density=True)
    ax[0].hist(r,  bins=bins, alpha=0.6, label='real', color='C0', density=True)
    ax[0].axvline(summ['null_thresh'], color='k', ls=':', label='null 95th')
    ax[0].set_xlabel('per-neuron OOF r'); ax[0].set_ylabel('density')
    ax[0].legend(fontsize=8); ax[0].set_title(f'{dset_tag}\n{enc_blocks}')
    ax[1].plot(np.sort(r)[::-1], label='real')
    ax[1].plot(np.sort(rn)[::-1], label='null', color='gray')
    ax[1].set_xlabel('cell rank'); ax[1].set_ylabel('OOF r'); ax[1].legend(fontsize=8)
    ax[1].set_title(f"frac>null95 = {summ['frac_above_null']:.1%}")
    fig.tight_layout()

#%% Per-cell significance — circshift null distribution (FDR + null-referenced z)
# Statistical test in place of a noise ceiling: N circshift shuffles → per-cell p
# (one-sided), Benjamini-Hochberg FDR, and z = (r − null_mean)/null_std (ceiling-free
# effect size). Rank opto candidates by z among FDR-significant cells.
if 1:
    n_shuffles = 10
    nd = f_encode_null_dist(built_blocks, enc_blocks, Y_enc, lags,
                            np.random.default_rng(enc_shuffle_seed),
                            n_shuffles=n_shuffles, n_jobs=enc_n_jobs, **_mkw)
    sig = f_encode_signif(real, nd, alpha=0.05)
    _orig = enc_cell_ids[np.where(real['keep_cells'])[0]]
    print(f"{sig['n_sig']}/{sig['r'].size} cells significant (FDR<0.05, N={n_shuffles})")
    order = np.argsort(np.nan_to_num(sig['z'], nan=-np.inf))[::-1]
    print('top cells by null-referenced z:')
    for c in order[:10]:
        print(f"  cell{_orig[c]:<5d}  r={sig['r'][c]:+.3f}  z={sig['z'][c]:+.1f}  "
              f"q={sig['q'][c]:.1e}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(sig['z'][np.isfinite(sig['z'])], bins=40, color='C0', alpha=0.8)
    ax[0].axvline(0, color='gray', lw=0.8)
    ax[0].set_xlabel('null-referenced z'); ax[0].set_ylabel('# cells')
    ax[0].set_title('encoding z vs circshift null')
    nlq = -np.log10(np.clip(sig['q'], 1e-12, 1.0))
    sc = ax[1].scatter(sig['r'], nlq, s=10, c=sig['sig'], cmap='coolwarm', vmin=0, vmax=1)
    ax[1].axhline(-np.log10(0.05), color='k', ls=':', lw=0.8)
    ax[1].set_xlabel('OOF r'); ax[1].set_ylabel('−log10 FDR q')
    ax[1].set_title(f"volcano  ({sig['n_sig']} sig, FDR<0.05)")
    o = np.argsort(sig['r'])[::-1]
    xr = np.arange(o.size)
    ax[2].plot(xr, sig['r'][o], color='C0', label='real')
    ax[2].plot(xr, sig['null_mean'][o], color='gray', lw=0.8, label='null mean')
    ax[2].fill_between(xr, sig['null_mean'][o] - 2 * sig['null_std'][o],
                       sig['null_mean'][o] + 2 * sig['null_std'][o],
                       color='gray', alpha=0.3, label='null ±2σ')
    ax[2].set_xlabel('cell rank (by r)'); ax[2].set_ylabel('OOF r')
    ax[2].legend(fontsize=8); ax[2].set_title('real vs null (sorted)')
    fig.suptitle(f'{dset_tag}  —  per-cell encoding significance  ({enc_decoder}, {enc_blocks})')
    fig.tight_layout()

#%% Latency scan — encoding performance vs lag (single-lag models)
# Fits one model per lag (only that lag in the design). Population curve shows
# where the drive concentrates; per-cell argmax = preferred encoding latency.
if 1:
    # range × #blocks × #shuffles drives ALL scan-cell cost; physio range ~ −300..+600
    scan_lags_ms = list(range(-4000, 4001, 200))            # n_lags = len(this)
    scan_blocks  = enc_blocks    # SPEED LEVER: set ['self_mot','beh','agg'] to drop pix (~100 PCs)
    scan_n_folds = 3             # fewer folds for the diagnostic curve (5→3 ≈ 1.6x faster)
    scan_lags_fr = f_lags_from_ms(scan_lags_ms, fs_img)
    _skw = {**_mkw, 'n_folds': scan_n_folds}
    scan = f_encode_latency_scan(built_blocks, scan_blocks, Y_enc, scan_lags_fr,
                                 n_jobs=enc_n_jobs, **_skw)
    lag_ms = np.array(scan['lags']) / fs_img * 1000.0
    R = scan['r_by_lag']                              # (n_lags, n_kept)
    Rf = np.where(np.isfinite(R), R, -np.inf)
    med = np.nanmedian(R, axis=1)
    q25, q75 = np.nanpercentile(R, [25, 75], axis=1)
    q90 = np.nanpercentile(R, 90, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].fill_between(lag_ms, q25, q75, alpha=0.2, label='IQR')
    ax[0].plot(lag_ms, med, '-o', label='median r')
    ax[0].plot(lag_ms, q90, '--', color='C1', label='90th pct')
    ax[0].axhline(0, color='gray', lw=0.6)
    ax[0].set_xlabel('latency (ms)'); ax[0].set_ylabel('per-cell OOF r')
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'{dset_tag}\nencoding vs latency  {enc_blocks}  ({enc_decoder})')

    best_lag = lag_ms[Rf.argmax(axis=0)]; maxr = Rf.max(axis=0)  # per-cell preferred
    good = maxr > 0.05
    ax[1].hist(best_lag[good], bins=lag_ms.size, color='C0', alpha=0.7)
    ax[1].set_xlabel('preferred latency (ms)')
    ax[1].set_ylabel('# cells (max r>0.05)')
    ax[1].set_title(f'per-cell preferred latency  ({int(good.sum())} cells)')
    fig.tight_layout()
    print('latency(ms):', np.round(lag_ms, 0).astype(int))
    print('median r   :', np.round(med, 3))

#%% Top cells by peak encoding r — overlay performance vs lag (combined model)
# Rank kept cells by best r over lags, overlay each one's r-vs-lag curve so their
# latency profiles compare directly. Marker = each cell's preferred lag / peak r.
if 1:
    st = f_encode_latency_stats(scan)
    _orig = enc_cell_ids[np.where(scan['keep_cells'])[0]]   # kept-idx → original id
    lag_ms = np.array(scan['lags']) / fs_img * 1000.0
    R = scan['r_by_lag']                              # (n_lags, n_kept)
    n_top = 8
    top = np.argsort(st['r_pref'])[::-1][:n_top]      # kept indices, best first

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(top)))
    for c, col in zip(top, cmap):
        pl = st['lag_pref'][c] / fs_img * 1000.0
        ax.plot(lag_ms, R[:, c], '-', color=col, lw=1.4,
                label=f"cell{_orig[c]}  r={st['r_pref'][c]:+.2f}@{pl:.0f}ms")
        ax.plot(pl, st['r_pref'][c], 'o', color=col, ms=5)
    ax.axvline(0, color='gray', lw=0.6); ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('lag (ms)   ←  activity LEADS feature      |      activity LAGS feature  →')
    ax.set_ylabel('per-cell OOF r')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f'{dset_tag}\ntop {n_top} cells by peak r — performance vs lag  {enc_blocks}')
    fig.tight_layout()
    print('top cells (orig id, peak r, pref lag ms):')
    for c in top:
        print(f'  cell{_orig[c]:<5d}  r={st["r_pref"][c]:+.3f}  lag={st["lag_pref"][c]/fs_img*1000:+.0f} ms')

#%% Top cells — latency split by input type (one panel per enc block)
# Same top cells as above (ranked by COMBINED r), but per block: each panel overlays
# those cells' r-vs-lag using ONLY that block's features, so you see which input type
# drives each cell at which lag. Runs len(enc_blocks) full-cell scans (real only).
if 1:
    st = f_encode_latency_stats(scan)
    _orig = enc_cell_ids[np.where(scan['keep_cells'])[0]]
    lag_ms = np.array(scan['lags']) / fs_img * 1000.0
    n_top = 8
    top = np.argsort(st['r_pref'])[::-1][:n_top]          # ranked by combined r

    # per-block r-vs-lag for all kept cells (real only). _skw = diagnostic folds (faster);
    # pix (~100 PCs) is the costly block — drop it from enc_blocks here if too slow.
    pb_real = {b: f_encode_latency_scan(built_blocks, [b], Y_enc, scan['lags'],
                                        n_jobs=enc_n_jobs, **_skw)['r_by_lag']
               for b in enc_blocks}

    nb = len(enc_blocks); ncol = min(3, nb); nrow = int(np.ceil(nb / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.4*nrow),
                             squeeze=False, sharex=True, sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(top)))
    for bi, b in enumerate(enc_blocks):
        ax = axes[bi // ncol][bi % ncol]
        Rb = pb_real[b]
        for c, col in zip(top, cmap):
            ax.plot(lag_ms, Rb[:, c], '-', color=col, lw=1.2,
                    label=f'cell{_orig[c]}' if bi == 0 else None)
        ax.axvline(0, color='gray', lw=0.6); ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(b, fontsize=10)
        if bi % ncol == 0:
            ax.set_ylabel('per-cell OOF r')
    for j in range(nb, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.supxlabel('lag (ms)   ←  activity LEADS feature   |   activity LAGS feature  →')
    fig.suptitle(f'{dset_tag}  —  top {n_top} cells, latency per input type  ({enc_decoder})')
    fig.tight_layout()

#%% Individual-cell latency profiles — r vs lag, split by feature block + shuffle null
# Per top cell: one real curve per feature block (so you see whether a cell is driven by
# vision (+lag) vs self-motion (−lag)), plus a per-block circshift-null curve (dashed).
# Runs len(blocks)*(1+ind_n_shuf) full-cell scans over scan['lags'] (reused across panels).
if 1:
    st = f_encode_latency_stats(scan)
    _orig = enc_cell_ids[np.where(scan['keep_cells'])[0]]   # kept-idx → original cell id
    lag_ms = np.array(scan['lags']) / fs_img * 1000.0
    step_ms = float(np.median(np.diff(lag_ms))) if lag_ms.size > 1 else 1.0
    ind_n_shuf = 2                                     # shuffle repeats for per-block null
    ind_rng = np.random.default_rng(enc_shuffle_seed + 1)

    # per-block real + null r-vs-lag for ALL kept cells (index panels into these).
    # _skw = diagnostic folds (faster); pix (~100 PCs) is the costly block.
    pb_real = {b: f_encode_latency_scan(built_blocks, [b], Y_enc, scan['lags'],
                                        n_jobs=enc_n_jobs, **_skw)['r_by_lag'] for b in enc_blocks}
    pb_null = {b: np.nanmedian([f_encode_latency_scan(
                   built_blocks, [b], f_shuffle_neural(Y_enc, ind_rng), scan['lags'],
                   n_jobs=enc_n_jobs, **_skw)['r_by_lag'] for _ in range(ind_n_shuf)], axis=0)
               for b in enc_blocks}

    ind_cells = 'top'                                  # 'top' or list of ORIGINAL ids
    n_ind = 9
    if ind_cells == 'top':
        sel = list(np.argsort(st['r_pref'])[::-1][:n_ind])
    else:
        _id2k = {o: k for k, o in enumerate(_orig)}
        sel = [_id2k[c] for c in ind_cells if c in _id2k]

    ncol = 3; nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8*ncol, 2.6*nrow), squeeze=False,
                             sharex=True)
    for ax, k in zip(axes.ravel(), sel):
        for ci, b in enumerate(enc_blocks):
            ax.plot(lag_ms, pb_real[b][:, k], '-', color=f'C{ci}', lw=1.3,
                    label=b if k == sel[0] else None)
            ax.plot(lag_ms, pb_null[b][:, k], ':', color=f'C{ci}', lw=0.7, alpha=0.6)
        ax.axvline(0, color='gray', lw=0.5); ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(f"cell{_orig[k]}  r={st['r_pref'][k]:+.2f} "
                     f"prom={st['prominence'][k]:+.2f}", fontsize=8)
    for ax in axes.ravel()[len(sel):]:
        ax.axis('off')
    axes.ravel()[0].legend(fontsize=7, title='block (— real, ⋯ null)')
    fig.suptitle(f'{dset_tag}\nindividual-cell latency profiles by block  '
                 f'(x=lag ms, y=OOF r)')
    fig.tight_layout()

#%% Preference-strength summary — prominence / benefit / width across cells
if 1:
    st = f_encode_latency_stats(scan)
    lag_ms = np.array(scan['lags']) / fs_img * 1000.0
    step_ms = float(np.median(np.diff(lag_ms))) if lag_ms.size > 1 else 1.0
    pref_r_thr = 0.05
    g = st['r_pref'] > pref_r_thr                      # well-encoded cells only
    pref_lag = st['lag_pref'][g] / fs_img * 1000.0
    prom, ben = st['prominence'][g], st['benefit'][g]
    wid = st['width'][g] * step_ms

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(prom[np.isfinite(prom)], bins=30, color='C1', alpha=0.7)
    ax[0, 0].set_xlabel('prominence (peak − median r)'); ax[0, 0].set_ylabel('# cells')
    ax[0, 0].set_title(f'prominence  (med {np.nanmedian(prom):+.3f})')
    ax[0, 1].hist(ben[np.isfinite(ben)], bins=30, color='C2', alpha=0.7)
    ax[0, 1].set_xlabel('predictive benefit (r_pref − r@0)'); ax[0, 1].set_ylabel('# cells')
    ax[0, 1].set_title(f'benefit  (med {np.nanmedian(ben):+.3f})')
    ax[1, 0].hist(wid[np.isfinite(wid)], bins=30, color='C4', alpha=0.7)
    ax[1, 0].set_xlabel('tuning width (ms, half-max)'); ax[1, 0].set_ylabel('# cells')
    ax[1, 0].set_title(f'width  (med {np.nanmedian(wid):.0f} ms)')
    sc = ax[1, 1].scatter(pref_lag, prom, c=st['r_pref'][g], cmap='viridis', s=12)
    ax[1, 1].axhline(0, color='gray', lw=0.6); ax[1, 1].axvline(0, color='gray', lw=0.5)
    ax[1, 1].set_xlabel('preferred lag (ms)'); ax[1, 1].set_ylabel('prominence')
    ax[1, 1].set_title('where × how strongly')
    fig.colorbar(sc, ax=ax[1, 1], fraction=0.046, label='r_pref')
    fig.suptitle(f'{dset_tag}\npreference strength  {enc_blocks}  '
                 f'({enc_decoder}, {int(g.sum())} cells r>{pref_r_thr})')
    fig.tight_layout()
    print(f'pref-strength medians — prom {np.nanmedian(prom):+.3f}  '
          f'benefit {np.nanmedian(ben):+.3f}  width {np.nanmedian(wid):.0f} ms')

#%% Per-block latency — sweep negative→positive, each feature in OWN panel + shuffle null
# Visual features are causal (activity LAGS stimulus → +lag); behavioral/self-motion
# can be predictive (activity LEADS the act → −lag). One panel per block; each shows the
# real median-r-vs-lag curve plus a circshift-null band (per-cell shifts break alignment,
# preserve autocorr). Runs len(blocks)*(1+pb_n_shuf)*len(lags) fits.
if 1:
    pb_lags_ms = list(range(-300, 401, 50))           # −300..+400 ms
    pb_lags_fr = f_lags_from_ms(pb_lags_ms, fs_img)
    pb_lag_ms = np.array(pb_lags_fr) / fs_img * 1000.0
    pb_n_shuf = 5                                      # shuffle repeats for null band
    pb_rng = np.random.default_rng(enc_shuffle_seed)

    nb = len(enc_blocks)
    ncol = min(3, nb); nrow = int(np.ceil(nb / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for bi, b in enumerate(enc_blocks):
        ax = axes[bi // ncol][bi % ncol]
        med_r = np.nanmedian(
            f_encode_latency_scan(built_blocks, [b], Y_enc, pb_lags_fr,
                                  n_jobs=enc_n_jobs, **_mkw)['r_by_lag'],
            axis=1)
        nulls = np.array([
            np.nanmedian(
                f_encode_latency_scan(built_blocks, [b], f_shuffle_neural(Y_enc, pb_rng),
                                      pb_lags_fr, n_jobs=enc_n_jobs, **_mkw)['r_by_lag'], axis=1)
            for _ in range(pb_n_shuf)])
        n_lo, n_hi = np.nanpercentile(nulls, [2.5, 97.5], axis=0)
        ax.fill_between(pb_lag_ms, n_lo, n_hi, color='gray', alpha=0.3, label='null 95%')
        ax.plot(pb_lag_ms, np.nanmedian(nulls, axis=0), color='gray', lw=0.8, ls='--')
        ax.plot(pb_lag_ms, med_r, '-o', ms=3, color='C0', label='real')
        ax.axvline(0, color='gray', lw=0.8, ls=':'); ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(b, fontsize=10)
        if bi % ncol == 0:
            ax.set_ylabel('median per-cell OOF r')
    for j in range(nb, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    axes[0][0].legend(fontsize=8)
    fig.supxlabel('lag (ms)   ←  activity LEADS feature      |      activity LAGS feature  →')
    fig.suptitle(f'{dset_tag}  —  per-block latency vs circshift null  ({enc_decoder})')
    fig.tight_layout()

#%% Feature importance — which blocks drive each cell (drop-one + block-alone)
# Runs 1 + 2*len(enc_blocks) encoding fits (each a multi-output ridge CV).
if 1:
    imp = f_encode_block_importance(built_blocks, enc_blocks, Y_enc, lags, **_mkw)
    bn = imp['block_names']
    med_u = np.nanmedian(imp['unique'], axis=0)      # population median unique Δr
    med_m = np.nanmedian(imp['marginal'], axis=0)    # population median marginal r
    # per-cell winner by unique contribution (NaN-safe)
    vr = np.isfinite(imp['unique']).any(axis=1)
    win = np.full(imp['unique'].shape[0], -1)
    win[vr] = np.nanargmax(imp['unique'][vr], axis=1)
    counts = [int((win == bi).sum()) for bi in range(len(bn))]
    print('block        median_unique  median_marginal  #cells_best')
    for bi, b in enumerate(bn):
        print(f'  {b:10s}   {med_u[bi]:+.3f}         {med_m[bi]:+.3f}           {counts[bi]}')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(bn))
    ax[0].bar(x - 0.2, med_u, 0.4, label='unique Δr')
    ax[0].bar(x + 0.2, med_m, 0.4, label='marginal r')
    ax[0].set_xticks(x); ax[0].set_xticklabels(bn, rotation=30, ha='right')
    ax[0].axhline(0, color='gray', lw=0.6); ax[0].set_ylabel('median over cells')
    ax[0].legend(fontsize=8); ax[0].set_title(f'{dset_tag}\nblock importance')
    ax[1].bar(x, counts)
    ax[1].set_xticks(x); ax[1].set_xticklabels(bn, rotation=30, ha='right')
    ax[1].set_ylabel('# cells best-predicted'); ax[1].set_title('per-cell winner (unique)')
    fig.tight_layout()

#%% Sanity gate — self_mot-only (running strongly modulates mouse V1)
if 1:
    sm = f_encode_per_neuron(built_blocks, ['self_mot'], Y_enc, lags, **_mkw)
    print(f"self_mot-only: median r = {np.nanmedian(sm['r_per_cell']):+.3f}, "
          f"max r = {np.nanmax(sm['r_per_cell']):+.3f}  "
          f"({int((sm['r_per_cell'] > 0.1).sum())} cells r>0.1)")
    # If even this is ~0, the problem is upstream (alignment/data), not the encoder.

#%% OOF predicted-vs-true for the top-r cells (slope ≈ 1 = honest)
if 1:
    orig_ids = enc_cell_ids[np.where(real['keep_cells'])[0]]   # kept-index → original cell id
    top = np.argsort(np.nan_to_num(real['r_per_cell'], nan=-1))[::-1][:6]
    f_plot_pred_scatter(real['oof_true'][:, top], real['oof_pred'][:, top],
                        [f'cell{orig_ids[c]} r={real["r_per_cell"][c]:+.2f}' for c in top],
                        suptitle=f'{dset_tag}\nencoding OOF (top cells)', max_panels=6)

#%% OOF real (blue) vs predicted (red) TRACES for top cells — does it track dynamics?
# Windowed for visibility (the full OOF session is long).
if 1:
    _orig = enc_cell_ids[np.where(real['keep_cells'])[0]]
    _top  = np.argsort(np.nan_to_num(real['r_per_cell'], nan=-1))[::-1][:6]
    t_oof = frame_t[real['row_offset'] + real['idx']]   # imaging time (s) per OOF row
    trace_win_sec = 120                           # length of the window to show
    w = t_oof <= (t_oof[0] + trace_win_sec)
    f_plot_pred_traces(real['oof_true'][w][:, _top], real['oof_pred'][w][:, _top],
                       [f'cell{_orig[c]} r={real["r_per_cell"][c]:+.2f}' for c in _top],
                       t_oof[w], max_panels=6,
                       suptitle=f'{dset_tag}\nencoding OOF traces (top cells, first {trace_win_sec}s)')

#%% Top-cell feature breakdown — which blocks drive the best-reconstructed cells
# For the SAME top cells shown in the scatter/trace plots, show each block's
# per-cell unique Δr. Needs the encode cell (real) + the importance cell (imp).
if 1:
    _orig = enc_cell_ids[np.where(real['keep_cells'])[0]]
    _top  = np.argsort(np.nan_to_num(real['r_per_cell'], nan=-1))[::-1][:6]
    _bn = imp['block_names']
    U = imp['unique'][_top]        # (n_top, n_blocks) unique Δr for the top cells

    print('top cell    full_r   ' + '  '.join(f'{b[:7]:>7s}' for b in _bn) + '    best')
    for i, c in enumerate(_top):
        best = _bn[int(np.nanargmax(U[i]))] if np.isfinite(U[i]).any() else 'none'
        print(f'  cell{_orig[c]:<5d}  {real["r_per_cell"][c]:+.2f}   '
              + '  '.join(f'{u:+7.2f}' for u in U[i]) + f'    {best}')

    fig, ax = plt.subplots(figsize=(1.3*len(_bn)+3, 0.55*len(_top)+2))
    im = ax.imshow(U, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(_bn))); ax.set_xticklabels(_bn, rotation=30, ha='right')
    ax.set_yticks(range(len(_top)))
    ax.set_yticklabels([f'cell{_orig[c]} (r={real["r_per_cell"][c]:+.2f})' for c in _top])
    for i in range(len(_top)):
        for j in range(len(_bn)):
            ax.text(j, i, f'{U[i,j]:+.2f}', ha='center', va='center', color='w', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, label='unique Δr')
    ax.set_title(f'{dset_tag}\ntop-cell block importance (unique Δr)')
    fig.tight_layout()
