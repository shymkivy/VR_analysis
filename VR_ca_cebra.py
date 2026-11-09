# -*- coding: utf-8 -*-
"""
VR_ca_cebra.py

PURPOSE: Runs the reverse direction, decoding stimulus and behavior features from the neural
population, as an independent check that the same features are represented.
PIPELINE: step 3 of 3 — VR_feature_search → VR_ca_encoding → VR_ca_cebra. See SCRIPTS.md.

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

import f_utils as util
import f_analysis as ana
import f_functions as ff              # rendering/loaders (mostly used indirectly via f_pipeline)
import f_pipeline as pipe
import f_feature_source as fsrc
import f_feature_helpers as fh
import f_visual_features as vf
import f_cebra_helpers as ceb
import f_decoding as dec
import f_ensembles as ens
import f_ensemble_plots as ensp
import f_ensemble_pipeline as enspipe
import f_encoding as enc
import f_encoding_plots as encp
import f_tuning as tun
import f_conv_repo as cr

from sklearn.decomposition import PCA

import time

#%%
# ==========================================================================
# SECTION 1 · SETUP — config, load data, pick dataset, smooth + hclust-sort
# ==========================================================================
mouse_id = 'L'
n_dset   = 5        # dataset index into the do_proc==1 rows (selects est1 below)

data_dir     = 'F:/VR/data_proc/' + mouse_id    # edit this
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'   # dataset table (pipe.f_load_session reads it)

lick_col = 'red'
lick_col2 = 'lightcoral'
rew_col = 'green'
rew_col2 = 'limegreen'

# Colormap used by ALL raster plots (encoding raster + neural-cleanup raster). Choose one:
raster_cmap = 'viridis'      # 'viridis' — sequential (S_smn / non-negative scale; ca_analysis default)
                             # 'RdBu_r'  — diverging, 0-centered (for z-scored / centered data)
# Any matplotlib cmap works; diverging names (RdBu_r/coolwarm/bwr/seismic) auto-use a symmetric scale.
_RASTER_DIVERGING = ('rdbu_r', 'rdbu', 'coolwarm', 'bwr', 'seismic', 'piyg', 'prgn')

obj_size = {'x':        5,
            'y':        20, # for cylinder 20, for cube 10
            'z':        5,
            'height':   2}

src = fsrc.f_default_source(mouse_id=mouse_id)   # feature source: where-from + save (defaults inside)

# Raster color scale for the CEBRA-latent ensemble plots. ens_fix_scale=True uses fixed ens_vmin/ens_vmax (nicer,
# comparable across figures, e.g. 0..0.5 for peak-normed S); False = per-figure percentile (ens_vmax_pct).
ens_fix_scale = True
ens_vmin      = 0.0
ens_vmax      = 0.5
ens_vmax_pct  = 99   # percentile brightness clip when ens_fix_scale=False (lower = brighter)

#%% Loading data + cheap behavior/monitor vectorization (f_load_session)
# One call loads calcium + behavior, picks est1, and runs the cheap vectorization
# (mov_data / lr_data / vec_data_l/r). Shared with the other scripts via f_pipeline.
# NOT included: neural smoothing/sort (below) and the render (build cell).
session = pipe.f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx)
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
T_img        = session['T_img']

#%% Preprocessing — NEURAL only (drop silent → smooth → hclust-sort → Y_neu)
# Self-contained neural loading: builds the neural matrix Y_neu / Y_neu_raw from est1 WITHOUT any
# feature build, so you can run dim-red / ensembles / tuning on cells alone. The feature cell only
# re-narrows these when near_dist_thresh is set (it couples features + neural on one frame mask).
n_jobs = 5

est1 = session['est1']          # the selected dataset (data_ca holds only it unless load_all)
# Short tag used in every figure title: mouse + dset index + session name.
dset_tag = f"{mouse_id} dset{n_dset}  {est1['dset_name']}"

# Cleanup: drop SILENT cells (sum of activity == 0 across the session) — they carry no signal for
# dim-red / decode / encode and can't be peak-normalized (max=0 → NaN). `active_cells` (bool over
# est1['S'] rows) keeps every downstream neural array in one consistent cell order.
_S_raw = np.asarray(est1['S'], dtype=float)
active_cells = _S_raw.sum(axis=1) > 0
_n_drop = int((~active_cells).sum())
if _n_drop:
    print(f'dropping {_n_drop} silent cells (sum activity = 0); {int(active_cells.sum())}/'
          f'{active_cells.size} remain')
S_active = _S_raw[active_cells]        # (n_active, T) — the neural matrix used from here on

S_sm = util.f_gauss_smooth(S_active, sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]

hclust_data = ana.f_hclust_firing_rates(S_smn, standardize=True, metric='kl', method='average', similarity_transform='auto')   # cosine, jsd, kl  # average for jsd kl

S_smn2 = S_smn[hclust_data['res_order'],:]

# Neural matrix (T_img × N), hclust-sorted — the canonical neural, available without features.
# The feature cell re-derives these under near_mask_img when near_dist_thresh is set; here (no
# filter) they span the full session so dim-red / ensembles / tuning can run on cells alone.
Y_neu        = S_smn2.T.astype(np.float32)
Y_neu_raw    = Y_neu                        # canonical raw (neural-cleanup cell restores from this)
near_mask_img = np.ones(T_img, dtype=bool)  # no frame filter yet (feature cell may narrow it)
S_raster     = S_smn2                       # filtered plotting views (match Y_neu length)
t_img_plot   = frame_t

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
    

#%% Neural cleanup + ensembles — PARAMS (the shared ensemble config)
# =============================================================================
# SECTION 2.5 · NEURAL CLEANUP + ENSEMBLES — one config for both (f_ensemble_pipeline). Every parameter
# and its options: _ens_defaults() at the top of functions/f_ensemble_pipeline.py. Sweeps, LNO ECDF,
# NMF consensus and all ensemble plots: VR_ensembles.py.
# Override by name (num_comp=12, alpha_W=1e-3, membership='cv_consensus') or by section
# (dimred={...}, stats={...}). Unknown keys raise.
ens_cfg = enspipe.f_default_ens_config(
    # num_comp   = 15,
    # alpha_W    = 1e-4,
    # membership = 'z_thresh',        # 'z_thresh' | 'cv_consensus'
    # sig_method = 'cv',              # 'cv' | 'mp' | 'shuffle'
)
dimred_params = ens_cfg['dimred']     # cleanup modes in ASSIGN + the encode/tuning selectors use it
print(f'ensemble config {enspipe.f_ens_config_tag(ens_cfg)} | dimred params:', dimred_params)


#%% Ensembles — from the SHARED config (ens_cfg): load from ens_cache, else fit + stats + save
# Tune and inspect ensembles in VR_ensembles.py (same config → the same cached result loads here).
# Gives ens_out / ens_table (ASSIGN, encode-over-inputs, LNO scatter, tuning) + ens_targets
# {'ens_all', 'ens_sig'} → (Y (T, n), labels). NMF fit only on a cache miss or ens_force=True.
if 1:
    ens_force = False                # True = refit + overwrite even if this config is cached
    _fs = dec.f_imaging_fs(frame_t)
    _cell_ids_ens = np.flatnonzero(active_cells)[hclust_data['res_order']]
    ens_res = enspipe.f_ensembles_cached(est1, ens_cfg, src['features_dir'], mouse_id, est1['dset_name'],
                                         fs=_fs, force=ens_force, dset_tag=dset_tag)
    ens_out, ens_table = ens_res['ens_out'], ens_res['ens_table']
    ens_targets = enspipe.f_ens_targets(ens_res)
    # downstream cells index Y_neu_raw with ens_out's cell order — must be the same cells/order
    if not np.array_equal(ens_res['cell_ids'], _cell_ids_ens):
        print("WARNING: ens_cfg['neural'] gives different cells/order than this script's Y_neu_raw — "
              "ens_cells / LNO / tuning would misalign. Keep ens_cfg neural at the defaults here.")
    print(f'{int(ens_table["sig"].sum())}/{len(ens_table)} significant ensembles  '
          f'(config {ens_res["tag"]}; targets {[(k, v[0].shape) for k, v in ens_targets.items()]})')


#%% Neural cleanup — ASSIGN Y_neu from the chosen source → feeds encode/decode/CEBRA
# Runs AFTER the ensembles cell so ensemble-based sources are available. neural_source:
#   'raw'            no-op / restore the raw neural
#   'svd_recon'      low-rank reconstruction (denoised, per-cell)
#   'components'     k×T latent scores (ALL components) → operate on components, not cells
#   'reliable_cells' raw cells with LNO-R² > r_reliable
#   'ens_components' score traces of ensembles (n_ens × T)  [needs ensembles cell]
#   'ens_cells'      raw cells that are MEMBERS of ensembles (union) [needs ensembles cell]
# ens_src_which selects WHICH ensembles feed ens_components/ens_cells: 'sig' (significant only) | 'all'.
# Reassigns Y_neu (keeps Y_neu_raw; 'raw' restores). Runs BEFORE CEBRA so it also cleans CEBRA input.
if 1:
    neural_source = 'ens_components'   # raw|svd_recon|components|reliable_cells|ens_components|ens_cells
    ens_src_which = 'sig'              # ens_*: 'sig' (significant only) | 'all' (every ensemble)
    _fs = dec.f_imaging_fs(frame_t)
    if 'Y_neu_raw' not in dir():
        Y_neu_raw = Y_neu                                   # fallback if the load cell wasn't rerun
    Y_neu, _info = enspipe.f_neural_source(neural_source, Y_neu_raw, dimred_params,
                                           ens_out=ens_out if 'ens_out' in dir() else None,
                                           ens_table=ens_table if 'ens_table' in dir() else None,
                                           which=ens_src_which, fs=_fs)
    print(_info)


#%% Neural cleanup — RASTER: original raw neural vs the SELECTED input (Y_neu)
# f_plot_source_raster: plots Y_neu_raw (top) vs the assigned Y_neu (bottom) — WHATEVER neural_source
# produced (svd_recon / components / reliable_cells / ens_components / ens_cells / raw). No refit; just
# renders the data that feeds encode/decode/CEBRA. cmap = raster_cmap; norm_orig=False → raw original.
if 1:
    raster_norm_orig = False          # False → ORIGINAL in RAW non-negative scale (own scaling)
    raster_norm_sel  = False          # False → SELECTED input in RAW non-negative scale (own scaling)
    raster_colorbar  = False         # True → color scale as a SEPARATE small figure
    _ft = frame_t if len(frame_t) == Y_neu_raw.shape[0] else None   # matched clock only if unfiltered
    ensp.f_plot_source_raster(Y_neu_raw, Y_neu, source_label=neural_source,
                         normalize=dimred_params.get('normalize', 'norm_mean_std'),
                         fs=dec.f_imaging_fs(frame_t), smooth_ms=dimred_params.get('smooth_ms', 0),
                         frame_t=_ft, cmap=raster_cmap, norm_orig=raster_norm_orig,
                         norm_sel=raster_norm_sel, colorbar=raster_colorbar, dset_tag=dset_tag)


#%% Feature plan — pick which features + params (all params live in f_feature_source.py)
# =============================================================================
# SECTION 2 · FEATURES — feature plan (source + params); build/load in the next cell
# =============================================================================
# feature_list = which blocks to load. side / near_dist_thresh = run knobs. Change a feature param
# by editing the f_default_* defaults or passing kwargs below. The render knobs below feed the render
# for any origin='build' block; the render itself (f_angles_to_movie_v2 [+terrain]) runs inside
# f_resolve_features → f_build_features.
# NON-VISUAL / legacy blocks — selected by NAME here. All resolve in the SAME f_resolve_features call
# as the visual banks below (one render shared; cheap blocks need none), so mix freely.
feature_list     = [
                    'agg',                   # cheap, no render
                    'pix',                     # needs the RENDER (movie PCA) — enable if wanted
                    # 'motion', 'pix_mot',      # per-monitor object motion (cols suffixed _L / _R)
                    #'pix_mot',
                    'self_mot', 'beh',         # mouse locomotion (speed+yaw / fwd-lat-yaw) + behavior
                    #'grid', 'flow',
                    # 'gabor', 'meng',              # grid-averaged visual (proc block)
                    # 'gaborG', 'mengG',            # OLD inline grid banks (superseded by vis_sets below)
                    # 'gaborS', 'mengS', 'gaborS+mengS',   # OLD sampled/derived blocks (pre-grid design)
                    ]

# VISUAL BANKS — the SAME self-contained dict form as VR_feature_search.py's `feat_sets`, so a config
# that wins the search PASTES HERE UNCHANGED. Both sides read it through the one canonical reader
# (f_feature_source.f_resolve_feat_sets), so a dict cannot mean two different things: same reserved
# '_common' defaults, same per-entry overrides, same `kind` validation, same cell names (f_num_tag).
# f_vis_sets_to_plan is the plan-side twin of fsearch.f_feature_configs.
#   '_common'   the knobs EVERY set inherits; an entry repeating a knob OVERRIDES it (that is all a
#               variant needs — e.g. layer2_proc=[] next to a pca default is the raw twin).
#   kind        'gabor' | 'fft' | 'meng' | 'fftglobal' | 'wavelet' | 'pix'   (VALIDATED — an unknown
#               kind, or a knob the kind ignores like tf_hz on gabor / stats on meng, RAISES instead
#               of being silently dropped). 'fftglobal' = whole-frame rfft2, no patch lattice (position
#               lives in the phase). 'wavelet' = separable DWT/SWT; its 3 detail subbands ARE the
#               orientations, so odegs must be None or length 3 (family / decimate are its knobs).
#   cyc_fov     cycles per FOV → a PYRAMID (one level per SF; patch/support scale with λ). A single
#               scalar sf_cyc_deg instead → one bank.
#   stats       gabor/fft: ('magnitude',) | ('quadrature',) = SIGNED even/odd re,ro (#74 — keeps mag AND
#               phase, no division, strict superset) | ('phase',) | ('luminance',). meng uses `output`.
#   marginalize axes to POOL ⊆ {'ori','sf'/'wl','tf'}: () → one block per FILTER, ('ori',) → per SF
#               level, 'all' → one block. Each entry expands to `<name>:<cell>` blocks.
#   layer1_proc per-filter f_proc step-list | layer2_proc per-cell (same ops as cell 1's procs).
#               kind='pix' has no filter axis, so it takes ONLY these two (run back-to-back on the raw
#               movie) and ignores the axis knobs — same as f_pix_config on the search side.
# Downstream (encode cell / decode sweep) address a bank by NAME — it expands to all its cells.
# ONE entry per feature type at IDENTICAL params, so the only thing differing is the REPRESENTATION.
# marginalize='all' pools ori+sf(+tf) → each bank is ONE block '<name>:all' at 50 dims, head-to-head.
# CAVEAT on raw width (equal AFTER the pca-50, unequal before it): gabor/fft magnitude = 1 channel per
# filter, quadrature = 2 (Qe+Qo), meng = 1 per (filter × tf) so its tf axis multiplies the bank.
vis_sets = {
    '_common': dict(cyc_fov=(1, 2, 4, 8, 16), odegs=(0, 45, 90), marginalize='all',
                    layer1_proc=None, layer2_proc=[('pca', 50)]),
    'gabmag':  dict(kind='gabor', stats=('magnitude',)),    # == the search's gabmag
    'gabquad': dict(kind='gabor', stats=('quadrature',)),   # signed re,ro (#74)
    'fftmag':  dict(kind='fft', stats=('magnitude',)),      # sector-mean |F| (#73)
    'fftquad': dict(kind='fft', stats=('quadrature',)),     # Re/Im at nearest coefficient
    'mengopp': dict(kind='meng', tf_hz=(0.25, 0.5, 1.0, 2), output='opponent',
                    transform='signed_sqrt'),               # motion energy (no `stats`)
    # 'pix100': dict(kind='pix'),                           # raw-movie baseline via the same dict
}
side             = 'both'       # 'L' | 'R' | 'both'
near_dist_thresh = None      # e.g. 30 to keep only near frames; None to keep all

# Render knobs — used only for origin='build' blocks; lat/vert also feed the recon/viz cells.
# monitor render resolution: lat_samp (WIDTH) / vert_samp (HEIGHT). lat:vert = hFOV:vFOV ≈ 1.49 →
# angularly-square pixels (~0.66 deg/px both axes at 150×101; the angle→pixel map is linear).
render_lat_samp    = 150     # WIDTH  (columns)
render_vert_samp   = 101     # HEIGHT (rows)
num_samp           = render_lat_samp   # legacy square fallback (kept for downstream refs)
max_frames         = None    # cap render length for debugging; None = full session
render_verbose     = False   # per-frame progress prints from the renderers
render_terrain     = False   # add heightmap ground plane behind objects (Z-composited)
terrain_clip_len   = None    # None = cam_params['clip_len']; else override terrain draw distance
terrain_stride     = 0.5     # int ≥1 keeps every Nth cell; float <1 bilinear-oversamples
terrain_point_size = 1       # half-size of pixel patch painted per terrain sample
terrain_mode       = 'surface'  # 'points' scatter (gappy) | 'surface' watertight fill → occlusion
terrain_shade      = 'none'  # 'none' invisible occluder (occlude only) | 'gray' | 'depth' visible
terrain_gray_level = 128     # intensity when terrain_shade='gray'
terrain_close_iter = 2       # 'surface' horizontal-close passes (smooth horizon + 1px slivers)
conv_backend       = 'gpu'   # gabor/meng build convolutions: 'gpu' (torch, fast) | 'cpu' (scipy)
conv_device        = 'cuda'  # torch device for conv_backend='gpu'
feat_whiten        = False   # whiten loaded PCA feature blocks (÷√eigenvalue → unit-variance PCs)
feat_pca_variant   = None    # None | 'pcacomb' | 'pcasep' | 'pcaraw' — load a pca_feat GEN cache
                             # variant (points cache_tag at <tag>_<variant>; gaborS/mengS in feature_list
                             # then load from it via origin='auto'/'cache'). Blocks not in the cache
                             # (agg/pix/self_mot) still build. Prints the cache's stored pca_specs to verify.

# plan = what + feature params | build_cfg = render/neural params. (src = where-from is in cell 1.)
# NEW feature-building design: each block's processing is an ordered f_proc step-list (applied
# GLOBALLY at build, before any CV) — block `proc` (== grid layer2) + optional per-input `input_proc`
# (== layer1). Ops: ('norm', zscore|standardize|total_var|first_sv) | ('pca', n|frac|(frac,cap)) |
# ('whiten', True); run in list order; [] = no proc. Override a block inline, e.g.
#   feature_plan['gaborS']['proc'] = [('norm','zscore'), ('pca', 300)]      # norm THEN pca
#   feature_plan['gaborS+mengS']['proc'] = [('pca', 500)]
# NB cache-loaded gaborS/mengS carry gen-time PCA — a 'pca' step in their proc is skipped on load.
feature_plan = fsrc.f_proc_feature_plan()
# feature_plan = fsrc.f_default_feature_plan()   # OLD design (n_pca/input_norm/zscore_joint per block)
# Merge the vis_sets banks into the plan and select them. Names collide-safe: a vis_sets key overrides
# a same-named plan block. Comment out to fall back to the inline gaborG/mengG entries in feature_list.
# The plan keys are the source of truth for the names — '_common' is a defaults entry, not a block, so
# reading them back off the plan (not off vis_sets) keeps it out of feature_list.
_vis_plan = fsrc.f_vis_sets_to_plan(vis_sets)
feature_plan.update(_vis_plan)
feature_list = feature_list + [n for n in _vis_plan if n not in feature_list]
# Pyramid / spec-list block (add 'pyr' to feature_list to use). f_pyramid_specs = param generator;
# f_specs_block = build a plan block from a list of feature dicts + level-2 marginalize/layer2_proc:
# feature_plan['pyr'] = fsrc.f_specs_block(fsrc.f_pyramid_specs('gabor', cyc_fov=(1, 2, 4, 8, 16)),
#                          marginalize=('ori',), layer1_proc=[('norm', 'zscore')], layer2_proc=[('pca', 50)])
build_cfg    = fsrc.f_default_build_cfg(side=side, num_samp=num_samp,
                                   lat_samp=render_lat_samp, vert_samp=render_vert_samp,
                                   max_frames=max_frames,
                                   render_terrain=render_terrain, terrain_clip_len=terrain_clip_len,
                                   terrain_stride=terrain_stride, terrain_point_size=terrain_point_size,
                                   terrain_mode=terrain_mode, terrain_shade=terrain_shade,
                                   terrain_gray_level=terrain_gray_level, terrain_close_iter=terrain_close_iter,
                                   conv_backend=conv_backend, conv_device=conv_device)
n_pix_pca = build_cfg['n_pix_pca']          # kept top-level for downstream plot cells
# SSD reuse of the CONVOLVED+POOLED gabor/meng/fft blocks. Cached PRE-PCA/proc, keyed on the resolved
# extractor params, in the ONE place every build path funnels through (build_visual_blocks) — so a bank
# already convolved by VR_feature_search reloads here instead of re-convolving. Root defaults to the
# source features_dir; conv_autosave_tag namespaces the store (bump for a fresh one).
build_cfg['conv_autosave']     = True
build_cfg['conv_autosave_tag'] = 'auto1'
# Drop point-samples keeping less than this fraction of their filter window on screen (the rest is
# zero pad). Geometry only. None = off. MUST MATCH VR_feature_search.py — it changes the columns a
# bank emits AND is part of the autosave key, so a mismatch silently rebuilds instead of reloading.
build_cfg['edge_min_frac'] = 0.45

#%% Build / load features + assign aliases
# f_resolve_features groups blocks by origin and runs each shared op once (one render for build
# blocks, one cache load, one repo open); only feature_list is loaded. Then legacy aliases + the
# optional near-dist frame filter + sanity prints.
# Side selection for the near-dist filter + multi-obj stats (vec_data from f_load_session).
_sel_side = {'L': ([vec_data_l], ['L']), 'R': ([vec_data_r], ['R']),
             'both': ([vec_data_l, vec_data_r], ['L', 'R'])}
if side not in _sel_side:
    raise ValueError(f'unknown side: {side!r}')
vec_data_use, side_tags = _sel_side[side]

# point the cache resolver at the pca_feat variant when requested (<tag>_<variant> file)
_src = ({**src, 'cache_tag': f"{src['cache_tag']}_{feat_pca_variant}"} if feat_pca_variant else src)
built_blocks, frame_t, side_tags, _movie = fsrc.f_resolve_features(
    session, feature_plan, build_cfg, dset_name=est1['dset_name'], only=feature_list,
    render_verbose=render_verbose, whiten=feat_whiten, **_src)
if _movie is not None:        # restore movies for recon/flow viz
    two_mon_frames = _movie
    _W = two_mon_frames.shape[2] // 2
    left_mon_frames, right_mon_frames = two_mon_frames[:, :, :_W], two_mon_frames[:, :, _W:]

# Legacy aliases — downstream cells (raster, near-distance filter, CEBRA
# supervision, etc.) reference these names. Keep them in sync with the
# registry; the registry cell below is now redundant for the agg/pix blocks
# but still wraps things for combine_blocks consumers.
# agg/pix are optional now (a plan may omit them) — .get() so the load doesn't crash; downstream
# agg/pix-specific cells guard on `X_agg is not None` / `'pix' in built_blocks`.
X_agg              = built_blocks.get('agg', {}).get('X')
X_pix              = built_blocks.get('pix', {}).get('X')
X_mot              = built_blocks.get('motion',  {}).get('X')
X_per_obj          = built_blocks.get('per_obj', {}).get('X')
agg_feat_names     = built_blocks.get('agg', {}).get('names')
mot_feat_names     = built_blocks.get('motion',  {}).get('names')
per_obj_feat_names = built_blocks.get('per_obj', {}).get('names')
n_per_side         = built_blocks.get('agg', {}).get('n_per_side')
pres_cols          = built_blocks.get('agg', {}).get('pres_cols')
side_feat_names_single = built_blocks.get('agg', {}).get('side_feat_names_single')
pca_pix            = built_blocks.get('pix', {}).get('pca_pix')
n_sides            = len(side_tags)

# Optional near-distance frame filter (Option C). Build the "any near" mask on the behavior clock,
# resample to imaging clock, then apply to every feature block + the neural (Y_neu) so they stay
# aligned. The neural base (Y_neu/Y_neu_raw/S_raster) is set in the Preprocessing cell; here we
# RE-DERIVE it from the source (S_smn2) under the current mask so re-running is idempotent.
if near_dist_thresh:
    near_per_side = [(v['obj_mon_idx'] & (v['obj_dist'] < near_dist_thresh)).any(axis=1)
                     for v in vec_data_use]
    near_beh = np.any(near_per_side, axis=0).astype(np.float32)
    near_mask_img = fh.f_resample_to_imaging(beh_t, near_beh[:, None], frame_t, pulse_delay,
                                          kind='nearest').ravel() > 0.5
    kept = int(near_mask_img.sum())
    print(f'near-distance filter (d<{near_dist_thresh}): keeping {kept}/{T_img} frames'
          f' ({100*kept/T_img:.1f}%) — NB: hard cuts break temporal contiguity')
    for _bname in built_blocks:
        built_blocks[_bname]['X'] = built_blocks[_bname]['X'][near_mask_img]
    # Refresh legacy aliases to point at the filtered arrays.
    X_agg = built_blocks.get('agg', {}).get('X')
    X_pix = built_blocks.get('pix', {}).get('X')
    X_mot     = built_blocks.get('motion',  {}).get('X')
    X_per_obj = built_blocks.get('per_obj', {}).get('X')
else:
    near_mask_img = np.ones(T_img, dtype=bool)

# Neural views. This cell now runs AFTER the ASSIGN cell (which sets Y_neu to the chosen cleaned /
# ensemble source), so with NO near filter we must NOT touch Y_neu / Y_neu_raw — only refresh the
# full-length plotting views. Under a near filter (Option C), re-derive the neural under the mask so
# features + neural stay aligned; NB ASSIGN + ensembles were built on the FULL neural, so re-run the
# ensemble + ASSIGN cells afterwards to re-align (frame_t stays full so f_imaging_fs sees true spacing).
if near_dist_thresh:
    if 'ens_out' in dir() and ens_out['scores'].shape[1] != int(near_mask_img.sum()):
        print('  WARNING: near filter changed T; ens_out/Y_neu were built on full neural — '
              're-run the ensemble + ASSIGN cells.')
    Y_neu_raw  = S_smn2.T.astype(np.float32)[near_mask_img]
    Y_neu      = Y_neu_raw
    S_raster   = S_smn2[:, near_mask_img]
    t_img_plot = frame_t[near_mask_img]
else:
    S_raster   = S_smn2
    t_img_plot = frame_t

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

# Quick shape / range check before training (agg/pix printed only when present).
print(f'side={side!r}  agg_mode={build_cfg["agg_mode"]!r}  obj_collapse={build_cfg["obj_collapse"]!r}')
print('Y_neu     :', Y_neu.shape,   'min/max:', Y_neu.min(),  Y_neu.max())
if X_agg is not None:
    agg_nan_frac = float(np.isnan(X_agg).mean())
    print(f'X_agg     : {X_agg.shape}  n_per_side={n_per_side}  NaN frac={agg_nan_frac:.2%}  '
          f'min/max: {np.nanmin(X_agg):.3g} {np.nanmax(X_agg):.3g}')
if X_pix is not None:
    _pix_n_used = pca_pix.n_components_                      # PCs kept (= n_pix_pca)
    _pix_n_max  = min(pca_pix.n_samples_, pca_pix.n_features_in_)  # full pixel rank ceiling
    print(f'X_pix     : {X_pix.shape}  min/max: {X_pix.min():.3g} {X_pix.max():.3g}  '
          f'PCA exp_var(cum) {np.cumsum(pca_pix.explained_variance_ratio_)[-1]:.3f} '
          f'({_pix_n_used}/{_pix_n_max} PCs)')

# Multi-object stats per side — tells us how often the collapse matters.
for v, tag in zip(vec_data_use, side_tags):
    n_in_fov = v['obj_mon_idx'].sum(axis=1)
    any_obj  = (n_in_fov >= 1).mean()
    multi    = (n_in_fov >= 2).mean()
    max_sim  = int(n_in_fov.max()) if n_in_fov.size else 0
    print(f'  side {tag}: any-obj-in-FOV {any_obj:.1%}, multi-obj (≥2) {multi:.1%}, '
          f'max simultaneous {max_sim}')

# Feature registry summary (per-block dimensionality / NaN frac / PCA variance retained). To
# concatenate blocks use build_target_matrix / build_target_columns (f_decoding) or
# make_cebra_supervision (f_cebra_helpers).
def _block_pca_var(b):
    """Fraction of total (raw) variance the block's kept PCs capture: a float, a dict {group:frac}
    for grouped-PCA blocks (grid gabor), or None for raw (un-PCA'd) blocks."""
    p = b.get('pca_pix', b.get('pca'))
    if p is None:
        return None
    if isinstance(p, dict):
        return {g: float(pp.explained_variance_ratio_.sum())
                for g, pp in p.items() if pp is not None}
    return float(p.explained_variance_ratio_.sum()) if hasattr(p, 'explained_variance_ratio_') else None

print(f'\n{dset_tag} — feature registry (built_blocks):')
_tot_dim = 0
for name, b in built_blocks.items():
    n_nan_frac = float(np.isnan(b['X']).mean())
    _dim = b['X'].shape[1]; _tot_dim += _dim
    _raw = b.get('raw_dim')
    _v = _block_pca_var(b)
    if _v is None:
        _vstr = 'raw (no PCA)'
    elif isinstance(_v, dict):
        _vstr = 'PCA var kept ' + ', '.join(f'{g}:{f:.1%}' for g, f in _v.items())
    else:
        _vstr = f'PCA var kept {_v:.1%}' + (f' (of {_raw} raw dims)' if _raw else '')
    print(f'  {name:10s}  dim={_dim:>4}  {b["X"].shape}  NaN frac={n_nan_frac:5.2%}   {_vstr}')
print(f'  {"TOTAL":10s}  dim={_tot_dim:>4}  across {len(built_blocks)} blocks')


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
#   'flow'     — optical flow on the grid (visual, Tier 2)(build_visual + visual_specs)
#   'gabor'    — Gabor oriented-edge energy (visual, Tier 3)(build_visual + visual_specs)
#                V1-style edges × SF × location; stats magnitude/phase/even
#   'meng'     — Adelson–Bergen motion energy (visual, Tier 4a)(build_visual + visual_specs)
#                direction+speed selective; A/B vs 'flow'/'gabor'
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
    'gabor': ['gabor'], # Gabor oriented-edge energy (Tier 3) — needs build_visual=True
    'meng': ['meng'],   # motion energy, dir+speed (Tier 4a) — needs build_visual=True
    'gaborS': ['gaborS'],
    'mengS': ['mengS'],
}
# Keep only models whose supervision blocks were actually loaded (feature_list can be a subset).
cebra_supervisions = {k: v for k, v in cebra_supervisions.items()
                      if all(b in built_blocks for b in v)}

# Old hardcoded calls kept commented:
# cebra_agg = f_run_cebra(Y_neu, X_agg_z, out_dim=3, max_iter=5000)
# cebra_pix = f_run_cebra(Y_neu, X_pix_z, out_dim=3, max_iter=5000)

cebra_models = {}
for _cname, _blocks in cebra_supervisions.items():
    _sup, _sup_names = ceb.make_cebra_supervision(_blocks, built_blocks)
    cebra_models[_cname] = ceb.f_run_cebra(Y_neu, _sup, out_dim=3, max_iter=5000)
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
    
    #plt.close('all')


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
visual_color_blocks = [b for b in ('grid', 'flow', 'gabor', 'meng') if b in built_blocks]
visual_color_specs = {}
for _vb in visual_color_blocks:
    _vx = built_blocks[_vb]['X']; _vn = built_blocks[_vb]['names']
    _k = min(n_grid_plot, _vx.shape[1])
    visual_color_specs[_vb] = [(_vn[i], _vx[:, i], 'viridis') for i in range(_k)]


# Iterate over cebra_models — for each, plot one figure per coloring scheme matching that model's
# supervision (agg-supervised → agg-channel coloring; pix → pix-PC; visual blocks → their comps).
# f_plot_emb_scatter lives in f_decoding.py; pass dset_tag + plot_2d explicitly.
for _cname, _m in cebra_models.items():
    _blocks = _m.get('supervision_blocks', [])
    if 'agg' in _blocks:
        dec.f_plot_emb_scatter(_m, agg_color_specs, agg_rows,
                           f'CEBRA-{_cname} embedding  |  color: agg inputs  '
                           f'(supervised by {"+".join(_blocks)}, side={color_tag!r})',
                           dset_tag=dset_tag, plot_2d=plot_2d)
    if 'pix' in _blocks:
        dec.f_plot_emb_scatter(_m, pix_color_specs, pix_rows,
                           f'CEBRA-{_cname} embedding  |  color: top {n_pix_plot} pix-PCs  '
                           f'(supervised by {"+".join(_blocks)})',
                           dset_tag=dset_tag, plot_2d=plot_2d)
    for _vb in visual_color_blocks:
        if _vb in _blocks and visual_color_specs[_vb]:
            dec.f_plot_emb_scatter(_m, visual_color_specs[_vb], grid_rows,
                               f'CEBRA-{_cname} embedding  |  color: top '
                               f'{len(visual_color_specs[_vb])} {_vb} comps  '
                               f'(supervised by {"+".join(_blocks)})',
                               dset_tag=dset_tag, plot_2d=plot_2d)


#%% Decoder params — shared struct (single source of truth; future 'best params' home)
# =============================================================================
# SECTION 3.5 · DECODER PARAMS + GENERIC SWEEP (input × target × #PCs × score)
# =============================================================================
# f_default_decoder_params → a plain dict of every decode/CV/score setting. Edit here; the sweep
# cell below overrides ONE key at a time. Options per knob:
#   input_source   'neural' | 'cebra:<name>'    <name> = any key in cebra_models (e.g. agg, pix)
#   n_input_pcs    int                          neural PCs to decode from (ignored for cebra input)
#   neural_pca     'global' | 'within_fold'     within_fold = per-fold PCA (leak-free; disables lags)
#   target_block   'pix' | 'gaborS' | 'mengS' | 'meng'   single-PCA blocks (grid 'gabor' is skipped)
#   n_target_pcs   None | int                   None = all target cols; int = top-k target PCs
#   decoder        'ridge' | 'ridgecv' | 'knn' | 'pls' | 'cca'
#   ridge_alpha    float                        penalty (decoder ridge/ridgecv)
#   k_nn           int                          neighbors (decoder knn)
#   n_components   int                          latent rank (decoder pls/cca)
#   ridgecv_alphas None | array                 α grid (decoder ridgecv); None → logspace(-2,4,13)
#   n_folds        int                          blocked-CV folds
#   fold_scheme    'contiguous' | 'interleaved' interleaved = chop T into n_blocks small blocks
#                                               round-robin'd to folds (no fold = one quiet epoch)
#   n_blocks       None | int                   interleaved only; None → n_folds*10
#   embargo_sec    float                        purge gap (s) each side of the test block
#   lags_ms        list of ms                   decoder FIR window; [0] = instantaneous
#   standardize    bool                         z-score input cols (train-fold stats)
#   scores         subset of ['r2','r2_vw','rv','procrustes','rsa']
#                    r2 = per-dim mean | r2_vw = variance-weighted | rv = multivariate r² |"
#                    procrustes = shape (rotation/scale-free) | rsa = timepoint-RDM correlation
#   rsa_subsample  int                          timepoints sampled for the RSA RDM
#   seed           int                          RNG seed (RSA subsample)
dec_params = dec.f_default_decoder_params(
    input_source  = 'neural',    # 'neural' | 'cebra:<name>'
    n_input_pcs   = 100,          # neural PCs (input); ignored for cebra input
    neural_pca    = 'global',    # 'global' | 'within_fold' (leak-free; no lags)
    target_block  = 'gaborS',       # pix | gaborS | mengS | meng
    n_target_pcs  = None,        # None = all cols | int = top-k
    decoder       = 'cca',     # ridge | ridgecv | knn | pls | cca
    standardize   = True,
    embargo_sec   = 0.0,
    lags_ms       = list(range(-400, 101, 100)),   # [0] = instantaneous
    n_folds       = 5,
    fold_scheme   = 'interleaved',   # 'contiguous' | 'interleaved' (chop → round-robin)
    n_blocks      = None,           # interleaved: # small blocks (None → n_folds*10)
    scores        = ['r2', 'r2_vw', 'rv', 'procrustes', 'rsa'],
    rsa_subsample = 2000,
)
print('decoder params:', dec_params)


#%% Decode sweep — vary ONE param across a list, watch how the scores change
# Pick any key of dec_params as the axis + a list of values. f_decoder_sweep re-runs the decoder
# for each value; f_plot_decoder_sweep draws one panel per score (numeric axis → line vs x;
# categorical axis (block/decoder/input names) → bars). Uncomment the experiment you want:
#   sweep_var, sweep_values = 'n_target_pcs', [1, 2, 3, 5, 8, 12, 16, 24, 32, 50]  # #feature-PCs
#   sweep_var, sweep_values = 'target_block', ['pix', 'gaborS', 'mengS']           # per-feature
#   sweep_var, sweep_values = 'n_input_pcs',  [1, 3, 8, 16, 32, 50]                # #neural-PCs
#   sweep_var, sweep_values = 'input_source', ['neural', 'cebra:agg', 'cebra:pix'] # neural vs CEBRA
#   sweep_var, sweep_values = 'decoder',      ['ridge', 'knn', 'pls']              # decoder A/B
#   sweep_var, sweep_values = 'fold_scheme',  ['contiguous', 'interleaved']        # CV scheme A/B
#   sweep_var, sweep_values = 'lag_ms', list(range(-500, 501, 100))                # single-lag tuning
#     ('lag_ms' = virtual axis: decode at ONE lag; +τ uses PAST neural, i.e. neural leads feature)
if 1:
    
    sweep_var    = 'n_target_pcs'           # n_target_pcs, lag_ms
    sweep_values = [1, 2, 3, 5, 8, 12, 16, 24, 32, 50]
    #sweep_values =  list(range(-2000, 501, 100))

    # CPU cap — leave a core free (same reason as the sweep/null cells).
    from threadpoolctl import threadpool_limits
    _n_threads = max(1, (os.cpu_count() or 2) - 1)
    t0 = time.perf_counter()
    with threadpool_limits(limits=_n_threads):
        dec_rows = dec.f_decoder_sweep(dec_params, sweep_var, sweep_values,
                                   built_blocks, Y_neu, cebra_models, frame_t)
    print(f'decode sweep over {sweep_var!r} ({len(sweep_values)} pts, '
          f'≤{_n_threads}/{os.cpu_count()} threads): {time.perf_counter()-t0:.1f}s')
    dec.f_plot_decoder_sweep(dec_rows, sweep_var, dec_params['scores'], dset_tag=dset_tag,
                         config=dec_params)

    # --- CEBRA-at-matched-dims (deferred) ---------------------------------------
    # To compare CEBRA vs PCA at EQUAL dims, retrain a CEBRA per dim and feed it as the input
    # under sweep_var='n_input_pcs' logic. Not wired yet (one f_run_cebra fit per dim = minutes);
    # decide retrain-per-dim vs truncate-one-embedding when we get there. For now, sweeping
    # input_source over the existing 3-D cebra_models works.


#%% (optional) COMBINED gabor+meng conv block → single joint PCA from RAW ('convS')
# gabor magnitude ≈ motion-energy at temporal-freq ≈ 0, so they share spatial structure. This samples
# BOTH kinds' RAW responses from the conv_repo and fits ONE PCA over the concatenation → a single
# 'convS' block. Unlike feature_block=['gaborS','mengS'] + feature_pca='joint' (which re-PCAs the
# already-separately-PCA'd 200+200), this does the feature PCA in a SINGLE step from raw, so no
# per-kind low-variance tail is dropped first. Then set enc_params feature_block='convS',
# n_feature_pcs=None (this IS the only feature PCA). Re-reads the repo (sampling) — run once.
if 0:
    convS_ds        = 8                        # spatial sample stride (match gaborS/mengS)
    convS_n_pca     = 200                      # joint components to keep from RAW gabor+meng
    convS_waves     = None                     # None = all wavelengths in repo | e.g. (12, 15, 18)
    convS_odegs     = None
    convS_tfreqs    = None                     # meng temporal freqs; None = all
    _repo = cr.f_conv_repo_dir(src['features_dir'], est1['dset_name'], src.get('conv_tag'))
    _cb, _ = cr.f_load_conv_combined(_repo, convS_ds, n_pca=convS_n_pca, kinds=('gabor', 'meng'),
                                  wavelengths=convS_waves, odegs=convS_odegs, tfreqs=convS_tfreqs,
                                  label='convS', verbose=False)
    built_blocks.update(_cb)
    # align to the current feature/neural clock: the repo is FULL-length, but the other blocks / Y_neu
    # may be shorter (near_dist_thresh time filter, or max_frames). Match them so the encoder aligns.
    _T = Y_neu_raw.shape[0] if 'Y_neu_raw' in dir() else built_blocks['agg']['X'].shape[0]
    _cx = built_blocks['convS']['X']
    if _cx.shape[0] != _T:
        if 'near_mask_img' in dir() and near_mask_img.size == _cx.shape[0] and int(near_mask_img.sum()) == _T:
            built_blocks['convS']['X'] = _cx[near_mask_img]          # near-distance frame filter
        else:
            built_blocks['convS']['X'] = _cx[:_T]                    # max_frames truncation
        print(f'  convS: aligned {_cx.shape[0]} → {built_blocks["convS"]["X"].shape[0]} frames')
    _cs = built_blocks['convS']
    print(f"convS: {_cs['raw_dim']} raw gabor+meng cols -> {_cs['X'].shape[1]} joint PCs"
          + (f"  ({np.cumsum(_cs['pca'].explained_variance_ratio_)[-1]:.3f} var kept)"
             if _cs['pca'] is not None else ''))


#%% Encoder params — shared struct (feature → neural; mirror of the decoder struct)
# =============================================================================
# SECTION 3.6 · ENCODING SCAN (feature → neural) — same sweep, reverse direction
# =============================================================================
# f_default_encoder_params → a plain dict of the encoding settings. Options per knob:
#   feature_block     'pix' | 'gaborS' | 'mengS' | 'meng'   the block under test (single-PCA)
#   n_feature_pcs     None | int                            None = all cols | top-k feature PCs
#   n_neural_pcs      None | int                            None = per-neuron target | reduce neural
#                                                           target to top-k PCs (predict → inverse-
#                                                           transform → per-neuron r kept; denoised)
#   covariate_blocks  list                                  always-in predictors, e.g. ['self_mot']
#                                                           to regress in locomotion ([] = pure)
#   decoder           'ridge' | 'ridgecv' | 'elasticnet' | 'lasso' | 'huber' | 'lad' | 'knn' | 'pls' | 'cca'
#                       L1-LOSS (min |y-ŷ|, robust to outlier frames): 'lad' (pure abs-error, SLOW) |
#                       'huber' (abs-error in tails, faster). L1-PENALTY (sparse weights): 'lasso'|'elasticnet'.
#   ridge_alpha       float                                 penalty α (elasticnet/lasso strength; L2 for
#                                                            huber; L1-penalty for lad — 0 = pure LAD)
#   l1_ratio          float                                 elasticnet only: 0=L2(ridge)…1=L1(lasso)
#   n_folds / fold_scheme / n_blocks / embargo_sec          CV knobs (interleaved = chop → round-robin)
#   lags_ms           list of ms                            CAUSAL (stimulus precedes response)
#   scores            per-neuron r_med / frac_sig / r2_med  +  population r2_vw / rv / procrustes / rsa
#   r_sig             float                                 frac_sig = fraction of cells with r > r_sig
enc_params = enc.f_default_encoder_params(
    feature_block    = ['gabmag'],  # a vis_sets BANK name → expands to ALL its `<bank>:<cell>` blocks
                                    #   (concatenated). An explicit cell scores ONE level, e.g.
                                    #   ['gabmag:sf0.01016']. Legacy names still work verbatim
                                    #   (pix | gaborS | mengS | meng | gaborS+mengS), and a LIST mixes
                                    #   them, e.g. ['gabmag', 'self_mot'].
    n_feature_pcs    = 150,       # None = all cols | int = top-k
    feature_pca      = 'joint',# 'per_block' = top-k of EACH block, then lags | 'joint' = concat
                                   #   blocks → ONE PCA(k) → then lags | 'joint_postlag' = concat →
                                   #   lag-embed → PCA(k) (caps the feature×lag dimensionality)
    n_neural_pcs     = None,       # None = per-neuron | int = reduce neural target to top-k PCs
    target_smooth_ms = 0,          # Gaussian σ (ms) on the neural TARGET; 0 = OFF (raw trace, matches
                                   #   how the dim-red was done). >0 → predict the smoothed rate
                                   #   (bias-for-variance; NOT ridge — raises r, hold fixed to compare)
    roughness_lambda = 0,          # temporal OUTPUT-roughness penalty λ·‖D(Xw)‖²; 0 = OFF (plain ridge)
                                   #   >0 penalizes SPIKY predictions (forces contiguous folds)
    roughness_order  = 2,          # 1 = penalize jumps | 2 = penalize spikes / curvature
    covariate_blocks = [],         # e.g. ['self_mot'] to regress in locomotion
    decoder          = 'ridge',    # 'ridge' (fixed λ=ridge_alpha) | 'ridgecv' (auto-λ per fold) |
                                   #   'elasticnet' (L2+L1 penalty, mix via l1_ratio) | 'lasso' (L1 pen) |
                                   #   'huber' (L1-error/robust loss) | 'lad' (pure abs-error, SLOW) |
                                   #   'knn' | 'pls' | 'cca'
    ridge_alpha      = 1e3,        # ridge λ / penalty strength — LARGER = smoother, more shrinkage.
    # for ridge 1e3                #   Sweep it ([0.1,1,10,100,1000]); on lots of data λ=1 is ~OLS.
                                   #   (also elasticnet/lasso strength — but sklearn scales it
                                   #   differently, so RE-TUNE much smaller, e.g. [1e-3,1e-2,0.1,1];
                                   #   also the ε in the roughness solver.)
    l1_ratio         = 1e-4,        # decoder='elasticnet': 0 = pure L2 (ridge) … 1 = pure L1 (lasso)
    ridgecv_alphas   = None,       # decoder='ridgecv': α grid to search (None → logspace(-2,4,13))
    standardize      = True,      # z-score the FEATURE design (global + per-fold). False = keep raw
                                   #   PC-score variance (eigenvalue weighting) so ridge favors the top
                                   #   PCs. NB CCA/PLS ignore this (scale=True internally).
    embargo_sec      = 0.0,
    lags_ms          = [0, 100, 200, 300, 400, 500], # [0],   # causal , 100, 200, 300
    n_folds          = 5,
    fold_scheme      = 'interleaved',   # 'contiguous' | 'interleaved'
    n_blocks         = None,
    scores           = ['r_med', 'r2_med', 'frac_sig', 'r2_vw', 'rv', 'procrustes', 'rsa'],
    rsa_subsample    = 2000,
)
print('encoder params:', enc_params)

# end params:
    # ridge; alpha 1e-3; 150pcfeat; gamor+meng
    # elasticnet; l1 rat - 1e-4; alpha = 10; 300pcfeat


# plt.close('all')
#%% Encoding frame filter — restrict feature→neural encoding to MOVEMENT / OBJECT-ON-SCREEN frames
# Built HERE (not at load) so the ensemble / dim-red neuron→neuron part keeps FULL traces. Applied
# AFTER lag embedding inside the encoder (lags reference true adjacent frames), to BOTH train & test.
# Static gabor frames during stationary / empty-screen epochs add redundant samples that can distort
# the encoding CV — drop them for the ENCODING only. Both None = use all frames.
if 1:
    from f_feature_helpers import f_self_motion_features
    enc_move_thresh = None      # keep frames with mouse speed > this (world units/s); None = off
    enc_near_thresh = None      # keep frames with any object closer than this on screen; None = off
    _emasks = []
    if enc_move_thresh is not None:
        _spd = f_self_motion_features(mov_data, beh_t=frame_t)[:, 0]        # (T_img,) linear speed
        _emasks.append(np.asarray(_spd, float) > float(enc_move_thresh))
        print(f'  speed: median={np.median(_spd):.3f} p90={np.percentile(_spd,90):.3f} '
              f'>{enc_move_thresh} → {100*np.mean(_spd>enc_move_thresh):.1f}% of frames')
    if enc_near_thresh is not None:
        _npers = [(v['obj_mon_idx'] & (v['obj_dist'] < enc_near_thresh)).any(axis=1) for v in vec_data_use]
        _nbeh = np.any(_npers, axis=0).astype(np.float32)
        _emasks.append(fh.f_resample_to_imaging(beh_t, _nbeh[:, None], frame_t, pulse_delay,
                                             kind='nearest').ravel() > 0.5)
    if _emasks:
        _efull = np.logical_and.reduce(_emasks)          # (T_img,)
        enc_frame_mask = _efull[near_mask_img]           # align to current Y_neu rows (load filter)
        print(f'encoding frame filter: keeping {int(enc_frame_mask.sum())}/{enc_frame_mask.size} '
              f'frames ({100*enc_frame_mask.mean():.1f}%) — NB: hard cuts break temporal contiguity')
    else:
        enc_frame_mask = None
        print('encoding frame filter: OFF (all frames)')


# (diagnostic) PCA variance of the feature block in enc_params — how many components to keep.
# Runs PCA on the SAME block matrix the encoder consumes (built_blocks[feature_block]['X'], i.e.
# the pipeline's PCA scores) and plots per-component + cumulative explained variance, so you can
# read where the block's variance saturates and set n_feature_pcs. The orange curve (if the block
# kept its fitted PCA) is cumulative variance vs the RAW stimulus = the representation ceiling
# (<1 because only n_pca components were kept at build). Grouped-PCA blocks (grid 'gabor') skip.
if 0:
    from sklearn.decomposition import PCA as _PCA
    _fb = enc_params['feature_block']
    _fb = _fb[0] if isinstance(_fb, (list, tuple)) else _fb   # combined blocks → show the first
    _b  = built_blocks[_fb]
    if isinstance(_b.get('pca'), dict) or _b.get('pca_groups'):
        print(f'{_fb!r} uses grouped PCA — single-spectrum plot skipped')
    else:
        _X  = np.asarray(_b['X'], float)
        _Xc = _X[np.isfinite(_X).all(axis=1)]             # drop NaN rows (like the encoder)
        _pc = _PCA(n_components=min(_Xc.shape)).fit(_Xc)   # run PCA on the block matrix
        _evr, _cum = _pc.explained_variance_ratio_, np.cumsum(_pc.explained_variance_ratio_)
        _k  = int(enc_params.get('n_feature_pcs') or _X.shape[1])
        _ks = np.arange(1, _evr.size + 1)

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(_ks, _evr, '-o', ms=3)
        ax[0].set_xlabel('component'); ax[0].set_ylabel('explained variance ratio')
        ax[0].set_title(f'{_fb}: per-component variance')
        ax[1].plot(_ks, _cum, '-o', ms=3, color='C0', label='vs block variance')
        _pca_obj = _b.get('pca') if _b.get('pca') is not None else _b.get('pca_pix')
        if _pca_obj is not None and hasattr(_pca_obj, 'explained_variance_ratio_'):
            _rawcum = np.cumsum(_pca_obj.explained_variance_ratio_)
            ax[1].plot(np.arange(1, _rawcum.size + 1), _rawcum, '-s', ms=3, color='C1',
                       label='vs raw stimulus (ceiling)')
        ax[1].axvline(_k, color='C3', ls='--', lw=0.8, label=f'n_feature_pcs={_k}')
        ax[1].set_xlabel('# components'); ax[1].set_ylabel('cumulative explained variance')
        ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=8, loc='lower right')
        ax[1].set_title(f'cumulative (k={_k} → {_cum[min(_k, _cum.size) - 1]:.2f} of block var)')
        fig.suptitle(f'{dset_tag} — feature PCA variance: {_fb!r}  '
                     f'({_X.shape[1]} block PCs, raw dim {_b.get("raw_dim", "?")})')
        fig.tight_layout()


#%% Encoding sweep — vary ONE param, watch how the encoding scores change (feature → neural)
# Shared sweep wrapper (f_encode_sweep): re-runs the per-neuron encoder for each value × target;
# f_plot_encode_sweep draws metric vs value (one line per target / 2nd-axis value). r_med/frac_sig/r2_med = per-neuron encoding
# fidelity; r2_vw/rv/procrustes/rsa = how well the feature rebuilds the whole population state.
#   sweep_var, sweep_values = 'n_feature_pcs', [1, 2, 3, 5, 8, 12, 16, 24, 32, 50]  # #feature-PCs
#   sweep_var, sweep_values = 'feature_block', ['pix', 'gaborS', 'mengS', ['gaborS','mengS']] # combos
#   sweep_var, sweep_values = 'feature_block', ['gabmag','gabquad','fftmag','fftquad','mengopp']
#                                              # vis_sets BANKS head-to-head (each expands to its cells)
#   sweep_var, sweep_values = 'lag_ms', list(range(0, 501, 100))                    # SINGLE-lag tuning
#   sweep_var, sweep_values = 'lags_ms', [[0],[0,100],[0,100,200],[0,100,200,300]]  # COMBINED lag SETS
#   sweep_var, sweep_values = 'feature_pca', ['per_block','joint','joint_postlag']  # PCA scoping
#   sweep_var, sweep_values = 'ridge_alpha', [0.1, 1, 10, 100, 1000]                # ridge λ (numeric)
#   sweep_var, sweep_values = 'decoder',      ['ridge', 'knn', 'pls']               # decoder A/B
#   sweep_var, sweep_values = 'fold_scheme',  ['contiguous', 'interleaved']         # CV scheme A/B
if 1:
    # 'lag_ms' = SINGLE-lag tuning (each value → lags_ms=[v]); 'lags_ms' = COMBINED lag SETS (each
    # value is a LIST, used verbatim). For combined sets use 'lags_ms' and make every value a list.
    sweep_var    = 'decoder'     #   n_feature_pcs, n_neural_pcs, lag_ms, lags_ms, roughness_lambda, decoder
    #sweep_values = [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]   # (single-lag: use sweep_var='lag_ms')
    #sweep_values =  list(range(-1000, 1001, 100))
    #sweep_values = [[100, 200, 300, 400], [0, 100, 200, 300, 400], [0, 100, 200, 300, 400, 500], [100, 200, 300, 400, 500]]
    #sweep_values = ['agg', 'pix', 'self_mot', 'beh', 'gaborS', 'mengS', 'gaborS+mengS']   
    sweep_values = ['ridge', 'knn', 'pls', 'cca']
    #sweep_values = [1e-5, 1e-3, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
    #sweep_values = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]

    # targets: 'cells' = Y_neu | 'ens_all' / 'ens_sig' = ensemble traces (ensembles cell above)
    enc_sweep_targets = ['cells']    # e.g. ['cells', 'ens_sig']
    enc_line_axis     = None         # optional 2nd axis → one line per value, e.g. ('ridge_alpha', [10, 1e3])
    enc_plot_metrics  = ['r_med', 'r2_vw']   # one line plot per metric (any score in enc_params['scores'])
    enc_panels        = True         # also the old one-panel-per-score view (1 axis, 'cells' only)

    from threadpoolctl import threadpool_limits
    try:    _efm = enc_frame_mask
    except NameError:  _efm = None
    _tg = {'cells': Y_neu}
    for _k in enc_sweep_targets:
        if _k == 'cells':
            continue
        _ets = ens_targets if 'ens_targets' in dir() else {}
        if _k in _ets and _ets[_k][0].shape[0] == Y_neu.shape[0]:
            _tg[_k] = _ets[_k]
        else:
            print(f'  target {_k!r} unavailable (run the ensembles cell; T must match Y_neu) — skipped')
    _tg = {k: v for k, v in _tg.items() if k in enc_sweep_targets}
    _axes = {sweep_var: sweep_values}
    if enc_line_axis:
        _axes[enc_line_axis[0]] = enc_line_axis[1]
    _n_threads = max(1, (os.cpu_count() or 2) - 1)
    t0 = time.perf_counter()
    with threadpool_limits(limits=_n_threads):
        enc_sweep_df, enc_sweep_units = enc.f_encode_sweep(enc_params, _axes, _tg, built_blocks, frame_t,
                                                           frame_mask=_efm, keep_units=True)
    print(f'encoding sweep over {list(_axes)} × {list(_tg)} ({len(enc_sweep_df)} fits, '
          f'≤{_n_threads}/{os.cpu_count()} threads): {time.perf_counter()-t0:.1f}s')
    for _m in enc_plot_metrics:
        encp.f_plot_encode_sweep(enc_sweep_df, metric=_m, title=f'{dset_tag} — encoding: {_m} vs {sweep_var}')
    if enc_panels and not enc_line_axis and 'cells' in _tg:
        _rows = (enc_sweep_df[enc_sweep_df['target'] == 'cells']
                 .rename(columns={sweep_var: 'x'}).to_dict('records'))
        dec.f_plot_decoder_sweep(_rows, sweep_var, enc_params['scores'], dset_tag=dset_tag,
                                 config=enc_params, logx='auto')


#%% Encoding over INPUT SETS — run the encoder on cells / responsive / ensembles → JOINT r/r² ECDF
# Sweep the neural TARGET (what's encoded), not a config key: build each target set with the same
# selector the tuning/ASSIGN cells use (f_select_tuning_units), run f_run_encoder on each (same
# enc_params + feature side), and overlay their per-unit score ECDFs. Compare how well features
# encode single cells vs responsive cells vs ensemble score traces.
#   sources: 'cells' (all active) | 'reliable_cells' (LNO-R²>r_reliable) | 'ensembles' (score traces)
#            | 'ens_cells' (member cells of selected ensembles).  ens sets need the ensembles cells run.
if 1:
    enc_input_sources = ['cells', 'ensembles']   # target sets to compare  ens_cells
    enc_input_which   = 'sig'          # ensembles/ens_cells: 'sig' (significant) | 'all'
    enc_ecdf_metric   = 'r2'           # 'r2' | 'r' (joint ECDF metric; a 2nd plot draws the other)
    enc_input_recompute = False        # True → rebuild target sets even if cached this session
    _fsimg = dec.f_imaging_fs(frame_t)
    _eo = ens_out if 'ens_out' in dir() else None
    _et = ens_table if 'ens_table' in dir() else None
    # session cache: reliable_cells re-runs the LNO dim-red (slow); reuse across re-runs. Keyed by
    # (source, which, n_frames) so it invalidates if the neural changes; clear via enc_input_recompute.
    if enc_input_recompute or '_enc_target_cache' not in dir():
        _enc_target_cache = {}
    _targets = {}
    for _src in enc_input_sources:
        _key = (_src, enc_input_which, int(Y_neu_raw.shape[0]))
        if _key in _enc_target_cache:
            _targets[_src] = _enc_target_cache[_key]
            print(f'  input {_src!r}: reusing cached target {_targets[_src].shape}')
            continue
        try:
            _Yt, _lab, _ = tun.f_select_tuning_units(_src, Y_neu_raw, ens_out=_eo, ens_table=_et,
                                                 dimred_params=dimred_params,
                                                 ens_src_which=enc_input_which, fs=_fsimg)
            _targets[_src] = _enc_target_cache[_key] = _Yt
        except Exception as _e:
            print(f'  skip input {_src!r}: {type(_e).__name__}: {_e}')
    try:    _efm = enc_frame_mask
    except NameError:  _efm = None
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1, (os.cpu_count() or 2) - 1)):
        enc_inputs_out = enc.f_encode_over_inputs(enc_params, built_blocks, _targets,
                                              frame_t, frame_mask=_efm)
    enc.f_plot_encode_ecdf_joint(enc_inputs_out, metric=enc_ecdf_metric, dset_tag=dset_tag)
    enc.f_plot_encode_ecdf_joint(enc_inputs_out, metric=('r' if enc_ecdf_metric == 'r2' else 'r2'),
                             dset_tag=dset_tag)


#%% Encoding — LNO R² vs encoding score SCATTER, overlaid across groups (cells / ensembles / …)
# One point per unit: x = LNO R² (reliability — cells: population leave-one-neuron-out; ensembles:
# within-ensemble member LNO = ens_table cv_r2), y = encoding score (feature→unit). Overlays the
# chosen groups so you see reliability vs encodability together. Same selector as the ECDF cell.
if 1:
    lnoenc_groups  = ['cells', 'ensembles']   # + 'reliable_cells' | 'ens_cells'
    lnoenc_metric  = 'r2'                      # y-axis (encoding): 'r2' | 'r'
    lnoenc_which   = 'sig'                     # ensembles: 'sig' (significant) | 'all'
    lnoenc_ens_lno = 'cv_r2'                   # ensemble LNO on x: 'cv_r2' | 'cv_r2_pool'
    try:    _efm = enc_frame_mask
    except NameError:  _efm = None
    _eo = ens_out if 'ens_out' in dir() else None
    _et = ens_table if 'ens_table' in dir() else None
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1, (os.cpu_count() or 2) - 1)):
        lnoenc = enc.f_lno_vs_encode_groups(lnoenc_groups, enc_params, built_blocks, Y_neu_raw,
                    frame_t, frame_mask=_efm, ens_out=_eo, ens_table=_et,
                    dimred_params=dimred_params, fs=dec.f_imaging_fs(frame_t),
                    ens_lno_metric=lnoenc_ens_lno, which=lnoenc_which)
    enc.f_plot_lno_vs_encode(lnoenc, metric=lnoenc_metric, dset_tag=dset_tag)


#%% Encoding raster — run ONE encoder (enc_params) → true vs feature-reconstructed neural raster
# Fits ONE encoder with the current enc_params (feature → neural), collects OOF predictions, and
# shows the TRUE neural raster next to the raster ENCODED from features. Cells sorted by per-cell
# encoding r (best on top); each row z-scored by its true stats so both panels share a color scale.
# With enc_params['n_neural_pcs'] set, TRUE is the rank-k projection (what the encoder can reach).
if 1:
    enc_trace_nshow = 25             # # units in the stacked true-vs-encoded trace overlay (top by r)
    from threadpoolctl import threadpool_limits
    try:    _efm = enc_frame_mask
    except NameError:  _efm = None
    with threadpool_limits(limits=max(1, (os.cpu_count() or 2) - 1)):
        enc_one = enc.f_run_encoder(enc_params, built_blocks, Y_neu, frame_t, frame_mask=_efm)
    print(f"encoded {enc_params['feature_block']!r} → neural:  "
          f"median r={np.nanmedian(enc_one['r_per_cell']):+.3f}  "
          f"median R²={np.nanmedian(enc_one['r2_per_cell']):+.3f}  "
          f"({enc_one['n_valid']} OOF frames, {enc_one['oof_true_cells'].shape[1]} cells)")
    _erd = raster_cmap.lower() in _RASTER_DIVERGING   # symmetric scale for a diverging cmap
    # model tag for titles: decoder (or roughness) + feature block + lags
    _mdl = ('rough' if enc_params.get('roughness_lambda') else enc_params.get('decoder', 'ridge'))
    _etag = f"{dset_tag} [{_mdl}, {enc_params['feature_block']}, {len(enc_params.get('lags_ms',[0]))} lag]"
    # under a frame_mask the OOF frames are a subset → absolute-frame placement is invalid; show the
    # OOF matrices directly (Y_neu=None) instead of the continuous raster.
    _Yn = None if _efm is not None else Y_neu
    _ft = None if _efm is not None else frame_t
    enc.f_plot_encode_rasters(enc_one, Y_neu=_Yn, frame_t=_ft, dset_tag=_etag,
                          vmin=(-0.5 if _erd else 0.0), vmax=0.5, cmap=raster_cmap, sort=None)
    # stacked TRUE vs ENCODED per-unit traces overlaid (best-r on top, offset going down)
    enc.f_plot_encode_traces(enc_one, n_show=enc_trace_nshow, frame_t=_ft, dset_tag=_etag)
    # per-neuron R² (and r) distribution across the population — separate ECDF figure
    enc.f_plot_encode_ecdf(enc_one, dset_tag=_etag, metrics=('r2', 'r'))
    # per-cell score TABLE (r / R² / Spearman / nRMSE) + r-vs-R² scatter — different metrics rank
    # cells differently (a cell can encode with high r but negative R² = right shape, wrong scale)
    enc_cell_scores = enc.f_encode_cell_scores(enc_one)
    with pd.option_context('display.max_rows', 25, 'display.width', 160):
        print(f'{_etag} — per-cell encoding scores (top 25 by R²):')
        print(enc_cell_scores.head(25).to_string(index=False))
    enc.f_plot_encode_score_scatter(enc_one, dset_tag=_etag)


#%% ============================================================================
# SECTION — FEATURE TUNING (reverse-correlation of activity vs conv Gabor/meng filters)
# =============================================================================
# For each unit (cell / ensemble / …), reverse-correlate its CONTINUOUS activity (no spike detection)
# against the dense per-filter conv responses (whole-monitor magnitude per orientation × wavelength
# [× temporal-freq]) → preferred tuning. Regressors from the conv_repo (f_conv_activity), on the same
# imaging clock as the neural data. Both STA (literal response-weighted avg) and ridge (whitened,
# sharper for correlated filters) are computed. Uses Y_neu_raw / ens_out (run the ensembles cell first
# for ensemble sources). Compute in f_tuning.py; plots in f_encoding_plots.py.

if 1:
    tuning_source   = 'ensembles'    # 'cells' | 'ensembles' | 'ens_cells' | 'reliable_cells'
    tuning_kind     = 'meng'        # 'gabor' | 'meng' (meng adds the temporal-freq axis)
    tuning_conv_tag = src.get('conv_tag')   # conv_repo GEN_TAG (folder <dset>_<tag>)
    tuning_wavelengths = None         # None = all in repo | e.g. (12, 15, 18)
    tuning_odegs       = None         # None = all | e.g. (0, 45, 90, 135)
    tuning_tfreqs      = None         # meng only; None = all
    tuning_lags_ms     = [0, 100, 200, 300, 400]   # causal; peak lag picked per unit (calcium delay)
    tuning_methods     = ['sta']          # compare literal STA vs whitened ridge  , 'ridge'
    tuning_transform   = 'sqrt' if tuning_kind == 'meng' else None  # regressor map: meng ENERGY→amplitude
                                     #   | None | 'sqrt' | 'signed_sqrt' | 'log1p'
    tuning_standardize = False       # z-score each filter regressor → magnitudes comparable across
                                     #   convs (recommended for method='sta'; ridge z-scores internally)
    tuning_ens_which   = 'sig'        # ensembles/ens_cells: 'sig' (significant) | 'all'
    _repo_dir = cr.f_conv_repo_dir(src['features_dir'], est1['dset_name'], tuning_conv_tag)
    print('tuning repo:', _repo_dir)


#%% Tuning — compute (both methods) → per-unit preferred orientation / wavelength [/ tfreq]
if 1:
    _fs = dec.f_imaging_fs(frame_t)
    # CACHE the conv regressors (reading the repo dense maps is the slow I/O step). Re-read only when
    # the repo / kind / wavelength / odeg / tfreq changes — NOT when tuning_source/methods/lags change.
    _tun_key = (str(_repo_dir), tuning_kind, str(tuning_wavelengths), str(tuning_odegs),
                str(tuning_tfreqs), str(tuning_transform), bool(tuning_standardize))
    try:    _reuse_reg = (_tun_key_prev == _tun_key) and (X_stim_full is not None)
    except NameError:  _reuse_reg = False
    if _reuse_reg:
        print('tuning: reusing cached conv regressors (repo/kind/filters/transform unchanged)')
    else:
        X_stim_full, filt_df = tun.f_conv_regressors(_repo_dir, kind=tuning_kind, wavelengths=tuning_wavelengths,
                                                 odegs=tuning_odegs, tfreqs=tuning_tfreqs, frame_ds=1,
                                                 transform=tuning_transform, standardize=tuning_standardize)
        _tun_key_prev = _tun_key
    X_stim = X_stim_full
    Y_units, tun_labels, tun_kind = tun.f_select_tuning_units(
        tuning_source, Y_neu_raw, ens_out=(ens_out if 'ens_out' in dir() else None),
        ens_table=(ens_table if 'ens_table' in dir() else None), dimred_params=dimred_params,
        ens_src_which=tuning_ens_which, fs=_fs)
    _n = min(X_stim.shape[0], Y_units.shape[0])   # crop to shared frames (repo may be full session)
    if X_stim.shape[0] != Y_units.shape[0]:
        print(f'  tuning: aligning conv {X_stim.shape[0]} / neural {Y_units.shape[0]} → {_n} frames')
    X_stim, Y_units = X_stim[:_n], Y_units[:_n]
    _lags = enc.f_lags_from_ms(tuning_lags_ms, _fs)
    from threadpoolctl import threadpool_limits
    tun_readouts = {}
    with threadpool_limits(limits=max(1, (os.cpu_count() or 2) - 1)):
        for _m in tuning_methods:
            _fit = tun.f_reverse_corr_tuning(X_stim, filt_df, Y_units, _lags, method=_m)
            tun_readouts[_m] = tun.f_tuning_readout(_fit, fs=_fs, labels=tun_labels)
    _rd = tun_readouts[tuning_methods[-1]]        # display the last method (ridge by default)
    _cols = ['unit', 'pref_odeg', 'pref_wavelength', 'pref_tfreq', 'peak_abs', 'peak_lag_ms',
             'osi', 'sf_sel', 'tf_sel']
    _cols = [c for c in _cols if c in _rd['table'].columns]
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(f'{dset_tag} — {tuning_kind} tuning ({tun_kind}, {tuning_methods[-1]}, '
              f'{Y_units.shape[1]} units):')
        print(_rd['table'].sort_values('peak_abs', ascending=False)[_cols].to_string(index=False))


#%% Tuning — plots (population + example units + STA vs ridge)
if 1:
    tuning_plot_z = True            # z-score tuning across filters for the heatmap + all-filters bars
                                    #   (peak in SD units, scale-free vs cross-frequency magnitude)
    _rd = tun_readouts[tuning_methods[-1]]
    encp.f_plot_tuning_population(_rd['table'], dset_tag=f'{dset_tag} {tuning_kind}', method=_rd.get('method', ''))
    encp.f_plot_tuning_marginals(_rd, u=None, dset_tag=f'{dset_tag} {tuning_kind}')   # population mean
    encp.f_plot_tuning_lags(_rd, dset_tag=f'{dset_tag} {tuning_kind}')                # tuning strength vs lag
    _top = _rd['table'].sort_values('peak_abs', ascending=False).index.to_numpy()[:3]   # 3 strongest
    for _u in _top:
        encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=f'{dset_tag} {tuning_kind}',
                                   zscore=tuning_plot_z)                                    # ori × wl grid
        encp.f_plot_tuning_allfilters(_rd, int(_u), dset_tag=f'{dset_tag} {tuning_kind}',
                                      zscore=tuning_plot_z)                                 # ALL filters, selected marked
    if len(tuning_methods) >= 2 and 'sta' in tun_readouts and 'ridge' in tun_readouts:
        encp.f_plot_tuning_compare(tun_readouts['sta'], tun_readouts['ridge'], dset_tag=f'{dset_tag} {tuning_kind}')
    # STA: for the top unit, show WHICH high-activity frames drive the average, on the activity trace
    if 'sta' in tun_readouts:
        sta_trigger_pct = 90                      # frames above this activity percentile = 'triggers'
        _sr = tun_readouts['sta']; _sk = _sr.get('keep_cells')
        _Yk = Y_units[:, np.asarray(_sk, bool)] if _sk is not None else Y_units
        _uS = int(_sr['table'].sort_values('peak_abs', ascending=False).index[0])   # top unit
        _ftS = frame_t if len(frame_t) == Y_units.shape[0] else None
        #encp.f_plot_sta_triggers(_Yk[:, _uS], frame_t=_ftS, thresh_pct=sta_trigger_pct,
        #                    dset_tag=f'{dset_tag} u{_sr["table"].iloc[_uS]["unit"]}')


#%% Tuning — VERIFY in monitor space: WHERE the tuning sits + the selected feature (re-reads repo)
# For the top ensembles, load the PREFERRED filter's responses from the conv_repo and show (a) the
# LOCATION map (reverse-corr of each screen location with the unit activity → RF blob) and (b) the
# activity-weighted STA (the selected feature rendered in monitor shape) — eyeball verification.
if 1:
    # plt.close('all')
    tuning_verify_n     = 2      # # top ensembles to verify (each re-reads the repo — a bit slow)
    tuning_loc_ds       = 8      # spatial stride for the LOCATION map (larger = coarser/faster)
    tuning_sta_frame_ds = 20     # read every Nth frame for the STA (larger = faster/less RAM)
    tuning_side         = 'both' # 'both' = L‖R monitors side by side (wide) | 'L' | 'R'
    tuning_monitor_z    = True   # z-score the monitor maps across pixels (SD units, diverging) so
                                 #   absolute magnitude doesn't set the color scale
    _rd = tun_readouts[tuning_methods[-1]]
    _keep = _rd.get('keep_cells')
    _Ykept = Y_units[:, np.asarray(_keep, bool)] if _keep is not None else Y_units   # kept-unit activity
    _topv = _rd['table'].sort_values('peak_abs', ascending=False).index.to_numpy()[:tuning_verify_n]
    for _u in _topv:
        _u = int(_u); _frow = tun.f_preferred_filt_row(_rd, _u); _act = _Ykept[:, _u]
        _lab = _rd['table'].iloc[_u]['unit']
        _tfv = _frow.get('tfreq', np.nan)                 # meng: show the preferred temporal freq
        _tfs = f' tf{_tfv:g}' if not (_tfv is None or (isinstance(_tfv, float) and np.isnan(_tfv))) else ''
        _tag = (f'{dset_tag} {tuning_kind} u{_lab} (o{_frow["odeg"]:.0f} wl{_frow["wavelength"]:g}{_tfs}) '
                f'[{_rd.get("method", "")}]')
        try:
            _nmon = 2 if tuning_side == 'both' else 1
            _mlab = ['L', 'R'] if tuning_side == 'both' else [tuning_side]
            _loc = tun.f_tuning_location_map(_repo_dir, _frow, _act, ds=tuning_loc_ds, side=tuning_side)
            if _loc is not None:
                encp.f_plot_tuning_monitor(_loc, dset_tag=_tag, title=f'location (corr, {tuning_side})',
                                      diverging=True, n_monitors=_nmon, labels=_mlab, zscore=tuning_monitor_z)
            _sta = tun.f_tuning_feature_sta(_repo_dir, _frow, _act, frame_ds=tuning_sta_frame_ds, side=tuning_side)
            if _sta is not None:
                encp.f_plot_tuning_monitor(_sta, dset_tag=_tag, title=f'preferred-filter STA ({tuning_side})',
                                      diverging=False, n_monitors=_nmon, labels=_mlab, zscore=tuning_monitor_z)
        except Exception as _e:
            print(f'  tuning verify u{_lab}: skipped ({type(_e).__name__}: {_e})')

    # (viz) a gabor KERNEL centered on a blank monitor frame — to eyeball its SPATIAL SCALE
    # (σ = sigma_scale·wavelength) relative to the monitor. Spatial kernel is the same for gabor and
    # meng (Adelson-Bergen uses a gabor spatial front-end). By default uses the top unit's PREFERRED
    # filter; set kern_wl / kern_odeg to OVERRIDE and draw any size you want (kern_wl can be a LIST
    # to compare sizes side by side).
    if 0:
        import matplotlib.pyplot as plt
        kern_wl          = None      # None = preferred wavelength | a number | a LIST e.g. [6, 12, 24]
        kern_odeg        = None      # None = preferred orientation (deg) | a number
        kern_sigma_scale = 0.5       # σ = sigma_scale·wavelength (registry default 0.5)
        kern_aspect      = 16 / 9    # display each monitor at this W:H (physical); None = square render grid
        _fr = tun.f_preferred_filt_row(_rd, int(_topv[0]))
        _odeg = float(_fr['odeg']) if kern_odeg is None else float(kern_odeg)
        _wls = ([float(_fr['wavelength'])] if kern_wl is None
                else ([float(w) for w in kern_wl] if np.ndim(kern_wl) else [float(kern_wl)]))
        _mon_hw = (render_vert_samp, render_lat_samp)      # one monitor (H, W render resolution)
        _ch, _cw = _mon_hw; _th = np.radians(_odeg)
        _nc = len(_wls)
        fig, _axes = plt.subplots(1, _nc, figsize=(4.2 * _nc, 4.4), squeeze=False)
        for _ax, _wl in zip(_axes[0], _wls):
            _sig = kern_sigma_scale * _wl
            _ev, _od = vf._gabor_pair(_th, _wl, _sig, 1.0)     # even (cosine-phase) kernel
            _kh, _kw = _ev.shape
            _canvas = np.zeros(_mon_hw, np.float32)            # blank ONE-monitor frame
            _ys, _xs = max(0, (_ch - _kh) // 2), max(0, (_cw - _kw) // 2)   # center-embed (crop if bigger)
            _kys, _kxs = max(0, (_kh - _ch) // 2), max(0, (_kw - _cw) // 2)
            _hh, _ww = min(_kh, _ch), min(_kw, _cw)
            _canvas[_ys:_ys + _hh, _xs:_xs + _ww] = _ev[_kys:_kys + _hh, _kxs:_kxs + _ww]
            _v = float(np.abs(_ev).max()) or 1.0
            _ax.imshow(_canvas, cmap='RdBu_r', vmin=-_v, vmax=_v, interpolation='nearest')
            if kern_aspect:                                # stretch square render grid → physical W:H
                _ax.set_aspect((_cw / _ch) / kern_aspect)
            _ax.set_title(f'o{_odeg:.0f} wl{_wl:g}  σ={_sig:g}px\nkernel {_kh}×{_kw} in {_ch}×{_cw}', fontsize=8)
            _ax.set_xticks([]); _ax.set_yticks([])
        fig.suptitle(f'{dset_tag} — gabor kernel(s) on monitor', fontsize=9)
        fig.tight_layout()


#%% Decode behavior from CEBRA embedding — time-blocked KNN cross-validation
# =============================================================================
# SECTION 4 · DECODING · core — bar A/B, PCA sweep (R²+AUC), shuffled null
# =============================================================================
# takes neural data and embeds it with supervision from some behavior input.. 
# then uses that data to decode behavior, complares to equivalent number of pca comp


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
bar_decoder        = 'ridge'   # 'knn'|'ridge'|'ridgecv'|'pls'|'cca' — KNN = local
                             # nonlinearity; pls/cca = joint low-rank fit.
bar_k_nn           = 15
bar_n_components   = 20      # pls/cca latent rank (auto-clamped per fold)
bar_standardize    = True    # z-score embedding cols (train-fold stats) — stops
                             # one embedding axis from dominating KNN distance.
                             # (New default; the old cell did not standardize.)
bar_embargo_sec    = 0.0     # purge gap (s) each side of the test block.
bar_fold_scheme    = 'interleaved'  # 'contiguous' = whole-epoch folds |
                             # 'interleaved' = chop into bar_n_blocks small blocks
                             # round-robin'd to folds (no fold = one quiet/odd epoch).
bar_n_blocks       = None    # interleaved only: # small blocks (None → n_folds*10).
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
_fs_bar = dec.f_imaging_fs(frame_t)
bar_embargo_fr     = int(round(bar_embargo_sec * _fs_bar))
# Decoder FIR: lag-embed the embedding over a window (bidirectional). [0] = off.
bar_lags           = dec.f_lags_from_ms(list(range(-300, 301, 100)), _fs_bar)

# PCA-3 baseline on neural data — same out_dim as CEBRA.
pca_neu = dec.f_pca_prefix(Y_neu, 3)

# Which feature blocks (from built_blocks registry) to decode. Each entry
# in target_blocks_bar gets its own bar group, with bars colored by embedding.
# Examples:
#   ['agg', 'pix']                — original A/B comparison
#   ['agg', 'motion']             — position vs velocity from same embeddings
#   ['agg', 'motion', 'pix']      — three target groups, three bar clusters
target_blocks_bar = ['agg', 'pix', 'grid', 'flow', 'gabor', 'meng']   # 'gabor'=edge energy, 'meng'=motion energy (A/B vs agg/pix)
target_blocks_bar = [b for b in target_blocks_bar if b in built_blocks]   # keep only loaded blocks

# How to aggregate per-column R² when a block has multiple output dims.
# 'uniform_average'   — current default; each column counts equally. Drags
#                       low when binary / low-variance columns fail to decode.
# 'variance_weighted' — joint R² (task #26); weights each column by its
#                       variance. More honest when high- and low-variance
#                       targets are mixed (e.g. lat/vert vs presence).
bar_multioutput = 'variance_weighted'   # 'uniform_average' = equal per-column mean |
                                      # 'variance_weighted' = joint R² (high-variance cols dominate)
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
    Xb = dec.f_apply_block_detrend(Xb, grp, bar_detrend_blocks,   # #8/#10 high-pass
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
        m, s = ceb.f_blocked_cv_r2(emb, target, n_folds=5, decoder=bar_decoder,
                               k_nn=bar_k_nn, n_components=bar_n_components,
                               embargo=bar_embargo_fr,
                               standardize=bar_standardize,
                               multioutput=bar_multioutput, return_std=True,
                               lags=bar_lags,
                               fold_scheme=bar_fold_scheme, n_blocks=bar_n_blocks)
        r2_table[i, j] = m
        r2_std[i, j]   = s
        print(f'  decode {fname:18s}  from  {ename:18s}  R² = {m:+.3f} ± {s:.3f}')

# Grouped bar plot — feature groups × embeddings.
dec.f_plot_decode_bars(
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
    decoder     = 'knn'   # 'knn' | 'ridge' | 'ridgecv' | 'pls' | 'cca'
    k_nn        = 15        # used when decoder == 'knn'
    ridge_alpha = 1.0       # used when decoder == 'ridge'
    n_components = 20       # pls/cca latent rank (auto-clamped per fold)
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
                                # INCOMPATIBLE with decode_lags (lag-after-PCA) —
                                # f_blocked_cv_r2 raises if both are set.
    # ─────────────────────────────────────────────────────────────────────────

    # Imaging sample rate from frame_times, for the sec→frame conversions above.
    fs_img = dec.f_imaging_fs(frame_t)
    embargo_fr       = int(round(embargo_sec * fs_img))
    detrend_sigma_fr = (detrend_sigma_sec * fs_img) if detrend_sigma_sec else None
    # f_detrend_col is imported from f_feature_helpers (unified 2026-06-03;
    # was an inline def here).

    n_pcs_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    n_folds = 5
    # Decoder FIR: lag-embed the neural PCs over a window → decode feature(t) from
    # the neural trajectory [t−Δ,t+Δ] (bidirectional: neural may LEAD or LAG the
    # feature). [0] = instantaneous (old behavior). Lag-after-PCA; needs
    # pca_within_fold=False (else f_blocked_cv_r2 raises).
    decode_lags_ms = list(range(-300, 301, 100))
    decode_lags    = dec.f_lags_from_ms(decode_lags_ms, fs_img)
    sweep_target_blocks = ['agg', 'pix', 'motion', 'pix_mot', 'beh', 'self_mot', 'grid', 'flow', 'gabor', 'meng']    # sweep over these ('gabor'=edge energy, 'meng'=motion energy)
    # Keep only loaded blocks. A vis_sets BANK is stored as `<bank>:<cell>` keys, so match those too —
    # f_run_decode_sweep expands the bank name itself (f_expand_blocks); a plain `in built_blocks`
    # test drops it here before that can happen.
    sweep_target_blocks = [b for b in sweep_target_blocks
                           if b in built_blocks or any(k.startswith(b + ':') for k in built_blocks)]
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
                                            #   'gabor'    — Gabor oriented-edge energy (Tier 3). build_visual.
                                            #   'meng'     — motion energy, dir+speed (Tier 4a). build_visual.
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
    # targets + pca_neu_full + the r2/auc sweep are computed by
    # f_decoding.f_run_decode_sweep below (same code the cross-dataset analyzer
    # uses); they're unpacked back into the globals the null / grid / diagnostic
    # cells expect.

    # f_cv_r2_single is now a thin wrapper over the unified f_blocked_cv_r2
    # (f_cebra_helpers) so the sweep, the shuffle-null cell, and the detrend
    # grid all share ONE blocked-CV implementation (no drift). The shuffle +
    # grid cells call f_cv_r2_single unchanged. (Inline blocked-CV loop removed
    # 2026-06-03; logic preserved verbatim inside f_blocked_cv_r2.)
    def f_cv_r2_single(emb, y, n_folds=5, embargo=0, standardize=True,
                       Y_raw=None, n_pcs=None, task='regression', lags=None):
        return ceb.f_blocked_cv_r2(emb, y, n_folds=n_folds, embargo=embargo,
                               standardize=standardize, decoder=decoder,
                               ridge_alpha=ridge_alpha, k_nn=k_nn,
                               ridgecv_alphas=ridgecv_alphas, n_components=n_components,
                               Y_raw=Y_raw, n_pcs=n_pcs, task=task, lags=lags)

    # decode_cfg gathers the knobs above; f_run_decode_sweep does targets + PCA +
    # the feature × n_pcs sweep (real). Keep the `sweep` dict for the null cell.
    decode_cfg = {
        'sweep_target_blocks': sweep_target_blocks, 'n_pcs_sweep': n_pcs_sweep,
        'n_folds': n_folds, 'decoder': decoder, 'k_nn': k_nn,
        'ridge_alpha': ridge_alpha, 'n_components': n_components,
        'ridgecv_alphas': ridgecv_alphas, 'standardize_emb': standardize_emb,
        'embargo_sec': embargo_sec, 'detrend_sigma_sec': detrend_sigma_sec,
        'detrend_blocks': detrend_blocks, 'detrend_embedding': detrend_embedding,
        'pca_within_fold': pca_within_fold, 'decode_lags_ms': decode_lags_ms,
    }
    # ── CPU cap ──────────────────────────────────────────────────────────────
    # sklearnex/MKL/OpenMP otherwise grab EVERY core for the KNN/Ridge/PCA fits in
    # the sweep → machine unresponsive / kernel crash. Cap the thread pools so
    # `leave_cores_free` core(s) stay idle. threadpoolctl (a sklearn dep) limits
    # BLAS / OpenMP / oneDAL threads at RUNTIME — no kernel restart / re-import.
    leave_cores_free = 1
    _n_threads = max(1, (os.cpu_count() or 2) - leave_cores_free)
    from threadpoolctl import threadpool_limits
    t0 = time.perf_counter()
    with threadpool_limits(limits=_n_threads):
        sweep = dec.f_run_decode_sweep(built_blocks, Y_neu, frame_t, decode_cfg)
    print(f'PCA decoding sweep ({decoder}, ≤{_n_threads}/{os.cpu_count()} threads): '
          f'{time.perf_counter()-t0:.1f}s')
    # Unpack into the globals the plots + null / grid / diagnostic cells expect.
    r2_mat       = sweep['r2_mat'];         auc_mat = sweep['auc_mat']
    targets_all  = sweep['targets_all'];    targets_bin = sweep['targets_bin']
    group_rows   = sweep['group_rows'];     sweep_groups_present = sweep['groups_present']
    pca_neu_full = sweep['pca_neu_full'];   max_pcs = sweep['max_pcs']
    n_agg = len(group_rows.get('agg', []))
    r2_agg = r2_mat[group_rows.get('agg', [])]
    r2_pix = r2_mat[group_rows.get('pix', [])]

    # Heatmap — single overview (clipped ±0.5; dividers between feature groups).
    dec.f_plot_feature_heatmap(
        r2_mat, n_pcs_sweep, [f'{g}:{n}' for g, n, _ in targets_all],
        group_sizes=[len(group_rows[g]) for g in sweep_groups_present],
        title=f'{dset_tag}\nper-feature blocked-CV R² (PCA → {decoder} decoding)',
        cbar_label='R²  (clipped at ±0.5)')

    # Per-feature R²-vs-n_pcs line plots, one panel per group.
    dec.f_plot_sweep_lines(n_pcs_sweep, r2_mat, targets_all, group_rows,
                       sweep_groups_present, dset_tag=dset_tag)

    # Aggregate decoding-performance-vs-n_pcs summary (2-axis: analog R² left,
    # binary AUC right) — see f_decoding.f_plot_decode_sweep_summary.
    dec.f_plot_decode_sweep_summary(n_pcs_sweep, r2_mat, group_rows,
                                sweep_groups_present, targets_bin, auc_mat,
                                dset_tag=dset_tag, decoder=decoder)


#%% Score feature representations — one decode R² + encode r per built block
# f_score_representations: decode_r2 (neural PCs → feature, leak-free within-fold PCA) +
# encode_r_med (feature → neural, f_encode_per_neuron). Compact single-value-per-method format.
# smp_compare=None scores EVERY loaded block (whatever feature_list built — gaborS/mengS/pix/…);
# pass a list to restrict, e.g. ['gaborS', 'mengS', 'pix'] or +['agg','gabor','meng'] for grid-pooled.
if 1:
    from f_dataset_scan import f_score_representations
    _fs = dec.f_imaging_fs(frame_t)
    _emb = int(round(0.5 * _fs))
    smp_compare = None      # None = every block in built_blocks; else a list of block names
    smp_scores = f_score_representations(built_blocks, Y_neu, smp_compare, n_pcs=30,
                                         lags_ms=(0, 100, 200), embargo=_emb, n_folds=5, fs=_fs)
    print(smp_scores.to_string(index=False))
    _d = smp_scores.set_index('method')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    _d['decode_r2'].plot.barh(ax=ax[0], color='steelblue')
    ax[0].set_title('decode R²  (neural → feature)')
    _d['encode_r_med'].plot.barh(ax=ax[1], color='indianred')
    ax[1].set_title('encode r  (feature → neural)')
    for a in ax:
        a.axvline(0, color='gray', lw=0.6)
    fig.suptitle(f'{dset_tag} — feature-representation scorecard (decode + encode)')
    fig.tight_layout()


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

# Same shared compute as the real sweep — f_run_decode_null reuses the sweep's
# targets + PCA and breaks the neural↔behavior alignment per shuffle_mode.
null_cfg = {'n_shuffles': n_shuffles, 'shuffle_mode': shuffle_mode,
            'min_shift_frames': min_shift_frames, 'shuffle_seed': shuffle_seed,
            'independent_target_shifts': independent_target_shifts}
print(f'shuffled-control sweep ({shuffle_mode}, n={n_shuffles}, decoder={decoder})...')
# Same CPU cap as the sweep cell — leave a core free (see leave_cores_free there).
from threadpoolctl import threadpool_limits
_n_threads = max(1, (os.cpu_count() or 2) - globals().get('leave_cores_free', 1))
t_null = time.perf_counter()
with threadpool_limits(limits=_n_threads):
    null = dec.f_run_decode_null(sweep, built_blocks, Y_neu, null_cfg)
r2_mat_null  = null['r2_mat_null']
auc_mat_null = null['auc_mat_null']
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
dec.f_plot_decode_real_vs_null(n_pcs_sweep, r2_mat, r2_mat_null, group_rows,
                           sweep_groups_present, targets_bin, auc_mat, auc_mat_null,
                           dset_tag=dset_tag, decoder=decoder,
                           shuffle_mode=shuffle_mode, n_shuffles=n_shuffles)

# Per-feature signal-above-chance heatmap: real − null_mean.
# Reads as "how much R² each feature gains over chance, at each n_pcs."
null_mean_per_feat = r2_mat_null.mean(axis=0)        # (n_features, n_pcs)
delta = r2_mat - null_mean_per_feat
dec.f_plot_feature_heatmap(
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
dec.f_plot_perfeature_null_grid(n_pcs_sweep, r2_mat, r2_mat_null, group_rows,
                            sweep_groups_present, targets_all, dset_tag=dset_tag,
                            decoder=decoder, shuffle_mode=shuffle_mode,
                            n_shuffles=n_shuffles)


#%% Save decode results (real + null) → cross-dataset analysis (VR_decode_analysis.py)
# Writes <decode_dir>/<mouse>_<dset>_<decode_tag>_decode.joblib (R²/AUC matrices +
# labels + decode_cfg + meta; no y arrays / neural). Run per dataset to build the
# set the analyzer loads. Bump decode_tag when the decode method changes.
save_decode = False
decode_tag  = 'v1'
decode_dir  = 'F:/VR/decode/' + mouse_id
if save_decode:
    from f_decode_cache import f_save_decode, f_decode_path
    _dmeta = {'mouse_id': mouse_id, 'n_dset': n_dset, 'dset_name': est1['dset_name'],
              'region': None, 'n_cells': int(Y_neu.shape[1]),
              'fs': float(dec.f_imaging_fs(frame_t)), 'n_frames': int(frame_t.shape[0])}
    _dpath = f_decode_path(decode_dir, mouse_id, est1['dset_name'], decode_tag)
    f_save_decode(_dpath, sweep, null, meta=_dmeta, gen_config=decode_cfg)
    print(f'saved decode [{decode_tag}] -> {_dpath}')


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
    grid_targets_raw = dec.build_target_columns(built_blocks, grid_blocks)
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
            ys.append(fh.f_detrend_col(yraw, ds_fr) if _apply else yraw)

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
    dec.f_plot_grid_delta_heatmap(grid_delta, grid_p, combo_labels, grid_feat_labels,
                              dset_tag=dset_tag, n_pcs_grid=n_pcs_grid, decoder=decoder)
    dec.f_plot_grid_focus_lines(detrend_grid, embargo_grid, combos, grid_real,
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
        return ceb.f_blocked_cv_r2(pca_neu_full[:, :npc], y_diag, n_folds=diag_n_folds,
                               embargo=embargo_fr, standardize=standardize_emb,
                               decoder=decoder, ridge_alpha=ridge_alpha, k_nn=k_nn,
                               return_predictions=True, lags=decode_lags)

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
    ins = dec.f_insample_predict(pca_neu_full[:, :diag_insample_n_pcs], y_diag,
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
    _sig_fr = fh.f_resolve_detrend_sigma(pred_block, detrend_blocks,
                                      detrend_sigma_sec, fs_img)
    _oof_targets = []
    for ci in range(min(pred_max_feats, _b['X'].shape[1])):
        y = _b['X'][:, ci]
        if _sig_fr:
            y = fh.f_detrend_col(y, _sig_fr)
        _oof_targets.append((_b['names'][ci], y))

    dec.f_plot_oof_trace_scatter(
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
    dec.f_plot_input_raster(
        S_raster, t_img_plot,
        built_blocks['agg']['X'], built_blocks['pix']['X'],
        side_tags, n_per_side, dist_name=dist_name, n_pix_plot=3,
        dset_tag=dset_tag, side=side, agg_mode=build_cfg['agg_mode'])


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

    if plot_block_name not in built_blocks:                      # fall back to a loaded block
        _fallback = next(iter(built_blocks))
        print(f'{plot_block_name!r} not loaded; showing {_fallback!r} instead '
              f'(available: {list(built_blocks)})')
        plot_block_name = _fallback
    _pb = built_blocks[plot_block_name]
    dec.f_plot_block_traces(
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
    Y_shuf = dec.f_shuffle_neural(Y_neu, _rng_pca, min_shift=_min_shift_pca)
    cum_shuf_runs[s] = PCA(n_components=max_k, random_state=42).fit(Y_shuf) \
                          .explained_variance_ratio_.cumsum()
ks = np.arange(1, max_k + 1)
dec.f_plot_real_vs_shuffle_line(
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
decoder_dec           = 'ridge'  # 'ridge' | 'ridgecv' | 'knn' | 'pls' | 'cca'
k_nn_dec              = 15        # used when decoder_dec == 'knn'
n_components_dec      = 20        # pls/cca latent rank (auto-clamped per fold)
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
    target_blocks_dec = dec.legacy_target_blocks(target_choice)
    print(f'  legacy target_choice={target_choice!r} → target_blocks_dec={target_blocks_dec!r}')

# Imaging fs for sec→frame conversions.
_fs_dec = dec.f_imaging_fs(frame_t)
embargo_fr_dec = int(round(embargo_sec_dec * _fs_dec))
# Decoder FIR: lag-embed the neural PCs over a window (bidirectional). [0] = off.
decode_lags_dec = dec.f_lags_from_ms(list(range(-300, 301, 100)), _fs_dec)

# Concatenated multi-output target, block-selectively high-passed (#8/#10).
# Old single-call form: target_dec, _ = combine_blocks(target_blocks_dec)
target_dec, target_dec_names = dec.build_target_matrix(
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
                 embargo=0, standardize=True, Y_raw=None, n_pcs=None, lags=None):
    return ceb.f_blocked_cv_r2(emb, target, n_folds=n_folds, embargo=embargo,
                           standardize=standardize, decoder=decoder_dec,
                           ridge_alpha=alpha, k_nn=k_nn_dec,
                           n_components=n_components_dec,
                           multioutput=multioutput, Y_raw=Y_raw, n_pcs=n_pcs, lags=lags)


max_pcs_dec = max(n_pcs_sweep_dec)

# Real
pca_real_full = dec.f_pca_prefix(Y_neu, max_pcs_dec)
r2_real_dec = np.array([_cv_r2_multi(pca_real_full[:, :k], target_dec, n_folds, ridge_alpha,
                                      multioutput=multioutput_dec,
                                      embargo=embargo_fr_dec, standardize=standardize_dec,
                                      lags=decode_lags_dec)
                        for k in n_pcs_sweep_dec])

# Shuffled — per-cell circshift; refit PCA each shuffle.
# uniform circshift null (seeded rng for reproducibility).
r2_shuf_runs_dec = np.zeros((n_shuf_dec, len(n_pcs_sweep_dec)))
t0 = time.perf_counter()
_rng_dec = np.random.default_rng(42)
_min_shift_dec = 0
for s in range(n_shuf_dec):
    Y_shuf = dec.f_shuffle_neural(Y_neu, _rng_dec, min_shift=_min_shift_dec)
    pca_shuf_full = dec.f_pca_prefix(Y_shuf, max_pcs_dec)
    for ki, k in enumerate(n_pcs_sweep_dec):
        r2_shuf_runs_dec[s, ki] = _cv_r2_multi(pca_shuf_full[:, :k], target_dec, n_folds, ridge_alpha,
                                                multioutput=multioutput_dec,
                                                embargo=embargo_fr_dec, standardize=standardize_dec,
                                                lags=decode_lags_dec)
    print(f'  shuffle {s+1}/{n_shuf_dec}  cumulative {time.perf_counter()-t0:.1f}s')
_valid_dec = int((~np.isnan(target_dec).any(axis=1)).sum())
dec.f_plot_real_vs_shuffle_line(
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
target_blocks      = ['gaborS']    # list of block names from built_blocks registry
                                # Examples:
                                #   ['agg']                — position channels
                                #   ['motion']             — velocity channels
                                #   ['pix']                — pix-PCA components
                                #   ['grid']               — retinotopic grid (visual, Tier 1)
                                #   ['flow']               — optical flow on grid (visual, Tier 2)
                                #   ['gabor']              — Gabor oriented-edge energy (visual, Tier 3)
                                #   ['meng']               — motion energy, dir+speed (visual, Tier 4a)
                                #   ['agg', 'motion']      — position + velocity
                                #   ['agg', 'pix']         — position + pixel scene
                                #   ['agg', 'grid', 'motion', 'pix'] — everything
                                # Available block names depend on the build cell:
                                #   'agg', 'pix' are always present.
                                #   'motion' present if build_motion=True.
                                #   'per_obj' present if build_per_obj=True.
                                #   'grid'/'flow'/'gabor'/'meng' present if
                                #     build_visual=True and named in visual_specs.
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
    target_blocks = dec.legacy_target_blocks(target_choice_rec)
    print(f'  legacy target_choice_rec={target_choice_rec!r} → target_blocks={target_blocks!r}')

# Concatenated target (fresh array — no view mutation), optionally dropping the
# binary presence channels. No detrend here (fs omitted). skip_presence affects
# both the decoder fit and the plots.
target_rec, target_names_rec = dec.build_target_matrix(
    built_blocks, target_blocks, skip_presence=skip_presence)

# Build decoder input based on input_source.
if input_source == 'pca_neural':
    if use_shuffle:
        # uniform circshift null (seeded rng).
        _rng_rec = np.random.default_rng(shuf_seed)
        Y_use = dec.f_shuffle_neural(Y_neu, _rng_rec, min_shift=0)
        src_tag = 'SHUFFLED neural (per-cell circshift)  ←  PCA'
    else:
        Y_use = Y_neu
        src_tag = 'real neural  ←  PCA'
    pca_rec_full = dec.f_pca_prefix(Y_use, n_pcs_rec)
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
    # Estimator from the shared factory (pls/cca rank auto-clamped to the dims).
    if decoder_rec in ('pls', 'cca'):
        _n_c = dec.f_resolve_n_components(
            n_components_pls if decoder_rec == 'pls' else n_components_cca,
            pca_rec_full.shape[1], target_rec.shape[1])
        reg = dec.make_decoder(decoder_rec, n_components=_n_c)
    else:
        reg = dec.make_decoder(decoder_rec, ridge_alpha=ridge_alpha_rec, k_nn=k_nn_rec)
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
dec.f_plot_pred_scatter(target_rec, preds_rec, target_names_rec,
                    suptitle=f'{dset_tag}\nreal vs predicted  |  {_rec_tag}',
                    max_panels=max_targets_plot)
t_img = t_img_plot
dec.f_plot_pred_traces(target_rec, preds_rec, target_names_rec, t_img,
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
pca_neu3 = dec.f_pca_prefix(Y_neu, 3)
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
    from sklearn.linear_model import Ridge

    # Pick which CEBRA model to dissect. Use the agg-supervised one by
    # default; swap to 'pix' / 'grid' / others as needed.
    cebra_key = 'agg' if 'agg' in cebra_models else list(cebra_models)[0]
    cm = cebra_models[cebra_key]
    emb = cm['embedding']                         # (T, d_emb)
    scores_c = emb.T                              # (d_emb, T) — k × T
    S = S_active                                   # (n_active, T) — silent cells dropped, matches emb

    # ridge fit: S.T (T, n_cells) on emb (T, d_emb)
    W = Ridge(alpha=1.0).fit(emb, S.T).coef_      # (n_cells, d_emb)
    coeffs_c = W

    thresh_c, thresh_s = ens.f_ens_get_thresh(
        S, coeffs_c, scores_c, mode='signal_z',
        signal_z_thresh=2.5, dred_method='nmf',   # one-sided shuff doesn't apply
    )
    ens_out_cebra = ens.f_apply_thresh(coeffs_c, scores_c, thresh_c, thresh_s)

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

    ensp.f_plot_ens_overview(ens_out_cebra, S, mouse_dset_tag=dset_tag, cmap=raster_cmap,
                             **(dict(vmin=ens_vmin, vmax=ens_vmax) if ens_fix_scale
                                else dict(vmax_pct=ens_vmax_pct)))
    ensp.f_plot_comp_scatter(coeffs_c, ens_out_cebra['cells']['clust_ident'],
                        dim=3, title=f'{dset_tag} — CEBRA-{cebra_key} coeff scatter')
