# -*- coding: utf-8 -*-
"""
VR_feature_search.py — feature-PARAMETER search on ONE dataset, scored by ENCODE + DECODE.

PURPOSE: Tries many visual feature representations on one dataset (gabor magnitude/quadrature, FFT,
motion energy, wavelet, raw pixels) and ranks them by how well they predict neurons (encode) and how
well neurons predict them (decode).
PIPELINE: step 1 of 3 — VR_feature_search → VR_ca_encoding → VR_ca_cebra. See SCRIPTS.md.

Take one dataset and ask "which feature representation (and at what PCA granularity /
sampling / combination) encodes+decodes the neurons best?". Every candidate is REGENERATED
fresh on the GPU (force_build=True — the cache is bypassed so the search spans the full
parameter space, not the few cached settings); the monitor render runs ONCE and is shared
across configs, only the conv re-runs per config. Each config is scored by
f_dataset_scan.f_score_representations (decode R² neural→feature + encode r feature→neural).
The neural-scored sibling of VR_feature_optim.py (movie recon).

ONE builder `fsearch.f_feature_grid(kind, marginalize=(...), sf_cyc_deg, tf_hz, ...)`: the bank's
filters form an ori × wl(SF) × tf grid; `marginalize` pools a chosen set of axes and the REMAINING
axes become a param matrix scored cell-by-cell (marginalize=() → per filter, {'ori'} → per (wl,tf),
{'ori','wl','tf'} → whole bank). Two optional PCA layers (per-filter, per-cell). See cell 2; pivot a
per-filter result into the 3D score grid with `fsearch.f_score_grid`.

Spyder cell-based (#%% markers). Created 2026-07-18. Compute in functions/f_feature_search.py.
"""

import sys
import os
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

_thread_cap = 14                     # cap BLAS threads before numpy import (encode/PCA saturate all
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(_thread_cap)  # cores otherwise → machine locks up mid-sweep)

import numpy as np
import matplotlib.pyplot as plt

import f_feature_source as fsrc
from f_pipeline import f_load_session
from f_feature_source import f_default_source, f_default_build_cfg
import f_feature_search as fsearch
import f_ensemble_pipeline as enspipe
import f_encoding as enc
from f_decoding import f_imaging_fs


#%% Cell 1 — load ONE dataset + source / build config (slow: session load; run once)
mouse_id  = 'L'
n_dset    = 5                        # index into do_proc==1 rows for this mouse
base_tag  = 'gen_7_25_26'            # pca_feat cache GEN_TAG (…_pcacomb / …_pcasep live under it)

paths = {'data_dir_base': 'F:/VR/data_proc/', 'data_dir_bh': 'F:/VR/Bh_data/mice_gcamp/',
         'vr_data_xlsx': 'F:/VR/data_proc/VR_data.xlsx', 'features_dir': 'F:/VR/features/'}

session = f_load_session(mouse_id, n_dset, paths['data_dir_base'] + mouse_id,
                         paths['data_dir_bh'], paths['vr_data_xlsx'], behavior=True)
dset_tag = f"{mouse_id}/{session['dset_name']}"
print(f'dataset: {dset_tag}  ({session["T_img"]} frames)')

# Shared source (WHERE blocks load from) + build_cfg (for the fresh GPU build).
# Render size / obj_size / side / conv_backend all come from f_default_build_cfg defaults
# (lat_samp=150, vert_samp=101 = angularly-square pixels); only conv_device is overridden.
base_source = f_default_source(mouse_id=mouse_id, feat_root=paths['features_dir'], cache_tag=base_tag,
                               conv_tag=base_tag)
build_cfg = f_default_build_cfg(conv_device='cuda')
build_cfg['cam_params'] = session['cam_params']
# Terrain occlusion on the generated movie (feeds pix/gabor/meng). render_terrain=True composites the
# heightmap; terrain_shade='none' = invisible occluder (objects behind terrain vanish, no terrain
# content added) | 'gray'/'depth' = visible surface. terrain_stride dense enough to avoid bare columns.
build_cfg['render_terrain']  = False
build_cfg['terrain_mode']    = 'surface'
build_cfg['terrain_shade']   = 'none'
build_cfg['terrain_stride']  = 0.5
# Reload convolved+pooled gabor/meng/fft blocks from SSD instead of reconvolving. Saved PRE
# proc/marginalize, so proc variants of one bank reload. Keyed by extraction recipe × render geometry
# → <features_dir>/conv_autosave/<mouse>_<dset>_<geomtag>/. conv_autosave_tag namespaces the store.
build_cfg['conv_autosave']     = True
build_cfg['conv_autosave_tag'] = 'auto1'
# Drop point-samples keeping less than this fraction of their filter window on screen (the rest is
# zero pad). Geometry only. None = off.
build_cfg['edge_min_frac'] = 0.45
# ONE render shared by every sweep cell below: each passes prerendered=fsearch.f_render_once(...),
# which caches IN THE MODULE (so cells work in any order, no cross-cell variable) and re-renders only
# when a render-relevant build_cfg knob or the dataset changes. fsearch.f_render_clear() frees it.


#%% Cell 1a — ENCODER config + neural target + scoring procs (fast: re-run after any change here)
# Separate from cell 1 so changing an encoder / scoring knob never reloads the session or re-renders.
# ENCODER = the shared encoder config (f_encoding; defaults = VR_ca_cebra's enc_params, every key and its
# options documented in _enc_defaults()). Each scored block is encoded with f_run_encoder under this config
# (feature_block = that block, n_feature_pcs = None — the feature set's own layer1/layer2 procs set the
# dims). The neural target (Y_neu) also comes from it: target_sigma_frames / target_peak_norm.
# Override by name, e.g. f_default_encoder_params(ridge_alpha=10, lags_ms=[0, 200], target_sigma_frames=0).
enc_cfg = enc.f_default_encoder_params(
    # OLD feature-search encoder (before 2026-11-06), restored for comparison. Delete a line to use the
    # shared default instead (shown on the right).
    #ridge_alpha         = 1.0,            # default 1e3
    lags_ms             = [200],          # default [0, 100, 200, 300, 400, 500]
    #fold_scheme         = 'contiguous',   # default 'interleaved' - default improves
    #embargo_sec         = 0.5,            # default 0.0
    #standardize         = False,          # default True
    #target_sigma_frames = 0,              # default 6 (also the decode predictor's smoothing)
)
print(f'encoder config {enc.f_enc_config_tag(enc_cfg)}: ridge α={enc_cfg["ridge_alpha"]}, '
      f'lags={enc_cfg["lags_ms"]} ms, {enc_cfg["n_folds"]}-fold {enc_cfg["fold_scheme"]}, '
      f'target σ={enc_cfg["target_sigma_frames"]}')
# f_proc STEP-LIST ops (ordered, run in list order, any subset). [] = identity.
#   ('norm','zscore')       per-column mean 0 / std 1
#   ('norm','standardize')  global mean 0 / std 1
#   ('norm','total_var')    ÷ √Σ col-variance (no centering)
#   ('norm','first_sv')     ÷ top singular value (no centering)
#   ('pca', n)              CENTERED PCA. int comps | float var-frac | (frac,cap) | None
#   ('svd', n)              UNCENTERED truncated SVD — same n forms, mean NOT subtracted
#   ('whiten', True)        scores ÷ √eigenvalue (needs a preceding 'pca'/'svd')
# ALL procs are applied GLOBALLY, THEN the CV splits — the folds add NO further normalization, so a
# norm step in the list is what standardizes the predictor.
# pca centers before decomposing; svd does not, so svd's first component carries the mean/energy
# direction. var_kept is defined differently for the two — compare within an op.
layer1_proc = [('norm', 'zscore')]                  # feature grid: per filter (global)
layer2_proc = [('pca', 200)]                        # feature grid: per cell (global)

neural_proc = []                                    # DECODE predictor (neurons): global proc, e.g.
#neural_proc = [('norm', 'zscore'), ('pca', 30)]    #   [('norm','zscore'),('pca',30)]; [] = raw cells
encode_neural_proc = [('norm', 'zscore'), ('pca', 20)]   # OLD encoder only (enc_cfg=None); unused with enc_cfg

# Neural rate proxy (T, n_active) — same recipe as VR_ca_cebra's Y_neu (drop silent → smooth → peak-norm).
_tgt = enc.f_enc_target(session['est1'], enc_cfg)        # also the DECODE predictor
Y_neu, active_cells = _tgt['Y'], _tgt['active']


#%% Cell 1b — ENCODING TARGETS: cells and/or NMF ensembles (shared ensemble config + cache)
# Ensembles come from f_ensemble_pipeline's shared config (defaults = VR_ca_cebra's settings), loaded
# from <features_dir>/ens_cache/ if VR_ensembles.py (or an earlier run) already built this config,
# else fit + saved. Each feature block is built ONCE and scored against every target in enc_targets;
# the sweep plot draws one line style per target.
#   'cells'    the per-cell neural target from cell 1 (Y_neu)
#   'ens_all'  every ensemble's NMF score trace
#   'ens_sig'  significant ensembles only
use_targets = ['cells']                     # e.g. ['cells', 'ens_sig'] | ['cells', 'ens_all', 'ens_sig']
ens_cfg = enspipe.f_default_ens_config()    # same overrides as VR_ensembles.py, e.g. num_comp=12
enc_targets = {'cells': Y_neu}
if any(t.startswith('ens') for t in use_targets):
    ens_res = enspipe.f_ensembles_cached(session['est1'], ens_cfg, base_source['features_dir'], mouse_id,
                                         session['dset_name'], fs=f_imaging_fs(session['frame_t']),
                                         dset_tag=dset_tag)
    enc_targets.update({k: v[0] for k, v in enspipe.f_ens_targets(ens_res).items()})
missing = [t for t in use_targets if t not in enc_targets]
if missing:
    print(f'targets not available (e.g. no significant ensembles): {missing}')
enc_targets = {k: enc_targets[k] for k in use_targets if k in enc_targets}
print('encoding targets:', {k: v.shape for k, v in enc_targets.items()})


#%% Cell 2 — FEATURE SETS (ONE dict holds every parameter → f_feature_configs)
# '_common' = knobs every set inherits; each other key = one set stating only what differs.
# The dict key names the config → scorecard label '<kind>:specs:<key>' (pix → 'pix:<key>').
#
# HOW TO:
#   add a set            'myset': dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,))
#   vary one knob        copy the entry, change that knob, give it a new key (entry beats '_common')
#   unreduced variant    add layer2_proc=[]            (overrides the '_common' reduction)
#   pin a size           layer1_proc=L1(60)            (L1()/L2() with no arg use l1_n / l2_n)
#   combine levels       cyc_fov=(4, 8, 16)            marginalize='all' pools them into ONE block
#   combine orientations odegs=(0, 45, 90)
#   centered ↔ uncentered   flip l1_op / l2_op below   ('pca' | 'svd')
#   stop one filter dominating   l1_equalize='total_var'
#   sweep component counts       cell 3c: sweep_layer, sweep_names=['myset'], sweep_vals=[10, 50, 200]
#   check widths / RAM           fsearch.f_feature_sizes(res, ram_gb=128)
#
#   kind='gabor'  stats=('magnitude',) → gabM | ('quadrature',) → gabQe+gabQo | +('luminance',) DC
#   kind='fft'    stats as above (fftM / fftQe+fftQo) + n_sf_sub / n_ori_sub / patch_snap
#   kind='meng'   tf_hz=(…), output='opponent'|'both'   [no `stats`]
#   kind='pix'    layer1_proc/layer2_proc only — no ori/sf axes, ignores the AXIS knobs
# `kind` is validated: an unknown kind, or a knob that kind ignores, raises.
#
# marginalize = axes to POOL ⊆ {'ori','wl'/'sf','tf'}; the REMAINING axes form a param matrix, one
# scored CELL each:  ()→per filter (ori×sf×tf grid) | ('ori',)→per SF level | 'all'→whole bank (1 cell).
# cyc_fov = cycles per FOV; hFOV=98.4° so cyc_fov 12.8 ≈ 0.13 cyc/deg.
# ds='nyquist' (default) → per-SF stride λ/2, so each level is sampled at its own scale.
# Memory: convolves ONE (sf,tf) at a time, reduces, frees the raw → peak ≈ one conv's raw.
feat_sets = {
    # ── shared by every set below ────────────────────────────────────────────────────────────────
    '_common':   dict(cyc_fov=(1, 2, 4, 8, 16), odegs=(0, 45, 90), marginalize='all',
                      layer1_proc=None, layer2_proc=[('pca', 100)]),
    # ── five conv representations at identical axes — only the REPRESENTATION differs ────────────
    'gabmag':    dict(kind='gabor', stats=('magnitude',)),
    # quadrature = signed even/odd (re, ro); carries magnitude AND phase jointly, 2 channels per
    # filter vs magnitude's 1, so the layer-2 pca is what makes the head-to-head fair.
    'gabquad':   dict(kind='gabor', stats=('quadrature',)),
    # fft keeps the whole patch lattice binned onto the bank's (ori × SF) sectors; at
    # n_sf_sub=n_ori_sub=1 its columns align index-for-index with gabor. CPU only.
    'fftmag':    dict(kind='fft', stats=('magnitude',)),
    'fftquad':   dict(kind='fft', stats=('quadrature',)),
    # meng adds a tf axis (4 TF × 5 SF = 20 conv-units vs 5). transform='signed_sqrt' maps its ±8e9
    # energy onto the gabor scale. output='both' keeps each direction, retaining the SUM.
    'mengopp':   dict(kind='meng', tf_hz=(0.25, 0.5, 1.0, 2.0), output='opponent',
                      transform='signed_sqrt'),
    # ── pixel baseline: the real visual input ────────────────────────────────────────────────────
    'pix':       dict(kind='pix'),
    'pix_raw':       dict(kind='pix',   layer2_proc=[]),
    # ── RAW variants: layer2_proc=[] overrides the '_common' reduction ───────────────────────────
    # Decode peak = 3.7·T·n_dim·8 B (f_dataset_scan.DECODE_PEAK_FACTOR, measured, linear in n_dim).
    # Raw widths at the '_common' axes (nyquist ds → 12/54/210/1122/3780 per SF level, 3 ori,
    # 2 monitors), against 128 GB total / ~102 GB free:
    #                            cols     T=96,383      T=140,426
    #     gabmag / fftmag        5178      15 GB ok      22 GB ok
    #     gabquad / fftquad     10356      30 GB ok      43 GB tight
    #     mengopp (×4 tf)       20712      59 GB tight   86 GB CRASH
    #     pix                   30300      86 GB CRASH  126 GB CRASH
    # pix is worse: its block is 11.7/17.0 GB float32 and DOUBLES inside f_resample_to_imaging. Use
    # 'pix_pca2000' for the pixel ceiling — under ridge that is the SAME MODEL as raw pixels
    # (measured: predictions agree to 2e-10, decode_r2 to 1e-16), valid only with no ('norm',…) after
    # the pca. Check real numbers per run with fsearch.f_feature_sizes(res).
    'gabmag_raw': dict(kind='gabor', stats=('magnitude',),   layer2_proc=[]),
    'fftmag_raw': dict(kind='fft',   stats=('magnitude',),   layer2_proc=[]),
    # 'gabquad_raw': dict(kind='gabor', stats=('quadrature',), layer2_proc=[]),
    # 'fftquad_raw': dict(kind='fft',   stats=('quadrature',), layer2_proc=[]),
    # 'mengopp_raw': dict(kind='meng', tf_hz=(0.25, 0.5, 1.0, 2.0), output='opponent',
    #                     transform='signed_sqrt', layer2_proc=[]),
    # 'pix_pca2000': dict(kind='pix', layer2_proc=[('pca', 2000)]),   # the tractable raw-pixel ceiling
}
    
    
# Reduction op + default size for each layer, used by every entry below via L1()/L2().
# 'pca' = centered | 'svd' = uncentered. Size: float = variance fraction (count adapts per unit) |
# int = fixed count | (frac, cap) | None = unreduced.
#
# LAYER 1 runs PER FILTER — one group = one (orientation, SF), BOTH monitors, all its spatial samples
#   ['gabM_o0_sf0', 'gabM_o0_sf1', …]. LAYER 2 runs on the CONCATENATED cell.
# The reduction returns SCORES, not unit-norm components: score variance = the eigenvalue, so each
#   filter enters the concat at its own scale and layer 2 weights filters by their retained variance.
#   MEASURED, two filters 100x apart in variance: the quiet one contributes 0.0% and PC1 loads 100%
#   on the loud one. A norm step fixes it, but ONLY at the END OF LAYER 1 — the same step in
#   layer2_proc rescales the already-mixed block by one scalar and leaves the split at 100/0.
#
# ('norm', …) scope, appended after the reduction:      ratio = spread between scores, kept or flattened
#   total_var    one scalar, ÷√Σ col-var          no centering   ratio kept    -> equalize filters
#   first_sv     one scalar, ÷ top singular value no centering   ratio kept    -> equalize filters
#   standardize  one scalar, global mean & std    CENTERS        ratio kept
#   zscore       per score, mean 0 / std 1        CENTERS        ratio -> 1
#   whiten       per score, ÷√eigenvalue          no centering   ratio -> 1    -> = taking the PCs
# Under svd the scores are NOT zero-mean, so zscore removes the mean/energy direction svd was chosen
#   to keep (measured: PC1 loading fell to 4%). Use whiten for per-score normalization under svd.
# MEASURED n_dim at layer2 svd .85 on a 2-filter block: none 17 | total_var 33 | whiten 53 — whitening
#   inflates each filter's low-variance tail, so layer 2 spends its budget on near-noise directions.
l1_op = 'svd'          # per-filter reduction (layer 1)
l2_op = 'svd'          # per-cell reduction   (layer 2)
l1_n = 0.95            # default layer-1 size
l2_n = None            # default layer-2 size (cell 3c sweeps this)
l1_equalize = None     # None | 'total_var' | 'first_sv' | 'whiten' — appended after the layer-1 reduction
def L1(n=None):        # L1() uses l1_n; L1(60) overrides
    return [(l1_op, l1_n if n is None else n)] + (
        [('whiten', True)] if l1_equalize == 'whiten' else
        [('norm', l1_equalize)] if l1_equalize else [])
def L2(n=None): return [(l2_op, l2_n if n is None else n)]   # L2() uses l2_n; L2(40) overrides

feat_sets = {
    # ── shared by every set below ────────────────────────────────────────────────────────────────
    '_common':   dict(cyc_fov=(1, 2, 4, 8, 16), odegs=(0,), marginalize='all',      #  45, 90
                      layer1_proc=None, layer2_proc=None),  # L2(100)
    # # ── the five conv representations — identical axes, so ONLY the representation differs ───────
    
    # 'gabmag1.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1,), odegs=(0,)),
    # 'gabmag2.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(2,), odegs=(0,)),
    # 'gabmag4.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,)),
    # 'gabmag8.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,)),
    # 'gabmag16.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16,), odegs=(0,)),
    # 'gabmag1.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1,), odegs=(45,)),
    # 'gabmag2.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(2,), odegs=(45,)),
    # 'gabmag4.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(45,)),
    # 'gabmag8.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(45,)),
    # 'gabmag16.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16,), odegs=(45,)),
    # 'gabmag1.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1,), odegs=(90,)),
    # 'gabmag2.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(2,), odegs=(90,)),
    # 'gabmag4.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(90,)),
    # 'gabmag8.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(90,)),
    # 'gabmag16.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16,), odegs=(90,)),
    # 'gabmag4.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,)),
    # 'gabmag4.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(45,)),
    # 'gabmag4.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(90,)),
    # 'gabmag4.0+45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,45,)),
    # 'gabmag4.45+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(45,90,)),
    # 'gabmag4.0+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,90,)),
    # 'gabmag4.0+45+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,45,90,)),
    'gabmag8.0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,)),
    'gabmag8.45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(45,)),
    'gabmag8.90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(90,)),
    'gabmag8.0+45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,45,)),
    'gabmag8.45+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(45,90,)),
    'gabmag8.0+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,90,)),
    'gabmag8.0+45+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,45,90,)),
    # 'gabmag1+2':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1,2), odegs=(0,)),
    # 'gabmag1+2+4':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1,2, 4), odegs=(0,)),
    # 'gabmag1+2+4+8':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1,2,4,8), odegs=(0,)),
    # 'gabmag4_0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,)),     # 22, .99
    # 'gabmag4_45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(45,)),     # 22, .99
    # 'gabmag4_90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(90,)),     # 20, .6
    # 'gabmag4_0+45+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0, 45, 90,)),
    # 'gabmag4_0+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0, 90,)),
    # #'gabmag4_0_pca40':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,), layer1_proc=L2(40)),
    # # 'gabmag4_0_pca20':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(0,), layer1_proc=L2(20)),
    # # 'gabmag4_90_pca40':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(90,), layer1_proc=L2(40)),
    # # 'gabmag4_90_pca20':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4,), odegs=(90,), layer1_proc=L2(20)),
    
    # 'gabmag8_0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,)),        
    # 'gabmag8_45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(45,)),        
    # 'gabmag8_90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(90,)),        
    # 'gabmag8_0+90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0, 90,)),
    
    # 'gabmag8_0':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0,), layer1_proc=L1(60)),  # .96, 40
    # 'gabmag8_45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(45,), layer1_proc=L1(60)),  # .90, 40
    # 'gabmag8_90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(90,), layer1_proc=L1(60)),  # .90, 40
    # 'gabmag8_0_90_60':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0, 90), layer1_proc=L1(40), layer2_proc=L2(60)),
    # 'gabmag8_0_45_90_60':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0, 45, 90), layer1_proc=L1(40), layer2_proc=L2(60)),
    # 'gabmag8_0_45_60':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8,), odegs=(0, 45), layer1_proc=L1(40), layer2_proc=L2(60)),
    
    # 'gabmag4_8_3o':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4, 8,), odegs=(0, 45, 90), layer1_proc=L1(.95), layer2_proc=L2(60)),
    # 'gabmag8_16_3o':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(8, 16), odegs=(0, 45, 90), layer1_proc=L1(.95), layer2_proc=L2(60)),
    # 'gabmag4_8_16_3o':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(4, 8, 16), odegs=(0, 45, 90), layer1_proc=L1(.95), layer2_proc=L2(.85)), # L2(60)
    # 'gabmag1_2_4_8_16_3o':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1, 2, 4, 8, 16), odegs=(0, 45, 90), layer1_proc=L1(.95), layer2_proc=L2(.85)), # L2(60)
    
    # 'gabmag1_4':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(1, 4), layer1_proc=L1(100)),
    # 'gabmag16_0_pca100':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16,), odegs=(0,), layer1_proc=L1(100)), # 120, .96
    # 'gabmag16_45_pca100':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16,), odegs=(45,), layer1_proc=L1(100)), # 100, .8
    # 'gabmag16_90_pca100':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16,), odegs=(90,), layer1_proc=L1(100)), # 100, .85
    # 'gabmag16_0_90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16), odegs=(0, 90), layer1_proc=L1(100)), # 50 .8
    # 'gabmag16_0_45':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16), odegs=(0, 45), layer1_proc=L1(100)), # 127 ,60 .9
    # 'gabmag16_0_45_90':    dict(kind='gabor', stats=('magnitude',), cyc_fov=(16), odegs=(0, 45, 90), layer1_proc=L1(100)),  # 67 0.85
    
    # #'fftmag_raw': dict(kind='fft',   stats=('magnitude',),   layer2_proc=[]),
    # 'fftmag1': dict(kind='fft',   stats=('magnitude',), cyc_fov=(1,), layer1_proc=L1(100)),
    # 'fftmag2': dict(kind='fft',   stats=('magnitude',), cyc_fov=(2,), layer1_proc=L1(100)),
    # 'fftmag4': dict(kind='fft',   stats=('magnitude',), cyc_fov=(4,), layer1_proc=L1(100)),
    # 'fftmag8': dict(kind='fft',   stats=('magnitude',), cyc_fov=(8,), layer1_proc=L1(100)),
    # 'fftmag1_2_4': dict(kind='fft',   stats=('magnitude',), cyc_fov=(1, 2, 4,), layer1_proc=L1(100)),
    # 'fftmag1_2': dict(kind='fft',   stats=('magnitude',), cyc_fov=(1, 2, ), layer1_proc=L1(100)),
    # 'fftmag2_4': dict(kind='fft',   stats=('magnitude',), cyc_fov=(2, 4,), layer1_proc=L1(100)),
}
print(f'reduction: layer1={l1_op}  layer2={l2_op}')

# Overwrite every layer-1 size with the variance fraction l1_n. `if 0:` keeps the hard-wired counts.
# Only entries that already set layer1_proc are touched; each is renamed with the live layer-1 setting
# APPENDED (never stripping the old name — a trailing _pca<N> may describe layer 2, and removing it
# would collide two entries into one). sweep_names may use the base name either way.
if 0:
    _out, _ren = {}, {}
    for _k, _v in feat_sets.items():
        _n = _k
        if _k != '_common' and _v.get('layer1_proc'):
            _v['layer1_proc'] = L1(l1_n)
            _n = f'{_k}_l1{l1_op}{l1_n:g}'
            _ren[_k] = _n
        if _n in _out:
            raise ValueError(f'layer1 override: name collision on {_n!r}')
        _out[_n] = _v
    feat_sets = _out                      # rebuilt in order, so the as-built plot order is unchanged
    print(f'layer1 OVERRIDE ({l1_op}, {l1_n}): {len(_ren)} of {len(feat_sets) - 1} entries '
          f'[set `if 0:` for hard-wired counts]')

configs = fsearch.f_feature_configs(feat_sets)

print(f'{len(configs)} configs: {[c["name"] for c in configs]}')
# A PROGRAMMATIC sweep (many values × all sets, without writing each out) is the other idiom: a `common`
# KWARG overrides '_common' for one whole pass, and `tag_suffix` keeps the names distinct. Precedence is
# entry > kwarg > '_common', so a set that PINS the knob (the _raw entries above) stays put:
#   configs = sum([fsearch.f_feature_configs(feat_sets, layer2_proc=[('pca', n)], tag_suffix=f'pca{n}')
#                  for n in (50, 100, 200, 400)], [])

# ── PREVIOUS SETS (reference — the proc/axis variants swept before the format was centralized) ─────
# Every one of these is now an entry in the single dict above (or a kwarg pass, as just shown):
#   layer1/layer2 proc variants on ONE magnitude bank, marginalize=('ori',):
#     'mag'          layer1 None                  layer2 None
#     'mag2pca'      layer1 None                  layer2 [('pca', 50)]
#     'mag1norm'     layer1 [('norm','zscore')]   layer2 None
#     'mag2norm'     layer1 None                  layer2 [('norm','zscore')]
#     'mag1norm2pca' layer1 [('norm','zscore')]   layer2 [('pca', 50)]
#   the same five with marginalize='all' ('magcomb', 'magcomb1norm', 'magcomb2pca', …),
#   'magcomb2pca2norm' = layer2 [('pca',50),('norm','zscore')], and the quadrature twins
#   ('quad', 'quad2pca', 'quad1norm2pca', 'quadcomb2pca', 'quadcomb1norm2pca').
#   SF-extent variants: 'mag32comb2pca' cyc_fov=(1,2,4,8,16,32) / 'mag8comb2pca' cyc_fov=(1,2,4,8).
#   pix proc sets: [('pca',100)] / [('pca',150)] / [('pca',500)] / [('pca',50),('norm','zscore')].
#   'maglum'  = stats=('magnitude','luminance')  → adds the DC wavelet.
#   'fftmq'   = stats=('magnitude','quadrature') → both fft stats in one block.
#   'fftsub2' = n_sf_sub=2, n_ori_sub=2          → sub-binned sectors, strictly richer than the bank
#               (breaks the 1:1 column correspondence with gabor).
#   meng tf banks: tf_hz=(1.0,) at sf_cyc_deg=(0.13,) ds=4 odegs=(0,90); 'tf123' tf_hz=(1.0,2.0,3.0).
# A marginalize sweep on the same sets = one f_feature_configs call per value, e.g.
#   sum([fsearch.f_feature_configs(feat_sets, marginalize=m, tag_suffix=f'm{i}')
#        for i, m in enumerate([(), ('ori',), 'all'])], [])


#%% Cell 3 — run the search + scorecard (force_build=True → fresh GPU, render once)
# verbose = the standard per-config / per-cell summary (unchanged). timing = EXTENDED log: flushed
# timestamped START + duration markers for each MAJOR step (render, per conv-unit, decode, encode) so
# you can read what's running while watching CPU. Leave timing=False for the compact output.
res = fsearch.f_feature_search(session, configs, Y_neu, base_source, build_cfg,
                               lags_ms=(200,), embargo_sec=0.5, n_folds=5, r_sig=0.1, force_build=True,
                               neural_proc=neural_proc, encode_neural_proc=encode_neural_proc,
                               prerendered=fsearch.f_render_once(session, build_cfg, shape_dir=base_source['features_dir']),
                               verbose=True, timing=True, targets=enc_targets, enc_cfg=enc_cfg)
print('\n=== feature-param search (build order) ===')
print(res[['config', 'block', 'decode_r2', 'decode_r2_uniform', 'decode_r2_med',
           'encode_r_med', 'encode_r2_med', 'encode_frac_sig', 'n_dim', 'status']].to_string(index=False))

#%% Cell 3a — results of cell 3: size table + scorecard plots + per-filter score grid
# SIZE monitor — one row per scored block: n_frames × n_dim (width actually decoded), raw_dim (width
# BEFORE the layer/PCA proc, so `reduction` shows what the proc bought), block_mb, and the projected
# decode peak (3.7·T·n_dim·8 B, measured). `ram_gb` flags ok / tight (>50%) / OVER (>90%).
# The per-cell lines during the run carry the same shapes inline.
# size_sort: any column (descending) | 'none' = keep the INPUT order (the order feat_sets/configs
# were listed in — res comes back in build order).
size_sort = 'none'          # 'decode_peak_gb' to rank by projected RAM instead
fsearch.f_feature_sizes(res, ram_gb=128, sort_by=size_sort)
# Scorecard (clean bars). plot_sort: 'none' = INPUT order (as listed in feat_sets) | 'decode' |
# 'encode' | any column name. show_n_dim appends the scored width to each y label as [Nd].
#   decode_metric ∈ r2 | r2_uniform | r2_med | validfrac ; encode_metric ∈ r_med | r_mean | r2_med | r2_mean | frac_sig
plot_sort = 'none'          # 'encode' / 'decode' to rank by that panel's metric
fsearch.f_plot_feature_search(res, sort_by=plot_sort, title=f'{dset_tag} — feature-param search',
                              decode_metric='r2', encode_metric='r_med', show_n_dim=True)
# Distribution view: same bars + percentile error bars over feature cols (decode) / cells (encode) —
# reveals sparse encoders the median hides. Toggle each overlay: show_iqr (25–75 box) / show_pct
# (5–95 whiskers) / show_minmax (◆ min/max).

# fsearch.f_plot_feature_dist(res, sort_by=plot_sort, title=f'{dset_tag} — feature-param search (dist)',
#                             decode_metric='r2_med', encode_metric='r_med',
#                             show_iqr=True, show_pct=True, show_minmax=False)

# more dist examples (any DECODE_METRICS / ENCODE_METRICS key):
# fsearch.f_plot_feature_dist(res, decode_metric='r2', encode_metric='r2_med')          # encode R² spread
# fsearch.f_plot_feature_dist(res, decode_metric='r2_med', encode_metric='r_med',       # IQR only
#                             show_pct=False, show_minmax=False)

# e.g. robust decode + significant-cell fraction:
# fsearch.f_plot_feature_search(res, sort_by='encode_frac_sig',
#                               decode_metric='r2_med', encode_metric='frac_sig')

# Per-FILTER grid → pivot the marginalize=() cells into the ori × (sf,tf) score matrix.
_g = fsearch.f_score_grid(res, 'decode_r2')
if _g['ori'].notna().any():
    print('\ndecode R² grid (ori × sf,tf):')
    print(_g.pivot_table('decode_r2', 'ori', ['sf', 'tf']).to_string())


#%% Cell 3c — COMPONENT SWEEP: performance vs components kept, layer 1 or layer 2, encode or decode
# sweep_layer: 'layer1' = per-filter reduction | 'layer2' = reduction of the combined cell.
#   For a single-filter set the two give identical blocks; they differ for combinations.
#   layer1 int counts apply PER FILTER (n_dim = sum over filters).
# sweep_vals forms: int count | float variance fraction | (frac, cap) | None unreduced.
# f_proc_sweep strips a pinned layer1_proc/layer2_proc from the selected sets and prints which.
# Points past a set's full width (fewer components than requested) are clipped from the line plot.
# Speed: only sweep_score is fitted (both if show_grid); each filter's decomposition is fit once and
# sliced per value; a value giving an already-scored block reuses that score.
sweep_layer  = 'layer1'                 # 'layer1' | 'layer2'
sweep_score  = 'encode'                 # 'encode' | 'decode'
sweep_metric = None                     # None → encode 'r_med' / decode 'r2'; any ENCODE/DECODE_METRICS key
sweep_names  = ['all']          # keys from feat_sets in cell 2 (several = several lines); 'all' = every set
sweep_vals   = [1, 2, 4, 8, 16, 32, 64,                 # component counts
                0.5, 0.8, 0.9, 0.95, 0.99, .999]              # variance fractions
                #(0.99, 150),                           # fraction, capped
                #None]                                  # unreduced
sweep_ops    = (l1_op if sweep_layer == 'layer1' else l2_op,)   # ('pca', 'svd') = one line per op
# Processing steps around the reduction. sweep_pre × sweep_post → one line per pair; an entry may chain
# steps, e.g. [('norm','total_var'), ('whiten', True)].
#   step                     pre (before reduction)                  post (after reduction)
#   ('norm', 'zscore')       each column mean 0 / std 1 → corr-PCA   each component mean 0 / std 1
#   ('norm', 'standardize')  one global mean / std (same pca comps)  one global mean / std
#   ('norm', 'total_var')    ÷ √Σ col-var (same comps)               equalizes filters (layer1)
#   ('norm', 'first_sv')     ÷ top singular value (same comps)       equalizes filters (layer1)
#   ('whiten', True)         not allowed                             ÷ √variance of each component
# '(same comps)' = rescale only, components unchanged. 'whiten' is dropped for an unreduced value.
sweep_pre    = [[],
                #[('norm', 'zscore')],
                #[('norm', 'standardize')],
                ]
sweep_post   = [[],
                #[('norm', 'zscore')],
                #[('norm', 'standardize')],
                #[('norm', 'total_var')],
                #[('norm', 'first_sv')],
                #[('whiten', True)],
                ]
show_grid    = False                    # also draw the 2×2 decode/encode × n_dim/var_kept panel

# previous layer-2 settings:
# sweep_layer = 'layer2'; sweep_names = ['gabmag4_8_16_3o']; sweep_ops = ('pca', 'svd')
# sweep_vals = [0.5, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, .995, .999]   # or 10, 25, 50, 100, 200

configs_pc = sum([fsearch.f_proc_sweep(feat_sets, sweep_vals, layer=sweep_layer, names=sweep_names, op=_o,
                                       post=sweep_post, pre=sweep_pre)
                  for _o in sweep_ops], [])
print(f'{len(configs_pc)} sweep configs: {[c["name"] for c in configs_pc]}')

# Same bank across configs → share the render and keep the pooled conv raw in RAM (keep_raw),
# else each value re-reads it from the conv_autosave store.
res_pc = fsearch.f_feature_search(session, configs_pc, Y_neu, base_source, build_cfg,
                                  lags_ms=(200,), embargo_sec=0.5, n_folds=5, r_sig=0.1,
                                  force_build=True, neural_proc=neural_proc,
                                  encode_neural_proc=encode_neural_proc,
                                  prerendered=fsearch.f_render_once(session, build_cfg, shape_dir=base_source['features_dir']),
                                  keep_raw=True, verbose=True, timing=False,
                                  scores=('decode', 'encode') if show_grid else (sweep_score,),
                                  targets=enc_targets, enc_cfg=enc_cfg)
print(res_pc[['config', 'block', 'n_dim', 'l1_dim_in', 'raw_dim', 'l1_var_kept', 'var_kept',
              'decode_r2', 'encode_r_med', 'status']].to_string(index=False))
fsearch.f_plot_comp_sweep(res_pc, score=sweep_score, metric=sweep_metric, layer=sweep_layer,
                          clip_saturated=True,
                          title=f'{dset_tag} — {sweep_score} vs {sweep_layer} components')
if show_grid:                           # 2×2: decode/encode × (n_dim, variance kept of the swept layer)
    fsearch.f_plot_pca_sweep(res_pc, decode_metric='r2', encode_metric='r_med', logx=False,
                             var_col='l1_var_kept' if sweep_layer == 'layer1' else 'var_kept',
                             title=f'{dset_tag} — performance vs components kept')
fsearch.f_feature_sizes(res_pc, ram_gb=128, sort_by='none')   # 'none' = sweep order; or any column


#%% Cell 3b (OLD, superseded by cell 3c `sweep_post`) — normalize / whiten test: expand configs × feat_prep
# Old config format (gaborS, ds=4). Use cell 3c's sweep_post for normalize/whiten on feat_sets banks.
# f_prep_configs adds a '|<prep>' variant per base config. NOTE: the search applies all procs before
# the CV split with NO within-fold standardization (proc_before_cv=True), so prep moves ENCODE too —
# the ridge penalty depends on column scale — as well as DECODE (feature = target).
if 0:
    gab_in = [{'source': 'gabor', 'reduce': 'sample', 'ds': 4, 'sf_cyc_deg': (0.13,), 'odegs': None}]
    base = fsearch.f_granularity_configs('gaborS', gab_in, ('combined',), n_pca=300)
    prep_cfgs = fsearch.f_prep_configs(base, preps=(None, 'zscore', 'whiten'))
    res_prep = fsearch.f_feature_search(session, prep_cfgs, Y_neu, base_source, build_cfg, verbose=True)
    print(res_prep[['config', 'feat_prep', 'decode_r2', 'encode_r_med', 'n_dim']].to_string(index=False))
    fsearch.f_plot_feature_search(res_prep, title=f'{dset_tag} — feature prep (normalize/whiten)')


#%% Cell 4 (example, opt-in) — marginal sweep over sampling ds (all fresh GPU build)
# Vary ONE axis (ds), hold the rest. The render is reused across these too (rendered once above).
if 0:
    ds_list = [2, 4, 6]
    sweep_cfgs = [{'name': f'gaborS_ds{ds}:combined', 'blocks': ['gaborS'],
                   'plan': {'gaborS': {'origin': 'build', 'n_pca': 300,
                                       'inputs': [{'source': 'gabor', 'reduce': 'sample', 'ds': ds,
                                                   'sf_cyc_deg': (0.13,), 'odegs': None}]}}}
                  for ds in ds_list]
    res_ds = fsearch.f_feature_search(session, sweep_cfgs, Y_neu, base_source, build_cfg, verbose=True)
    print(res_ds[['config', 'decode_r2', 'encode_r_med', 'n_dim']].to_string(index=False))
    fsearch.f_plot_feature_search(res_ds, title=f'{dset_tag} — gaborS ds sweep')
