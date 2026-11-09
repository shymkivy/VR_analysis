# -*- coding: utf-8 -*-
"""
VR_feature_gen.py — build + cache feature blocks ONCE for every dataset (TODO #66).

Render (f_angles_to_movie_v2) + build_feature_blocks + build_visual_blocks is slow
(minutes/dataset, gabor/meng dominate). This batch script runs it once over all
do_proc datasets and saves each as a joblib cache; the analysis scripts
(VR_ca_cebra.py / VR_ca_encoding.py) then LOAD instead of rebuilding
(load_cached_features=True). Caches store FEATURES only — neural stays in data_ca;
the raw movie is skipped unless SAVE_MOVIE.

Config is the inline CONFIG dict below — edit it in Spyder. Bump GEN_TAG when the
feature method changes (new cache filenames + staleness detection).

Created 2026-06-24.
"""

import sys
import os
import gc
import time
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

_thread_cap = 6                      # cap BLAS threads before numpy import (PCA/scipy saturate all
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(_thread_cap)  # cores otherwise → machine locks up mid-run)

import numpy as np
import pandas as pd

from f_utils import f_load_caim_data
from f_functions import f_load_bh_data_all, f_comp_FOV_adj
from f_decoding import f_imaging_fs
from f_feature_cache import (f_save_features, f_features_path, f_features_exist,
                             f_write_config_json, f_pick_frames)
from f_feature_gen import (f_build_one_dataset, f_build_conv_repo_one_dataset,
                           f_build_pca_feat_one_dataset, f_resolve_dsets)

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:                  # progress bar optional
    _TQDM = False

def _log(msg):
    """Print without clobbering the tqdm bar (tqdm.write when available)."""
    if _TQDM:
        tqdm.write(msg)
    else:
        print(msg, flush=True)

#%% Config — edit inline. Bump GEN_TAG when the feature method changes.
cfg = {
    'GEN_TAG': 'gen_7_16_26', 'MICE': ['L'], 'SAVE_MOVIE': True, 'OVERWRITE': False,
    # GEN_MODE: 'blocks' (reduced+PCA'd feature-block caches, f_save_features) | 'conv_repo' (dense
    # per-filter convolved maps for decoder optimization; uses cfg['conv_specs']; incremental) |
    # 'pca_feat' (convolve → PCA → save ONLY small reduced blocks; uses cfg['pca_specs'] + the SAVE_*
    # / n_pca_* knobs; two granularities to separate files: '_pcacomb' per-input, '_pcasep' per-filter).
    'GEN_MODE': 'pca_feat',
    # gabor/meng convolution backend for BOTH modes (conv_repo dense maps AND blocks-mode gabor/meng
    # build): 'cpu' (scipy) | 'gpu' (torch, f_conv_gpu — parity-tested, much faster on the RTX 3090).
    # conv_device = torch device for backend='gpu' (None → auto = CUDA if available else CPU).
    'conv_backend': 'gpu', 'conv_device': 'cuda',
    # which datasets per mouse (indices are into the do_proc==1 rows, same as n_dset):
    #   None                  ALL do_proc datasets
    #   4                     just do_proc index 4
    #   [0, 4, 7]             several by index
    #   ['2024_01_15']        by session name (exact, else substring match)
    #   {'L': [0, 4], 'M': None}   per-mouse selection
    'DSETS': None,              # None = ALL do_proc datasets (build 1-by-1)
    'MAX_DATASETS': None,       # cap on TOTAL datasets built this run (None = no cap;
                               # cache-skips don't count). Handy for a partial/test run.
    'paths': {'data_dir_base': 'F:/VR/data_proc/', 'data_dir_bh': 'F:/VR/Bh_data/mice_gcamp/',
              'vr_data_xlsx': 'F:/VR/data_proc/VR_data.xlsx', 'features_dir': 'F:/VR/features/'},
    'render': {'cam_params': {'aspect': 16 / 9, 'FOV_axis': 'horizontal', 'FOV_deg': 98.4,
                              'cam_rotation_deg': 49.2, 'clip_len': 60, 'num_mon': 2,
                              'cam_height': 0.6},
               'obj_size': {'x': 5, 'y': 20, 'z': 5, 'height': 2}, 'num_samp': 101,
               # render width/height. lat:vert = hFOV:vFOV ≈ 1.49 → angularly-square pixels
               # (~0.66 deg/px both axes; the angle→pixel map is linear, NOT the physical 16:9).
               'lat_samp': 150, 'vert_samp': 101,
               # terrain occlusion on the generated movie (feeds pix/gabor/meng). render_terrain=True
               # composites the heightmap; terrain_shade='none' = invisible occluder (objects behind
               # terrain vanish, no terrain content) | 'gray'/'depth' visible.
               'render_terrain': False, 'terrain_mode': 'surface', 'terrain_shade': 'none',
               'terrain_stride': 0.5, 'terrain_point_size': 1, 'terrain_clip_len': None,
               'terrain_gray_level': 128, 'terrain_close_iter': 2,
               'interp_step': 0.1, 'max_frames': None},
    'feature_blocks': {'side': 'both', 'agg_mode': 'angular', 'n_pix_pca': 50,
                       'obj_collapse': 'salience_mean', 'nan_absent': True,
                       'build_motion': True, 'motion_smooth_sigma': 2.0,
                       'build_self_motion': True, 'self_motion_smooth_sigma': 2.0,
                       'build_pix_motion': True, 'pix_motion_smooth_sigma': 2.0,
                       'build_lick_reward': True, 'lick_reward_smooth_sigma': 0.0,
                       'build_per_obj': False},
    'visual_specs': {
        #'grid':  {'grid': (12, 12), 'n_pca': 100, 'stats': ('occupancy', 'mean', 'edge')},
        #'flow':  {'grid': (12, 12), 'n_pca': 100, 'method': 'lucas_kanade',
        #          'stats': ('speed', 'dir', 'div')},
        #'gabor': {'grid': (12, 12), 'n_orient': 4, 'wavelengths': (6,),
        #          'stats': ('magnitude', 'phase'), 'n_pca': {'gabM': 50, 'gabP': 50}},
        #'meng':  {'grid': (12, 12), 'n_orient': 4, 'wavelengths': (8,),
        #          'temporal_freqs': (0.15,), 'output': 'opponent', 'n_pca': 100}
        },
    # conv_repo mode: which dense filters to convolve + save (GEN_MODE='conv_repo' only).
    # PHYSICAL units — gabor spatial freq in cycles/degree (sf_cyc_deg), meng temporal freq in Hz
    # (tf_hz); f_conv_filter_specs_phys converts to px/frame at the render geometry + imaging fs, and
    # the manifest stores sf_cyc_deg/tf_hz so f_load_conv_blocks selects filters in cyc/deg + Hz.
    # sf grid ≈ legacy 18/15/12 px at 150×101 (0.085/0.10/0.13 cyc/deg).
    'conv_specs': {
        'gabor': {'n_orient': 4, 'sf_cyc_deg': (0.13,)},
        'meng':  {'n_orient': 4, 'sf_cyc_deg': (0.13,), 'tf_hz': (1.0, 2.0, 3.0),
                  'temporal_support': 'auto', 'n_cycles': 2, 'support_max': 91,  # per-tf ~2 cycles/tf
                  'output': 'opponent'}
        },
    # pca_feat mode (GEN_MODE='pca_feat'): independent save granularities, each to its own tagged
    # cache. n_pca_* = int (comp count) | float(0,1) (variance) | (frac, cap) (variance capped). Saved
    # PCA blocks are UNWHITENED (whiten is a load-time toggle in f_resolve_features / cebra).
    'SAVE_COMBINED': True,    # '_pcacomb': one PCA per INPUT (all orientations, L‖R combined)
    'SAVE_SEPARATED': True,   # '_pcasep' : one PCA per FILTER (orientation×SF×TF, L‖R combined)
    'SAVE_RAW': False,        # '_pcaraw' : sampled conv WITHOUT PCA (large; the full point-sample)
    'n_pca_combined': (0.995, 600),
    'n_pca_separated': (0.995, 300),
    # Also cache the pix movie-PCA (reuses the render already done for gabor/meng) so cebra/encode-scan
    # load pix from the cache with NO re-render. Saved into every granularity cache (passthrough).
    'SAVE_PIX': True,
    'n_pix_pca': 300,
    # pca_specs: which sampled gabor/meng INPUTS to build (feature-source 'inputs' schema; physical
    # sf_cyc_deg / tf_hz; ds = point-sample stride). Each becomes one saved block per granularity.
    # ONE block per tf (mengS=1Hz, mengS2=2Hz, mengS3=3Hz) so the derived composites can from_block
    # a chosen tf subset without reconvolving. Must match the mengS/mengS2/mengS3 blocks in the plan.
    'pca_specs': {
        'gaborS': {'inputs': [{'source': 'gabor', 'reduce': 'sample', 'ds': 4,
                               'sf_cyc_deg': (0.13,), 'odegs': None}]},
        'mengS':  {'inputs': [{'source': 'meng', 'reduce': 'sample', 'ds': 4, 'transform': 'signed_sqrt',
                               'sf_cyc_deg': (0.13,), 'tf_hz': (1.0,), 'odegs': None, 'output': 'opponent'}]},
        'mengS2': {'inputs': [{'source': 'meng', 'reduce': 'sample', 'ds': 4, 'transform': 'signed_sqrt',
                               'sf_cyc_deg': (0.13,), 'tf_hz': (2.0,), 'odegs': None, 'output': 'opponent'}]},
        'mengS3': {'inputs': [{'source': 'meng', 'reduce': 'sample', 'ds': 4, 'transform': 'signed_sqrt',
                               'sf_cyc_deg': (0.13,), 'tf_hz': (3.0,), 'odegs': None, 'output': 'opponent'}]},
        },
}
print(f'GEN_TAG={cfg["GEN_TAG"]}  mice={cfg["MICE"]}')


#%% Batch — loop mice × selected datasets, build + save (1-by-1, up to MAX_DATASETS)
summary = []
n_run = 0                                   # datasets actually built this run (not skips)
max_dsets = cfg.get('MAX_DATASETS')         # None = no cap
stop = False
for mouse_id in cfg['MICE']:
    if stop:
        break
    data_dir = cfg['paths']['data_dir_base'] + mouse_id
    vr_data = pd.read_excel(cfg['paths']['vr_data_xlsx'])
    params_xls = pd.read_excel(cfg['paths']['data_dir_bh'] + mouse_id + '_params.xlsx')
    vr_data2 = vr_data.loc[(vr_data.mouse_id == mouse_id) &
                           (vr_data.do_proc == 1)].reset_index(drop=True)
    data_ca = f_load_caim_data(data_dir, vr_data2.dset_name, caiman_tag='results_cnmf.hdf5',
                               cuts_tag='h5cutsinfo', frame_data_tag='framedata',
                               r_values_min=0.5, min_SNR=1.5, thresh_cnn_min=0.8)
    bh_data = f_load_bh_data_all(cfg['paths']['data_dir_bh'] + '/' + mouse_id + '/',
                                 vr_data2.bh_dset_name, params_xls, data_ca)

    features_dir = cfg['paths']['features_dir'] + mouse_id
    f_write_config_json(features_dir, cfg['GEN_TAG'], cfg)   # provenance sidecar
    cam_params = f_comp_FOV_adj(dict(cfg['render']['cam_params']))
    cfg['_features_dir'], cfg['_mouse_id'] = features_dir, mouse_id   # for conv_repo mode

    _sel = f_resolve_dsets(cfg.get('DSETS'), vr_data2, mouse_id)
    print(f'{mouse_id}: {len(_sel)}/{len(vr_data2)} datasets → '
          f'{[vr_data2.dset_name[i] for i in _sel]}')
    _bar = tqdm(_sel, desc=f'{mouse_id} feature-gen', unit='dset') if _TQDM else _sel
    for n_dset in _bar:
        if max_dsets is not None and n_run >= max_dsets:
            _log(f'reached MAX_DATASETS={max_dsets} → stopping')
            stop = True
            break
        est1, bh = data_ca[n_dset], bh_data[n_dset]
        dname = est1['dset_name']

        # conv_repo mode — dense per-filter maps (incremental at the FILTER level, so no
        # dataset-level exists-check; f_build_conv_repo skips filters already present).
        if cfg.get('GEN_MODE', 'blocks') == 'conv_repo':
            n_run += 1
            t0 = time.perf_counter()
            try:
                _man, _nw, _ns = f_build_conv_repo_one_dataset(est1, bh, cam_params, cfg)
            except Exception as e:
                _log(f'  FAIL(repo) {mouse_id}/{dname}: {type(e).__name__}: {e}')
                summary.append((mouse_id, dname, f'FAIL:{type(e).__name__}', 0.0))
                continue
            dt = time.perf_counter() - t0
            _log(f'  repo {mouse_id}/{dname}: +{_nw} filters, {_ns} skipped  {dt:.1f}s')
            summary.append((mouse_id, dname, f'repo+{_nw}', dt))
            gc.collect()
            continue

        # pca_feat mode — convolve → PCA → save small reduced blocks (two granularities to separate
        # tagged caches). Skip only when EVERY enabled granularity's cache already exists (the
        # render+conv is shared, so a missing one forces a full rebuild anyway); OVERWRITE forces redo.
        if cfg.get('GEN_MODE', 'blocks') == 'pca_feat':
            _pca_tags = [cfg['GEN_TAG'] + sfx for sfx, on in
                         (('_pcacomb', cfg.get('SAVE_COMBINED', True)),
                          ('_pcasep',  cfg.get('SAVE_SEPARATED', True)),
                          ('_pcaraw',  cfg.get('SAVE_RAW', False))) if on]
            if (not cfg['OVERWRITE'] and _pca_tags
                    and all(f_features_exist(features_dir, mouse_id, dname, _t) for _t in _pca_tags)):
                _log(f'  skip {mouse_id}/{dname} (pca_feat caches exist: '
                     f'{[t.rsplit("_", 1)[-1] for t in _pca_tags]})')
                summary.append((mouse_id, dname, 'skip', 0.0))
                continue
            n_run += 1
            t0 = time.perf_counter()
            try:
                _nc, _nsp, _nr = f_build_pca_feat_one_dataset(est1, bh, cam_params, cfg)
            except Exception as e:
                _log(f'  FAIL(pca) {mouse_id}/{dname}: {type(e).__name__}: {e}')
                summary.append((mouse_id, dname, f'FAIL:{type(e).__name__}', 0.0))
                gc.collect()
                continue
            dt = time.perf_counter() - t0
            _log(f'  pca {mouse_id}/{dname}: {_nc} comb + {_nsp} sep + {_nr} raw blocks  {dt:.1f}s')
            summary.append((mouse_id, dname, f'pca+{_nc}/{_nsp}/{_nr}', dt))
            gc.collect()
            continue

        path = f_features_path(features_dir, mouse_id, dname, cfg['GEN_TAG'])
        if not cfg['OVERWRITE'] and f_features_exist(features_dir, mouse_id, dname, cfg['GEN_TAG']):
            _log(f'  skip {mouse_id}/{dname} (cache exists)')
            summary.append((mouse_id, dname, 'skip', 0.0))
            continue

        n_run += 1                                   # this dataset is being built
        t0 = time.perf_counter()
        try:
            bb, frame_t, side_tags, movie = f_build_one_dataset(est1, bh, cam_params, cfg)
        except Exception as e:                       # don't let one bad dataset kill the batch
            _log(f'  FAIL {mouse_id}/{dname}: {type(e).__name__}: {e}')
            summary.append((mouse_id, dname, f'FAIL:{type(e).__name__}', 0.0))
            continue

        _reg = (vr_data2.region[n_dset] if 'region' in vr_data2.columns else None)
        meta = {'mouse_id': mouse_id, 'n_dset': int(n_dset), 'dset_name': dname,
                'region': _reg, 'n_cells': int(est1['S'].shape[0]),
                'fs': float(f_imaging_fs(frame_t)), 'n_frames': int(frame_t.size),
                'blocks': {k: list(bb[k]['X'].shape) for k in bb}}
        _exf, _exi = f_pick_frames(movie, cfg.get('N_EXAMPLE_FRAMES', 8))
        f_save_features(path, bb, frame_t, side_tags, meta=meta, gen_config=cfg,
                        movie=(movie if cfg['SAVE_MOVIE'] else None),
                        example_frames=_exf, example_frame_idx=_exi)
        dt = time.perf_counter() - t0
        _log(f'  saved {mouse_id}/{dname}  {meta["n_cells"]}cells {meta["n_frames"]}fr '
             f'{len(bb)}blocks  {dt:.1f}s -> {path.name}')
        summary.append((mouse_id, dname, 'saved', dt))

        # Free this dataset's render + blocks before the next one (the rendered
        # movie is large) so an all-datasets run doesn't accumulate memory.
        del bb, movie
        gc.collect()

#%% Summary
print('\n=== feature-gen summary ===')
for m, d, status, dt in summary:
    print(f'  {m:3s} {d:24s} {status:14s} {dt:6.1f}s')
# count each status kind (conv_repo→'repo+<n>', pca_feat→'pca+<c>/<s>', blocks→'saved'/'skip', 'FAIL:*')
_n_repo = sum(1 for _, _, s, _ in summary if str(s).startswith('repo'))
_n_pca = sum(1 for _, _, s, _ in summary if str(s).startswith('pca'))
_n_saved = sum(1 for _, _, s, _ in summary if s == 'saved')
_n_skip = sum(1 for _, _, s, _ in summary if s == 'skip')
_n_fail = sum(1 for _, _, s, _ in summary if str(s).startswith('FAIL'))
print(f'total: {sum(dt for *_, dt in summary):.1f}s, '
      f'{_n_repo} repo, {_n_pca} pca, {_n_saved} saved, {_n_skip} skipped, {_n_fail} failed')
