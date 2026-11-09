# -*- coding: utf-8 -*-
"""
VR_feature_tuning.py — WIDE gabor/motion-energy feature-tuning SCAN, per cell & per ensemble.

Unlike VR_ca_cebra.py (which loads one/two KNOWN-GOOD feature sets for encoding/decoding), this
script SURVEYS a wide filter grid: render the monitor movie once, convolve a wide gabor/meng grid
on the GPU (generated as we go — no pre-built conv_repo, magnitude-only), and for EACH CELL and
EACH ENSEMBLE read out the preferred filter (orientation × wavelength [× temporal-freq]) by
reverse-correlation. Compute: f_conv_gpu (torch conv) + f_feature_scan (scan engine + pluggable
scorer) + f_ensemble_pipeline (ensembles) + f_tuning (readout) + f_encoding_plots (plots).

The scan is scorer-agnostic (f_feature_scan.f_score_scan): scorer='tuning' here; encode/decode
performance scorers are planned extension points reusing f_score_representations / f_run_encoder /
f_run_decode_sweep.

Spyder cell-based (#%% markers). Created 2026-07-16.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))
sys.path.append(str(path1 / 'RNN_scripts' / 'functions'))

import f_functions as ff
import f_analysis as ana
import f_utils as util
import f_pipeline as pipe
import f_feature_source as fsrc
import f_decoding as dec
import f_ensembles as ens
import f_ensemble_pipeline as enspipe
import f_conv_repo as cr
import f_feature_scan as fscan
import f_tuning as tun
import f_encoding as enc
import f_encoding_plots as encp


#%% =========================================================================
# SECTION 1 · SETUP — config + dataset
# ==========================================================================
mouse_id = 'L'
n_dset   = 5        # dataset index into the do_proc==1 rows

data_dir     = 'F:/VR/data_proc/' + mouse_id
data_dir_bh  = 'F:/VR/Bh_data/mice_gcamp/'
vr_data_xlsx = 'F:/VR/data_proc/VR_data.xlsx'

raster_cmap = 'RdBu_r'
obj_size = {'x': 5, 'y': 20, 'z': 5, 'height': 2}
src = fsrc.f_default_source(mouse_id=mouse_id)   # features_dir / conv_tag / cache_tag
scan_tag = 'gen_7_16_26'         # scan-generation tag (versions the scan file; checked on load)


#%% Load data + cheap behavior/monitor vectorization (f_load_session)
session = pipe.f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx)
est1       = session['est1']
cam_params = session['cam_params']
frame_t    = session['frame_t']
T_img      = session['T_img']


#%% Preprocessing — NEURAL only (drop silent → smooth → hclust-sort → Y_neu_raw)
dset_tag = f"{mouse_id} dset{n_dset}  {est1['dset_name']}"

_S_raw = np.asarray(est1['S'], dtype=float)
active_cells = _S_raw.sum(axis=1) > 0
_n_drop = int((~active_cells).sum())
if _n_drop:
    print(f'dropping {_n_drop} silent cells; {int(active_cells.sum())}/{active_cells.size} remain')
S_active = _S_raw[active_cells]

S_sm  = util.f_gauss_smooth(S_active, sigma_frames=6)
S_smn = S_sm / np.max(S_sm, axis=1)[:, None]
hclust_data = ana.f_hclust_firing_rates(S_smn, standardize=True, metric='kl',
                                         method='average', similarity_transform='auto')
S_smn2 = S_smn[hclust_data['res_order'], :]

Y_neu_raw = S_smn2.T.astype(np.float32)                       # (T_img, n_active), hclust order
# true neuron ID per row of Y_neu_raw.T (extraction row → est1 cell id)
_cell_ids_ens = np.flatnonzero(active_cells)[hclust_data['res_order']]


#%% Ensembles — from the SHARED ensemble config (f_ensemble_pipeline; defaults = VR_ca_cebra)
# Loaded from <features_dir>/ens_cache/ (built by VR_ensembles.py / cebra) or fit + saved there.
# Every parameter: _ens_defaults() in f_ensemble_pipeline.py. Ensemble plots: VR_ensembles.py.
ens_cfg = enspipe.f_default_ens_config()      # override by name, e.g. num_comp=12, membership='cv_consensus'
dimred_params = ens_cfg['dimred']
_fs = dec.f_imaging_fs(frame_t)
ens_res = enspipe.f_ensembles_cached(est1, ens_cfg, src['features_dir'], mouse_id, est1['dset_name'],
                                     fs=_fs, dset_tag=dset_tag)
ens_out, ens_table = ens_res['ens_out'], ens_res['ens_table']
ens_params = {'tag': ens_res['tag'], 'cfg': ens_res['cfg']}     # recorded into the scan file
# downstream cells index Y_neu_raw with ens_out's cell order — must be the same cells/order
if not np.array_equal(ens_res['cell_ids'], _cell_ids_ens):
    print("WARNING: ens_cfg['neural'] gives different cells/order than this script's Y_neu_raw — "
          "keep ens_cfg neural at the defaults here.")
print(f'{int(ens_table["sig"].sum())}/{len(ens_table)} significant ensembles  (config {ens_res["tag"]})')


#%% Render the monitor movie ONCE (L/R), on the imaging clock
# render width/height. lat:vert = hFOV:vFOV ≈ 1.49 → angularly-square pixels (~0.66 deg/px both axes).
render_lat_samp  = 150           # WIDTH  (columns)
render_vert_samp = 101           # HEIGHT (rows)
render_cfg = {'obj_size': obj_size,
              'lat_samp': render_lat_samp, 'vert_samp': render_vert_samp, 'max_frames': None,
              'render_terrain': False, 'cam_params': cam_params}
left_mon, right_mon, _ = pipe.f_render_session(session, render_cfg, render_verbose=True)
movies, side_tags = [left_mon, right_mon], ['L', 'R']
print(f'movie: {left_mon.shape[0]} frames, {left_mon.shape[1]}×{left_mon.shape[2]}/side '
      f'(neural T={Y_neu_raw.shape[0]})')


#%% SCAN — convolve the WIDE grid on GPU → whole-monitor magnitude regressors (X_stim)
# The wide parameter grid to survey. Each conv is generated on the fly (no stored repo) and reduced
# to its whole-monitor magnitude time series. filt_batch = how many gabor filters per GPU call
# (the resource knob; larger = faster + more VRAM). meng shares the spatial gabor across tfreqs.

scan_backend = 'gpu'             # 'gpu' (torch) | 'cpu' (scipy fallback, n_workers threads)
scan_device  = 'cuda'            # torch device for backend='gpu'
scan_filt_batch = 8              # gabor filters per GPU conv call (VRAM knob)
scan_n_workers  = 4              # CPU-backend threads (ignored on GPU)
scan_standardize = True          # z-score each filter column (comparable across freqs)
scan_sample_ds  = None           # None = convolve at full render; int / (ds_h, ds_w) → point-sample
                                 #   the movie before conv (coarser/faster scan from the same render)
scan_chunk_t    = 2000           # frames per GPU chunk (VRAM knob; movie streams from CPU, so peak
                                 #   VRAM ∝ chunk_t·filt_batch·H·W — INDEPENDENT of session length)

# NYQUIST LIMIT — resolvable sf_cyc_deg / wavelength / tf_hz for the current render × downsample + fs.
if 0:
    fscan.f_nyquist_limits(cam_params, render_lat_samp, render_vert_samp,
                           dec.f_imaging_fs(frame_t), scan_sample_ds, temporal_support=11)

scan_spatial    = True           # also record per-filter time-mean spatial map (where activity sits)
# RESUMABLE single-file store: rewritten atomically after every batch → crash-safe; a re-run SKIPS
# filters already in the file and only convolves NEW ones (widen scan_specs freely — old values are
# reused). The file also EMBEDS the dataset identity + cell traces (Y_neu_raw) + ensemble outputs
# (store_neural=True) so another script can reload + score with NO session reload / re-extraction.
# canonical path in <features_dir>/scan/ (parallel to conv_repo/), versioned by scan_tag
scan_file = fscan.f_scan_path(src['features_dir'], est1['dset_name'], scan_tag,
                              render_lat_samp, render_vert_samp, scan_sample_ds)
os.makedirs(os.path.dirname(scan_file), exist_ok=True)
scan_store_neural = True         # embed Y_neu_raw + ens_out/ens_table (self-contained; larger file)
_fs = dec.f_imaging_fs(frame_t)  # imaging Hz (for tf_hz→cycles/frame conversion + tuning lags)

# resolvable limits for the current render × downsample combo (play with render_lat/vert + scan_sample_ds):
# effective deg/px = render deg/px × scan_sample_ds → max sf_cyc_deg = 1/(2·deg_per_px); max tf_hz = fs/2.
_dpp_eff = cr.f_deg_per_px(cam_params, render_lat_samp, render_vert_samp, scan_sample_ds)
print(f'effective deg/px={_dpp_eff:.3f} (render {render_lat_samp}x{render_vert_samp}, ds={scan_sample_ds}) '
      f'→ max sf={1/(2*_dpp_eff):.2f} cyc/deg, max tf_hz={_fs/2:.0f}')

# grid in PHYSICAL units (cyc/degree + Hz), converted to px/frame by f_conv_filter_specs_phys.
scan_specs = cr.f_conv_filter_specs_phys(
    {'gabor': {'n_orient': 4, 'sf_cyc_deg': (0.19, 0.17, 0.15, 0.13, 0.11, 0.09)},        # mon width ~98.4deg
     'meng':  {'n_orient': 4, 'sf_cyc_deg': (0.13, 0.11, 0.09), 'tf_hz': (0.5, 1, 3, 6, 9, 12, 15),
               'temporal_support': 'auto', 'n_cycles': 2, 'support_max': 91,  # per-tf: ~2 cycles/tf
               'output': 'opponent'}},
    cam_params, render_lat_samp, render_vert_samp, fs=_fs, sample_ds=scan_sample_ds)
_ng = sum(s['kind'] == 'gabor' for s in scan_specs)
_nm = sum(s['kind'] == 'meng' for s in scan_specs)
print(f'scan grid: {len(scan_specs)} filters ({_ng} gabor, {_nm} meng) → {scan_file}')

# data bundle embedded in the scan file (self-contained analysis artifact)
scan_data = {'mouse_id': mouse_id, 'n_dset': n_dset, 'dset_name': est1['dset_name'],
             'dset_tag': dset_tag, 'gen_tag': scan_tag, 'frame_t': frame_t, 'fs': _fs,
             'active_cells': active_cells, 'cell_ids': _cell_ids_ens,
             'dimred_params': dimred_params, 'seed': dimred_params.get('seed'),
             'ens_params': ens_params,
             'Y_neu_raw': Y_neu_raw, 'ens_out': ens_out, 'ens_table': ens_table}

X_stim, filt_df = fscan.f_scan_conv_activity(
    movies, side_tags, scan_specs, backend=scan_backend, device=scan_device,
    filt_batch=scan_filt_batch, reduce='absmean', combine_sides='mean', transform_meng='sqrt',
    standardize=scan_standardize, sample_ds=scan_sample_ds, spatial_mean=scan_spatial,
    chunk_t=scan_chunk_t, n_workers=scan_n_workers, scan_file=scan_file, data=scan_data,
    store_neural=scan_store_neural, verbose=True)
print(f'X_stim {X_stim.shape}  filt_df {filt_df.shape}')


#%% DATASET STATS — per-filter magnitude distribution + where activity sits on the monitor
# f_scan_stats: per-filter mean/std/max/percentiles of the whole-monitor magnitude (RAW) — the
# dataset's convolved-activity levels at each freq/orientation. f_scan_spatial: per-filter time-mean
# spatial map (L‖R) → where each feature's activity concentrates (interpret tuning vs where/how often
# that feature appeared on screen). Both read from the scan file (recompute-free).
if 1:
    # plt.close('all')
    scan_stats = fscan.f_scan_stats(scan_file, percentiles=(5, 25, 50, 75, 95))
    _cols = [c for c in ['kind', 'odeg', 'sf_cyc_deg', 'tf_hz', 'mean', 'std', 'max',
                         'p5', 'p50', 'p95'] if c in scan_stats.columns]
    with pd.option_context('display.max_rows', 40, 'display.width', 200):
        print(f'{dset_tag} — filter magnitude stats ({len(scan_stats)} filters):')
        print(scan_stats[_cols].to_string(index=False))
    if scan_spatial:
        _maps, _mdf = fscan.f_scan_spatial(scan_file)
        _top = scan_stats.sort_values('mean', ascending=False).head(6)   # highest-activity filters
        fig, axs = plt.subplots(2, 3, figsize=(12, 5)); axs = axs.ravel()
        for ax, (_, rr) in zip(axs, _top.iterrows()):
            _mask = (_mdf['kind'] == rr['kind']) & (_mdf['odeg'] == rr['odeg'])
            if 'sf_cyc_deg' in _mdf.columns:
                _mask &= np.isclose(_mdf['sf_cyc_deg'], rr['sf_cyc_deg'])
            _idx = np.flatnonzero(_mask.to_numpy())
            if _idx.size:
                ax.imshow(_maps[_idx[0]], aspect='auto', cmap='magma')
                _tf = f" tf{rr['tf_hz']:g}" if 'tf_hz' in rr and not pd.isna(rr.get('tf_hz')) else ''
                ax.set_title(f"{rr['kind']} o{rr['odeg']:.0f} sf{rr.get('sf_cyc_deg', np.nan):.2f}{_tf}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{dset_tag} — time-mean activity (L‖R), top-6 by mean magnitude'); fig.tight_layout()


#%% TUNING — score X_stim per CELL and per ENSEMBLE (reverse-correlation)
if 1:
    tuning_sources = ('cells', 'ensembles')   # also 'ens_cells' | 'reliable_cells'
    tuning_methods = ('sta',)                  # 'sta' (literal) and/or 'ridge' (whitened)
    tuning_lags_ms = [0, 100, 200, 300, 400]   # causal; peak lag picked per unit
    tuning_ens_which = 'sig'                   # ensembles: 'sig' | 'all'
    _fs = dec.f_imaging_fs(frame_t)
    tun_scan = fscan.f_score_scan(
        X_stim, filt_df, Y_neu_raw, scorer='tuning', sources=tuning_sources,
        ens_out=ens_out, ens_table=ens_table, dimred_params=dimred_params,
        ens_which=tuning_ens_which, fs=_fs, methods=tuning_methods, lags_ms=tuning_lags_ms)
    for _src, _r in tun_scan.items():
        _rd = _r['readouts'][tuning_methods[-1]]
        _cols = [c for c in ['unit', 'pref_odeg', 'pref_sf_cyc_deg', 'pref_tf_hz', 'pref_wavelength',
                             'pref_tfreq', 'peak_abs', 'peak_lag_ms', 'osi', 'sf_sel', 'tf_sel']
                 if c in _rd['table'].columns]
        with pd.option_context('display.max_rows', 40, 'display.width', 200):
            print(f'\n{dset_tag} — {_src} ({_r["kind"]}, {tuning_methods[-1]}, '
                  f'{_rd["table"].shape[0]} units):')
            print(_rd['table'].sort_values('peak_abs', ascending=False)[_cols].to_string(index=False))

    # SAVE the fitted results (the ridge/STA response W + per-unit tables/marginals) + ALL params
    # used, so they can be reloaded and re-plotted later (f_load_scan_results → f_encoding_plots).
    tuning_results_file = scan_file.replace('_scan.joblib', '_tuning.joblib')
    tuning_params = {'scorer': 'tuning', 'sources': tuning_sources, 'methods': tuning_methods,
                     'lags_ms': tuning_lags_ms, 'ens_which': tuning_ens_which, 'fs': _fs,
                     'scan': {'reduce': 'absmean', 'combine_sides': 'mean', 'transform_meng': 'sqrt',
                              'standardize': scan_standardize, 'n_filt': int(X_stim.shape[1])},
                     'ens_params': ens_params, 'seed': dimred_params.get('seed'),
                     'dset_tag': dset_tag, 'scan_file': scan_file}
    fscan.f_save_scan_results(tuning_results_file, tun_scan, params=tuning_params)
    print(f'saved tuning results (W response + params) → {tuning_results_file}')


#%% TUNING — plots (population + example units), per source
if 1:
    tuning_plot_z = True             # z-score across filters for the heatmap + all-filters bars
    tuning_marg_norm = (True, False) # per-unit marginals: peak-normalized shape and/or raw |tuning|
    for _src, _r in tun_scan.items():
        _rd = _r['readouts'][tuning_methods[-1]]
        _tag = f'{dset_tag} [{_src}]'
        encp.f_plot_tuning_population(_rd['table'], dset_tag=_tag, method=_rd.get('method', ''))
        encp.f_plot_tuning_marginals(_rd, u=None, dset_tag=_tag)     # population mean marginals
        if _r.get('source') == 'ensembles' or _rd['table'].shape[0] <= 20:  # per-unit overlay (one line/ensemble)
            for _nrm in tuning_marg_norm:
                encp.f_plot_tuning_marginals_all(_rd, dset_tag=_tag, normalize=_nrm)
        encp.f_plot_tuning_lags(_rd, dset_tag=_tag)                  # tuning strength vs lag
        _top = _rd['table'].sort_values('peak_abs', ascending=False).index.to_numpy()[:3]
        _has_tf = _rd['by_tfreq'] is not None and np.size(_rd['by_tfreq'])
        for _u in _top:
            encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=_tag, zscore=tuning_plot_z, axis='sf')
            if _has_tf:                                              # meng: also orientation × temporal-freq
                encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=_tag, zscore=tuning_plot_z, axis='tf')
            encp.f_plot_tuning_allfilters(_rd, int(_u), dset_tag=_tag, zscore=tuning_plot_z)


#%% TUNING — SIGNIFICANT CELLS only (circshift shuffle test)
# Per-cell tuning significance = peak |tuning| vs a per-cell circular-shift null (BH-FDR). Keep the
# significant cells, refit the readout on JUST them, and draw the same views (population + per-unit
# overlay + top-cell heatmaps) — legible on the sig subset instead of the all-cell hairball.
if 1:
    sig_n_shuffle = 200              # circshift null draws per cell
    sig_alpha     = 0.05             # BH-FDR level
    sig_method    = tuning_methods[-1]
    sig_seed      = 42
    sig_marg_norm = (True, False)    # per-unit marginal overlays (normalized shape / raw)

    _fs   = dec.f_imaging_fs(frame_t)
    _lags = enc.f_lags_from_ms(list(tuning_lags_ms), _fs)
    Yc, cell_labels, _ = tun.f_select_tuning_units('cells', Y_neu_raw)
    # per FEATURE FAMILY (gabor / meng) so significance + plots aren't mixed across families
    for _fk in list(dict.fromkeys(filt_df['kind'].tolist())):
        _cmask = (filt_df['kind'] == _fk).to_numpy()
        _Xk = X_stim[:, _cmask]; _fdfk = filt_df.iloc[_cmask].reset_index(drop=True)
        sig_df = tun.f_tuning_significance(_Xk, _fdfk, Yc, _lags, method=sig_method,
                                           n_shuffle=sig_n_shuffle, alpha=sig_alpha, seed=sig_seed,
                                           labels=cell_labels)
        _sig = sig_df[sig_df['sig']].sort_values('z', ascending=False)
        print(f'{dset_tag} [{_fk}] — significant cells: {_sig.shape[0]}/{sig_df.shape[0]} '
              f'(shuffle test, FDR<{sig_alpha}, {sig_method})')
        with pd.option_context('display.max_rows', 50, 'display.width', 200):
            print(_sig[['unit', 'peak_abs', 'z', 'p_fdr']].to_string(index=False))
        if not _sig.shape[0]:
            continue
        _cols = _sig['col'].to_numpy(int)
        _rd = tun.f_tuning_readout(
            tun.f_reverse_corr_tuning(_Xk, _fdfk, Yc[:, _cols], _lags, method=sig_method),
            fs=_fs, labels=[cell_labels[c] for c in _cols])
        _tag = f'{dset_tag} [sig cells:{_fk}]'
        encp.f_plot_tuning_population(_rd['table'], dset_tag=_tag, method=sig_method)
        encp.f_plot_tuning_marginals(_rd, u=None, dset_tag=_tag)
        if _rd['table'].shape[0] <= 40:                              # overlay only if legible
            for _nrm in sig_marg_norm:
                encp.f_plot_tuning_marginals_all(_rd, dset_tag=_tag, normalize=_nrm)
        encp.f_plot_tuning_lags(_rd, dset_tag=_tag)
        _has_tf = _rd['by_tfreq'] is not None and np.size(_rd['by_tfreq'])
        for _u in _rd['table'].sort_values('peak_abs', ascending=False).index.to_numpy()[:3]:
            encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=_tag, zscore=True, axis='sf')
            if _has_tf:
                encp.f_plot_tuning_heatmap(_rd, int(_u), dset_tag=_tag, zscore=True, axis='tf')
            encp.f_plot_tuning_allfilters(_rd, int(_u), dset_tag=_tag, zscore=True)


#%% TUNING — VERIFY in monitor space (re-convolve the preferred filter on the fly)
# For the top units of each source, re-convolve their PREFERRED filter over the movie (GPU) and show
# (a) the LOCATION map (per-pixel corr with activity → RF blob) and (b) the activity-weighted STA.
if 1:
    verify_n      = 2                # top units per source to verify
    verify_side   = 'both'          # 'both' = L‖R side by side | 'L' | 'R'
    verify_z      = True            # z-score the monitor maps (SD units)
    for _src, _r in tun_scan.items():
        _rd = _r['readouts'][tuning_methods[-1]]
        _keep = _rd.get('keep_cells')
        _Yu, _lab, _ = tun.f_select_tuning_units(_r.get('source', _src), Y_neu_raw, ens_out=ens_out,
                                                 ens_table=ens_table, dimred_params=dimred_params,
                                                 ens_src_which=tuning_ens_which, fs=_fs)
        _Yk = _Yu[:, np.asarray(_keep, bool)] if _keep is not None else _Yu
        _topv = _rd['table'].sort_values('peak_abs', ascending=False).index.to_numpy()[:verify_n]
        _mlab = ['L', 'R'] if verify_side == 'both' else [verify_side]
        _nmon = len(_mlab)
        for _u in _topv:
            _u = int(_u); _frow = tun.f_preferred_filt_row(_rd, _u); _act = _Yk[:, _u]
            _tfv = _frow.get('tf_hz', _frow.get('tfreq', np.nan))   # Hz if physical
            _tfs = f' tf{_tfv:g}Hz' if not (_tfv is None or (isinstance(_tfv, float) and np.isnan(_tfv))) else ''
            _sfs = (f'sf{_frow["sf_cyc_deg"]:g}cyc/deg' if 'sf_cyc_deg' in _frow
                    else f'wl{_frow["wavelength"]:g}')
            _tag = (f'{dset_tag} [{_src}] u{_rd["table"].iloc[_u]["unit"]} '
                    f'({_frow["kind"]} o{_frow["odeg"]:.0f} {_sfs}{_tfs})')
            try:
                _maps = fscan.f_filter_activity_maps(movies, side_tags, _frow, _act,
                                                     backend=scan_backend, device=scan_device)
                _order = [s for s in _mlab if s in _maps]
                _loc = np.hstack([_maps[s]['location'] for s in _order])
                _sta = np.hstack([_maps[s]['sta'] for s in _order])
                encp.f_plot_tuning_monitor(_loc, dset_tag=_tag, title=f'location (corr, {verify_side})',
                                           diverging=True, n_monitors=_nmon, labels=_order, zscore=verify_z)
                encp.f_plot_tuning_monitor(_sta, dset_tag=_tag, title=f'preferred-filter STA ({verify_side})',
                                           diverging=False, n_monitors=_nmon, labels=_order, zscore=verify_z)
            except Exception as _e:
                print(f'  verify {_src} u{_lab[_u] if _u < len(_lab) else _u}: skipped '
                      f'({type(_e).__name__}: {_e})')


#%% SUMMARY — combined per-cell + per-ensemble preferred-tuning table + cell↔ensemble comparison
if 1:
    _rows = []
    for _src, _r in tun_scan.items():
        _t = _r['readouts'][tuning_methods[-1]]['table'].copy()
        _t.insert(0, 'source', _src)
        _rows.append(_t)
    tuning_summary = pd.concat(_rows, ignore_index=True) if _rows else pd.DataFrame()
    if not tuning_summary.empty:
        print(f'\n{dset_tag} — tuning summary ({len(tuning_summary)} units across '
              f'{tuning_summary["source"].nunique()} sources)')
        # preferred-orientation distribution per source
        _piv = (tuning_summary.groupby(['source', 'pref_odeg']).size()
                .unstack(fill_value=0))
        print('preferred orientation counts:')
        print(_piv.to_string())
        # cell vs ensemble OSI / peak comparison
        if 0:
            plt.close('all')
            fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
            _sfcol = 'pref_sf_cyc_deg' if 'pref_sf_cyc_deg' in tuning_summary.columns else 'pref_wavelength'
            for _src, _g in tuning_summary.groupby('source'):
                axs[0].hist(_g['osi'].dropna(), bins=20, histtype='step', label=_src, density=True)
                axs[1].hist(_g[_sfcol].dropna(), bins=20, histtype='step', label=_src, density=True)
            axs[0].set_xlabel('OSI'); axs[1].set_xlabel(_sfcol); axs[0].legend()
            fig.suptitle(f'{dset_tag} — cell vs ensemble tuning'); fig.tight_layout()
