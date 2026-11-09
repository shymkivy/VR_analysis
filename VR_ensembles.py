# -*- coding: utf-8 -*-
"""
VR_ensembles.py

PURPOSE: Load one dataset, build its NMF ensembles from the shared ensemble config, inspect and plot
them, and save them to the ensemble cache so every other script reloads the same ensembles.
PIPELINE: shared TARGETS piece (PLAN_ensemble_config.md). Config + cache in
functions/f_ensemble_pipeline.py (f_default_ens_config / f_ensembles_cached). Plots are cebra's
ensemble plots (f_ensemble_plots). Calcium only — no render, no features.

Spyder cell-based (#%% markers). Created 2026-11-06.
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

from f_pipeline import f_load_session
import f_ensembles as ens
import f_ensemble_plots as ensp
import f_ensemble_pipeline as enspipe
import f_decoding as dec


#%% Cell 1 — dataset + ensemble config
mouse_id = 'L'
n_dset   = 5                         # index into do_proc==1 rows for this mouse

paths = {'data_dir_base': 'F:/VR/data_proc/', 'data_dir_bh': 'F:/VR/Bh_data/mice_gcamp/',
         'vr_data_xlsx': 'F:/VR/data_proc/VR_data.xlsx', 'features_dir': 'F:/VR/features/'}

# Defaults = VR_ca_cebra's settings. Override a key by name or a section by dict, e.g.
#   num_comp=12 | membership='cv_consensus' | neural={'sigma_frames': 0} | dimred={'alpha_W': 1e-3}
# Sections: neural (input prep) / dimred (NMF) / apply (threshold) / stats (membership, significance)
# / select (which ensembles feed encode/decode). Unknown keys raise.
ens_cfg = enspipe.f_default_ens_config(
    # num_comp   = 15,
    # membership = 'z_thresh',        # 'z_thresh' | 'cv_consensus'
    # sig_method = 'cv',              # 'cv' | 'mp' | 'shuffle'
)
ens_force = False                    # True = refit even if this config is cached
ens_root  = paths['features_dir'] + mouse_id   # per-mouse dir (same as cebra / feature search) → ens_cache/ inside
ens_save  = True                     # write the result to <ens_root>/ens_cache/

# plot settings
raster_cmap = 'viridis'
ens_vmax_pct = 99                    # raster color scale percentile
print(f'ensemble config tag: {enspipe.f_ens_config_tag(ens_cfg)}')
for _s in ('neural', 'apply', 'stats', 'select'):
    print(f'  {_s}: {ens_cfg[_s]}')
print(f"  dimred: {ens_cfg['dimred']}")


#%% Cell 2 — load the dataset (calcium only) + neural input from the config
session = f_load_session(mouse_id, n_dset, paths['data_dir_base'] + mouse_id, paths['data_dir_bh'],
                         paths['vr_data_xlsx'], behavior=False)
est1     = session['est1']
dset_tag = f"{mouse_id}/{session['dset_name']}"
frame_t  = np.asarray(session['frame_t']).ravel()
fs       = float(dec.f_imaging_fs(frame_t))
ens_in   = enspipe.f_ens_neural_input(est1, ens_cfg)      # X (T, N) + cell_ids (est1 rows)
X        = ens_in['X']
print(f'dataset: {dset_tag}  ({X.shape[0]} frames, {X.shape[1]} active cells, fs {fs:.2f} Hz)')


#%% Cell 3 (opt-in) — choose num_comp: dim-red sweep by CV-LNO score
# test_err = LNO reconstruction error (lower = better); lno_r2_med = median per-cell leave-one-neuron-
# out R² (higher = better). sweep_var = x-axis; outer_var = one curve per value.
if 0:
    sweep_var    = 'num_comp'
    sweep_values = [2, 4, 6, 7, 8, 9, 10, 11, 12, 14]
    #sweep_var, sweep_values = 'alpha_W', [0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
    outer_var    = 'beta_loss'                         # 'method' | 'beta_loss' | 'normalize' | ...
    outer_values = ['frobenius', 'kullback-leibler']
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1, (os.cpu_count() or 2) - 1)):
        dimred_curves = ens.f_dimred_sweep2d(ens_cfg['dimred'], outer_var, outer_values,
                                             sweep_var, sweep_values, X.T, fs=fs)
    dec.f_plot_sweep_curves(dimred_curves, sweep_var, ens_cfg['dimred']['scores'],
                            dset_tag=dset_tag, config=ens_cfg['dimred'])


#%% Cell 4 (opt-in) — NMF consensus stability: which components reproduce across random inits
if 0:
    ens_cons_runs = 30
    _dp = ens_cfg['dimred']
    _sb = int(round(_dp['smooth_ms'] * fs / 1000.0)) if _dp.get('smooth_ms') else 0
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1, (os.cpu_count() or 2) - 1)):
        cons = ens.f_nmf_consensus(X.T, _dp['num_comp'], n_runs=ens_cons_runs,
                                   normalize=_dp.get('normalize', 'norm_mean_std'), smooth_sigma_bins=_sb,
                                   dred_kwargs=ens._dimred_kwargs(_dp), random_state=_dp.get('seed', 42))
    print(f'{dset_tag} — NMF consensus (k={_dp["num_comp"]}, {ens_cons_runs} runs): '
          f'cophenetic={cons["cophenetic"]:.3f}')
    ensp.f_plot_nmf_stability(cons, dset_tag=dset_tag)


#%% Cell 5 — ensembles from the config (cache load, or fit + stats + save)
ens_res = enspipe.f_ensembles_cached(est1, ens_cfg, ens_root, mouse_id,
                                     session['dset_name'], fs=fs, force=ens_force, save=ens_save,
                                     dset_tag=dset_tag)
ens_out, ens_table = ens_res['ens_out'], ens_res['ens_table']
_cols = [c for c in ('ensemble', 'n_cells', 'memb_frac', 'cv_r2', 'cv_r2_pool', 'mp_ratio',
                     'frac_active', 'sig') if c in ens_table.columns]
with pd.option_context('display.max_rows', None, 'display.width', 200):
    print(ens_table[_cols].to_string(index=False))
print(f'{int(ens_table["sig"].sum())}/{len(ens_table)} significant')

# Re-run only the stats (membership / significance) without refitting NMF:
#   ens_cfg2 = enspipe.f_default_ens_config(membership='cv_consensus')
#   ens_res  = enspipe.f_ensembles_from_config(X, ens_cfg2, cell_ids=ens_in['cell_ids'], fs=fs,
#                                              prior=ens_res, dset_tag=dset_tag)


#%% Cell 6 — per-ensemble scores + significance
ensp.f_plot_ens_scores(ens_table, dset_tag=dset_tag)


#%% Cell 7 — sorted raster + one panel per ensemble
ensp.f_plot_ens_overview(ens_out, X.T, mouse_dset_tag=dset_tag, cmap=raster_cmap,
                         vmax_pct=ens_vmax_pct, ens_table=ens_table)


#%% Cell 8 — reliability: per-cell LNO-R² ECDF with the per-ensemble LNO overlaid
ens_lno_metric = 'cv_r2'             # 'cv_r2' | 'cv_r2_pool'
lno_r2 = ensp.f_plot_lno_ceiling(X.T, ens_cfg['dimred'], fs=fs, dset_tag=dset_tag,
                                 ens_lno=ens_table[ens_lno_metric].to_numpy(float),
                                 ens_label=f'ensembles ({ens_lno_metric})')


#%% Cell 9 — per-ensemble CV reconstruction trace (what cv_r2 measures)
ens_cv_which = 'sig'                 # 'sig' | 'all' | [0, 2, 5]
ensp.f_plot_ens_cv_trace(ens_out, X.T, ens_table, ens_cfg['dimred'], fs=fs, which=ens_cv_which,
                         cv_method=ens_cfg['stats']['cv_method'], cv_num_comp=1, frame_t=frame_t,
                         dset_tag=dset_tag, max_panels=8)


#%% Cell 10 — FOV map: all selected cells (gray) + significant-ensemble members per color
ensp.f_plot_ensemble_fov(est1['A'], est1['dims'], np.flatnonzero(ens_in['active']), ens_table,
                         dset_tag=dset_tag, which='sig', thresh_frac=0.1, gamma=0.7)


#%% Cell 11 — targets other scripts consume (sanity view)
ens_targets = enspipe.f_ens_targets(ens_res)
for _k, (_Y, _lab) in ens_targets.items():
    print(f'{_k}: {_Y.shape}  ensembles {list(_lab)}')
