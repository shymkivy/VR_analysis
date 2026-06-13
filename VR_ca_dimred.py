# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:01:39 2026

@author: ys2605
"""

import sys
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
from f_analysis import f_hclust_firing_rates, f_circshift_rates
from f_functions import f_load_bh_data, f_get_session_data, f_plot_session2, f_proc_movement, f_proc_lick_rew, f_proc_lick_rew_df, f_comp_FOV_adj, f_add_phase, f_get_monitor_coords, f_angles_to_movie, f_angles_to_movie_v2 #, f_plot_session
from f_render_diagnostics import f_plot_monitor_outline, f_plot_lateral_over_time, f_plot_vertical_over_time, f_plot_dist_over_time
from f_RNN_dred import f_run_dred
from f_ensembles import (f_NMF, f_PCA, f_sparsePCA, f_mini_batch_sparsePCA,
                          f_dred_add_error, f_hoyer_sparsity, f_component_stability,
                          f_normalize_rows, f_shuffle_data, f_make_cv_groups,
                          f_estimate_dim_corr, f_cv_estimate_grid,
                          f_ens_get_thresh, f_apply_thresh, f_extract_clust,
                          f_ensemble_extract,
                          f_residualize_on_behavior, f_NMF_constrained)


from sklearn.decomposition import SparsePCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import MiniBatchSparsePCA

import time


# =============================================================================
# SECTION 1 — Setup & data loading
# =============================================================================

#%% Config — mouse, paths, object size
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

#%% Load calcium data
data_ca = f_load_caim_data(data_dir, vr_data2.dset_name, caiman_tag = 'results_cnmf.hdf5', cuts_tag='h5cutsinfo', frame_data_tag = 'framedata', r_values_min = 0.5, min_SNR=1.5, thresh_cnn_min=0.8)

#%% Pick dataset, smooth + hclust-sort + raster preview
n_dset = 3

num_comp = 50
n_jobs = 5

est1 = data_ca[n_dset]
     
S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]

hclust_data = f_hclust_firing_rates(S_smn, standardize=True, metric='kl', method='average', similarity_transform='auto')   # cosine, jsd, kl  # average for jsd kl

S_smn2 = S_smn[hclust_data['res_order'],:]
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



# =============================================================================
# SECTION 2 — In-sample dim-red sweep (NMF / PCA variants × k)
# =============================================================================

#%% NMF — vestigial elastic_net_decomposition (kept as reference)

# def elastic_net_decomposition(X, n_components=10, alpha=0.1, l1_ratio=0.5, n_iter=100, random_state=42):
#     """
#     Alternating elastic net matrix decomposition: X ≈ U @ V
#     """
#     rng = np.random.RandomState(random_state)
#     n_samples, n_features = X.shape

#     # Initialize factor matrices
#     U = rng.randn(n_samples, n_components)
#     V = rng.randn(n_components, n_features)

#     en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
#                     fit_intercept=False, max_iter=5000)

#     for iteration in range(n_iter):

#         # --- Step 1: Fix V, solve for U row-by-row ---
#         for i in range(n_samples):
#             en.fit(V.T, X[i])
#             U[i] = en.coef_

#         # --- Step 2: Fix U, solve for V column-by-column ---
#         for j in range(n_features):
#             en.fit(U, X[:, j])
#             V[:, j] = en.coef_

#         # Monitor reconstruction error
#         if iteration % 10 == 0:
#             error = np.linalg.norm(X - U @ V, 'fro')
#             print(f"Iter {iteration:3d} | Reconstruction error: {error:.4f}")

#     return U, V


# U, V = elastic_net_decomposition(S_smn2, n_components=10, alpha=0.1, l1_ratio=0.7)


#%% decoding
# Dim-red wrappers (f_NMF, f_PCA, f_sparsePCA, f_mini_batch_sparsePCA,
# f_dred_add_error) moved to functions/f_ensembles.py 2026-06-01. They're
# imported at the top of this script. Reason: the new CV-grid /
# auto-num-comp / threshold-shuffle cells (see end of this file) refit
# NMF/PCA from inside the module, which forces them out of the script
# to avoid a circular import. Behavior preserved verbatim.



#%% Sweep NMF/PCA variants × k — build dred_all

# inputs should be observations x features, to time x cells
# output to fits are sample coefficients, and components are feature basis vectors
# for t x n orientation, the loss function sums over timepoints, do the temporal structure is directly in the loss function

start_time = time.perf_counter()
dred_all = []

nmf_alpha_w = 0.0001
nmf_max_iter = 2000

# Meaningful alpha_W sweep ranges per NMF variant (data is row-max-normalized, so
# X entries are in [0, 1] and ||X||_F is on the order of sqrt(n_t)). At
# alpha_W = 0.0001 all variants look ~identical because the penalty sits orders
# of magnitude below the reconstruction loss — that's the prior observation.
# To actually exercise each variant, sweep alpha_W log-spaced and watch the
# Hoyer-sparsity panel + the metric panel diverge across alphas:
#
#   ridge NMF (l1_ratio=0):    [0, 0.001, 0.01, 0.1, 1.0]
#       — shrinks W toward zero proportionally, no hard zeros.
#         Need bigger alphas to see effect than lasso.
#   elastic net NMF (l1=0.5):  [0, 0.001, 0.01, 0.05, 0.1]
#       — mixes shrinkage + sparsity; sparsity onset slightly higher than lasso.
#   lasso NMF (l1_ratio=1):    [0, 0.001, 0.01, 0.05, 0.1]
#       — hard zeros in W typically kick in around 0.01-0.05 on this data.
#         Most informative variant for sparsity sweeps.
#   KL-NMF (l1_ratio=0):       [0, 0.0001, 0.001, 0.01]
#       — KL loss is on a different (smaller) scale than Frobenius, so
#         smaller alphas are needed for comparable relative regularization.
#
# Run AFTER the plain-vs-KL comparison settles, on the winning beta_loss.

n_jobs = 5

for num_comp2 in [5, 10, 20, 30, 50]:#np.arange(1,16,5): , 100, 150, 200
    print('analyzing %d comp' % num_comp2)
    start_time2 = time.perf_counter()
    X = S_smn2.T# .astype(np.float32)

    if 1:
        dred_out = f_PCA(X, num_comp2, random_state = 42)
        dred_all.append(dred_out)

    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, l1_ratio = 0, alpha_W = nmf_alpha_w, alpha_H = 'same')             # ridge NMF      | sweep alpha_W: [0, 0.001, 0.01, 0.1, 1.0]
        dred_all.append(dred_out)

    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, l1_ratio = 0.5, alpha_W = nmf_alpha_w, alpha_H = 'same')           # elastic net NMF | sweep alpha_W: [0, 0.001, 0.01, 0.05, 0.1]
        dred_all.append(dred_out)

    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, l1_ratio = 1, alpha_W = nmf_alpha_w, alpha_H = 'same')             # lasso NMF      | sweep alpha_W: [0, 0.001, 0.01, 0.05, 0.1]  (hard zeros ~0.01-0.05)
        dred_all.append(dred_out)

    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, solver='mu', beta_loss='kullback-leibler', l1_ratio = 0, alpha_W = nmf_alpha_w, alpha_H = 'same')   # KL NMF         | sweep alpha_W: [0, 0.0001, 0.001, 0.01]  (KL loss smaller scale → smaller alphas)
        dred_all.append(dred_out)

    if 0:
        dred_out = f_sparsePCA(X, num_comp2, random_state = 42, n_jobs=n_jobs)
        dred_all.append(dred_out)
    
    if 0:
        dred_out = f_mini_batch_sparsePCA(X, num_comp2, batch_size = 30, random_state = 42, n_jobs=n_jobs)
        dred_all.append(dred_out)
    
    print('elapsed time: total %.2f, loop %.2f' % (time.perf_counter() - start_time, time.perf_counter() - start_time2))

#%% Compute reconstruction-error metrics + organize sweep results
# add quality check
for n_d in range(len(dred_all)):
    f_dred_add_error(S_smn2.T, dred_all[n_d])

algo_labels = []
num_comp = np.zeros(len(dred_all), dtype=int)
# identify the networks and reorganize
for n_d in range(len(dred_all)):
    algo_labels.append(dred_all[n_d]['algo'])
    num_comp[n_d] = dred_all[n_d]['num_comp']

dred_all = np.array(dred_all)
algo_labels = np.array(algo_labels)
algo_uq = np.unique(algo_labels)
num_comp_uq = np.unique(num_comp)
num_algo = algo_uq.shape[0]

#%% Exploratory: reconstruction + component visualization (large if 0: block)

if 0:
    plt.close('all')
    
    data_out2 = dred_all[algo_labels =='elastic_net_NMF'][3]
    
    data_rec = data_out2['scores'] @ data_out2['components']
    
    plt.figure()
    plt.imshow(data_rec, aspect='auto', interpolation='none')

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[1].plot(S_smn2s[0,:])
    ax[0].plot(data_rec[0,:])

    plt.figure()
    plt.plot(data_out2['scores'][1,:])
    
    plt.figure()
    plt.imshow(data_rec.T, aspect='auto', interpolation='none')
    
    n_comp = 0
    plt.figure()
    plt.plot(data_out2['components'][n_comp,:])
    plt.title('components')
    
    n_comp = 0
    plt.figure()
    plt.plot(data_out2['scores'][:,n_comp])
    plt.title('scores')
    
    fig, ax = plt.subplots(2,1, sharex=True, )
    
    data_out = dred_all[algo_labels =='elastic_net_NMF']
    
    data_plot = data_out
    max_comp = 5
    for n_an in range(len(data_plot)):
        d = data_plot[n_an]
        num_comp_plot = np.min([max_comp, d['num_comp']])
        fig, ax = plt.subplots(num_comp_plot, 1, sharex=True)
        for n_comp in range(num_comp_plot):
            if d['num_comp'] == 1:
                ax = (ax,)
            ax[n_comp].plot(d['scores'][:,n_comp])
            ax[n_comp].set_ylabel('comp %d' % (n_comp+1))
        fig.suptitle('scores; %d comp total' % d['num_comp'])

    for n_an in range(len(data_plot)):
        d = data_plot[n_an]
        num_comp_plot = np.min([max_comp, d['num_comp']])
        fig, ax = plt.subplots(num_comp_plot, 1, sharex=True)
        for n_comp in range(num_comp_plot):
            if d['num_comp'] == 1:
                ax = (ax,)
            ax[n_comp].plot(d['components'][n_comp,:])
            ax[n_comp].set_ylabel('comp %d' % (n_comp+1))
        fig.suptitle('components; %d comp total' % d['num_comp'])
    
    
    ens_data = data_out[-1]['scores'][:,:10].T
    ens_data = ens_data - np.min(ens_data, axis=1)[:,None]
    ens_data = ens_data / np.max(ens_data, axis=1)[:,None]
    
    fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].imshow(S_smn2, aspect='auto', vmin=0, vmax=0.5, interpolation='none')
    ax[1].imshow(ens_data, aspect='auto', vmin=0, vmax=0.5, interpolation='none')  # 
    fig.suptitle('NMF')
    
    ens_data = data_out[2]['scores'][:,:10].T
    ens_data = ens_data - np.min(ens_data, axis=1)[:,None]
    ens_data = ens_data / np.max(ens_data, axis=1)[:,None]
    
    fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].imshow(S_smn2, aspect='auto', vmin=0, vmax=0.5, interpolation='none')
    ax[0].set_title('Calcium data')
    ax[0].set_ylabel('CS sorted cells')
    ax[1].imshow(ens_data, aspect='auto', vmin=0, vmax=0.5, interpolation='none')  # 
    ax[1].set_title('%s' % data_out[1]['algo'])
    ax[1].set_ylabel('ensambles')
    ax[1].set_xlabel('frames (60Hz)')
    
    
    data_all = dred_all
    leg = algo_uq
    
    num_comp_all = np.zeros((len(num_comp_uq), len(leg)), dtype=int)
    frob_data_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    frob_rec_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    frob_error_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    rel_error_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    exp_var_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    ss_tot_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    ss_res_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    kl_loss_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    dur_all = np.zeros((len(num_comp_uq), len(leg)), dtype=float)
    leg_all = np.empty((len(leg),), dtype='<U20')
    for n_dred in range(len(leg)):
        data_out = data_all[algo_labels == leg[n_dred]]
        leg_all[n_dred] = data_out[0]['algo']
        for n_comp in range(len(data_out)):
            num_comp_all[n_comp, n_dred] = data_out[n_comp]['num_comp']
            frob_data_all[n_comp, n_dred] = data_out[n_comp]['frob_data']
            frob_rec_all[n_comp, n_dred] = data_out[n_comp]['frob_rec']
            frob_error_all[n_comp, n_dred] = data_out[n_comp]['frob_error']
            rel_error_all[n_comp, n_dred] = data_out[n_comp]['rel_error']
            exp_var_all[n_comp, n_dred] = data_out[n_comp]['explained_var']
            ss_tot_all[n_comp, n_dred] = data_out[n_comp]['ss_tot']
            ss_res_all[n_comp, n_dred] = data_out[n_comp]['ss_res']
            kl_loss_all[n_comp, n_dred] = data_out[n_comp]['kl_loss']
            dur_all[n_comp, n_dred] = data_out[n_comp]['duration']

    # 5-panel comparison: each algo evaluated on both Frobenius (frob_error,
    # rel_error, explained_var) and KL (kl_loss) reconstruction. KL-NMF should
    # win kl_loss; frob-NMF / PCA win frob_error. Disagreement between the two
    # is the interesting signal — same data, different objective.
    fig, ax = plt.subplots(5,1, sharex=True)
    ax[0].plot(num_comp_all, frob_rec_all)
    ax[0].set_ylabel('frob rec')
    ax[0].legend(leg_all)
    ax[1].plot(num_comp_all, frob_error_all)
    ax[1].set_ylabel('frob error')
    ax[2].plot(num_comp_all, rel_error_all)
    ax[2].set_ylabel('rel error')
    ax[3].plot(num_comp_all, exp_var_all)
    ax[3].set_ylabel('explained var')
    ax[4].plot(num_comp_all, kl_loss_all)
    ax[4].set_ylabel('KL loss')
    ax[4].set_xlabel('components')
    
    
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(num_comp_all, ss_tot_all)
    ax[0].set_ylabel('ss_tot')
    ax[0].legend(leg_all)
    ax[1].plot(num_comp_all, ss_res_all)
    ax[1].set_ylabel('ss_res')
    
    fig, ax = plt.subplots(1,1, sharex=True)
    ax.plot(num_comp_all, dur_all)
    ax.legend(leg_all)
    ax.set_ylabel('time (sec)')
    ax.set_xlabel('components')


# =============================================================================
# SECTION 3 — Per-method comparison diagnostics (Tier 1)
# =============================================================================

#%% Tier 1 — component correlation heatmaps (per algo, fixed k)
# At a fixed k, |corrcoef| between components answers: how redundant is each
# decomposition? PCA off-diagonal is ~0 by construction (orthogonal components).
# NMF can produce highly correlated components at large k — bright off-diagonal
# blobs flag pairs of nearly-redundant ensembles, i.e. k may be too high for
# this algo. Compare across algos at the same k to see which one packs more
# distinct ensembles into the same k.

if 0:
    k_show = 20  # pick from num_comp_uq

    fig, axes = plt.subplots(1, len(algo_uq), figsize=(3*len(algo_uq), 3.2))
    if len(algo_uq) == 1:
        axes = [axes]
    im = None
    for n_a, algo in enumerate(algo_uq):
        d_algo = dred_all[algo_labels == algo]
        match = [d for d in d_algo if d['num_comp'] == k_show]
        if not match:
            continue
        comps = match[0]['components']  # (k, n_features = n_cells)
        cm = np.abs(np.corrcoef(comps))
        im = axes[n_a].imshow(cm, vmin=0, vmax=1, cmap='viridis', interpolation='none')
        axes[n_a].set_title(algo, fontsize=9)
    fig.suptitle(f'|corr| between components, k={k_show}')
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02)


#%% Tier 1 — Hoyer sparsity per algo (vs k)
# Hoyer index in [0, 1]: 0 = fully dense, 1 = single non-zero entry.
# Computed on |W| (column-wise → temporal sparsity of each component's score)
# and |H| (row-wise → cell-loading sparsity). abs() makes PCA (signed) directly
# comparable to NMF (non-negative). KL-NMF's main theoretical claim is that it
# produces sparser components than frob-NMF — this plot tests that claim
# directly. Expect: PCA dense (low Hoyer); NMF sparser; KL-NMF sparsest if
# claim holds.

# f_hoyer_sparsity moved to functions/f_ensembles.py (imported above).

if 0:
    sparsity_W = np.zeros((len(num_comp_uq), len(algo_uq)))
    sparsity_H = np.zeros((len(num_comp_uq), len(algo_uq)))
    for n_a, algo in enumerate(algo_uq):
        d_algo = dred_all[algo_labels == algo]
        for d in d_algo:
            n_k = np.where(num_comp_uq == d['num_comp'])[0][0]
            W = d['scores']        # (T, k) — per-component temporal activation
            H = d['components']    # (k, n_cells) — per-component cell loading
            sparsity_W[n_k, n_a] = np.mean([f_hoyer_sparsity(W[:, i]) for i in range(W.shape[1])])
            sparsity_H[n_k, n_a] = np.mean([f_hoyer_sparsity(H[i, :]) for i in range(H.shape[0])])

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(num_comp_uq, sparsity_W)
    ax[0].set_ylabel('mean Hoyer(W)')
    ax[0].legend(algo_uq, fontsize=8)
    ax[0].set_title('temporal sparsity (per-component score)')
    ax[1].plot(num_comp_uq, sparsity_H)
    ax[1].set_ylabel('mean Hoyer(H)')
    ax[1].set_xlabel('components')
    ax[1].set_title('cell-loading sparsity (per component)')


#%% Tier 1 — component activation distribution (per algo, fixed k)
# For each component, fraction of frames where |score| exceeds 10% of its max.
# Low fraction = transient / event-locked ensemble (fires occasionally).
# High fraction = always-on baseline component (e.g., overall arousal).
# Use abs() so PCA (signed scores) is comparable to NMF.
# Expectation: KL-NMF should produce more low-fraction (transient) components
# than frob-NMF, because KL down-weights silent-bin errors and lets components
# specialize to active periods instead of smearing across silence.

if 0:
    k_show = 20
    thresh_frac = 0.1  # active if |score| > thresh_frac * max(|score|)

    fig, ax = plt.subplots(1, 1)
    for n_a, algo in enumerate(algo_uq):
        d_algo = dred_all[algo_labels == algo]
        match = [d for d in d_algo if d['num_comp'] == k_show]
        if not match:
            continue
        scores = match[0]['scores']  # (T, k)
        fracs = []
        for i in range(scores.shape[1]):
            s = np.abs(scores[:, i])
            mx = s.max()
            if mx == 0:
                fracs.append(0.0)
                continue
            fracs.append(np.mean(s > thresh_frac * mx))
        ax.hist(fracs, bins=20, alpha=0.5, label=algo, range=(0, 1))
    ax.set_xlabel(f'fraction of frames active (>{int(thresh_frac*100)}% of max)')
    ax.set_ylabel('# components')
    ax.set_title(f'component activation distribution, k={k_show}')
    ax.legend(fontsize=8)


#%% Tier 1 — reconstruction overlay (per algo, fixed k)
# Stacked imshows: top row is the data S_smn2 (cells x time, hclust-sorted),
# remaining rows are each algo's full reconstruction at fixed k_show, in the
# SAME cell order and SAME color scale as the data. Direct visual A/B test.
#
# What to look for:
#  - Does the reconstruction preserve the discrete event-locked bursts visible
#    in the data, or smear them into smooth baselines?
#  - Are co-firing blocks (the diagonal stripes from hclust ordering) preserved?
#  - Does any algo introduce ringing / negative artifacts (PCA can; clipped
#    here at vmin=0 so they show as flat black).
#  - Are silent regions kept silent, or does the reconstruction hallucinate
#    weak baseline activity everywhere?
#
# The numerical metrics (frob_error, kl_loss, exp_var, Hoyer) rank algos on
# objective functions. This plot tells you whether that ranking corresponds to
# preserving structure you actually care about. If KL-NMF wins kl_loss but its
# reconstruction smears events, the metric ranking isn't biologically useful.

if 0:
    k_show = 20

    n_rows = 1 + len(algo_uq)
    fig, ax = plt.subplots(n_rows, 1, sharex=True, figsize=(8, 1.6*n_rows))

    ax[0].imshow(S_smn2, aspect='auto', vmin=0, vmax=0.5, interpolation='none')
    ax[0].set_ylabel('data')
    ax[0].set_title(f'reconstruction overlay, k={k_show}')

    for n_a, algo in enumerate(algo_uq):
        d_algo = dred_all[algo_labels == algo]
        match = [d for d in d_algo if d['num_comp'] == k_show]
        if not match:
            ax[1+n_a].set_ylabel(algo + '\n(missing)')
            continue
        d = match[0]
        # full reconstruction in the same units as S_smn2 (cells x time):
        # X_fit = scores @ components + min_val(NMF) + mean(PCA)
        min_val = d.get('min_val', 0)
        mean = d['model'].mean_ if hasattr(d['model'], 'mean_') else 0
        rec = d['scores'] @ d['components'] + min_val + mean   # (T, n_cells)
        ax[1+n_a].imshow(rec.T, aspect='auto', vmin=0, vmax=0.5, interpolation='none')
        ax[1+n_a].set_ylabel(algo)

    ax[-1].set_xlabel('frames')
    fig.tight_layout()


#%% Tier 1 — Frobenius vs KL loss scatter (per algo, all k)
# Each point is one (algo, k); points within an algo are connected in k order
# so you can see the trajectory as components are added. Both losses are
# in-sample reconstruction error on the same data, just with different noise
# models (frob = Gaussian, KL = Poisson).
#
# How to read it:
#  - BOTTOM-LEFT corner = best on both losses. If one algo's curve sits there
#    across all k, it dominates — no trade-off, that's your method.
#  - L-SHAPE / Pareto front = real trade-off. Some algos low frob / high KL,
#    others the reverse. Method choice depends on which loss reflects the
#    noise model you actually believe (Gaussian → frob, Poisson → KL for
#    spike-rate-like data).
#  - TIGHT CLUSTER = all algos give similar reconstruction by both measures.
#    Either k is too low (everyone fits noise the same way) or the data isn't
#    discriminative — pick on speed / interpretability instead.
#  - OUTLIER on both axes = different regime (e.g., KL-NMF found a much
#    sparser solution that costs more on both losses but might be more
#    interpretable — corroborate with the Hoyer plot).
#
# The 5-panel metric figure shows trends per loss; this scatter shows
# CONSENSUS across losses. Single most diagnostic single plot for "does the
# choice of loss change which method wins."

if 0:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for algo in algo_uq:
        d_algo = dred_all[algo_labels == algo]
        # sort by k so the connecting line goes in increasing-k order
        ks = np.array([d['num_comp'] for d in d_algo])
        order = np.argsort(ks)
        d_algo = d_algo[order]
        ks = ks[order]
        fe = np.array([d['frob_error'] for d in d_algo])
        kl = np.array([d['kl_loss']    for d in d_algo])
        ax.plot(fe, kl, 'o-', label=algo, alpha=0.7)
        # annotate first and last k so the trajectory direction is readable
        ax.annotate(f'k={ks[0]}',  (fe[0],  kl[0]),  fontsize=7, alpha=0.7)
        ax.annotate(f'k={ks[-1]}', (fe[-1], kl[-1]), fontsize=7, alpha=0.7)
    ax.set_xlabel('Frobenius error  ||X - X_rec||_F')
    ax.set_ylabel('KL loss  Σ X·log(X/X_rec) - X + X_rec')
    ax.set_title('reconstruction loss by both objectives, per (algo, k)')
    ax.legend(fontsize=8)


# =============================================================================
# SECTION 4 — Stability across seeds
# =============================================================================

#%% Stability across seeds — helpers + sweep + plots
# Asks a different question than the loss / sparsity plots above: not "which
# method fits well" but "which COMPONENTS are real ensembles vs arbitrary
# basis vectors that fit data + noise". A low-error decomposition with k=30
# can be 8 stable + 22 noise components — reconstruction error can't see this.
#
# Method: refit each (algo, k) n_seeds times with different random_state. For
# each component in seed 0 (reference), find its best match in each other
# seed via Hungarian assignment on |corr|. Stability per ref component =
# mean |corr| of its matched partner across seeds. Range [0, 1]; stable
# components ~0.9+, noise ~random matchings.
#
# Picks k for you: the "mean stability vs k" curve typically rises, plateaus,
# then drops as k exceeds the data's effective dimensionality. The plateau
# height = effective k. Most principled answer to "how many components".

# f_component_stability moved to functions/f_ensembles.py (imported above).


#%% Stability sweep — refits each (algo, k) n_seeds times
# Cost: n_seeds × n_algos × len(ks_stab) fits. KL-NMF dominates runtime.
# Subset ks_stab to keep this manageable; the curve shape is what matters,
# not per-point precision.

if 0:
    n_seeds = 10
    ks_stab = [10, 20, 50]   # subset of full sweep grid

    # algo specs mirror the main sweep (line ~351 onward); each entry is
    # (label, callable taking X, k, random_state)
    algo_specs = [
        ('PCA',             lambda X, k, rs: f_PCA(X, k, random_state=rs)),
        ('NMF',             lambda X, k, rs: f_NMF(X, k, random_state=rs, max_iter=nmf_max_iter, l1_ratio=0,   alpha_W=nmf_alpha_w, alpha_H='same')),
        ('elastic_net_NMF', lambda X, k, rs: f_NMF(X, k, random_state=rs, max_iter=nmf_max_iter, l1_ratio=0.5, alpha_W=nmf_alpha_w, alpha_H='same')),
        ('lasso_NMF',       lambda X, k, rs: f_NMF(X, k, random_state=rs, max_iter=nmf_max_iter, l1_ratio=1,   alpha_W=nmf_alpha_w, alpha_H='same')),
        ('KL_NMF',          lambda X, k, rs: f_NMF(X, k, random_state=rs, max_iter=nmf_max_iter, solver='mu', beta_loss='kullback-leibler', l1_ratio=0, alpha_W=nmf_alpha_w, alpha_H='same')),
    ]

    X = S_smn2.T

    # nested dict: stability_results[algo_name][k] = (n_seeds-1, k) sim matrix
    stability_results = {name: {} for name, _ in algo_specs}
    t_total = time.perf_counter()
    for algo_name, algo_fn in algo_specs:
        for k in ks_stab:
            t0 = time.perf_counter()
            comps_list = [algo_fn(X, k, s)['components'] for s in range(n_seeds)]
            stability_results[algo_name][k] = f_component_stability(comps_list)
            print(f'  {algo_name:18s} k={k:3d}  {time.perf_counter()-t0:6.1f}s')
    print(f'stability sweep total: {time.perf_counter()-t_total:.1f}s')


#%% Stability plot — mean stability vs k, one curve per algo
# Main diagnostic. Stable plateau height = effective dimensionality.
# Two algos can match on exp_var while one finds 15 stable + 5 noise
# components and the other finds 5 stable + 15 noise — loss curves look
# identical, this curve is very different. KL-NMF is theorized to be
# more stable than frob-NMF on count-like data; this tests it.

if 0:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for algo_name in stability_results:
        ks = sorted(stability_results[algo_name].keys())
        means = [stability_results[algo_name][k].mean() for k in ks]
        ax.plot(ks, means, 'o-', label=algo_name, alpha=0.8)
    ax.set_xlabel('components (k)')
    ax.set_ylabel('mean component stability  (|corr| across seeds)')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color='gray', ls=':', alpha=0.5)  # rule-of-thumb "stable" threshold
    ax.set_title(f'component stability vs k  ({n_seeds} seeds)')
    ax.legend(fontsize=8)


#%% Stability per-component heatmap — for one chosen (algo, k)
# Rows = comparison seeds; columns = ref components (sorted by mean stability,
# best first). Color = matched |corr|. Reads as: "for each ref component, how
# reliably does it reappear across seeds?"
#  - Solid bright columns on the left, fading to mottled on the right →
#    a few real ensembles + many noise components. Effective k = column
#    where brightness drops.
#  - Uniformly bright everywhere → all components are real; consider larger k.
#  - Uniformly dim → algo is unstable at this k; either k is too high or
#    the algo is fundamentally noisy at this regime.

if 0:
    chosen_algo = 'KL_NMF'
    chosen_k = 20

    sim = stability_results[chosen_algo][chosen_k]   # (n_seeds-1, k)
    mean_stab = sim.mean(axis=0)                     # per ref component
    order = np.argsort(mean_stab)[::-1]              # best first
    sim_sorted = sim[:, order]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    im = ax.imshow(sim_sorted, vmin=0, vmax=1, aspect='auto', cmap='viridis', interpolation='none')
    ax.set_xlabel('ref component (sorted by stability, best first)')
    ax.set_ylabel('comparison seed')
    ax.set_title(f'per-component stability — {chosen_algo}, k={chosen_k}')
    fig.colorbar(im, ax=ax, fraction=0.03, label='|corr| with matched partner')


# =============================================================================
# SECTION 5 — Shuffled-null sweep (exp_var floor)
# =============================================================================

#%% Shuffled null — sweep on circshift-shuffled data
# Lightweight noise floor: refit every (algo, k) on data where each cell's
# time series is independently circularly shifted by a random offset. This
# DESTROYS cross-cell timing (the structure ensembles capture) but PRESERVES
# per-cell rate, autocorrelation, and burstiness — so any exp_var the model
# achieves on shuffled data reflects flexibility / per-cell baseline fitting,
# not real co-firing structure.
#
# Real signal = exp_var(real) - exp_var(shuffled). Without this baseline,
# absolute exp_var is uninterpretable (exp_var = 0.6 could be all signal or
# all per-cell baseline; can't tell). Shuffled gap separates the two.
#
# Single shuffle is enough for a baseline; for error bars on the noise
# floor, loop n_shuf times and average exp_var_shuf per (algo, k).

if 0:
    X_shuf = f_circshift_rates(S_smn2, min_shift=0,
                               rng=np.random.default_rng(42)).T   # (T, N), main-sweep orientation

    # algo specs mirror the main sweep settings (line ~351 onward).
    # Each entry is (label, callable taking X, k).
    algo_specs_shuf = [
        ('PCA',             lambda X, k: f_PCA(X, k, random_state=42)),
        ('NMF',             lambda X, k: f_NMF(X, k, random_state=42, max_iter=nmf_max_iter, l1_ratio=0,   alpha_W=nmf_alpha_w, alpha_H='same')),
        ('elastic_net_NMF', lambda X, k: f_NMF(X, k, random_state=42, max_iter=nmf_max_iter, l1_ratio=0.5, alpha_W=nmf_alpha_w, alpha_H='same')),
        ('lasso_NMF',       lambda X, k: f_NMF(X, k, random_state=42, max_iter=nmf_max_iter, l1_ratio=1,   alpha_W=nmf_alpha_w, alpha_H='same')),
        ('KL_NMF',          lambda X, k: f_NMF(X, k, random_state=42, max_iter=nmf_max_iter, solver='mu', beta_loss='kullback-leibler', l1_ratio=0, alpha_W=nmf_alpha_w, alpha_H='same')),
    ]

    dred_all_shuf = []
    t_total = time.perf_counter()
    for k in num_comp_uq:
        for algo_name, algo_fn in algo_specs_shuf:
            t0 = time.perf_counter()
            d = algo_fn(X_shuf, int(k))
            f_dred_add_error(X_shuf, d)
            dred_all_shuf.append(d)
            print(f'  {algo_name:18s} k={int(k):3d}  {time.perf_counter()-t0:6.1f}s')
    dred_all_shuf = np.array(dred_all_shuf)
    print(f'shuffled sweep total: {time.perf_counter()-t_total:.1f}s')


#%% Shuffled null — exp_var: real vs shuffled, and the gap
# Top panel: solid = real, dashed = shuffled (same color per algo). The gap
# between them at each k is the actual signal the algo extracts above what
# random data would produce.
# Bottom panel: explicit gap = exp_var(real) - exp_var(shuffled). Algos
# whose gap saturates earliest extract real cross-cell structure most
# efficiently. Gap ~ 0 = algo is just modeling per-cell rate distributions
# (preserved by shuffling), not co-firing.
#
# Cross-check with stability:
#  - High stability + large gap = real, reproducible structure. Trust.
#  - High stability + small gap = reliably finding artifacts (per-cell
#    baseline, not ensembles). Don't interpret biologically.
#  - Low stability + large gap = real signal but algo is unstable; try
#    different method or smaller k.
#  - Low stability + small gap = nothing useful at this k.

if 0:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    shuf_algos = np.array([d['algo'] for d in dred_all_shuf])

    for algo in algo_uq:
        d_real = dred_all[algo_labels == algo]
        d_shuf = dred_all_shuf[shuf_algos == algo]
        if len(d_shuf) == 0:
            continue
        ks_real = sorted(set(d['num_comp'] for d in d_real))
        ks_both = [k for k in ks_real if any(d['num_comp'] == k for d in d_shuf)]
        ev_real = [next(d for d in d_real if d['num_comp'] == k)['explained_var'] for k in ks_both]
        ev_shuf = [next(d for d in d_shuf if d['num_comp'] == k)['explained_var'] for k in ks_both]
        ev_gap  = np.array(ev_real) - np.array(ev_shuf)

        line, = ax[0].plot(ks_both, ev_real, 'o-', label=algo, alpha=0.8)
        ax[0].plot(ks_both, ev_shuf, '--', color=line.get_color(), alpha=0.5)
        ax[1].plot(ks_both, ev_gap, 'o-', label=algo, alpha=0.8, color=line.get_color())

    ax[0].set_ylabel('explained variance')
    ax[0].set_title('solid = real, dashed = shuffled (same color per algo)')
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel('exp_var(real) - exp_var(shuffled)')
    ax[1].set_xlabel('components')
    ax[1].axhline(0, color='gray', ls=':', alpha=0.5)
    ax[1].set_title('signal above noise floor')


# =============================================================================
# SECTION 6 — Ensemble extraction (MATLAB port; see notes.txt 2026-06-01)
# =============================================================================

#%% Method B — auto-num-comp via shuffle PCA eigenvalues
# Counts components whose explained-variance ratio exceeds the max
# eigenvalue of the per-cell circular-shifted null. Cheap; gives a single
# number to seed num_comp before running the full CV grid.

if 0:
    from f_ensemble_plots import f_plot_dim_estimate
    dim_info = f_estimate_dim_corr(S_smn2, n_shuf=50, normalize='norm_mean_std',
                                   random_state=42, verbose=True)
    f_plot_dim_estimate(dim_info, title=f'{mouse_id} dset{n_dset} — Method B')
    print(f"dimensionality_corr = {dim_info['dimensionality_corr']:.2f}  "
          f"→ recommended num_comp = {int(np.ceil(dim_info['dimensionality_corr']))}")


#%% Method A — CV grid (smooth_SD × num_comp) with leave-neuron-out test error
# Port of MATLAB f_ens_estimate_dim_params. Picks the (smooth_SD, num_comp)
# that minimises held-out test error from per-cell predictions made via the
# factor basis fit on the other folds. with_shuffle=True overlays a
# circshift-null surface for reference; the gap real–shuf at each (s, k)
# is the actual signal the model extracts above per-cell baseline.

if 0:
    from f_ensemble_plots import f_plot_cv_grid

    frame_rate_hz = 60
    smooth_SDs_ms = [50, 100, 150, 200]
    smooth_SDs_bins = [int(round(s * frame_rate_hz / 1000)) for s in smooth_SDs_ms]
    num_comps = [4, 8, 12, 16, 20]

    cv_df = f_cv_estimate_grid(
        S_smn2, smooth_SDs_bins, num_comps,
        method='nmf', k_folds=5, reps=2, with_shuffle=True,
        chunked_shuffle=False, normalize='norm_mean_std',
        random_state=42, verbose=True,
    )
    f_plot_cv_grid(cv_df[~cv_df.is_shuf], cv_df[cv_df.is_shuf],
                   x='smooth_SD', y='num_comp', z='test_err',
                   title=f'{mouse_id} dset{n_dset} — CV (LNO test err)')

    best = (cv_df[~cv_df.is_shuf].groupby(['smooth_SD', 'num_comp'])
                                  .test_err.mean().idxmin())
    print(f'best (smooth_SD_bins, num_comp) = {best}')


#%% Methods C / D — ensemble extraction (single best params)
# Pick smooth_SD + num_comp from Method A's CV grid (or set by hand from
# Method B). Two extraction paths:
#   thresh  — per-component cutoff via signal_z (median + 2.5 spread) or
#             shuff (95th percentile of 50-shuffle NMF null). Default for NMF.
#   clust   — hclust on coeffs after shuffle-pwcorr cell filtering. Default
#             for PCA/ICA; alt cross-check for NMF.

if 0:
    ens_out = f_ensemble_extract(
        S_smn2, num_comp=10, smooth_sigma_bins=6,
        dred_method='nmf', extraction='thresh', thresh_mode='signal_z',
        normalize='norm_mean_std', signal_z_thresh=2.5, n_shuf=50,
        random_state=42, verbose=True,
    )
    print(f'kept {ens_out["active_cells_mask"].sum()} / '
          f'{len(ens_out["active_cells_mask"])} cells')
    for i, ci in enumerate(ens_out['cells']['ens_list']):
        print(f"  ens {i+1}: {len(ci)} cells, "
              f"{len(ens_out['trials']['ens_list'][i])} active frames")

    # cluster-based cross-check
    if 0:
        ens_out_c = f_ensemble_extract(
            S_smn2, num_comp=10, smooth_sigma_bins=6,
            dred_method='nmf', extraction='clust',
            normalize='norm_mean_std', random_state=42, verbose=True,
        )

    # shuffle-thresh alternative (slow — refits NMF n_shuf times)
    if 0:
        ens_out_s = f_ensemble_extract(
            S_smn2, num_comp=10, smooth_sigma_bins=6,
            dred_method='nmf', extraction='thresh', thresh_mode='shuff',
            n_shuf=50, random_state=42, verbose=True,
        )


#%% Ensemble visualization — MATLAB driver-style figure set
# 1) sorted-cell raster + mean trace
# 2) per-ensemble 4-panel deet (raster sorted by coeff, sorted-coeff bar,
#    mean trace with pink active-trial background, ensemble score trace)
# Optionally also a 2D/3D component scatter coloured by ensemble.

if 0:
    from f_ensemble_plots import f_plot_ens_overview, f_plot_comp_scatter
    f_plot_ens_overview(ens_out, S_smn2,
                        mouse_dset_tag=f'{mouse_id} dset{n_dset}')
    f_plot_comp_scatter(ens_out['coeffs'], ens_out['cells']['clust_ident'],
                        dim=3, title=f'{mouse_id} dset{n_dset} — comp scatter')


# =============================================================================
# SECTION 7 — Behavior-clamped extraction (exploratory)
# =============================================================================

#%% Behavior-residualized NMF — exploratory (branch b)
# Regress neural data on a behavior block (built via the CEBRA registry),
# then run the full ensemble pipeline on the residuals. The remaining
# structure is the behavior-independent ensembles.
# REQUIRES bh_data — load via VR_ca_cebra.py-style cells first.

if 0:
    # Build behavior features via the same registry the CEBRA script uses.
    # Adjust target_blocks to the behavior dimensions you want to clamp.
    from f_feature_helpers import build_feature_blocks, f_zscore
    from f_cebra_helpers import make_cebra_supervision

    # Minimal call — assumes bh_data, est1, two_mon_frames, vec_data_l/r exist.
    built = build_feature_blocks(
        bh_data[n_dset], est1, vec_data_l, vec_data_r,
        two_mon_frames=None, n_pix_pca=10,
        build_agg=True, build_motion=True, build_self_motion=True,
    )
    B, _ = make_cebra_supervision(['agg', 'motion', 'self_mot'], built)
    # B is (T, d_total); we need (n_feat, T) for f_residualize_on_behavior.
    X_resid, W_beh = f_residualize_on_behavior(S_smn2, B.T, ridge_alpha=1e-3)
    print(f'beh-explained var = '
          f'{1 - np.var(X_resid) / np.var(S_smn2):.3f} of total')

    ens_resid = f_ensemble_extract(
        X_resid, num_comp=10, smooth_sigma_bins=6,
        dred_method='nmf', extraction='thresh', thresh_mode='signal_z',
        random_state=42, verbose=True,
    )
    from f_ensemble_plots import f_plot_ens_overview
    f_plot_ens_overview(ens_resid, X_resid,
                        mouse_dset_tag=f'{mouse_id} dset{n_dset} — beh-resid')


#%% Constrained NMF — exploratory (branch c)
# Semi-supervised NMF where a subset of H rows is CLAMPED to behavior
# targets (B as built above). Useful as a sanity check: does the
# behavior-clamped block reconstruct behavior? Do the free components
# recover non-behavior structure? Heavier than (b) — alternating
# multiplicative-update loop with no convergence guarantees.

if 0:
    # Build behavior block H_clamp = B (n_feat, T). Normalize rows to
    # comparable scale so the free components can fit alongside.
    Bn = f_normalize_rows(B.T, mode='norm_rms')   # keeps non-negative-ish
    Bn = np.maximum(Bn, 0)                         # NMF needs ≥ 0

    cnmf = f_NMF_constrained(
        np.maximum(S_smn2, 0), n_free=8, H_clamp=Bn,
        max_iter=300, tol=1e-4, ridge=1.0, random_state=42, verbose=True,
    )
    print(f"final rel err = {cnmf['recon_err'][-1]:.3f}  "
          f"(n_free={cnmf['n_free']}, n_clamp={cnmf['n_clamp']})")

    # treat free components as a regular NMF result and run threshold
    # extraction on them so we can see what they encode
    coeffs_free = cnmf['W'][:, :cnmf['n_free']]    # (n_cells, n_free)
    scores_free = cnmf['H_free']                    # (n_free, n_t)
    thresh_c, thresh_s = f_ens_get_thresh(
        S_smn2, coeffs_free, scores_free, mode='signal_z',
        signal_z_thresh=2.5, dred_method='nmf')
    ens_cnmf = f_apply_thresh(coeffs_free, scores_free, thresh_c, thresh_s)
    ens_cnmf.update({
        'coeffs': coeffs_free, 'scores': scores_free,
        'ord_cell': np.argsort(-coeffs_free.max(axis=1)),
        'num_comps': cnmf['n_free'], 'extraction_method': 'thresh',
        'ensemble_method': 'NMF_constrained',
        'active_cells_mask': np.ones(S_smn2.shape[0], dtype=bool),
    })
    from f_ensemble_plots import f_plot_ens_overview
    f_plot_ens_overview(ens_cnmf, S_smn2,
                        mouse_dset_tag=f'{mouse_id} dset{n_dset} — cNMF free')

