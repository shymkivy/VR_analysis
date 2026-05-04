# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:01:39 2026

@author: ys2605
"""

import sys
import os

for user1 in ['ys2605', 'shymk']:
    if os.path.isdir('C:/Users/' + user1):
        path1 = 'C:/Users/' + user1 + '/Desktop/stuff/'
        
sys.path.append(path1 + '/VR/VR_analysis/')
sys.path.append(path1 + '/VR/VR_analysis/functions')
sys.path.append(path1 + '/RNN_scripts/functions')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearnex import patch_sklearn
patch_sklearn()

from f_utils import f_load_caim_data, f_gauss_smooth
from f_analysis import f_hclust_firing_rates, f_circshift_rates
from f_functions import f_load_bh_data, f_get_session_data, f_plot_session2, f_proc_movement, f_proc_lick_rew, f_proc_lick_rew_df, f_comp_FOV_adj, f_add_phase, f_get_monitor_coords, f_plot_monitor_outline, f_plot_lateral_over_time, f_plot_vertical_over_time, f_plot_dist_over_time, f_angles_to_movie #, f_plot_session
from f_RNN_dred import f_run_dred


from sklearn.decomposition import SparsePCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import MiniBatchSparsePCA

import time

#%%
mouse_id = 'L'

data_dir = 'F:/VR/data_proc/' + mouse_id    # edit this  
data_dir_bh = 'F:/VR/Bh_data/mice_gcamp/'
# search for files to load using tags in the filename

vr_data = pd.read_excel('F:/VR/data_proc/VR_data.xlsx')
params_xls = pd.read_excel(data_dir_bh + mouse_id + '_params.xlsx')

vr_data2 = vr_data.iloc[(vr_data.mouse_id == mouse_id) & (vr_data.do_proc == 1)].reset_index(drop=True)
num_dsets = len(vr_data2)

lick_col = 'red'
lick_col2 = 'lightcoral'
rew_col = 'green'
rew_col2 = 'limegreen'

obj_size = {'x':        5,
            'y':        10,
            'z':        5,
            'height':   2}


# cylinders radius 5, half height 10.. coordinate at center.. height is y + height

#%%
data_ca = f_load_caim_data(data_dir, vr_data2.dset_name, caiman_tag = 'results_cnmf.hdf5', cuts_tag='h5cutsinfo', frame_data_tag = 'framedata', r_values_min = 0.5, min_SNR=1.5, thresh_cnn_min=0.8)

#%%
n_dset = 3

num_comp = 50
n_jobs = 5

est1 = data_ca[n_dset]
     
S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]

hclust_data = f_hclust_firing_rates(S_smn, standardize=True, metric='cosine', method='average')

S_smn2 = S_smn[hclust_data['res_order'],:]
plt.figure()
plt.imshow(S_smn2, aspect='auto', vmin=0, vmax=0.5, interpolation='none')


# scale
S_smn2s = S_smn2 - np.mean(S_smn2, axis=1)[:,None]
S_smn2s = S_smn2s/np.std(S_smn2s, axis=1)[:,None]
# S_smn2s = scaler.fit_transform(S_smn2.T).T

plt.figure()
plt.plot(S_smn2s[0,:])
#plt.plot(X_scaled[0,:])

plt.figure()
plt.imshow(S_smn2s, aspect='auto', interpolation='none')



#%% NMF

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

def f_NMF(X, num_comp, max_iter = 500, random_state = None, tol=1e-4, beta_loss='frobenius', l1_ratio = 0.0, alpha_W = 0, alpha_H = 'same'):
    
    start_time = time.perf_counter()
    
    min_val = np.min(X)
    # Data must be non-negative
    model = NMF(
        n_components=num_comp,        # number of latent dimensions
        init='nndsvda',         # smart initialization (recommended)
        solver='cd',            # coordinate descent;   cd, mu 
        beta_loss=beta_loss,    # minimizes ||X - WH||²         frobenius, kullback-leibler, itakura-saito
                                # Frobenius — standard L2, sensitive to outliers 
                                # Kullback-Leibler — good for count data (text, bag-of-words)
                                # Itakura-Saito — good for audio/spectral data
        alpha_W=alpha_W,       # regularization on W
        alpha_H=alpha_H,       # regularization on H
        l1_ratio=l1_ratio,      # 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
        tol = tol,
        max_iter=max_iter,
        random_state=random_state,
    )
    
    W = model.fit_transform(X-min_val)   # shape: (n_samples, n_components) — reduced representation
    #H = nmf.components_        # shape: (n_components, n_features) — basis components
    
    
    if l1_ratio == 0:
        algo = 'NMF'
    elif l1_ratio == 0.5:
        algo = 'elastic_net_NMF'
    elif l1_ratio == 1:
        algo = 'lasso_NMF'
    elif l1_ratio == None:
        algo = 'NMF'
    
    out = {'algo':          algo,
           'num_comp':      num_comp,
           'components':    model.components_,
           'scores':        W,
           'min_val':       min_val,
           'model':         model,
           'max_iter':      max_iter,
           'random_state':  random_state,
           'beta_loss':     beta_loss,
           'l1_ratio':      l1_ratio,
           'alpha_W':       alpha_W,
           'alpha_H':       alpha_H,
           'tol':           tol,
           'duration':      time.perf_counter() - start_time,
           }
    
    return out

def f_PCA(X, num_comp, svd_solver='randomized', random_state=None, tol=0.0):
    
    start_time = time.perf_counter()
    
    # 2. Fit PCA
    model = PCA(
        n_components=num_comp,
        svd_solver=svd_solver,          # auto
        random_state = random_state,
        tol=tol
        )
    X_reduced = model.fit_transform(X)

    out = {'algo':          'PCA',
           'num_comp':      num_comp,
           'components':    model.components_,
           'scores':        X_reduced,
           'model':         model,
           'svd_solver':    svd_solver,
           'tol':           tol,
           'duration':      time.perf_counter() - start_time,
           }
    
    return out


def f_sparsePCA(X, num_comp, random_state=None, alpha=1.0, ridge_alpha=0.01, n_jobs=5, method = 'lars', tol=1e-08):
    
    start_time = time.perf_counter()
    
    model = SparsePCA(
        n_components=num_comp,
        method = method,                # lars, cd
        alpha=alpha,                    # elastic net regularization strength
        ridge_alpha=ridge_alpha,        # L2 component (stability)
        max_iter=500,
        tol = tol,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    X_reduced = model.fit_transform(X)   # shape: (n_samples, n_components)

    out = {'algo':          'sparsePCA',
           'num_comp':      num_comp,
           'components':    model.components_,
           'scores':        X_reduced,
           'model':         model,
           'tol':           tol,
           'duration':      time.perf_counter() - start_time,
           }
    
    return out

def f_mini_batch_sparsePCA(X, num_comp, random_state=None, batch_size = 3, alpha=1.0, ridge_alpha=0.01, n_jobs=5, method = 'lars', tol=1e-03):
    
    start_time = time.perf_counter()
    
    model = MiniBatchSparsePCA(
        n_components=num_comp,
        method = method,                # lars, cd
        alpha=alpha,                    # elastic net regularization strength
        ridge_alpha=ridge_alpha,        # L2 component (stability)
        max_iter=500,
        batch_size = batch_size,        # The number of features (neurons) to take in each mini batch.
        tol = tol,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    X_reduced = model.fit_transform(X)   # shape: (n_samples, n_components)

    out = {'algo':          'mini_batch_sparsePCA',
           'num_comp':      num_comp,
           'components':    model.components_,
           'scores':        X_reduced,
           'model':         model,
           'batch_size':    batch_size,
           'tol':           tol,
           'duration':      time.perf_counter() - start_time,
           }
    
    return out

def f_dred_add_error(X, dred_data):
    
    if 'min_val' in dred_data:
        min_val = dred_data['min_val']
    else:
        min_val = 0
    
    
    if hasattr(dred_data['model'], 'mean_'):
        mean = dred_data['model'].mean_
    else:
        mean = 0
    
    data_rec = dred_data['scores'] @ dred_data['components'] + min_val + mean
    
    frob_data = np.linalg.norm(X, 'fro')
    frob_error = np.linalg.norm(X - data_rec, 'fro')
    rel_error  = frob_error / np.linalg.norm(data_rec, 'fro')
    
    # Explained variance equivalent
    ss_res = np.sum((X - data_rec) ** 2)
    ss_tot = np.sum((X - X.mean()) ** 2)
    explained_var = 1 - ss_res/ss_tot

    dred_data['frob_data'] = frob_data
    dred_data['frob_rec'] = np.linalg.norm(data_rec, 'fro')
    dred_data['frob_error'] = frob_error
    dred_data['rel_error'] = rel_error
    dred_data['ss_res'] = ss_res
    dred_data['ss_tot'] = ss_tot
    dred_data['explained_var'] = explained_var
    

#%%

# inputs should be observations x features, to time x cells
# output to fits are sample coefficients, and components are feature basis vectors
# for t x n orientation, the loss function sums over timepoints, do the temporal structure is directly in the loss function

start_time = time.perf_counter()
dred_all = []

nmf_alpha_w = 0.0001
nmf_max_iter = 2000

n_jobs = 5

for num_comp2 in [5, 10, 20, 50, 100, 150, 200]:#np.arange(1,16,5):
    print('analyzing %d comp' % num_comp2)
    start_time2 = time.perf_counter()
    X = S_smn2.T# .astype(np.float32)
    
    if 1:
        dred_out = f_PCA(X, num_comp2, random_state = 42)
        dred_all.append(dred_out)
    
    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, l1_ratio = 0, alpha_W = nmf_alpha_w, alpha_H = 'same')             # ridge NMF
        dred_all.append(dred_out)
    
    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, l1_ratio = 0.5, alpha_W = nmf_alpha_w, alpha_H = 'same')           # elastic net NMF
        dred_all.append(dred_out)
    
    if 1:
        dred_out = f_NMF(X, num_comp2, random_state = 42, max_iter = nmf_max_iter, l1_ratio = 1, alpha_W = nmf_alpha_w, alpha_H = 'same')             # lasso NMF
        dred_all.append(dred_out)
    
    if 1:
        dred_out = f_sparsePCA(X, num_comp2, random_state = 42, n_jobs=n_jobs)
        dred_all.append(dred_out)
    
    if 1:
        dred_out = f_mini_batch_sparsePCA(X, num_comp2, batch_size = 30, random_state = 42, n_jobs=n_jobs)
        dred_all.append(dred_out)
    
    print('elapsed time: total %.2f, loop %.2f' % (time.perf_counter() - start_time, time.perf_counter() - start_time2))

#%%
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

#%%

if 0:
    plt.close('all')
    
    data_out = dred_all[algo_labels =='elastic_net_NMF']
    
    data_rec = data_out['scores'] @ data_out['components']
    
    plt.figure()
    plt.imshow(data_rec, aspect='auto', interpolation='none')

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[1].plot(S_smn2s[0,:])
    ax[0].plot(data_rec[0,:])

    plt.figure()
    plt.plot(data_out['scores'][1,:])
    
    plt.figure()
    plt.imshow(data_rec.T, aspect='auto', interpolation='none')
    
    n_comp = 0
    plt.figure()
    plt.plot(data_out['components'][n_comp,:])
    plt.title('components')
    
    n_comp = 0
    plt.figure()
    plt.plot(data_out['scores'][:,n_comp])
    plt.title('scores')
    
    fig, ax = plt.subplots(2,1, sharex=True, )
    
    
    data_plot = data_out
    max_comp = 5
    for n_an in range(len(data_plot)):
        data_out = data_plot[n_an]
        num_comp_plot = np.min([len(data_out), max_comp, data_out['num_comp']])
        fig, ax = plt.subplots(num_comp_plot, 1, sharex=True)
        for n_comp in range(num_comp_plot):
            if data_out['num_comp'] == 1:
                ax = (ax,)
            ax[n_comp].plot(data_out['scores'][:,n_comp])
            ax[n_comp].set_ylabel('comp %d' % (n_comp+1))
        fig.suptitle('scores; %d comp total' % data_out['num_comp'])
    
    for n_an in range(len(data_plot)):
        data_out = data_plot[n_an]
        num_comp_plot = np.min([len(data_out), max_comp, data_out['num_comp']])
        fig, ax = plt.subplots(num_comp_plot, 1, sharex=True)
        for n_comp in range(num_comp_plot):
            if data_out['num_comp'] == 1:
                ax = (ax,)
            ax[n_comp].plot(data_out['components'][n_comp,:])
            ax[n_comp].set_ylabel('comp %d' % (n_comp+1))
        fig.suptitle('components; %d comp total' % data_out['num_comp'])
    
    
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
            dur_all[n_comp, n_dred] = data_out[n_comp]['duration']
    
    fig, ax = plt.subplots(4,1, sharex=True)
    ax[0].plot(num_comp_all, frob_rec_all)
    ax[0].set_ylabel('frob rec')
    ax[0].legend(leg_all)
    ax[1].plot(num_comp_all, frob_error_all)
    ax[1].set_ylabel('frob error')
    ax[2].plot(num_comp_all, rel_error_all)
    ax[2].set_ylabel('rel error')
    ax[3].plot(num_comp_all, exp_var_all)
    ax[3].set_ylabel('explained var')
    ax[3].set_xlabel('components')
    
    
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
#%% error
# Reconstruct original matrix


