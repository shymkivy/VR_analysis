# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:25:46 2026

@author: ys2605
"""

import sys
import os

for user1 in ['ys2605', 'shymk']:
    if os.path.isdir('C:/Users/' + user1):
        path1 = 'C:/Users/' + user1 + '/Desktop/stuff/VR/VR_analysis/'
        path2 = 'C:/Users/' + user1 + '/Desktop/stuff/slow_dynamics_analysis/'
        path3 = 'C:/Users/' + user1 + '/Desktop/stuff/RNN_scripts/'
        
sys.path.append(path1)
sys.path.append(path1 + '/functions')
sys.path.append(path2 + '/functions')
sys.path.append(path3 + '/functions')

import numpy as np

#import pandas as pd
import matplotlib.pyplot as plt

from f_sd_utils import f_get_fnames_from_dir, f_load_caim_data, f_gauss_smooth
from f_analysis import f_hclust_firing_rates

#%%

data_dir = 'F:/VR/data_proc/L'    # edit this  
# search for files to load using tags in the filename
flist = f_get_fnames_from_dir(data_dir, ext_list = ['hdf5'], tags=['L', '_results_cnmf'])  # 'results_cnmf_sort'

#%%

data_ca = f_load_caim_data(data_dir, flist, r_values_min = 0.5, min_SNR=1.5, thresh_cnn_min=0.8)


#%%
n_dset = 9
est1 = data_ca[n_dset]
     
S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]
    
if 0:
    plt.close('all')
    
    n_cell = 0
    title_tag = 'cell %d, snr=%.2f' % (n_cell, est1['SNR_comp'][n_cell])
    plt.figure()
    plt.imshow(np.reshape(est1['A'][n_cell,:].toarray(), est1['dims']))
    plt.title(title_tag)
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(est1['C'][n_cell,:] + est1['YrA'][n_cell,:])
    ax1.set_title(title_tag)
    ax2.plot(est1['S'][n_cell,:])
    
    plt.figure()
    plt.imshow(np.reshape(est1['A'].sum(axis=0), est1['dims']).T)
    plt.title('%d cells' % est1['A'].shape[0])
    
    n_cell = 1
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(est1['C'][n_cell,:] + est1['YrA'][n_cell,:])
    ax2.plot(est1['S'][n_cell,:])
    ax2.plot(S_sm[n_cell,:])
    
    plt.figure()
    plt.imshow(S_smn, aspect='auto')

#%%

res_ord = f_hclust_firing_rates(S_smn, standardize=True, metric='cosine', method='average')

if 1:
    plt.figure()
    plt.imshow(S_smn[res_ord,:], aspect='auto', vmin=0, vmax=0.5, interpolation='none') 
    plt.title('dataset %d' % n_dset)
    plt.ylabel('CS sorted neurons')
    plt.xlabel('Frames')
    
    
    

