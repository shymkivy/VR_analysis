# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 12:23:44 2026

@author: ys2605
"""

import sys
import os

for user1 in ['ys2605', 'shymk']:
    if os.path.isdir('C:/Users/' + user1):
        path1 = 'C:/Users/' + user1 + '/Desktop/stuff/VR/VR_analysis/'

sys.path.append(path1)
sys.path.append(path1 + '/functions')

import numpy as np
import pandas as pd
import tifffile as tf
import json
import matplotlib.pyplot as plt

#%%
data_path = 'F:/VR/data_proc/2p_fov_reg/'

path_xls = ''

vr_data = pd.read_excel('F:/VR/data_proc/VR_data.xlsx')
vr_data_slice = vr_data.loc[0]


tp_locs = np.array(json.loads(vr_data_slice['tp_fov_locs']))
wf_locs = np.array(json.loads(vr_data_slice['wf_locs']))

wf_fov = tf.imread(os.path.join(data_path, vr_data_slice['wf_im']))
twop_fov = tf.imread(os.path.join(data_path, vr_data_slice['tp_im']))


#%%
plt.close('all')

plt.figure()
plt.imshow(np.rot90(wf_fov, k=0, axes=(1,0)), aspect='auto')
for n_p in range(wf_locs.shape[0]):     # wf_locs.shape[0]
    plt.plot(wf_locs[n_p,0], wf_locs[n_p,1], '.', color='red')
    plt.text(wf_locs[n_p,0], wf_locs[n_p,1], '%d' % (n_p))
plt.text(np.mean(wf_locs[:,0]), np.mean(wf_locs[:,1]) , vr_data_slice['dset_name'], horizontalalignment='center', verticalalignment='center', color='red')


plt.figure()
plt.imshow(np.rot90(twop_fov, k=0, axes=(1,0)), aspect='auto')
for n_p in range(wf_locs.shape[0]):     # wf_locs.shape[0]
    plt.plot(tp_locs[n_p,0], tp_locs[n_p,1], '.', color='red')
    plt.text(tp_locs[n_p,0], tp_locs[n_p,1], '%d' % (n_p))
plt.title(vr_data_slice['dset_name'])

if 0:
    plt.figure()
    plt.imshow(np.rot90(wf_fov, k=-1, axes=(1,0)), aspect='auto')




