# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:25:46 2026

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
import tifffile as tf

from f_utils import f_load_caim_data, f_gauss_smooth
from f_analysis import f_hclust_firing_rates, f_circshift_rates
from f_functions import f_load_bh_data, f_get_session_data, f_plot_session2, f_proc_movement, f_proc_lick_rew, f_proc_lick_rew_df, f_comp_FOV_adj, f_add_phase, f_get_monitor_coords, f_plot_monitor_outline, f_plot_lateral_over_time, f_plot_vertical_over_time, f_plot_dist_over_time, f_angles_to_movie #, f_plot_session
from f_RNN_dred import f_run_dred

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
pulse_diff_frames = []
pulse_diff_vid = []

bh_data = []

for n_dset in range(num_dsets):
    vr_data_slice = vr_data2.iloc[n_dset]
    est1 = data_ca[n_dset]
    pulse_times = est1['frame_times'][est1['vid_cuts'][:-1,1]]
    
    #flist = f_get_fnames_from_dir(data_dir_bh + mouse_id, ext_list = [], tags=[vr_data_slice.bh_dset_name])  # 'results_cnmf_sort'
    
    bh_data_slice = f_load_bh_data(data_dir_bh + '/' + mouse_id + '/', vr_data_slice.bh_dset_name, params_xls)
    
    idx1 = bh_data_slice['events'].event == 'AlignmentPulse'
    align_slice = bh_data_slice['events'].iloc[idx1]

    bh_data_slice['align_slice'] = align_slice
    bh_data_slice['bh_pulse_delay'] = align_slice.Time.iloc[0] - est1['frame_times'][est1['vid_cuts'][0,1]]
    bh_data.append(bh_data_slice)
    
    diff_frames = np.diff(pulse_times)
    diff_vid = np.diff(align_slice.Time)
    if len(diff_frames) and len(diff_vid):
        pulse_diff_frames.append(np.max(diff_frames))
        pulse_diff_vid.append(np.max(diff_vid))

if 0:
    plt.figure()
    plt.plot(np.array(pulse_diff_frames) - np.array(pulse_diff_vid), '.')
    plt.xlabel('dataset')
    plt.ylabel('frame - vid pulse diff (sec)')
    
    plt.figure()
    plt.plot(pulse_diff_vid, np.array(pulse_diff_frames) - np.array(pulse_diff_vid), '.')
    plt.ylabel('frame - vid pulse diff (sec)')
    plt.xlabel('recording duration (sec)')
    
    n_dset = 3
    est1 = data_ca[n_dset]
    plt.figure()
    plt.plot(est1['frame_times'], ~est1['vid_cuts_trace'])
    plt.xlabel('Time (sec)')
    for align_time in bh_data[n_dset]['align_slice'].Time:
        plt.plot(np.ones(2) * align_time - bh_data[n_dset]['bh_pulse_delay'], [0, 1], color='k')
    plt.title('dataset %d' % n_dset)

s_data = f_get_session_data(bh_data)

axs = f_plot_session2(s_data, y_size=2)
p1 = axs.plot(s_data['idx'], s_data['num_rewards']/np.max(s_data['num_rewards']))
p2 = axs.plot(s_data['idx'], s_data['num_licks']/np.max(s_data['num_licks']))
axs.legend([p1[0], p2[0]], ['rewards', 'licks'])

#%%
plt.close('all')

n_dset = 3
est1 = data_ca[n_dset]
     
S_sm = f_gauss_smooth(est1['S'], sigma_frames=6)
S_smn = S_sm/np.max(S_sm, axis=1)[:,None]

hclust_data = f_hclust_firing_rates(S_smn, standardize=True, metric='cosine', method='average')

if 0:
    plt.figure()
    plt.imshow(hclust_data['similarity_matrix'][hclust_data['res_order'],:][:,hclust_data['res_order']])
    plt.title(mouse_id + (' dset %d; ' % n_dset) + bh_data[n_dset]['dset_name'])
    
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
    
    plt.figure()
    plt.plot(est1['frame_times'], ~est1['vid_cuts_trace'])

ftag = bh_data[n_dset]['dset_name']
ftag2 = mouse_id + (' dset %d; ' % n_dset) + ftag

mov_data = f_proc_movement(bh_data[n_dset], frame_times = est1['frame_times'], do_interp=1, interp_step = 0.1, plot_stuff = False, title_tag = ftag2)

if 0:
    plt.figure()
    plt.plot(mov_data['time'][1:], mov_data['d_dist'])
    plt.plot(mov_data['time'][1:], mov_data['d_phi'])
    plt.title(ftag2)
    plt.xlabel('time')
    plt.legend(['distance','yaw'])

    plt.figure()
    plt.plot(mov_data['d_dist'], mov_data['d_phi'], '.')
    plt.xlabel('distance')
    plt.ylabel('yaw')
    plt.title(ftag2)
    
    plt.figure()
    plt.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_dist'])
    plt.plot(est1['frame_times'], mov_data['d_dist_frames'])
    
    
# lick reward data
lr_data = f_proc_lick_rew(bh_data[n_dset], mov_data, frame_times = est1['frame_times'], plot_stuff = False, title_tag = ftag2)

# find reward events
lr_data_idx = f_proc_lick_rew_df(bh_data[n_dset], plot_stuff = True, title_tag = ftag2)


#%%

if 0:
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, gridspec_kw={'height_ratios': [6, 1, 1]})
    ax1.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax1.set_title(ftag2)
    ax1.set_ylabel('CS sorted neurons')
    ax2.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_dist']+1)
    ax2.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_phi']*2)
    ax2.set_xlabel('time (sec)')
    ax2.legend(['distance', 'rotation'])
    ax3.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['lick_trace']), color=lick_col2)
    ax3.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['rew_trace']), color=rew_col2)
    ax3.set_xlabel('time (sec)')
    ax3.legend(['lick', 'reward'], loc='lower right')
else:
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax1.set_title(ftag2)
    ax1.set_ylabel('CS sorted neurons')
    ax2.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_dist']+1)
    ax2.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_phi']*3)
    ax2.set_xlabel('time (sec)')
    ax2.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['lick_trace'])*0.8-2, color=lick_col2)
    ax2.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['rew_trace'])*0.8-3, color=rew_col2)
    ax2.set_xlabel('time (sec)')
    ax2.legend(['distance', 'rotation', 'lick', 'reward'], loc='lower right')

if 0:
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax1.set_title(ftag2)
    ax1.set_ylabel('CS sorted neurons')
    ax2.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_dist'])
    ax2.plot(mov_data['time'][1:] - bh_data[n_dset]['bh_pulse_delay'], mov_data['d_phi'])
    ax2.set_xlabel('time (sec)')
    ax2.legend(['distance', 'rotation'])
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax1.set_title(ftag2)
    ax1.set_ylabel('CS sorted neurons')
    ax2.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['lick_trace']))
    ax2.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['rew_trace']))
    ax2.set_xlabel('time (sec)')
    ax2.legend(['lick', 'reward'])
    
    plt.figure()
    plt.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], lr_data['lick_trace'])

#%%
S_smn_shuff = f_circshift_rates(S_smn)
proj_data, exp_var, components, mean_all = f_run_dred(S_smn.T, subtr_mean=1, method=2)
proj_data_shuff, exp_var_shuff, components_shuff, mean_all_shuff = f_run_dred(S_smn_shuff.T, subtr_mean=1, method=2)

if 0:
    # plt.close('all')

    plt.figure()
    plt.plot(exp_var)
    plt.plot(exp_var_shuff)
    plt.legend(['data', 'shuff'])
    plt.title('Explained variance; dset %d' % n_dset)
    
    plt.figure()
    plt.plot(np.cumsum(exp_var))
    plt.plot(np.cumsum(exp_var_shuff))
    plt.legend(['data', 'shuff'])
    plt.title('Cumulative variance; dset %d' % n_dset)

    comp_pairs_plot = [[n*2, n*2+1] for n in range(3)]
    
    # plotting behavior on frames
    plot_bh = 1
    for comp_pair in comp_pairs_plot:
        plt.figure()
        plt.plot(proj_data[:,comp_pair[0]], proj_data[:,comp_pair[1]], color='k', linewidth= 1)
        if plot_bh:
            rew_frame_idx = lr_data['rew_frames'].astype(bool)
            lick_frame_idx = lr_data['lick_frames'].astype(bool)
            plt.plot(proj_data[lick_frame_idx,comp_pair[0]], proj_data[lick_frame_idx,comp_pair[1]], '.', color='pink')
            plt.plot(proj_data[rew_frame_idx,comp_pair[0]], proj_data[rew_frame_idx,comp_pair[1]], '.', color='lightgreen')
        plt.xlabel('comp %d' % (comp_pair[0]+1))
        plt.ylabel('comp %d' % (comp_pair[1]+1))
        plt.title('Neuronal activity; dset %d' % n_dset)
        plt.legend(['Acticity', 'Licks', 'Rewards'])
        
    # splitting movement and stationary frames
    idx_mov = mov_data['d_dist_frames'] > 0.01
    for comp_pair in comp_pairs_plot:
        plt.figure()
        plt.plot(proj_data[idx_mov,comp_pair[0]], proj_data[idx_mov,comp_pair[1]], linewidth= 1)
        plt.plot(proj_data[~idx_mov,comp_pair[0]], proj_data[~idx_mov,comp_pair[1]], linewidth= 1)
        plt.xlabel('comp %d' % (comp_pair[0]+1))
        plt.ylabel('comp %d' % (comp_pair[1]+1))
        plt.title('Neuronal activity; dset %d' % n_dset)
        plt.legend(['Locomoting', 'Stationary'])

    # splitting right and left rotations
    idx_rot = mov_data['d_phi_frames'] > 0
    for comp_pair in comp_pairs_plot:
        plt.figure()
        plt.plot(proj_data[idx_rot,comp_pair[0]], proj_data[idx_rot,comp_pair[1]], linewidth= 1)
        plt.plot(proj_data[~idx_rot,comp_pair[0]], proj_data[~idx_rot,comp_pair[1]], linewidth= 1)
        plt.xlabel('comp %d' % (comp_pair[0]+1))
        plt.ylabel('comp %d' % (comp_pair[1]+1))
    
    # plotting components, different groups of cells
    for comp_pair in comp_pairs_plot:
        plt.figure()
        plt.plot(components[:,comp_pair[0]], components[:,comp_pair[1]], '.')
        plt.xlabel('comp %d' % (comp_pair[0]+1))
        plt.ylabel('comp %d' % (comp_pair[1]+1))
    
    num_comp = 80
    data_rec = np.dot(proj_data[:,:num_comp], components[:,:num_comp].T).T + mean_all[:,None]
    plt.figure()
    plt.imshow(data_rec, aspect='auto', vmin=0, vmax=0.5)
    
#%% reward trial average plots

fr = 1/np.median(np.diff(est1['frame_times']))

rew_frames1 = np.where(lr_data['rew_frames'].astype(bool))[0]
time_win = [-60, 120]

trial_time = np.arange(time_win[0], time_win[1], 1)/fr

rew_frames2 = rew_frames1[(rew_frames1 > -time_win[0]) & ((rew_frames1 + time_win[1]) < len(lr_data['rew_frames']))]

(num_cells, num_t) = S_smn.shape
num_avr_t = np.sum(np.abs(time_win))
num_rew = len(rew_frames2)

S_smn_trials = np.zeros((num_cells, num_avr_t, num_rew))

for n_r in range(num_rew):
    idx1 = rew_frames2[n_r]
    S_smn_trials[:,:,n_r] = S_smn[:, idx1 + time_win[0] : idx1 + time_win[1]]
   
if 0:
    plt.figure()
    plt.imshow(np.mean(S_smn_trials, axis=2)[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.15, interpolation='none', extent=[float(trial_time[0]), float(trial_time[-1]), S_smn.shape[0],0])
    plt.title('trial ave reward response')
    plt.xlabel('time (sec)')
    
    plt.figure()
    plt.imshow(S_smn_trials[hclust_data['res_order'],:,0], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(trial_time[0]), float(trial_time[-1]), S_smn.shape[0],0])

#%%
S_smn_trials2d = S_smn_trials.reshape(num_cells, num_avr_t * num_rew, order='F')

proj_data_tr2d, exp_var_tr, components_tr, mean_all_tr = f_run_dred(S_smn_trials2d.T, subtr_mean=1, method=2)
proj_data_tr3d = proj_data_tr2d.T.reshape(num_cells, num_avr_t, num_rew, order='F')

if 0:
    plt.figure()
    plt.imshow(S_smn_trials2d[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none')
    plt.title('trial ave reward response')
    plt.xlabel('Frames')

    
    plt.close('all')
    comp_pairs_plot = [[n*2, n*2+1] for n in range(5)]
    
    # plotting behavior on frames
    for comp_pair in comp_pairs_plot:
        plt.figure()
        for n_tr in range(10):
            plt.plot(proj_data_tr3d[comp_pair[0],:,n_tr], proj_data_tr3d[comp_pair[1],:,n_tr], linewidth= 1)
            plt.plot(proj_data_tr3d[comp_pair[0],0,n_tr], proj_data_tr3d[comp_pair[1],0,n_tr], '.', color='k')
            plt.plot(proj_data_tr3d[comp_pair[0],-time_win[0],n_tr], proj_data_tr3d[comp_pair[1],-time_win[0],n_tr], '.', color=rew_col2)

        plt.xlabel('comp %d' % comp_pair[0])
        plt.ylabel('comp %d' % comp_pair[1])
        
    # centered at reward onset
    for comp_pair in comp_pairs_plot:
        plt.figure()
        for n_tr in range(10):
            plt.plot(proj_data_tr3d[comp_pair[0],:,n_tr] - proj_data_tr3d[comp_pair[0],-time_win[0],n_tr], proj_data_tr3d[comp_pair[1],:,n_tr] - proj_data_tr3d[comp_pair[1],-time_win[0],n_tr], linewidth= 1)
            plt.plot(proj_data_tr3d[comp_pair[0],0,n_tr] - proj_data_tr3d[comp_pair[0],-time_win[0],n_tr], proj_data_tr3d[comp_pair[1],0,n_tr] - proj_data_tr3d[comp_pair[1],-time_win[0],n_tr], '.', color='k')
        plt.plot(0, 0, '.', color=rew_col2)

        plt.xlabel('comp %d' % comp_pair[0])
        plt.ylabel('comp %d' % comp_pair[1])
     
    num_comp = 10
    data_rec = np.dot(proj_data_tr3d.reshape(num_cells, num_avr_t* num_rew, order='F').T[:,:num_comp], components_tr[:,:num_comp].T).T + mean_all[:,None]
    plt.figure()
    plt.imshow(data_rec[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5)
        
#%% analyze monitor movements

cam_params = {'aspect':             16/9,          # 1920/1080
              'FOV_axis':           'vertical',     # which axis is fixed
              'FOV_deg':            65.9,           # 80
              'cam_rotation_deg':   49.2,           # was 80/2
              'clip_len':           60,
              'num_mon':            2}

df_obj_data = bh_data[n_dset]['object_data']
#df_mov = bh_data[n_dset]['movement']
df_obj_events = bh_data[n_dset]['object_events']
#df_events = bh_data[n_dset]['events']
#df_terrain_data = bh_data[n_dset]['terrainData']

mov_data = f_proc_movement(bh_data[n_dset], do_interp=1, interp_step = 0.1, plot_stuff = False, title_tag = ftag2)

phi = mov_data['phi']
theta = mov_data['theta']

f_comp_FOV_adj(cam_params)

mouse_xyz = np.array((mov_data['x_pos'], mov_data['y_pos'], mov_data['z_pos'])).T
obj_locs = np.vstack((df_obj_data.ObjLocX.to_numpy(), df_obj_data.ObjLocY.to_numpy(), df_obj_data.ObjLocZ.to_numpy())).T

mon_r_phi = f_add_phase(phi, cam_params['cam_rotation_rad']*1.0)
mon_l_phi = f_add_phase(phi, -cam_params['cam_rotation_rad']*1.0)
if 1:
    vec_data_r = f_get_monitor_coords(mov_data, mon_r_phi, theta, bh_data[n_dset], cam_params, remove_outside_objects = True)
    vec_data_l = f_get_monitor_coords(mov_data, mon_l_phi, theta, bh_data[n_dset], cam_params, remove_outside_objects = True)
    
    fig, ax1 = plt.subplots()
    ax1.plot(mouse_xyz[:,0], mouse_xyz[:,2])
    ax1.plot(mouse_xyz[0,0], mouse_xyz[0,2], 'ko')
    ax1.plot(df_obj_data.ObjLocX, df_obj_data.ObjLocZ, '.')
    #plt.plot(df_obj_events.ObjLocX, df_obj_events.ObjLocZ, 'o')
    #plt.plot(df_mov['x_pos'][lick_mot_idx], df_mov['z_pos'][lick_mot_idx], 'ro')
    ax1.plot(mouse_xyz[:,0][lr_data['rew_trace'].astype(bool)], mouse_xyz[:,2][lr_data['rew_trace'].astype(bool)], 'go')
    ax1.set_title(ftag2)
    for n_ev in range(len(df_obj_events)):
        obev = df_obj_events.iloc[n_ev]
        
        ax1.text(obev.ObjLocX, obev.ObjLocZ, 'r%d' % n_ev)
        

    for n_pt in np.arange(0,5000, 5000):
        ax1.plot(mouse_xyz[n_pt,0], mouse_xyz[n_pt,2], 'o', color='lightgreen')
        if cam_params['num_mon'] == 1:
            # mouse
            f_plot_monitor_outline(mouse_xyz[:,n_pt], phi[n_pt], theta[n_pt], cam_params, axis=ax1, color_cent = 'gray', color_edge = 'blue')

        elif cam_params['num_mon'] == 2:
            # mouse
            f_plot_monitor_outline(mouse_xyz[n_pt,:], mon_l_phi[n_pt], theta[n_pt], cam_params, axis=ax1, color_cent = 'gray', color_edge = 'darkviolet')
            f_plot_monitor_outline(mouse_xyz[n_pt,:], mon_r_phi[n_pt], theta[n_pt], cam_params, axis=ax1, color_cent = 'gray', color_edge = 'blue')
            
            n_fr = 0
            obj_used = vec_data_r['obj_used']
            for n_obj in range(len(obj_used)):
                n_obj2 = obj_used[n_obj]
                if vec_data_r['obj_mon_idx'][n_fr,:][n_obj]:
                    obj_coords = obj_locs[n_obj2,:]
                    ax1.plot(obj_coords[0], obj_coords[2], '.', color='magenta')
            obj_used = vec_data_l['obj_used']
            for n_obj in range(len(obj_used)):
                n_obj2 = obj_used[n_obj]
                if vec_data_l['obj_mon_idx'][n_fr,:][n_obj]:
                    obj_coords = obj_locs[n_obj2,:]
                    ax1.plot(obj_coords[0], obj_coords[2], '.', color='red')

    if 0:
        plt.figure()
        plt.plot(np.sum(vec_data_r['obj_mon_idx'], axis=0))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    f_plot_lateral_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax1)
    f_plot_vertical_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], cam_params, axis=ax2)
    f_plot_dist_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax3)
    ax1.set_title('Right monitor')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    f_plot_lateral_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax1)
    f_plot_vertical_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], cam_params, axis=ax2)
    f_plot_dist_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax3)
    ax1.set_title('Left monitor')
 
    if 1:
        num_samp = 101
        left_mon_frames = f_angles_to_movie(vec_data_l, mov_data['time'], cam_params, obj_size, lat_samp = num_samp, vert_samp = num_samp)
        right_mon_frames = f_angles_to_movie(vec_data_r, mov_data['time'], cam_params, obj_size, lat_samp = num_samp, vert_samp = num_samp)
        
        two_mon_frames = np.concatenate((left_mon_frames, right_mon_frames), axis = 2)
        
        if 0:
            tf.imwrite(os.path.join('F:/test_mov', 'two_mon_' + est1['dset_name'] + '.tif'), two_mon_frames.astype(np.uint8))   

if 0:
    fig, ax = plt.subplots(1,1)
    ax.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax.set_title(ftag2)
    ax.set_ylabel('CS sorted neurons')
    ax.set_xlabel('time (sec)')
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, gridspec_kw={'height_ratios': [6, 1, 1,1]})
    ax1.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax1.set_title(ftag2)
    ax1.set_ylabel('CS sorted neurons')
    f_plot_lateral_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax2)
    #f_plot_vertical_ov_time(mouse_xyz, mon_l_phi, theta, obj_locs, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], cam_params, axis=ax2)
    f_plot_dist_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax3)
    ax4.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['lick_trace']), color=lick_col2)
    ax4.plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['rew_trace']), color=rew_col2)
    ax4.set_xlabel('time (sec)')
    ax4.legend(['lick', 'reward'], loc='lower right')
    
    # righ and left look flipped, transitions from right edge of right to left edge of left
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax1.imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax1.set_title(ftag2)
    ax1.set_ylabel('CS sorted neurons')
    f_plot_lateral_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax2)
    ax2.set_ylabel('Left mon')
    f_plot_lateral_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax3)
    ax3.set_ylabel('Right mon')
    
    # righ and left look flipped, transitions from right edge of right to left edge of left
    fig, ax = plt.subplots(6,1, sharex=True, gridspec_kw={'height_ratios': [7, 1, 1, 1, 1, 1]})
    ax[0].imshow(S_smn[hclust_data['res_order'],:], aspect='auto', vmin=0, vmax=0.5, interpolation='none', extent=[float(est1['frame_times'][0]), float(est1['frame_times'][-1]), S_smn.shape[0],0]) 
    ax[0].set_title(ftag2)
    ax[0].set_ylabel('CS sorted neurons')
    f_plot_lateral_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[1], xlabel=False)
    ax[1].set_ylabel('Left mon')
    f_plot_dist_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[2], xlabel=False)
    f_plot_lateral_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[3], xlabel=False)
    ax[3].set_ylabel('Right mon')
    f_plot_dist_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[4], xlabel=False)
    ax[5].plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['lick_trace']), color=lick_col2)
    ax[5].plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['rew_trace']), color=rew_col2)
    ax[5].set_xlabel('time (sec)')
    ax[5].legend(['lick', 'reward'], loc='lower right')
    
    
    fig, ax = plt.subplots(7,1, sharex=True)
    f_plot_lateral_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[0], xlabel=False)
    ax[0].set_title('Left monitor')
    f_plot_vertical_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], cam_params, axis=ax[1], xlabel=False)
    f_plot_dist_over_time(vec_data_l, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[2], xlabel=False)
    f_plot_lateral_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[3], xlabel=False)
    ax[3].set_title('Right monitor')
    f_plot_vertical_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], cam_params, axis=ax[4], xlabel=False)
    f_plot_dist_over_time(vec_data_r, mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], axis=ax[5], xlabel=False)
    ax[6].plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['lick_trace']), color=lick_col2)
    ax[6].plot(mov_data['time'] - bh_data[n_dset]['bh_pulse_delay'], np.sign(lr_data['rew_trace']), color=rew_col2)
    ax[6].set_xlabel('time (sec)')
    ax[6].legend(['lick', 'reward'], loc='lower right')
    
    
    






