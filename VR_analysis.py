# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:26:21 2025

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
import matplotlib.pyplot as plt

from f_functions import f_load_fnames, f_load_bh_data, f_get_session_data, f_plot_session2, f_proc_movement, f_proc_lick_rew, f_proc_lick_rew_df, f_spheric_to_cart, f_cart_to_spheric_np, f_comp_FOV_adj, f_plot_monitor_outline, f_plot_lateral_over_time, f_plot_vertical_ov_time, f_plot_dist_ov_time # f_plot_session

#%%

#dpath = path1 + '/UNITY/data_out_real/L_bh'

#m_tag0 = "mice1"
m_tag0 = "mice_gcamp"
#tag1 = 'L'
tag1 = 'L'

m_tag = m_tag0 + ' ' + tag1

dpath = 'F:/VR/Bh_data/' + m_tag0 + '/'
dpath2 = dpath + tag1 +'/'

flist, ftimes = f_load_fnames(dpath2)
params_xls = pd.read_excel(dpath + tag1 + '_params.xlsx')

#df_save = pd.DataFrame(flist)
#df_save.to_csv(dpath + '//../RL_data.csv', index=False)


bh_data = []
                                                
for ftag in flist:
    
    bh_data_slice = f_load_bh_data(dpath2, ftag, params_xls)
    bh_data.append(bh_data_slice)

#%%

s_data = f_get_session_data(bh_data)

if 0:
    plt.figure()
    plt.plot(s_data['lick_reward'])
    plt.plot(s_data['lick_reward_dist'])
    plt.title(tag1)
    
    plt.figure()
    plt.plot(s_data['zonal'])
    plt.plot(s_data['zone_size'])
    plt.title(tag1)
    
    plt.figure()
    plt.plot(s_data['fixed_wheel'])


#f_plot_session(s_data)

f_plot_session2(s_data, y_size=1)

#%%
plt.close('all')
num_dsets = len(bh_data)

peak_reward_rate = np.zeros(num_dsets)
#for n_dset in range(num_dsets):

n_dset = 70    

ftag = bh_data[n_dset]['dset_name']
ftag2 = tag1 + ftag

mov_data = f_proc_movement(bh_data[n_dset], do_interp=1, interp_step = 0.1, plot_stuff = False, title_tag = tag1 + ftag)

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


#licks = df_events[df_events.event == "Lick"].reset_index(drop=True) 
#objects = df_events[df_events.event == "Object"].reset_index(drop=True) 
#rewards = df_events[df_events.event == "Reward"].reset_index(drop=True)

# lick reward data
lr_data = f_proc_lick_rew(bh_data[n_dset], mov_data, plot_stuff = False, title_tag = ftag2)
peak_reward_rate[n_dset] = lr_data['peak_reward_rate']

# find reward events
lr_data_idx = f_proc_lick_rew_df(bh_data[n_dset], plot_stuff = True, title_tag = ftag2)

    
#%%
# plt.close('all')
if 0:
    ax = f_plot_session2(s_data, 1)
    ax.set_title(m_tag)
    ax.set_xlabel('Session')
    
    ax = f_plot_session2(s_data, np.max(s_data['num_rewards'])*1.2)
    ax.plot(s_data['idx'], s_data['num_rewards'], 'black')
    ax.set_xlabel('Session')
    ax.set_ylabel('Rewards per session')
    ax.set_title(m_tag)
    
    ax = f_plot_session2(s_data, np.max(s_data['num_licks'])*1.2)
    ax.plot(s_data['idx'], s_data['num_licks'], 'black')
    ax.set_xlabel('Session')
    ax.set_ylabel('Licks per session')
    ax.set_title(m_tag)
    
    ax = f_plot_session2(s_data, np.max(s_data['total_dist'])*1.2)
    ax.plot(s_data['idx'], s_data['total_dist'], 'black')
    ax.set_xlabel('Session')
    ax.set_ylabel('Dist traveled per session')
    ax.set_title(m_tag)
    
    
    ax = f_plot_session2(s_data, np.max(s_data['total_duration'])*1.2)
    ax.plot(s_data['idx'], s_data['total_duration'], 'black')
    ax.set_xlabel('Session')
    ax.set_ylabel('Session duration')
    ax.set_title(m_tag)
    
    
    ax = f_plot_session2(s_data, np.max(peak_reward_rate)*1.2)
    ax.plot(s_data['idx'], peak_reward_rate, 'black')
    ax.set_xlabel('Session')
    ax.set_ylabel('Peak reward rate')
    ax.set_title(m_tag)

if 0:
    plt.figure()
    plt.plot(s_data['num_rewards'])
    plt.ylabel('num rewards')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(s_data['num_licks'])
    plt.ylabel('num licks')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(s_data['total_dist'])
    plt.ylabel('total distance')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(s_data['total_duration'])
    plt.ylabel('session duration')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(peak_reward_rate)
    plt.ylabel('peak reward rate')
    plt.xlabel('session')
    plt.title(tag1)

    
#%%
if 0:
    n_fl = 50
    df_events = bh_data[n_fl]['events']
    
    licks = df_events[df_events.event == "Lick"].reset_index(drop=True) 
    objects = df_events[df_events.event == "Object"].reset_index(drop=True) 
    rewards = df_events[df_events.event == "Reward"].reset_index(drop=True)
    
    plt.figure()
    plt.plot(licks.Time, np.ones(len(licks.Time)), '.')
    plt.plot(rewards.Time, np.ones(len(rewards.Time)), '.')
    plt.plot(objects.Time, np.ones(len(objects.Time)), '.')

#%% analyze vid feed

cam_params = {'aspect':     16/9,
              'clip_len':   50,
              'FOV_deg':    80,
              'num_mon':    2}

df_obj_data = bh_data[n_dset]['object_data']
df_mov = bh_data[n_dset]['movement']

mov_data = f_proc_movement(bh_data[n_dset], do_interp=1, interp_step = 0.1, plot_stuff = False, title_tag = tag1 + ftag)


phi = mov_data['phi']
theta = mov_data['theta']

FOV_rad_adj, h_adj = f_comp_FOV_adj(cam_params)

mouse_xyz = np.array((mov_data['x_pos'], mov_data['y_pos'], mov_data['z_pos'])).T
obj_locs = np.vstack((df_obj_data.ObjLocX.to_numpy(), df_obj_data.ObjLocY.to_numpy(), df_obj_data.ObjLocZ.to_numpy())).T

mon_r_phi = phi+FOV_rad_adj/2
mon_l_phi = phi-FOV_rad_adj/2

if 0:
    
    fig, ax1 = plt.subplots()
    ax1.plot(mouse_xyz[:,0], mouse_xyz[:,2])
    ax1.plot(mouse_xyz[0,0], mouse_xyz[0,2], 'ko')
    ax1.plot(df_obj_data.ObjLocX, df_obj_data.ObjLocZ, '.')
    #plt.plot(df_obj_events.ObjLocX, df_obj_events.ObjLocZ, 'o')
    #plt.plot(df_mov['x_pos'][lick_mot_idx], df_mov['z_pos'][lick_mot_idx], 'ro')
    ax1.plot(mouse_xyz[:,0][lr_data['rew_trace'].astype(bool)], mouse_xyz[:,2][lr_data['rew_trace'].astype(bool)], 'go')
    ax1.set_title(ftag2)

    for n_pt in np.arange(0,50, 50):
        ax1.plot(mouse_xyz[n_pt,0], mouse_xyz[n_pt,2], 'o', color='lightgreen')
        if cam_params['num_mon'] == 1:
            # mouse
            f_plot_monitor_outline(mouse_xyz[:,n_pt], phi, theta, n_pt, 0, cam_params, axis=ax1, color_cent = 'gray', color_edge = 'blue')

        elif cam_params['num_mon'] == 2:
            # mouse
            f_plot_monitor_outline(mouse_xyz[:,n_pt], phi, theta, n_pt, -FOV_rad_adj/2, cam_params, axis=ax1, color_cent = 'gray', color_edge = 'green')
            f_plot_monitor_outline(mouse_xyz[:,n_pt], phi, theta, n_pt, FOV_rad_adj/2, cam_params, axis=ax1, color_cent = 'gray', color_edge = 'blue')


    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    f_plot_lateral_over_time(mouse_xyz, mon_l_phi, theta, obj_locs, mov_data['time'], cam_params, axis=ax1)
    f_plot_vertical_ov_time(mouse_xyz, mon_l_phi, theta, obj_locs, mov_data['time'], cam_params, axis=ax2)
    f_plot_dist_ov_time(mouse_xyz, mon_l_phi, theta, obj_locs, mov_data['time'], cam_params, axis=ax3)
    
        
    
    if 0:
        if cam_params['num_mon'] == 1:
            # version with 1 monitor
            rot_vec = f_spheric_to_cart(phi, theta)
            rot_vec_r = f_spheric_to_cart(phi+FOV_rad_adj/2, theta)
            rot_vec_l = f_spheric_to_cart(phi-FOV_rad_adj/2, theta)

        elif cam_params['num_mon'] == 2:
            # version with 2 monitors
            rot_vec = f_spheric_to_cart(phi, theta)
            mon_l_dir = f_spheric_to_cart(phi-FOV_rad_adj/2, theta)
            mon_l_edge_l = f_spheric_to_cart(phi-FOV_rad_adj, theta)
            mon_r_dir = f_spheric_to_cart(phi+FOV_rad_adj/2, theta)
            mon_r_edge_r = f_spheric_to_cart(phi+FOV_rad_adj, theta)
        
        n_pt = 0
        spher_vec_objs = f_cart_to_spheric_np(obj_locs - mouse_xyz[n_pt,:])
        spher_vec_objs[:,2] = spher_vec_objs[:,2]%(2*np.pi)
        
        obj_dist_ord = np.argsort(np.sqrt(np.sum((obj_locs - mouse_xyz[n_pt,:])**2, axis=1)))
        obj_dist_ord = np.argsort(spher_vec_objs[:,0])
        # plotting points within fov
        for n_pt3 in range(len(obj_dist_ord)):
            if np.abs((mon_l_phi[n_pt] - spher_vec_objs[n_pt3,2])) < FOV_rad_adj/2:
                if spher_vec_objs[n_pt3,0] < cam_params['clip_len']:
                    ax1.plot(obj_locs[n_pt3,x_pt], obj_locs[n_pt3,z_pt], '.', color='magenta')
        
        n_pt3 = 11
        plt.figure()
        for n_pt3 in range(12):
            plt.plot(mon_l_phi[:1000] - spher_vec_objs[n_pt3,2], theta[:1000] - spher_vec_objs[n_pt3,1])
        plt.xlim([-FOV_rad_adj/2, FOV_rad_adj/2])
        plt.ylim([-FOV_rad_adj/2/cam_params['aspect'], FOV_rad_adj/2/cam_params['aspect']])
        
        
        # plot how objects translate through horizontal fov over time
        num_t = 1000
        num_obj = 15
        plt.figure()
        for n_pt3 in range(num_obj):
            mon_l_phi[:num_t] - spher_vec_objs[n_pt3,2]
            
            spher_vec_objs[n_pt3,0]
            
            plt.plot(mov_data['time'][:num_t], mon_l_phi[:num_t] - spher_vec_objs[n_pt3,2])
        plt.ylim([-FOV_rad_adj/2, FOV_rad_adj/2])
        
        # testing
        for n_pt in range(500):
            plt.plot(mon_l_phi[n_pt] - spher_vec_objs[n_pt3,2], theta[n_pt] - spher_vec_objs[n_pt3,1], '.', color='red')
        plt.xlim([-FOV_rad_adj/2, FOV_rad_adj/2])
        plt.ylim([-FOV_rad_adj/2/cam_params['aspect'], FOV_rad_adj/2/cam_params['aspect']])

if 0:
    plt.figure()
    plt.plot(mov_data['phi'])
    
    plt.figure()
    plt.plot(np.sin(mov_data['phi']))
    
    plt.figure()
    plt.plot(rot_vec[:,0])
    
    

