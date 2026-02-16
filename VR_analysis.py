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
import scipy as sc
import matplotlib.pyplot as plt
import json

from f_functions import f_load_fnames, f_get_pd_value, f_plot_session, f_plot_session2

#%%

#dpath = path1 + '/UNITY/data_out_real/L_bh'

m_tag0 = "mice1"
#m_tag0 = "mice_gcamp"
#tag1 = 'L'
tag1 = 'RL'

m_tag = m_tag0 + ' ' + tag1

dpath = 'F:/VR/Bh_data/' + m_tag0 + '/'
dpath2 = dpath + tag1 +'/'

flist, ftimes = f_load_fnames(dpath2)
params_xls = pd.read_excel(dpath + tag1 + '_data.xlsx')

#df_save = pd.DataFrame(flist)
#df_save.to_csv(dpath + '//../RL_data.csv', index=False)

df_mov_all = []
df_obj_data_all = []
df_obj_events_all = []
df_events_all = []
params_all = []

for ftag in flist:
    fpath_movement = dpath2 + ftag + 'MovementData.csv'# 'tracking_2025_3_13_11h_46m_0s.csv'
    fpath_object_data = dpath2 + ftag + 'ObjectData.csv'
    fpath_obj_events = dpath2 + ftag + 'ObjectEvents.csv'
    fpath_terrain = dpath2 + ftag + 'Terrain4Data.csv'
    fpath_events = dpath2 + ftag + 'Events.csv'
    fpath_daq = dpath2 + ftag + 'DAQ.csv'
    
    params = {}
    fpath_params = dpath2 + ftag + 'params.json'
    idx3 = params_xls['dateTags'] == ftag
    if os.path.exists(fpath_params):
        with open(fpath_params, 'r') as f:
            params_load = json.load(f)
            params['zoneDist'] = params_load['objectRewardZoneDistance']
            if params['zoneDist'] > 100:
                params['zoneDist'] = params['zoneDist']**(1/2)
            params['zonal'] = int(params_load['rewardZonesOn'])
            params['lickReward'] = int(params_load['lickRewardOn'])
            params['minRewardDist'] = params_load['minRewardDistace']
    else:
        params['zoneDist'] = f_get_pd_value(params_xls['zoneDist'][idx3])
        params['zonal'] = int(f_get_pd_value(params_xls['zonal'][idx3]))
        params['lickReward'] = int(f_get_pd_value(params_xls['lickReward'][idx3]))
        params['minRewardDist'] = f_get_pd_value(params_xls['minDist'][idx3])
    
    if sum(idx3):
        params['fixedWheel'] = f_get_pd_value(params_xls['fixedWheel'][idx3])
    else:
        params['fixedWheel'] = False
    
    params_all.append(params)
    df_mov_all.append(pd.read_csv(fpath_movement))
    df_obj_data_all.append(pd.read_csv(fpath_object_data))
    df_obj_events_all.append(pd.read_csv(fpath_obj_events))
    df_events_all.append(pd.read_csv(fpath_events))

#%%
session_lr = np.zeros(len(df_mov_all), dtype=bool)
session_lr_dist = np.zeros(len(df_mov_all), dtype=float)
session_zonal = np.zeros(len(df_mov_all), dtype=bool)
session_zone_size = np.zeros(len(df_mov_all), dtype=float)
session_fixed_wheel = np.zeros(len(df_mov_all), dtype=bool)
for n_fl in  range(len(params_all)):
    par1 = params_all[n_fl]
    session_lr[n_fl] = bool(par1['lickReward'])
    session_lr_dist[n_fl] = par1['minRewardDist']
    session_zonal[n_fl] = bool(par1['zonal'])
    session_zone_size[n_fl] = par1['zoneDist']
    session_fixed_wheel[n_fl] = bool(par1['fixedWheel'])

sessions_id = np.arange(len(params_all))+1

if 0:
    plt.figure()
    plt.plot(session_lr)
    plt.plot(session_lr_dist)
    plt.title(tag1)
    
    plt.figure()
    plt.plot(session_zonal)
    plt.plot(session_zone_size)
    plt.title(tag1)
    
    plt.figure()
    plt.plot(session_fixed_wheel)


f_plot_session(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel)

f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, y_size=1)

#%%
# plt.close('all')

init_pos_xyzphi = []
num_rewards = np.zeros(len(df_mov_all))
num_licks = np.zeros(len(df_mov_all))
total_dist = np.zeros(len(df_mov_all))
total_duration = np.zeros(len(df_mov_all))
peak_reward_rate = np.zeros(len(df_mov_all))

idx_start = 1
ip_step = .1
do_interp = 1

for n_fl in range(len(df_mov_all)):
    df_mov = df_mov_all[n_fl]
    df_events = df_events_all[n_fl]
    df_obj_data = df_obj_data_all[n_fl]
    df_obj_events = df_obj_events_all[n_fl]
    ftag = flist[n_fl]
    
    ftag2 = tag1 + ftag
    #mouse_pos_xyz = (pd.concat([df_mov.x_pos, df_mov.y_pos, df_mov.z_pos], axis=1))
    mouse_pos_xyz = (pd.concat([df_mov.x_pos, df_mov.y_pos, df_mov.z_pos], axis=1))[idx_start:].reset_index(drop=True)     # df.x_pos.iloc[n_pt]
    mouse_pos_xyz1 = mouse_pos_xyz.values
    
    time1 = df_mov.Time[idx_start:].values
    
    # direction mouse is facing
    phi = df_mov.y_rot_eu[idx_start:].values/360*2*np.pi   # direction mouse is facing
    theta = df_mov.x_rot_eu[idx_start:].values/360*2*np.pi+np.pi/2  
    
    if do_interp:
        time2 = np.arange(time1[0], time1[-1], ip_step)
        x_pos2 = np.interp(time2, time1, mouse_pos_xyz1[:,0])
        y_pos2 = np.interp(time2, time1, mouse_pos_xyz1[:,1])
        z_pos2 = np.interp(time2, time1, mouse_pos_xyz1[:,2])
        sin_phi2 = np.interp(time2, time1, np.sin(phi))
        cos_phi2 = np.interp(time2, time1, np.cos(phi))

    else:
        time2 = time1
        x_pos2 = mouse_pos_xyz1[:,0]
        y_pos2 = mouse_pos_xyz1[:,1]
        z_pos2 = mouse_pos_xyz1[:,2]
        sin_phi2 = np.sin(phi)
        cos_phi2 = np.cos(phi)
    
    dtime = time2[1:]
    
    phi2 = np.atan2(sin_phi2, cos_phi2)
    dphi21 = np.diff(phi2)
    dphi2 = np.atan2(np.sin(dphi21), np.cos(dphi21))
    
    dpos1 = np.diff([x_pos2, y_pos2, z_pos2], axis=1).T
    ddist = np.sum(dpos1**2,axis=1)**(1/2)
    
    if 0:
        plt.figure()
        plt.plot(dtime, ddist)
        plt.plot(dtime, dphi2)
        plt.title(ftag2)
        plt.xlabel('time')
        plt.legend(['distance','yaw'])
    if 0:
        plt.figure()
        plt.plot(ddist, dphi2, '.')
        plt.xlabel('distance')
        plt.ylabel('yaw')
        plt.title(ftag2)
    
    if 0:
        plt.figure()
        plt.hist(np.log(ddist+1e-5)) # 
        
        plt.figure()
        plt.hist(dphi2, bins=50)
    
    if 0:
        plt.figure()
        plt.plot(time1, phi)

        plt.figure()
        plt.plot(time2, x_pos2)
        plt.plot(time1, mouse_pos_xyz.x_pos.values)
    
    init_pos_xyzphi.append([mouse_pos_xyz.x_pos[0], mouse_pos_xyz.y_pos[0], mouse_pos_xyz.z_pos[0], phi[0]])
    
    licks = df_events[df_events.event == "Lick"].reset_index(drop=True) 
    objects = df_events[df_events.event == "Object"].reset_index(drop=True) 
    rewards = df_events[df_events.event == "Reward"].reset_index(drop=True)
     
    num_rewards[n_fl] = len(rewards)
    num_licks[n_fl] = len(licks)
    total_dist[n_fl] = df_mov['totalDist'].values[-1]
    total_duration[n_fl] = df_mov['Time'].values[-1]
    
    if 1:
        rew_trace = np.zeros(len(time2))
        for n_r in range(len(rewards)):
            rew_trace[np.floor(rewards.Time[n_r]).astype(int)] += 1
        
        lick_trace = np.zeros(len(time2))
        for n_r in range(len(licks)):
            lick_trace[np.floor(licks.Time[n_r]).astype(int)] += 1
        
        if 0:
            rew_trace_sm = sc.ndimage.gaussian_filter(rew_trace, sigma = 10/ip_step)
            lick_trace_sm = sc.ndimage.gaussian_filter(lick_trace, sigma = 10/ip_step)
            
            fig, ax1 = plt.subplots()  # figsize=(8, 8)
            ax2 = ax1.twinx()
            ax1.plot(time2, lick_trace_sm, 'tab:blue')
            ax2.plot(time2, rew_trace_sm, 'tab:orange');
            ax1.set_ylabel("reward rate", color='tab:blue')
            ax1.tick_params(axis="y", labelcolor='tab:blue')
            ax2.set_ylabel("lick rate", color='tab:orange')
            ax2.tick_params(axis="y", labelcolor='tab:orange')
            plt.title(ftag2)
        
        if 0:
            rew_cum = np.cumsum(rew_trace)
            lick_cum = np.cumsum(lick_trace)
            
            lr_ratio = (lick_cum + 1e-5)/(rew_cum + 1e-5)
            
            plt.figure()
            plt.plot(time2, lr_ratio)
            
            plt.figure()
            plt.plot(rew_cum/rew_cum[-1])
            plt.plot(lick_cum/lick_cum[-1])
        
        time_cum = np.cumsum(np.ones(len(time2))*ip_step)
        rew_cum = np.cumsum(rew_trace)
        
        rew_rate = rew_cum/time_cum
        
        peak_reward_rate[n_fl] = np.max(rew_rate)
        
        if 1:
            plt.figure()
            plt.plot(time2, rew_rate)
            plt.ylabel("rewards per sec")
            plt.xlabel("time (sec)")
            plt.title(ftag2)
            
    rew_mot_idx = []
    for n_r in range(len(rewards)):
        time1 = rewards['Time'][n_r]
        rew_mot_idx.append(np.argmin((df_mov['Time'] - time1)**2))
    rew_mot_idx = np.array(rew_mot_idx)
    
    lick_mot_idx = []
    for n_l in range(len(licks)):
        time1 = licks['Time'][n_l]
        lick_mot_idx.append(np.argmin((df_mov['Time'] - time1)**2))
    lick_mot_idx = np.array(lick_mot_idx)
    
    if 0:
        plt.figure()
        plt.plot(mouse_pos_xyz.x_pos, mouse_pos_xyz.z_pos)
        plt.plot(mouse_pos_xyz.x_pos[0], mouse_pos_xyz.z_pos[0], 'ko')
        plt.plot(df_obj_data.ObjLocX, df_obj_data.ObjLocZ, '.')
        #plt.plot(df_obj_events.ObjLocX, df_obj_events.ObjLocZ, 'o')
        #plt.plot(df_mov['x_pos'][lick_mot_idx], df_mov['z_pos'][lick_mot_idx], 'ro')
        plt.plot(df_mov['x_pos'][rew_mot_idx], df_mov['z_pos'][rew_mot_idx], 'go')
        plt.title(ftag2)
    
    if 0:
        time_win = [-10, 20]
        fig, ax = plt.subplots()
        for n_r in range(len(rewards)):
            time1 = rewards['Time'][n_r]
            licks_idx = np.logical_and(licks['Time']>(time1+time_win[0]), licks['Time']<(time1+time_win[1]))
            trial_licks = licks['Time'][licks_idx].reset_index(drop=True)  - time1
            
            for n_l in range(len(trial_licks)):
                ax.plot(np.array([trial_licks[n_l], trial_licks[n_l]]), np.array([0, 0.5])+n_r+1, 'r')
            ax.plot(np.array([0, 0]), np.array([0, 0.5])+n_r+1, 'g')
        ax.invert_yaxis()
        ax.set_ylabel('Trials')
        ax.set_xlabel('Time (sec)')
        ax.set_title(ftag2)
    
#%%
# plt.close('all')
ax = f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, 1)
ax.set_title(m_tag)
ax.set_xlabel('Session')

ax = f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, np.max(num_rewards)*1.2)
ax.plot(sessions_id, num_rewards, 'black')
ax.set_xlabel('Session')
ax.set_ylabel('Rewards per session')
ax.set_title(m_tag)

ax = f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, np.max(num_licks)*1.2)
ax.plot(sessions_id, num_licks, 'black')
ax.set_xlabel('Session')
ax.set_ylabel('Licks per session')
ax.set_title(m_tag)

ax = f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, np.max(total_dist)*1.2)
ax.plot(sessions_id, total_dist, 'black')
ax.set_xlabel('Session')
ax.set_ylabel('Dist traveled per session')
ax.set_title(m_tag)


ax = f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, np.max(total_duration)*1.2)
ax.plot(sessions_id, total_duration, 'black')
ax.set_xlabel('Session')
ax.set_ylabel('Session duration')
ax.set_title(m_tag)


ax = f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, np.max(peak_reward_rate)*1.2)
ax.plot(sessions_id, peak_reward_rate, 'black')
ax.set_xlabel('Session')
ax.set_ylabel('Peak reward rate')
ax.set_title(m_tag)

if 0:
    plt.figure()
    plt.plot(num_rewards)
    plt.ylabel('num rewards')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(num_licks)
    plt.ylabel('num licks')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(total_dist)
    plt.ylabel('total distance')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(total_duration)
    plt.ylabel('session duration')
    plt.title(tag1)
    
    plt.figure()
    plt.plot(peak_reward_rate)
    plt.ylabel('peak reward rate')
    plt.xlabel('session')
    plt.title(tag1)

    
#%%
plt.figure()
plt.plot(licks.Time, np.ones(len(licks.Time)), '.')
plt.plot(rewards.Time, np.ones(len(rewards.Time)), '.')
plt.plot(objects.Time, np.ones(len(objects.Time)), '.')

#%%

