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

from f_functions import f_load_fnames, f_load_bh_data, f_get_session_data, f_plot_session2, f_proc_movement, f_proc_lick_rew, f_proc_lick_rew_df # f_plot_session

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

df_mov_all = []
df_obj_data_all = []
df_obj_events_all = []
df_events_all = []
params_all = []
                                                
for ftag in flist:
    
    bh_data_slice = f_load_bh_data(dpath2, ftag, params_xls)
    bh_data.append(bh_data_slice)
    
    params_all.append(bh_data_slice['params'])
    df_mov_all.append(bh_data_slice['movement'])
    df_obj_data_all.append(bh_data_slice['object_data'])
    df_obj_events_all.append(bh_data_slice['object_events'])
    df_events_all.append(bh_data_slice['events'])

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
# plt.close('all')
num_dsets = len(bh_data)

peak_reward_rate = np.zeros(num_dsets)
for n_dset in range(num_dsets):
    
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
    lr_data = f_proc_lick_rew(bh_data[n_dset], mov_data, plot_stuff = True, title_tag = ftag2)
    peak_reward_rate[n_dset] = lr_data['peak_reward_rate']
    
    # find reward events
    lr_data_idx = f_proc_lick_rew_df(bh_data[n_dset], plot_stuff = True, title_tag = ftag2)

    
#%%
# plt.close('all')
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

#%%

