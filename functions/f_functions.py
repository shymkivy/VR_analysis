# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:47:27 2025

@author: ys2605
"""
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import json
import scipy as sc

import matplotlib.pyplot as plt
import matplotlib.patches as ptc

#%%
def f_load_fnames(dir_path):
    f_list = os.listdir(dir_path)
    f_list2 = []
    f_times2 = []
    for fil1 in f_list:
        idx1 = fil1.find('s_')
        parts1 = fil1.split('_')
        tag1 = fil1[:idx1+1]
        if tag1 not in f_list2:
            time1 = np.datetime64('%d-%02d-%02dT%02d:%02d' % (int(parts1[0]), int(parts1[1]), int(parts1[2]), int(parts1[3][:-1]), int(parts1[4][:-1])), 'm')
            f_list2.append(tag1)
            f_times2.append(time1)
    
    f_list3 = np.array(f_list2)
    f_times3 = np.array(f_times2)
    
    idx2 = np.argsort(f_times3)
    
    f_list4 = f_list3[idx2]
    f_times4 = f_times3[idx2]
    
    return f_list4, f_times4

def f_get_pd_value(val_in):
    val1 = val_in.values[0]
    if np.isnan(val1):
        val_out = 0
    else:
        val_out = val1
    return val_out


def f_load_bh_data(data_dir, ftag, params_xls):
    fpath_movement = data_dir + ftag + '_MovementData.csv'# 'tracking_2025_3_13_11h_46m_0s.csv'
    fpath_object_data = data_dir + ftag + '_ObjectData.csv'
    fpath_obj_events = data_dir + ftag + '_ObjectEvents.csv'
    #fpath_terrain = dpath2 + ftag + '_Terrain4Data.csv'
    fpath_events = data_dir + ftag + '_Events.csv'
    #fpath_daq = dpath2 + ftag + '_DAQ.csv'
    
    params = {}
    fpath_params = data_dir + ftag + '_params.json'
    idx3 = (params_xls['dateTags'] == ftag) | (params_xls['dateTags'] == ftag + '_')
    if os.path.exists(fpath_params):
        with open(fpath_params, 'r') as f:
            params_load = json.load(f)
            params['zoneDist'] = params_load['objectRewardZoneDistance']
            if params['zoneDist'] > 100:
                params['zoneDist'] = params['zoneDist']**(1/2)
            params['zonal'] = int(params_load['rewardZonesOn'])
            params['lickReward'] = int(params_load['lickRewardOn'])
            params['minRewardDist'] = params_load['minRewardDistace']
    elif sum(idx3):
        if not np.isnan(params_xls['zonal'][idx3].to_numpy())[0]:
            params['zonal'] = int(f_get_pd_value(params_xls['zonal'][idx3]))
        else:
            params['zonal'] = 0
        if not np.isnan(params_xls['zoneDist'][idx3].to_numpy())[0]:
            params['zoneDist'] = f_get_pd_value(params_xls['zoneDist'][idx3])
        else:
            params['zoneDist'] = 0
        if not np.isnan(params_xls['lickReward'][idx3].to_numpy())[0]:
            params['lickReward'] = int(f_get_pd_value(params_xls['lickReward'][idx3]))
        else:
            params['lickReward'] = 0
        if not np.isnan(params_xls['minDist'][idx3].to_numpy())[0]:
            params['minRewardDist'] = f_get_pd_value(params_xls['minDist'][idx3])
        else:
            params['minRewardDist'] = 0
    else:
        print('missing params for file %s' % ftag)
    
    if sum(idx3):
        params['fixedWheel'] = f_get_pd_value(params_xls['fixedWheel'][idx3])
    else:
        params['fixedWheel'] = False
    
    data_out = {'dset_name':        ftag,
                'params':           params,
                'movement':         pd.read_csv(fpath_movement),
                'object_data':      pd.read_csv(fpath_object_data),
                'object_events':    pd.read_csv(fpath_obj_events),
                'events':           pd.read_csv(fpath_events)}
    return data_out

#%%
def f_plot_volt(data_in, time):
    cum_d = data_in - data_in[data_in.index[0]]
    cum_ds = sc.ndimage.gaussian_filter1d(cum_d, 10)
    #diff_ds = np.diff(cum_ds, prepend=0)

    plt.figure()
    plt.plot(time, cum_d/np.std(cum_d))
    plt.plot(time, cum_ds/np.std(cum_d))
    #plt.plot(df2.Time, cumvel_s/np.std(cumvel_s))
    plt.title('DAQ distance data')
    plt.legend(['dist volt', 'dist volt sm'])


#%%
def if_spheric_to_cart(phi, theta, rho=1):
    # 
    # spherical: (phi, theta, rho) - (lateral, up/down, mag)
    # cartesian: (z, x, y) - (forward/back, side to side, up/down)  
    
    z = np.cos(phi)*np.sin(theta)*rho       # normally x
    x = np.sin(phi)*np.sin(theta)*rho       # normally y
    y = np.cos(theta)*rho                   # normally z

    xyz_vec = (pd.concat([x, y, z], axis=1)) # np.array
    xyz_vec.columns = ['x', 'y', 'z']
    return xyz_vec

def if_cart_to_spheric(x, y, z):
    # spherical: (rho, theta, phi) - (mag, up/down, lateral)
    # cartesian: (z, x, y) - (forward/back, side to side, up/down) 
    
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.atan2(x, z)                  # normally y/x
    theta = np.acos(y/rho)              # normally z/rho
    
    #spher_vec = np.vstack([rho, theta, phi])
    spher_vec = pd.concat([rho, theta, phi], axis=1)
    spher_vec.columns = ['rho', 'theta', 'phi']
    
    return spher_vec

#%%
def if_parse_vert(vertEl):
    
    temp_vert = {'index':    [],
                'x':        [],
                'y':        [],
                'z':        []}
    
    for child2 in vertEl:
        temp_vert['index'].append(child2.attrib['index'])
        temp_vert['x'].append(float(child2.find('x').text))
        temp_vert['y'].append(float(child2.find('y').text))
        temp_vert['z'].append(float(child2.find('z').text))
        
    return temp_vert

#%%
def load_XML(path):
    
    tree = ET.parse(path)
    td = tree.getroot()

    #tags_all = []
    #for child1 in td:
    #    tags_all.append(child1.tag)

    # get locations of all vertices of mesh
    meshVert = if_parse_vert(td.find('meshVertList'))

    # get locations of objects and their types
    objList = {'index':    [],
                'x':        [],
                'y':        [],
                'z':        [],
                'type':     []}

    objListEl = td.find('objList')
    for child1 in objListEl:
        objList['index'].append(child1.attrib['index'])
        objList['x'].append(float(child1.find('x').text))
        objList['y'].append(float(child1.find('y').text))
        objList['z'].append(float(child1.find('z').text))
        objList['type'].append(float(child1.find('type').text))


    # get vertices of each object
    objTypeList = {'index':    [],
                   'vert':     []}


    objTypeListEl = td.find('objTypeList')
    for child1 in objTypeListEl:
        objTypeList['index'].append(child1.attrib['index'])
        temp_vert = if_parse_vert(child1.find('vertList'))
        objTypeList['vert'].append(temp_vert)
    
    xml_data = {'meshVert':     meshVert,
                'objList':      objList,
                'objTypeList':  objTypeList}
    
    return xml_data


#%%
def f_get_session_data(bh_data):
    num_dsets = len(bh_data)

    session_lr = np.zeros(num_dsets, dtype=bool)
    session_lr_dist = np.zeros(num_dsets, dtype=float)
    session_zonal = np.zeros(num_dsets, dtype=bool)
    session_zone_size = np.zeros(num_dsets, dtype=float)
    session_fixed_wheel = np.zeros(num_dsets, dtype=bool)
    
    num_dsets = len(bh_data)

    num_rewards = np.zeros(num_dsets, dtype=int)
    num_licks = np.zeros(num_dsets, dtype=int)
    total_dist = np.zeros(num_dsets, dtype=float)
    total_duration = np.zeros(num_dsets, dtype=float)

    for n_fl in  range(num_dsets):
        par1 = bh_data[n_fl]['params']
        
        session_lr[n_fl] = bool(par1['lickReward'])
        session_lr_dist[n_fl] = par1['minRewardDist']
        session_zonal[n_fl] = bool(par1['zonal'])
        session_zone_size[n_fl] = par1['zoneDist']
        session_fixed_wheel[n_fl] = bool(par1['fixedWheel'])
        
        df_events = bh_data[n_fl]['events']
        
        licks = df_events[df_events.event == "Lick"].reset_index(drop=True) 
        rewards = df_events[df_events.event == "Reward"].reset_index(drop=True)
        num_rewards[n_fl] = len(rewards)
        num_licks[n_fl] = len(licks)
        
        df_mov = bh_data[n_fl]['movement']
        
        total_dist[n_fl] = df_mov['totalDist'].values[-1]
        total_duration[n_fl] = df_mov['Time'].values[-1]

    sessions_id = np.arange(num_dsets)+1
    
    session_data = {'lick_reward':          session_lr,
                    'lick_reward_dist':     session_lr_dist,
                    'zonal':                session_zonal,
                    'zone_size':            session_zone_size,
                    'fixed_wheel':          session_fixed_wheel,
                    'idx':                  sessions_id,
                    'num_rewards':          num_rewards,
                    'num_licks':            num_licks,
                    'total_dist':           total_dist,
                    'total_duration':       total_duration}
    
    return session_data


def f_plot_session(session_data):
    
    session_lr = session_data['lick_reward']
    session_lr_dist = session_data['lick_reward_dist']
    session_zonal = session_data['zonal']
    session_zone_size = session_data['zone_size']
    session_fixed_wheel = session_data['fixed_wheel']
    
    sessions_id = np.arange(len(session_lr))+1
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if sum(session_lr):
        ax1.plot(sessions_id[session_lr], session_lr_dist[session_lr]/np.max(session_lr_dist[session_lr]), 'tab:orange')
        rect_lr = ptc.Rectangle([sessions_id[np.where(session_lr)[0][0]],0], np.sum(session_lr)-1, 1, facecolor='tab:orange', alpha=0.1) # , color='tab:orange'
        ax1.add_patch(rect_lr)
    if sum(session_zonal):
        ax1.plot(sessions_id[session_zonal], session_zone_size[session_zonal]/np.max(session_zone_size[session_zonal]), 'tab:green')
        rect_zone = ptc.Rectangle([sessions_id[np.where(session_zonal)[0][0]],0], np.sum(session_zonal)-1, 1, facecolor='tab:green', alpha=0.1) # , color='tab:orange'
        ax1.add_patch(rect_zone)   
    if sum(session_fixed_wheel):
        rect_fix = ptc.Rectangle([sessions_id[np.where(session_fixed_wheel)[0][0]],0], np.sum(session_fixed_wheel)-1, 1, facecolor='tab:blue', alpha=0.1) # , color='tab:orange'
        ax1.add_patch(rect_fix)
    ax1.yaxis.tick_right()
    ax2.yaxis.tick_left()
    
    return ax2

def f_plot_session2(session_data, y_size=1):
    
    session_lr = session_data['lick_reward']
    session_lr_dist = session_data['lick_reward_dist']
    session_zonal = session_data['zonal']
    session_zone_size = session_data['zone_size']
    session_fixed_wheel = session_data['fixed_wheel']
    
    sessions_id = np.arange(len(session_lr))+1
    
    fig, ax1 = plt.subplots()
    
    if sum(session_lr):
        norm_dist = session_lr_dist[session_lr]/np.max(session_lr_dist[session_lr])*y_size
        ax1.plot(sessions_id[session_lr], norm_dist, 'tab:orange', alpha=0.5)
        start_idx = np.where(session_lr)[0][0]
        end_idx = np.where(session_lr)[0][-1]
        lr_start = sessions_id[start_idx]
        rect_lr = ptc.Rectangle([lr_start, 0-y_size*0.1], np.sum(session_lr)-1, y_size*1.2, facecolor='tab:orange', alpha=0.1) # , color='tab:orange'
        ax1.add_patch(rect_lr)
        ax1.text(lr_start, 0-y_size*0.07, 'Lick-Reward', fontsize=12, color='tab:orange')
        ax1.text(lr_start, norm_dist[0]+0.02*y_size, str(int(session_lr_dist[0])), fontsize=12, color='tab:orange')
        ax1.text(sessions_id[end_idx]-1, norm_dist[-1]+0.02*y_size, str(int(session_lr_dist[end_idx])), fontsize=12, color='tab:orange')
        
    if sum(session_zonal):
        norm_dist = session_zone_size[session_zonal]/np.max(session_zone_size[session_zonal])*y_size
        ax1.plot(sessions_id[session_zonal], norm_dist, 'tab:green', alpha=0.5)
        start_idx = np.where(session_zonal)[0][0]
        end_idx = np.where(session_zonal)[0][-1]
        zone_start = sessions_id[start_idx]
        rect_zone = ptc.Rectangle([zone_start,0-y_size*0.1], np.sum(session_zonal)-1, y_size*1.2, facecolor='tab:green', alpha=0.1) # , color='tab:orange'
        ax1.add_patch(rect_zone)   
        ax1.text(zone_start, 0-y_size*0.07, 'Object recognition', fontsize=12, color='tab:green')
        ax1.text(zone_start, norm_dist[0]+y_size*0.02, str(int(session_zone_size[start_idx])), fontsize=12, color='tab:green')
        ax1.text(sessions_id[end_idx]-1, norm_dist[-1]+y_size*0.02, str(int(session_zone_size[end_idx])), fontsize=12, color='tab:green')
        
    if sum(session_fixed_wheel):
        fix_start = sessions_id[np.where(session_fixed_wheel)[0][0]]
        rect_fix = ptc.Rectangle([fix_start,0-y_size*0.1], np.sum(session_fixed_wheel)-1, y_size*1.2, facecolor='tab:blue', alpha=0.1) # , color='tab:orange'    
        ax1.add_patch(rect_fix)
        ax1.text(fix_start, 0, 'Fixed linear motion', fontsize=12, color='tab:blue')
    
    return ax1

#%%

def f_proc_movement(bh_data_slice, frame_times = None, do_interp = 1, interp_step = 0.1, idx_start = 1, plot_stuff = False, title_tag = ''):
    
    df_mov = bh_data_slice['movement']
    
    mouse_pos_xyz = (pd.concat([df_mov.x_pos, df_mov.y_pos, df_mov.z_pos], axis=1))[idx_start:].reset_index(drop=True)     # df.x_pos.iloc[n_pt]
    mouse_pos_xyz1 = mouse_pos_xyz.values
    
    time1 = df_mov.Time[idx_start:].values
    
    # direction mouse is facing
    phi = df_mov.y_rot_eu[idx_start:].values/360*2*np.pi   # direction mouse is facing
    #theta = df_mov.x_rot_eu[idx_start:].values/360*2*np.pi+np.pi/2
    
    if do_interp:
        time2 = np.arange(time1[0], time1[-1], interp_step)
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
    
    mov_data = {'time':         time2,
                'x_pos':        x_pos2,
                'y_pos':        y_pos2,
                'z_pos':        z_pos2,
                'd_phi':        dphi2,
                'd_dist':       ddist,
                'do_interp':    do_interp,
                'interp_step':  interp_step,
                }
    
    
    if frame_times is not None:
        
        time2_corr = time2 - bh_data_slice['bh_pulse_delay']
        
        ddist_frames = np.interp(frame_times, time2_corr[1:], ddist)
        dphi_frames = np.interp(frame_times, time2_corr[1:], dphi2)
        x_pos_frames = np.interp(frame_times, time2_corr, x_pos2)
        y_pos_frames = np.interp(frame_times, time2_corr, y_pos2)
        z_pos_frames = np.interp(frame_times, time2_corr, z_pos2)

        mov_data['d_dist_frames'] = ddist_frames
        mov_data['d_phi_frames'] = dphi_frames
        mov_data['x_pos_frames'] = x_pos_frames
        mov_data['y_pos_frames'] = y_pos_frames
        mov_data['z_pos_frames'] = z_pos_frames
    
    if plot_stuff:

        plt.figure()
        plt.plot(dtime, ddist)
        plt.plot(dtime, dphi2)
        plt.title(title_tag)
        plt.xlabel('time')
        plt.legend(['distance','yaw'])

        plt.figure()
        plt.plot(ddist, dphi2, '.')
        plt.xlabel('distance')
        plt.ylabel('yaw')
        plt.title(title_tag)
    
        plt.figure()
        plt.hist(np.log(ddist+1e-5)) # 
        
        plt.figure()
        plt.hist(dphi2, bins=50)
    
        plt.figure()
        plt.plot(time1, phi)

        plt.figure()
        plt.plot(time2, x_pos2)
        plt.plot(time1, mouse_pos_xyz.x_pos.values)
        
    return mov_data

def f_proc_lick_rew(bh_data_slice, mov_data, frame_times = None, plot_stuff = False, title_tag = ''):
    df_events = bh_data_slice['events']
    licks = df_events[df_events.event == "Lick"].reset_index(drop=True) 
    rewards = df_events[df_events.event == "Reward"].reset_index(drop=True)
    
    rew_trace = np.zeros(len(mov_data['time']))
    for n_r in range(len(rewards)):
        idx1 = np.where(rewards.Time[n_r] < mov_data['time'])[0][0]
        #idx1 = np.argmin((mov_data['time'] - rewards.Time[n_r])**2)
        rew_trace[idx1] += 1
    
    lick_trace = np.zeros(len(mov_data['time']))
    for n_r in range(len(licks)):
        idx1 = np.where(licks.Time[n_r] < mov_data['time'])[0][0]
        lick_trace[idx1] += 1
    
    rew_trace_sm = sc.ndimage.gaussian_filter(rew_trace, sigma = 10/mov_data['interp_step'])
    lick_trace_sm = sc.ndimage.gaussian_filter(lick_trace, sigma = 10/mov_data['interp_step'])
    
    rew_cum = np.cumsum(rew_trace)
    lick_cum = np.cumsum(lick_trace)
    time_cum = np.cumsum(np.ones(len(mov_data['time']))*mov_data['interp_step'])
    
    lr_ratio = (lick_cum + 1e-5)/(rew_cum + 1e-5)
    
    if frame_times is not None:
        rew_frames = np.zeros(len(frame_times))
        
        for n_r in range(len(rewards)):
            idx1 = (rewards.Time[n_r] - bh_data_slice['bh_pulse_delay']) < frame_times
            if sum(idx1):
                loc1 = np.where(idx1)[0][0]
                #idx1 = np.argmin((mov_data['time'] - rewards.Time[n_r])**2)
                rew_frames[loc1] += 1
        
        lick_frames = np.zeros(len(frame_times))
        for n_r in range(len(licks)):
            idx1 = (licks.Time[n_r] - bh_data_slice['bh_pulse_delay']) < frame_times
            if sum(idx1):
                loc1 = np.where(idx1)[0][0]
                #idx1 = np.argmin((mov_data['time'] - rewards.Time[n_r])**2)
                lick_frames[loc1] += 1
    
    if plot_stuff:
        plt.figure()
        plt.plot(mov_data['time'], lick_trace)
        plt.plot(mov_data['time'], rew_trace)
        plt.legend(['licks', 'rewards'])
    

        fig, ax1 = plt.subplots()  # figsize=(8, 8)
        ax2 = ax1.twinx()
        ax1.plot(mov_data['time'], lick_trace_sm, 'tab:blue')
        ax2.plot(mov_data['time'], rew_trace_sm, 'tab:orange');
        ax1.set_ylabel("lick rate", color='tab:blue')
        ax1.tick_params(axis="y", labelcolor='tab:blue')
        ax2.set_ylabel("reward rate", color='tab:orange')
        ax2.tick_params(axis="y", labelcolor='tab:orange')
        plt.title(title_tag)
    
        plt.figure()
        plt.plot(mov_data['time'], lr_ratio)
        
        plt.figure()
        plt.plot(rew_cum/rew_cum[-1])
        plt.plot(lick_cum/lick_cum[-1])
    
    rew_rate = rew_cum/time_cum
    
    if plot_stuff:
        plt.figure()
        plt.plot(mov_data['time'], rew_rate)
        plt.ylabel("rewards per sec")
        plt.xlabel("time (sec)")
        plt.title(title_tag)
        
    lr_data = {'rew_trace':         rew_trace,
               'lick_trace':        lick_trace,
               'rew_trace_sm':      rew_trace_sm,
               'lick_trace_sm':     lick_trace_sm,
               'peak_reward_rate':  np.max(rew_rate),
               'rew_rate':          rew_rate,
               'cum_rew':           rew_cum,
               'cum_lick':          lick_cum,
               'cum_lr_ratio':      lr_ratio,
               }
    
    if frame_times is not None:
        lr_data['rew_frames'] = rew_frames
        lr_data['lick_frames'] = lick_frames
    
    return lr_data


def f_proc_lick_rew_df(bh_data_slice, plot_stuff = True, title_tag = ''):
    
    df_mov = bh_data_slice['movement']
    df_events = bh_data_slice['events']
    df_obj_data = bh_data_slice['object_data']
    
    licks = df_events[df_events.event == "Lick"].reset_index(drop=True) 
    #objects = df_events[df_events.event == "Object"].reset_index(drop=True) 
    rewards = df_events[df_events.event == "Reward"].reset_index(drop=True)
    
    # find reward events
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
    
    if plot_stuff:
        plt.figure()
        plt.plot(df_mov['x_pos'], df_mov['z_pos'])
        plt.plot(df_mov['x_pos'][0], df_mov['z_pos'][0], 'ko')
        plt.plot(df_obj_data.ObjLocX, df_obj_data.ObjLocZ, '.')
        #plt.plot(df_obj_events.ObjLocX, df_obj_events.ObjLocZ, 'o')
        #plt.plot(df_mov['x_pos'][lick_mot_idx], df_mov['z_pos'][lick_mot_idx], 'ro')
        plt.plot(df_mov['x_pos'][rew_mot_idx], df_mov['z_pos'][rew_mot_idx], 'go')
        plt.title(title_tag)
    
    if plot_stuff:
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
        ax.set_title(title_tag)
        
    lr_data = {'rew_idx':       rew_mot_idx,
               'lick_idx':      lick_mot_idx}
    
    return lr_data
    
    
    
    
    
    