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
import tifffile as tf
from scipy.spatial import cKDTree
from scipy.ndimage import zoom as nd_zoom

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
    fpath_terrain = data_dir + ftag + '_Terrain4Data.csv'
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
                'events':           pd.read_csv(fpath_events),
                'terrainData':      pd.read_csv(fpath_terrain)}
    return data_out


def f_load_bh_data_all(bh_dir, bh_dset_names, params_xls, data_ca=None):
    # Convenience wrapper that mirrors f_load_caim_data: loads behavior for a
    # list of datasets. If `data_ca` is provided, each behavior dataset is also
    # aligned to its matching imaging session — adds these extra keys to each
    # returned dict:
    #   align_slice           : rows of events where event == 'AlignmentPulse'
    #   bh_pulse_delay        : first-pulse offset (seconds). Subtract from
    #                           behavior time to bring it onto imaging time.
    #   max_pulse_diff_frames : diagnostic — max inter-pulse spacing on the
    #                           imaging-frame clock (NaN if <2 pulses).
    #   max_pulse_diff_vid    : diagnostic — same on the video/behavior clock.
    # If `data_ca` is None, only the bare f_load_bh_data output is returned per
    # dataset (no alignment, no diagnostics). Use this for behavior-only
    # analyses where there's no matching imaging data.
    # When provided, data_ca must be aligned 1:1 with bh_dset_names.

    bh_data = []
    for n_fl in range(len(bh_dset_names)):
        bh_data_slice = f_load_bh_data(bh_dir, bh_dset_names[n_fl], params_xls)

        if data_ca is not None:
            est1 = data_ca[n_fl]

            idx1 = bh_data_slice['events'].event == 'AlignmentPulse'
            align_slice = bh_data_slice['events'].iloc[idx1]
            bh_data_slice['align_slice'] = align_slice
            bh_data_slice['bh_pulse_delay'] = align_slice.Time.iloc[0] - est1['frame_times'][est1['vid_cuts'][0,1]]

            pulse_times = est1['frame_times'][est1['vid_cuts'][:-1,1]]
            diff_frames = np.diff(pulse_times)
            diff_vid    = np.diff(align_slice.Time)
            bh_data_slice['max_pulse_diff_frames'] = float(np.max(diff_frames)) if len(diff_frames) else np.nan
            bh_data_slice['max_pulse_diff_vid']    = float(np.max(diff_vid))    if len(diff_vid)    else np.nan

        bh_data.append(bh_data_slice)
    return bh_data


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
def f_spheric_to_cart(phi, theta, rho=1):
    # 
    # spherical: (phi, theta, rho) - (lateral, up/down, mag)
    # cartesian: (z, x, y) - (forward/back, side to side, up/down)  
    
    z = np.cos(phi)*np.sin(theta)*rho       # normally x
    x = np.sin(phi)*np.sin(theta)*rho       # normally y
    y = np.cos(theta)*rho                   # normally z
    
    if np.isscalar(x):
        xyz_vec = np.hstack((x, y, z))
    else:
        if isinstance(x, np.ndarray):
            xyz_vec = np.vstack((x, y, z)).T
        elif isinstance(x, pd.Series):
            xyz_vec = (pd.concat([x, y, z], axis=1)) # np.array
            xyz_vec.columns = ['x', 'y', 'z']
    return xyz_vec

def f_cart_to_spheric(x, y, z):
    # spherical: (rho, theta, phi) - (mag, up/down, lateral)
    # cartesian: (z, x, y) - (forward/back, side to side, up/down) 
    
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(x, z)                  # normally y/x
    theta = np.arccos(y/rho)              # normally z/rho
    
    if np.isscalar(phi):
        spher_vec = np.hstack((rho, theta, phi))[None,:]
    else:
        if isinstance(rho, np.ndarray):
            spher_vec = np.vstack([rho, theta, phi]).T
        elif isinstance(rho, pd.Series) or isinstance(rho, pd.DataFrame):
            spher_vec = pd.concat([rho, theta, phi], axis=1)
            spher_vec.columns = ['rho', 'theta', 'phi']
    
    return spher_vec

def f_cart_to_spheric_np(xyz):
    # spherical: (rho, theta, phi) - (mag, up/down, lateral)
    # cartesian: (z, x, y) - (forward/back, side to side, up/down) 
    
    if len(xyz.shape) > 1:
        rho = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
        phi = np.arctan2(xyz[:,0], xyz[:,2])                  # normally y/x
        theta = np.arccos(xyz[:,1]/rho)              # normally z/rho
    else:
        rho = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
        phi = np.arctan2(xyz[0], xyz[2])                  # normally y/x
        theta = np.arccos(xyz[1]/rho)              # normally z/rho
    
    spher_vec = np.vstack([rho, theta, phi]).T
    
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
        norm_dist = session_lr_dist/np.max(session_lr_dist[session_lr])*y_size
        chunk_diff = np.diff(session_lr.astype(int), prepend=0, append=0)
        chunk_on = np.where(chunk_diff > 0.5)[0]
        chunk_off = np.where(chunk_diff < -0.5)[0]
        for n_ch in range(len(chunk_on)):
            ax1.plot(sessions_id[chunk_on[n_ch]:chunk_off[n_ch]], norm_dist[chunk_on[n_ch]:chunk_off[n_ch]], 'tab:orange', alpha=0.5)
            rect_lr = ptc.Rectangle([sessions_id[chunk_on[n_ch]], 0-y_size*0.1], chunk_off[n_ch]-chunk_on[n_ch]-1, y_size*1.2, facecolor='tab:orange', alpha=0.1) # , color='tab:orange'
            ax1.add_patch(rect_lr)
        ax1.text(sessions_id[chunk_on[0]], 0-y_size*0.07, 'Lick-Reward', fontsize=12, color='tab:orange')
        ax1.text(sessions_id[chunk_on[0]], norm_dist[chunk_on[0]]+0.02*y_size, str(int(session_lr_dist[chunk_on[0]])), fontsize=12, color='tab:orange')
        ax1.text(sessions_id[chunk_off[0]]-3, norm_dist[chunk_off[0]-1]+0.02*y_size, str(int(session_lr_dist[chunk_off[0]-1])), fontsize=12, color='tab:orange')
        
    if sum(session_zonal):
        
        norm_dist = session_zone_size/np.max(session_zone_size[session_zonal])*y_size
        chunk_diff = np.diff(session_zonal.astype(int), prepend=0, append=0)
        chunk_on = np.where(chunk_diff > 0.5)[0]
        chunk_off = np.where(chunk_diff < -0.5)[0]
        for n_ch in range(len(chunk_on)):
            ax1.plot(sessions_id[chunk_on[n_ch]:chunk_off[n_ch]], norm_dist[chunk_on[n_ch]:chunk_off[n_ch]], 'tab:green', alpha=0.5)
            rect_zone = ptc.Rectangle([sessions_id[chunk_on[n_ch]],0-y_size*0.1], chunk_off[n_ch]-chunk_on[n_ch]-1, y_size*1.2, facecolor='tab:green', alpha=0.1) # , color='tab:orange'
            ax1.add_patch(rect_zone)   
        ax1.text(sessions_id[chunk_on[-1]], 0-y_size*0.07, 'Object recognition', fontsize=12, color='tab:green')
        ax1.text(sessions_id[chunk_on[-1]], norm_dist[chunk_on[-1]]+y_size*0.02, str(int(session_zone_size[chunk_on[-1]])), fontsize=12, color='tab:green')
        ax1.text(sessions_id[chunk_off[-1]-1]-1, norm_dist[chunk_off[-1]-1]+y_size*0.02, str(int(session_zone_size[chunk_off[-1]-1])), fontsize=12, color='tab:green')
        
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
    theta = df_mov.x_rot_eu[idx_start:].values/360*2*np.pi + np.pi/2  # not using theta, not sure how it will work otherwise.. I guess 0 is looking straight
    
    # Delay correction: shift the behavior clock to the imaging clock when the
    # pulse offset is available. Data values are unchanged; only the time axis is
    # relabeled. delay_corrected flags whether the shift was applied.
    pulse_delay = bh_data_slice.get('bh_pulse_delay', None)
    delay_corrected = pulse_delay is not None and not np.isnan(pulse_delay)
    pd_shift = float(pulse_delay) if delay_corrected else 0.0

    if frame_times is not None:
        # Sample at the imaging frames; time axis is the imaging clock.
        frame_times = np.asarray(frame_times).ravel()
        samp_t = frame_times + pd_shift
        x_pos2 = np.interp(samp_t, time1, mouse_pos_xyz1[:,0])
        y_pos2 = np.interp(samp_t, time1, mouse_pos_xyz1[:,1])
        z_pos2 = np.interp(samp_t, time1, mouse_pos_xyz1[:,2])
        sin_phi2 = np.interp(samp_t, time1, np.sin(phi - np.pi))
        cos_phi2 = np.interp(samp_t, time1, np.cos(phi - np.pi))
        sin_theta2 = np.interp(samp_t, time1, np.sin(theta))
        cos_theta2 = np.interp(samp_t, time1, np.cos(theta))
        time2 = frame_times
        interp_step_eff = float(np.median(np.diff(time2)))

    elif do_interp:
        samp_t = np.arange(time1[0], time1[-1], interp_step)
        x_pos2 = np.interp(samp_t, time1, mouse_pos_xyz1[:,0])
        y_pos2 = np.interp(samp_t, time1, mouse_pos_xyz1[:,1])
        z_pos2 = np.interp(samp_t, time1, mouse_pos_xyz1[:,2])
        sin_phi2 = np.interp(samp_t, time1, np.sin(phi - np.pi))
        cos_phi2 = np.interp(samp_t, time1, np.cos(phi - np.pi))
        sin_theta2 = np.interp(samp_t, time1, np.sin(theta))
        cos_theta2 = np.interp(samp_t, time1, np.cos(theta))
        time2 = samp_t - pd_shift
        interp_step_eff = interp_step

    else:
        x_pos2 = mouse_pos_xyz1[:,0]
        y_pos2 = mouse_pos_xyz1[:,1]
        z_pos2 = mouse_pos_xyz1[:,2]
        sin_phi2 = np.sin(phi - np.pi)
        cos_phi2 = np.cos(phi - np.pi)
        sin_theta2 = np.sin(theta)
        cos_theta2 = np.cos(theta)
        time2 = time1 - pd_shift
        interp_step_eff = interp_step

    dtime = time2[1:]
    
    phi2 = np.arctan2(sin_phi2, cos_phi2) + np.pi
    dphi21 = np.diff(phi2)
    dphi2 = np.arctan2(np.sin(dphi21), np.cos(dphi21))
    
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    
    dpos1 = np.diff([x_pos2, y_pos2, z_pos2], axis=1).T
    ddist = np.sum(dpos1**2,axis=1)**(1/2)
    
    mov_data = {'time':         time2,
                'x_pos':        x_pos2,
                'y_pos':        y_pos2,
                'z_pos':        z_pos2,
                'xyz':          np.array((x_pos2, y_pos2, z_pos2)).T,
                'phi':          phi2,
                'theta':        theta2,
                'd_phi':        dphi2,
                'd_dist':       ddist,
                'do_interp':    do_interp,
                'interp_step':  interp_step_eff,   # median dt on the imaging grid
                'on_imaging_clock': frame_times is not None,
                'delay_corrected':  delay_corrected,
                }
    
    
    if frame_times is not None:
        # Frame-length (T) velocity aliases for indexing frame-length data;
        # d_dist/d_phi are length T-1. time2 is the imaging clock (== frame_times).
        mov_data['d_dist_frames'] = np.interp(frame_times, time2[1:], ddist)
        mov_data['d_phi_frames']  = np.interp(frame_times, time2[1:], dphi2)

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
    
    # Event times are behavior clock; shift them onto mov_data['time'] when it
    # has been delay-corrected to the imaging clock.
    ev_shift = bh_data_slice['bh_pulse_delay'] if mov_data.get('delay_corrected') else 0.0

    rew_trace = np.zeros(len(mov_data['time']))
    for n_r in range(len(rewards)):
        _hits = np.where((rewards.Time[n_r] - ev_shift) < mov_data['time'])[0]
        idx1 = _hits[0] if _hits.size else len(mov_data['time']) - 1   # trailing event → last frame
        rew_trace[idx1] += 1

    lick_trace = np.zeros(len(mov_data['time']))
    for n_r in range(len(licks)):
        _hits = np.where((licks.Time[n_r] - ev_shift) < mov_data['time'])[0]
        idx1 = _hits[0] if _hits.size else len(mov_data['time']) - 1
        lick_trace[idx1] += 1
    
    dt_med = float(np.median(np.diff(mov_data['time'])))   # grid spacing (s)
    # 10 s Gaussian smoothing, sigma in samples
    rew_trace_sm = sc.ndimage.gaussian_filter(rew_trace, sigma = 10/dt_med)
    lick_trace_sm = sc.ndimage.gaussian_filter(lick_trace, sigma = 10/dt_med)

    rew_cum = np.cumsum(rew_trace)
    lick_cum = np.cumsum(lick_trace)
    # time_cum = np.cumsum(np.ones(len(mov_data['time']))*mov_data['interp_step'])
    time_cum = mov_data['time'] - mov_data['time'][0]      # elapsed time (s)
    time_cum[0] = dt_med                                   # avoid /0 in rew_rate at t=0
    
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
    
    
#%%

def f_comp_FOV_adj(cam_params):
    # unity fov formula
    # first either horizontal or vertical axis is chosen as reference, then from there and aspec ratio the other is computed
    #  h Fov = 2*arctan(tan(v Fov * 0.5) * aspectRatio)
    # AspectRatio = tan(hFov * 0.5) / tan(vFov * 0.5);  width / height
    
    FOV_rad = cam_params['FOV_deg']/360*2*np.pi
    
    if cam_params['FOV_axis'] == 'vertical':
        vFOV = FOV_rad
        #hFOV = 2 * np.atan(np.tan(vFOV/2) * cam_params['aspect'])
        hFOV = 2 * np.arctan2(np.sin(vFOV/2) * cam_params['aspect'], np.cos(vFOV/2))
    elif cam_params['FOV_axis'] == 'horizontal':
        hFOV = FOV_rad
        vFOV = 2 * np.arctan2(np.sin(hFOV/2) / cam_params['aspect'], np.cos(hFOV/2))
    
    cam_params['vFOV_rad'] = vFOV
    cam_params['hFOV_rad'] = hFOV
    cam_params['cam_rotation_rad'] = cam_params['cam_rotation_deg']/360*2*np.pi

    return cam_params

    # d = cam_params['clip_len']
    # h = d/np.cos(FOV_rad/2)         # fov height/2
    # w = h*np.sin(FOV_rad/2)         # fov width/2
    # h_adj = np.sqrt((w*cam_params['aspect'])**2 + d**2)
    # FOV_rad_adj = np.asin((w*cam_params['aspect'])/(h_adj))*2

    # return FOV_rad_adj, h_adj

def f_add_phase(phase1, phase2):
    
    phase_out = np.angle(np.exp(1j*(phase1 + phase2)))
    
    return phase_out

def f_get_monitor_coords(mov_data, mon_phi, mon_theta, bh_data_slice, cam_params, remove_outside_objects = True):
    
    df_obj_data = bh_data_slice['object_data']
    df_obj_events = bh_data_slice['object_events']
    df_terrain_data = bh_data_slice['terrainData']

    mouse_xyz = np.array((mov_data['x_pos'], mov_data['y_pos'], mov_data['z_pos'])).T
    obj_locs = np.vstack((df_obj_data.ObjLocX.to_numpy(), df_obj_data.ObjLocY.to_numpy(), df_obj_data.ObjLocZ.to_numpy())).T
    obj_ev_locs = np.vstack((df_obj_events.ObjLocX.to_numpy(), df_obj_events.ObjLocY.to_numpy(), df_obj_events.ObjLocZ.to_numpy())).T
    
    num_obj = obj_locs.shape[0]
    (num_t, num_d) = mouse_xyz.shape

    obj_mouse_vec = np.zeros((num_t, num_d, num_obj), dtype=float)
    obj_mon_idx = np.zeros((num_t, num_obj), dtype=bool)
    obj_dist_all = np.zeros((num_t, num_obj), dtype=float)
    obj_lat_angle_all = np.zeros((num_t, num_obj), dtype=float)
    obj_vert_angle_all = np.zeros((num_t, num_obj), dtype=float)
    num_rewards = 0
    for n_obj in range(num_obj):
        
        is_reward = np.sqrt(np.sum((obj_locs[n_obj,:] - obj_ev_locs)**2, axis=1)) == 0

        spher_vec_objs = f_cart_to_spheric_np(obj_locs[n_obj,:] - mouse_xyz)        # object locations vs mouse
        #spher_vec_objs[:,2] = spher_vec_objs[:,2]%(2*np.pi)
        obj_mouse_vec[:,:,n_obj] = spher_vec_objs
        
        lat_angle = f_add_phase(mon_phi, - spher_vec_objs[:,2])                 # lateral angle from monitor center
        vert_angle = f_add_phase(mon_theta, - spher_vec_objs[:,1])              # vertical angle from monitor center
        
        obj_dist_all[:,n_obj] = spher_vec_objs[:,0]
        obj_lat_angle_all[:,n_obj] = lat_angle
        obj_vert_angle_all[:,n_obj] = vert_angle
        
        idx_dist = spher_vec_objs[:,0] < cam_params['clip_len']                 # distance from mouse
        idx_lat = np.abs(lat_angle) < cam_params['hFOV_rad']/2           
        #idx_lat = np.abs(mon_phi - spher_vec_objs[:,2]) < FOV_rad_adj/2
        idx_vert = np.abs(vert_angle) < cam_params['vFOV_rad']/2
        #idx_vert = np.abs(mon_theta - spher_vec_objs[:,1]) < FOV_rad_adj/2/cam_params['aspect']
        if np.sum(is_reward):
            # reward_time is behavior clock; shift onto mov_data['time'] when
            # it has been delay-corrected to the imaging clock.
            ev_shift = bh_data_slice['bh_pulse_delay'] if mov_data.get('delay_corrected') else 0.0
            reward_time = df_obj_events[is_reward].Time.values[0]
            idx_not_rewarded = mov_data['time'] < (reward_time - ev_shift)
            num_rewards += 1
        else:
            idx_not_rewarded = np.ones((num_t), dtype=bool)
        
        in_fov_idx = idx_dist & idx_lat & idx_vert & idx_not_rewarded
        obj_mon_idx[:,n_obj] = in_fov_idx
    
    if remove_outside_objects:
        obj_idx = np.sum(obj_mon_idx, axis=0) > 0
        obj_used = np.where(obj_idx)[0]
        obj_mouse_vec = obj_mouse_vec[:,:,obj_idx]
        obj_mon_idx = obj_mon_idx[:,obj_idx]
        obj_dist_all = obj_dist_all[:,obj_idx]
        obj_lat_angle_all = obj_lat_angle_all[:,obj_idx]
        obj_vert_angle_all = obj_vert_angle_all[:,obj_idx]
    else:
        obj_used = np.arange(num_obj)
    
    ovj_vec_data = {'obj_mouse_vec':    obj_mouse_vec,        # 
                    'obj_mon_idx':      obj_mon_idx,
                    'obj_used':         obj_used,
                    'obj_dist':         obj_dist_all,
                    'obj_lat_angle':    obj_lat_angle_all,
                    'obj_vert_angle':   obj_vert_angle_all,
                    'num_rewards':      num_rewards}
    return ovj_vec_data
    

def f_angles_to_movie(vec_data, time, cam_params, obj_size, lat_samp = 51, vert_samp = 51):
    # Legacy 4-edge OUTLINE renderer (rectangle border + center pixel).
    # Kept for visual debugging — for production rendering use
    # f_angles_to_movie_v2 (filled silhouettes, antialiased, depth-composited).
    #
    # 2026-05-20: corrected to match v2's conventions:
    #  - output shape (T, vert_samp, lat_samp) — rows = vert, cols = lat
    #    (was (T, lat_samp, vert_samp), only worked when lat==vert).
    #  - vert_angle NEGATED in pixel mapping → default ImageJ shows sky-up
    #    (positive vert_angle → small pix → top of display).
    #  - obj_size['y'] is the FULL vertical extent (half-extent above
    #    center = obj_size['y']/2).
    #  - clip_underground always on: bottom of mask sits at ground level
    #    (= obj_size['height'] below center), not at -y/2.
    #  - Edge naming matches display + world: edge_top = small pix =
    #    world up = top of object geometry (y/2 above center);
    #    edge_bottom = large pix = world down = ground (height below
    #    center).

    num_frames = time.shape[0]
    mon_frames = np.zeros((num_frames, vert_samp, lat_samp), dtype=np.uint8)

    for n_obj in range(vec_data['obj_used'].shape[0]):

        idx_mon_obj = vec_data['obj_mon_idx'][:, n_obj]
        obj_dist = vec_data['obj_dist'][:, n_obj]

        # Angular half-extents (pixels). obj_size['y'] is the FULL vertical
        # extent of the geometry, so the half-extent above center is y/2.
        # obj_size['height'] is the center-above-ground offset, used to clip
        # the silhouette at the terrain surface.
        obj_x_ang_pix = np.round(np.arctan2(obj_size['x'],       obj_dist) / cam_params['hFOV_rad'] * (lat_samp - 1))
        obj_y_ang_pix = np.round(np.arctan2(obj_size['y'] / 2.0, obj_dist) / cam_params['vFOV_rad'] * (vert_samp - 1))
        obj_h_ang_pix = np.round(np.arctan2(obj_size['height'],  obj_dist) / cam_params['vFOV_rad'] * (vert_samp - 1))

        # Pixel mapping NEGATED for both lat and vert so default ImageJ shows
        # natural sky-up (positive vert_angle = above camera → SMALL pix).
        lat_angle_idx  = np.round(((-vec_data['obj_lat_angle'][:, n_obj]  + cam_params['hFOV_rad']/2) / cam_params['hFOV_rad']) * (lat_samp  - 1)).astype(int)
        vert_angle_idx = np.round(((-vec_data['obj_vert_angle'][:, n_obj] + cam_params['vFOV_rad']/2) / cam_params['vFOV_rad']) * (vert_samp - 1)).astype(int)

        for n_fr in range(num_frames):
            if not idx_mon_obj[n_fr]:
                continue
            edge_left   = int(lat_angle_idx[n_fr]  - obj_x_ang_pix[n_fr])
            edge_right  = int(lat_angle_idx[n_fr]  + obj_x_ang_pix[n_fr] + 1)
            # Smaller pix = top of object in world (y/2 above center);
            # larger pix = ground (height below center).
            edge_top    = int(vert_angle_idx[n_fr] - obj_y_ang_pix[n_fr])
            edge_bottom = int(vert_angle_idx[n_fr] + obj_h_ang_pix[n_fr])

            # center pixel
            mon_frames[n_fr, vert_angle_idx[n_fr], lat_angle_idx[n_fr]] = 255

            # horizontal edges (top + bottom)
            if 0 <= edge_bottom < vert_samp:
                mon_frames[n_fr, edge_bottom, max(edge_left, 0):min(edge_right + 1, lat_samp)] = 255
            if 0 <= edge_top < vert_samp:
                mon_frames[n_fr, edge_top,    max(edge_left, 0):min(edge_right + 1, lat_samp)] = 255
            # vertical edges (left + right)
            if 0 <= edge_left < lat_samp:
                mon_frames[n_fr, max(edge_top, 0):min(edge_bottom + 1, vert_samp), edge_left]  = 255
            if 0 <= edge_right < lat_samp:
                mon_frames[n_fr, max(edge_top, 0):min(edge_bottom + 1, vert_samp), edge_right] = 255

    return mon_frames


def f_angles_to_movie_v2(vec_data, time, cam_params, obj_size, lat_samp = 51, vert_samp = 51, filled = True, antialias = True, chunk_t = None, return_per_object = False, total_n_obj = None, clip_underground = True, return_depth = False, max_frames = None, verbose = True):
    # v2 step 1: vertical extent of each object.
    # obj_size['y'] is the object's FULL vertical extent (so half-extent above
    # center = obj_size['y']/2), and obj_size['height'] is the offset of the
    # center above the ground. The visible portion runs from ground
    # (= center - height) up to the top of the geometry (= center + y/2);
    # the rest of the cylinder/cube sits below ground and should not render.
    # obj_size['x'] / obj_size['z'] remain LATERAL HALF-EXTENTS (radius for
    # cylinders, half-side for cubes) — convention mismatch with 'y' is
    # legacy but harmless because the renderer applies them as half-widths.
    #
    # Pixel mapping convention (this function and f_render_terrain): both
    # lat and vert pixel formulas NEGATE the angle, so default image viewers
    # (row 0 at top of screen) show natural sky-up — positive vert_angle
    # (object above camera) lands at SMALL pix = TOP of display. Same for
    # lateral (positive = object to the right of monitor → right column).
    #
    # Under this convention, the silhouette mask is:
    #   edge_top    (smaller pix, world-up direction)
    #   edge_bottom (larger pix, world-down direction)
    #
    #   clip_underground=True  (default):
    #       edge_top    = vert_angle_idx - arctan2(obj_size['y']/2,    dist)
    #       edge_bottom = vert_angle_idx + arctan2(obj_size['height'], dist)
    #       Bottom of the silhouette sits at world-y = ground.
    #   clip_underground=False:
    #       edge_top    = vert_angle_idx - arctan2(obj_size['y']/2, dist)
    #       edge_bottom = vert_angle_idx + arctan2(obj_size['y']/2, dist)
    #       Symmetric ±y/2 rendering — the full object mesh, including any
    #       portion below ground. Useful only for debugging.
    #
    # Convention history (legacy v1 f_angles_to_movie was also brought up
    # to date on 2026-05-20 — same conventions as v2 now):
    #  - 2026-05-19: discovered the h/y values were applied to the wrong
    #    sides of the silhouette. Initial fix swapped them (made the math
    #    correct in world coords but the display was still upside-down).
    #  - 2026-05-19: switched obj_size['y'] convention from "half-extent
    #    above center" to "FULL vertical extent" → added /2 in the formula.
    #  - 2026-05-20: added the vert_angle negation in the pixel mapping
    #    and reverted the h/y swap, since with the negated mapping the
    #    original v1 edge-naming (top=small pix, bottom=large pix) matches
    #    both world up/down AND display top/bottom under default origin.
    #
    # The horizontal half-extent is obj_size['x'] in both modes.
    #
    # v2 step 2: filled silhouettes instead of a 4-edge wireframe (default).
    # Massive sparsity reduction for downstream PCA / CEBRA — most of the
    # object is now signal instead of zero. Set filled=False to recover the
    # 4-edge outline rendering (still useful for visual debugging).
    #
    # v2 step 3: per-frame Python loop replaced by broadcast ops. Edge bounds
    # are computed as (T,) arrays once per object, then broadcast against
    # row/col coordinate grids to build the paint mask for the whole session
    # in one shot. Output shape changed from (T, lat_samp, vert_samp) to
    # (T, vert_samp, lat_samp) so axes match indexing (rows=vert, cols=lat);
    # numerically identical when lat_samp == vert_samp (the only case used
    # in practice). Output dtype changed to uint8 (was int) — values are
    # 0..255 anyway, and downstream code (PCA / imwrite) handles uint8 fine.
    #
    # v2 step 4: subpixel-coverage anti-aliasing (default, antialias=True).
    # Each pixel's intensity = 255 * fractional area of the object rectangle
    # that lies inside that pixel. Interior pixels stay at 255, edge pixels
    # get partial values. Smooth edges; graded intensity as objects move
    # across pixel boundaries.
    # Note: antialias objects are ~1 pixel narrower than antialias=False,
    # because the legacy hard-edge path has a +1 pad on edge_right that the
    # antialias path drops. Set antialias=False to keep the legacy behavior
    # with hard 0/255 edges and the +1 pad.
    # chunk_t bounds memory for long sessions by processing the time axis
    # in batches of chunk_t frames (default None = single pass).
    #
    # v2 step 5: depth-correct compositing for overlapping objects (antialias
    # path only). Two objects covering the same pixel no longer just take a
    # max — they combine via alpha compositing:
    #   out = 255 * (1 - prod_k(1 - coverage_k))
    # i.e. per pixel, "uncovered probability" = product across objects of
    # (1 - their fractional coverage), and lit-ness = 1 minus that. This is
    # the standard back-to-front composite for objects rendered at the same
    # intensity, and collapses to an *order-independent* product since the
    # final intensity (255) is uniform across objects. Side-effect: at edge
    # pixels where two objects partially overlap, intensity rises above
    # either object's individual coverage (e.g. 0.5 + 0.5 → 0.75, not 0.5).
    # If per-object intensity is added later (depth attenuation, identity
    # channel, color), this needs a real per-frame depth sort — flag it then.
    #
    # return_depth: if True, also return a (T, vert_samp, lat_samp) float32
    # depth buffer for Z-buffer compositing with f_render_terrain. Per pixel,
    # depth = distance to the center of the nearest object covering that
    # pixel (np.inf where no object). Object-center distance (not surface)
    # is what's stored — sufficient for occlusion at 51×51 monitor sampling.
    # Returns (mon_frames, mon_depth) — or (mon_frames, mon_per_obj, mon_depth)
    # if return_per_object is also True.
    #
    # v2 step 6: optional per-object identity output. With return_per_object=
    # True, the function returns (mon_frames, mon_per_obj), where mon_per_obj
    # has shape (T, vert_samp, lat_samp, n_obj_out) and channel k stores the
    # k-th object's coverage rendered *alone* (no compositing with other
    # objects, no depth ordering). Lets downstream code decode object
    # identity from neural activity, separate overlapping objects, and run
    # per-object PCA / CEBRA. Memory: n_obj_out × the size of mon_frames.
    # Channel layout:
    #   total_n_obj = None  → n_obj_out = vec_data['obj_used'].shape[0],
    #                         channels are local (this-monitor only).
    #   total_n_obj = N     → n_obj_out = N, channels indexed by the
    #                         ORIGINAL object id (vec_data['obj_used'][k]).
    #                         Pass len(bh_data[n_dset]['object_data']) when
    #                         you want left+right to share a channel space
    #                         so the two monitors' per-object tensors can be
    #                         stacked / compared.

    # max_frames: render only the first N frames (default None = full session).
    # Useful for fast debugging — output time axis becomes min(T, max_frames).
    num_frames = time.shape[0]
    if max_frames is not None:
        num_frames = min(num_frames, int(max_frames))

    mon_frames = np.zeros((num_frames, vert_samp, lat_samp), dtype=np.uint8)
    if return_depth:
        mon_depth = np.full((num_frames, vert_samp, lat_samp), np.inf, dtype=np.float32)

    use_antialias = antialias and filled

    # `time` is a parameter (the mov_data['time'] array), shadowing the
    # stdlib module — import the clock under an alias so verbose progress
    # printing works.
    import time as _t
    _t0 = _t.perf_counter()
    n_objs_total = vec_data['obj_used'].shape[0]
    if verbose:
        print(f'f_angles_to_movie_v2: T={num_frames}  n_obj={n_objs_total}  '
              f'lat={lat_samp} vert={vert_samp}  filled={filled}  '
              f'antialias={use_antialias}  chunk_t={chunk_t}  '
              f'return_depth={return_depth}  return_per_object={return_per_object}',
              flush=True)

    if not use_antialias:
        ys = np.arange(vert_samp)[None, :, None]  # row coords, shape (1, vert, 1)
        xs = np.arange(lat_samp)[None, None, :]   # col coords, shape (1, 1, lat)

    t_step = num_frames if chunk_t is None else int(chunk_t)

    n_objs = vec_data['obj_used'].shape[0]

    if return_per_object:
        if total_n_obj is None:
            n_obj_out = n_objs
            obj_dest_idx = np.arange(n_objs, dtype=int)
        else:
            n_obj_out = int(total_n_obj)
            obj_dest_idx = np.asarray(vec_data['obj_used'], dtype=int)
        mon_per_obj = np.zeros((num_frames, vert_samp, lat_samp, n_obj_out), dtype=np.uint8)

    if use_antialias:
        # Antialias path: chunk-outer, object-inner so the running
        # "uncovered" product accumulates across objects per time chunk.

        # Precompute float-valued bounds for every object, once each.
        # Each list entry is (T,) float32; we slice into them per chunk.
        obj_lefts   = []
        obj_rights  = []
        obj_tops    = []
        obj_bottoms = []
        obj_in_mon  = []
        for n_obj in range(n_objs):
            idx_mon_obj = vec_data['obj_mon_idx'][:, n_obj]
            obj_in_mon.append(idx_mon_obj)
            if not idx_mon_obj.any():
                obj_lefts.append(None); obj_rights.append(None)
                obj_tops.append(None);  obj_bottoms.append(None)
                continue
            obj_dist = vec_data['obj_dist'][:, n_obj]
            # obj_size convention: 'x' is the lateral HALF-extent (= radius
            # for cylinders, half-side for cubes). 'y' is the FULL vertical
            # extent of the geometry (so the half-extent above center =
            # obj_size['y']/2). 'height' is the offset of the registered
            # center above the ground.
            obj_x_ang_pix_f = np.arctan2(obj_size['x'],       obj_dist) / cam_params['hFOV_rad'] * (lat_samp - 1)
            obj_y_ang_pix_f = np.arctan2(obj_size['y'] / 2.0, obj_dist) / cam_params['vFOV_rad'] * (vert_samp - 1)
            # below-center extent: ground level if clip_underground, else full -y/2.
            if clip_underground:
                obj_h_ang_pix_f = np.arctan2(obj_size['height'], obj_dist) / cam_params['vFOV_rad'] * (vert_samp - 1)
            else:
                obj_h_ang_pix_f = obj_y_ang_pix_f
            # Pixel mapping is NEGATED for both lat and vert so default image
            # viewers (row 0 at top of screen) show natural sky-up: positive
            # vert_angle (object above camera) maps to SMALLER pix = TOP of
            # display; same logic for lateral.
            lat_angle_idx_f  = ((-vec_data['obj_lat_angle'][:, n_obj]  + cam_params['hFOV_rad']/2) / cam_params['hFOV_rad']) * (lat_samp - 1)
            vert_angle_idx_f = ((-vec_data['obj_vert_angle'][:, n_obj] + cam_params['vFOV_rad']/2) / cam_params['vFOV_rad']) * (vert_samp - 1)
            obj_lefts.append(  (lat_angle_idx_f  - obj_x_ang_pix_f).astype(np.float32))
            obj_rights.append( (lat_angle_idx_f  + obj_x_ang_pix_f).astype(np.float32))
            # With the negated vert mapping, SMALLER pix = world-up = TOP of
            # display. So obj_tops (smaller pix) = top of object in world
            # = vert_idx - obj_y_ang_pix_f (y/2 above center); obj_bottoms
            # (larger pix) = ground (clip_underground) or full -y/2 below
            # center = vert_idx + obj_h_ang_pix_f.
            obj_tops.append(   (vert_angle_idx_f - obj_y_ang_pix_f).astype(np.float32))
            obj_bottoms.append((vert_angle_idx_f + obj_h_ang_pix_f).astype(np.float32))

        xs_c = np.arange(lat_samp, dtype=np.float32)
        ys_c = np.arange(vert_samp, dtype=np.float32)

        n_chunks = (num_frames + t_step - 1) // t_step
        for chunk_idx, t_start in enumerate(range(0, num_frames, t_step)):
            t_end = min(t_start + t_step, num_frames)
            chunk = t_end - t_start
            uncovered = np.ones((chunk, vert_samp, lat_samp), dtype=np.float32)
            painted = False
            for n_obj in range(n_objs):
                if obj_lefts[n_obj] is None:
                    continue
                in_mon_chunk = obj_in_mon[n_obj][t_start:t_end]
                # Filter to in-FOV frames — each object is typically in FOV for
                # ~10-50% of frames. Coverage on out-of-FOV frames would be
                # all zero anyway.
                in_idx_local = np.flatnonzero(in_mon_chunk)
                n_in = in_idx_local.shape[0]
                if n_in == 0:
                    continue
                ol = obj_lefts[n_obj][t_start:t_end][in_idx_local]
                or_ = obj_rights[n_obj][t_start:t_end][in_idx_local]
                ot = obj_tops[n_obj][t_start:t_end][in_idx_local]
                ob = obj_bottoms[n_obj][t_start:t_end][in_idx_local]

                # Per-object spatial bounding box across all in-FOV frames.
                # Object typically spans 10-30 pixels in a 101×101 monitor,
                # so bbox clipping is ~10× fewer ops per object on top of the
                # in-FOV time filter. 1-px slack to keep partial-coverage edge
                # pixels.
                l_min = max(0,         int(np.floor(ol.min())) - 1)
                l_max = min(lat_samp,  int(np.ceil(or_.max())) + 1)
                v_min = max(0,         int(np.floor(ot.min())) - 1)
                v_max = min(vert_samp, int(np.ceil(ob.max())) + 1)
                if l_max <= l_min or v_max <= v_min:
                    continue   # object entirely outside the rendered grid

                xs_sub = xs_c[l_min:l_max]
                ys_sub = ys_c[v_min:v_max]
                cov_x = np.clip(
                    np.minimum(xs_sub[None, :] + 0.5, or_[:, None]) -
                    np.maximum(xs_sub[None, :] - 0.5, ol[:, None]),
                    0.0, 1.0,
                )  # (n_in, l_sub)
                cov_y = np.clip(
                    np.minimum(ys_sub[None, :] + 0.5, ob[:, None]) -
                    np.maximum(ys_sub[None, :] - 0.5, ot[:, None]),
                    0.0, 1.0,
                )  # (n_in, v_sub)
                coverage = cov_y[:, :, None] * cov_x[:, None, :]   # (n_in, v_sub, l_sub)

                # Cross-product fancy indexing via np.ix_ — picks the
                # (in_idx_local × v_min:v_max × l_min:l_max) slab. Reads
                # return a copy of that slab; assignment writes back in
                # place via __setitem__.
                _rows = np.arange(v_min, v_max)
                _cols = np.arange(l_min, l_max)
                ix = np.ix_(in_idx_local, _rows, _cols)

                # uncovered ← uncovered * (1 - coverage) over the bbox×in_idx slab
                sub_unc = uncovered[ix]
                sub_unc *= (1.0 - coverage)
                uncovered[ix] = sub_unc

                if return_per_object:
                    # per-object channel stores this object's coverage rendered
                    # alone (no compositing, no occlusion) — exactly what you
                    # want for identity decoding downstream.
                    chan_chunk = mon_per_obj[t_start:t_end, :, :, obj_dest_idx[n_obj]]
                    chan_chunk[ix] = (255.0 * coverage).astype(np.uint8)
                if return_depth:
                    # Object center distance per in-FOV frame; update depth
                    # only where this object actually covers the pixel.
                    obj_d_in = vec_data['obj_dist'][t_start:t_end, n_obj][in_idx_local].astype(np.float32)
                    mon_depth_chunk = mon_depth[t_start:t_end]
                    sub_d = mon_depth_chunk[ix]
                    hit = coverage > 0
                    # np.where(hit, obj_d, sub_d) uses sub_d as fallback so the
                    # subsequent np.minimum is a no-op where coverage==0.
                    # Saves the np.inf-filled temp the older code allocated.
                    np.minimum(sub_d,
                                np.where(hit, obj_d_in[:, None, None], sub_d),
                                out=sub_d)
                    mon_depth_chunk[ix] = sub_d
                painted = True
            if painted:
                mon_frames[t_start:t_end] = (255.0 * (1.0 - uncovered)).astype(np.uint8)
            if verbose:
                print(f'  chunk {chunk_idx+1}/{n_chunks}  '
                      f'frames {t_start}..{t_end}  '
                      f'elapsed {_t.perf_counter()-_t0:.1f}s', flush=True)
    else:
        _print_every = max(1, n_objs // 10)
        for n_obj in range(n_objs):

            idx_mon_obj = vec_data['obj_mon_idx'][:, n_obj]
            if not idx_mon_obj.any():
                if verbose and ((n_obj + 1) % _print_every == 0 or n_obj == n_objs - 1):
                    print(f'  obj {n_obj+1}/{n_objs}  '
                          f'elapsed {_t.perf_counter()-_t0:.1f}s', flush=True)
                continue

            obj_dist = vec_data['obj_dist'][:, n_obj]
            # obj_size: 'x' lateral half-extent, 'y' FULL vertical extent
            # (half above center = y/2), 'height' center-above-ground.
            obj_x_ang_pix_f = np.arctan2(obj_size['x'],       obj_dist) / cam_params['hFOV_rad'] * (lat_samp - 1)
            obj_y_ang_pix_f = np.arctan2(obj_size['y'] / 2.0, obj_dist) / cam_params['vFOV_rad'] * (vert_samp - 1)
            if clip_underground:
                obj_h_ang_pix_f = np.arctan2(obj_size['height'], obj_dist) / cam_params['vFOV_rad'] * (vert_samp - 1)
            else:
                obj_h_ang_pix_f = obj_y_ang_pix_f
            # Pixel mapping NEGATED for both axes so default viewers show
            # natural sky-up (positive vert_angle = above camera = SMALL pix
            # = TOP of display).
            lat_angle_idx_f  = ((-vec_data['obj_lat_angle'][:, n_obj]  + cam_params['hFOV_rad']/2) / cam_params['hFOV_rad']) * (lat_samp - 1)
            vert_angle_idx_f = ((-vec_data['obj_vert_angle'][:, n_obj] + cam_params['vFOV_rad']/2) / cam_params['vFOV_rad']) * (vert_samp - 1)

            obj_x_ang_pix = np.round(obj_x_ang_pix_f)
            obj_y_ang_pix = np.round(obj_y_ang_pix_f)
            obj_h_ang_pix = np.round(obj_h_ang_pix_f)
            lat_angle_idx = np.round(lat_angle_idx_f).astype(int)
            vert_angle_idx = np.round(vert_angle_idx_f).astype(int)

            edge_left = (lat_angle_idx - obj_x_ang_pix).astype(int)
            edge_right = (lat_angle_idx + obj_x_ang_pix + 1).astype(int)
            # With negated vert mapping, SMALL pix = world-up = TOP of display.
            # edge_top (smaller pix) = top of object in world = vert - obj_y_pix
            # (y/2 above center); edge_bottom (larger pix) = ground (clip) or
            # full -y/2 below center = vert + obj_h_pix.
            edge_top    = (vert_angle_idx - obj_y_ang_pix).astype(int)               # y/2 above center = top of geometry
            edge_bottom = (vert_angle_idx + obj_h_ang_pix).astype(int)               # height below center (= ground when clip_underground)

            et = edge_top[:, None, None]
            eb = edge_bottom[:, None, None]
            el = edge_left[:, None, None]
            er = edge_right[:, None, None]
            in_mon = idx_mon_obj[:, None, None]

            if filled:
                mask = in_mon & (ys >= et) & (ys <= eb) & (xs >= el) & (xs <= er)
                mon_frames[mask] = 255
                if return_per_object:
                    per_obj_slice = mon_per_obj[..., obj_dest_idx[n_obj]]
                    per_obj_slice[mask] = 255
                if return_depth:
                    obj_d = obj_dist.astype(np.float32)[:, None, None]
                    d_layer = np.where(mask, obj_d, np.inf)
                    np.minimum(mon_depth, d_layer, out=mon_depth)
            else:
                x_in = (xs >= el) & (xs <= er)
                y_in = (ys >= et) & (ys <= eb)
                edge_mask = in_mon & (
                    (((ys == et) | (ys == eb)) & x_in) |
                    (((xs == el) | (xs == er)) & y_in)
                )
                mon_frames[edge_mask] = 255
                # center pixel
                idx_vis = np.where(idx_mon_obj)[0]
                mon_frames[idx_vis, vert_angle_idx[idx_vis], lat_angle_idx[idx_vis]] = 255
                if return_per_object:
                    per_obj_slice = mon_per_obj[..., obj_dest_idx[n_obj]]
                    per_obj_slice[edge_mask] = 255
                    per_obj_slice[idx_vis, vert_angle_idx[idx_vis], lat_angle_idx[idx_vis]] = 255
                if return_depth:
                    obj_d = obj_dist.astype(np.float32)[:, None, None]
                    d_layer = np.where(edge_mask, obj_d, np.inf)
                    np.minimum(mon_depth, d_layer, out=mon_depth)
                    # center pixel always written if object is in monitor
                    mon_depth[idx_vis, vert_angle_idx[idx_vis], lat_angle_idx[idx_vis]] = np.minimum(
                        mon_depth[idx_vis, vert_angle_idx[idx_vis], lat_angle_idx[idx_vis]],
                        obj_dist[idx_vis].astype(np.float32),
                    )
            if verbose and ((n_obj + 1) % _print_every == 0 or n_obj == n_objs - 1):
                print(f'  obj {n_obj+1}/{n_objs}  '
                      f'elapsed {_t.perf_counter()-_t0:.1f}s', flush=True)

    if verbose:
        print(f'f_angles_to_movie_v2: done in {_t.perf_counter()-_t0:.1f}s', flush=True)

    out_tuple = [mon_frames]
    if return_per_object:
        out_tuple.append(mon_per_obj)
    if return_depth:
        out_tuple.append(mon_depth)
    if len(out_tuple) == 1:
        return mon_frames
    return tuple(out_tuple)


def f_render_terrain(terrain_data, mouse_xyz, mon_phi, mon_theta, cam_params,
                      lat_samp=51, vert_samp=51,
                      chunk_pitch=122.0, chunk_centered=True, cell_size=1.0,
                      eye_height=0.0, stride=4, max_intensity=200, chunk_t=2000,
                      flip_x=False, flip_z=True, swap_xz=False,
                      clip_len=None, point_size=1, max_diff_bytes=int(2e9),
                      return_depth=False, max_frames=None, verbose=True):
    # Render a heightmap as a depth-shaded ground-plane image on the monitor.
    # Sibling of f_angles_to_movie_v2 — composite with np.maximum to get
    # objects on top of ground:
    #   left_mon_frames = np.maximum(left_mon_frames, terrain_l)
    #
    # terrain_data : DataFrame with columns ChunkPosX, ChunkPosZ, x, z, height.
    #   ChunkPosX / ChunkPosZ are world-coord positions of each chunk's reference
    #   point: the chunk CENTER if chunk_centered=True (default; matches the
    #   Unity convention where chunks are placed by their centers), or the
    #   corner if False.
    #   chunk_pitch : world-unit spacing between adjacent chunk centers.
    #   cell_size : world units per cell (default 1.0 for this rig). With
    #     chunk_pitch=122, cell_max=124, and cell_size=1.0, adjacent chunks
    #     overlap by 3 cells per side = 2 world units (A[122..124, *] equal
    #     B[0..2, *]). See f_terrain_world_coords docstring.
    #   flip_x / flip_z : invert the cell index direction within a chunk.
    #     Needed when the heightmap's local axis runs opposite to world.
    #     Default flip_z=True based on visual stitching; toggle as needed.
    #   swap_xz : swap the role of the local 'x' and 'z' columns within each
    #     chunk before mapping to world (handy when local axes are transposed).
    # World coords (after flip and swap and optional centering):
    #     world_x = ChunkPosX + (x_local - center_x) * cell_size   (chunk_centered)
    #             = ChunkPosX +  x_local             * cell_size   (else)
    #     world_z = ChunkPosZ + (z_local - center_z) * cell_size
    #     world_y = height
    # mouse_xyz   : (T, 3) world position over time, columns (x, y, z) Unity-style
    #               (y is up). Eye offset is added internally via eye_height.
    # mon_phi     : (T,) monitor azimuth per frame (matches f_get_monitor_coords)
    # mon_theta   : (T,) monitor elevation per frame; constant in current usage
    #
    # cam_params  : dict with 'hFOV_rad', 'vFOV_rad', 'clip_len'.
    # clip_len    : override for the terrain visibility / intensity-falloff
    #               distance. None (default) → use cam_params['clip_len'],
    #               matching the object renderer. Set to a separate value
    #               to render terrain farther/closer than objects (e.g.,
    #               clip_len=120 to see ground twice as far while objects
    #               still cut off at the original 60).
    # stride      : cell-grid subsampling. Integer ≥ 1 keeps every Nth cell
    #               (current behavior). Float < 1 oversamples each chunk's
    #               heightmap via bilinear interpolation — useful when the
    #               raw 125×125 grid is too coarse. Memory cost scales as
    #               (1/stride)² per chunk, so stride=0.5 → 4× and 0.25 → 16×.
    #               Won't fully fix near-field gaps on its own — see
    #               `point_size`.
    # point_size  : pixel half-size of the square patch painted around each
    #               projected terrain sample. Default 1 = single pixel. Set
    #               to 2 or 3 to fill near-field gaps without exploding the
    #               sample count. Patch is constant in pixel space (not
    #               world-space), so far-field also gets the same patch —
    #               which is usually fine since far-field is already dense.
    # stride      : sub-sample every `stride`-th terrain cell along each axis.
    #               125x125 raw → ~31x31 at stride=4 (fast and visually dense).
    # max_intensity: peak terrain brightness on screen (kept below 255 so objects
    #               clearly sit on top after compositing). Intensity falls off
    #               linearly with distance: I = max_intensity*(1 - r/clip_len).
    # chunk_t     : initial time-axis chunking. Adaptive: if the diff array
    #               (chunk_t × N_local × 3) would exceed `max_diff_bytes`,
    #               the chunk is halved repeatedly until it fits. Halving also
    #               tightens the per-chunk bbox (mouse roams less in a smaller
    #               time window) → N_local drops too.
    # max_diff_bytes : memory budget for the (chunk_t, N_local, 3) float32 diff
    #               array. Default ~2 GB → ~5–6 GB peak with the dependent
    #               (chunk, N_local) intermediates. Lower this if you have less
    #               RAM headroom; raise it if you have plenty.
    #
    # Output shape (T, vert_samp, lat_samp) uint8, same convention as
    # f_angles_to_movie_v2.
    # return_depth : if True, also return per-pixel min-distance float32
    #               buffer (np.inf where nothing was drawn) for downstream
    #               Z-buffer compositing with the object renderer. Returns
    #               (out_intensity, out_depth) tuple instead of just out.

    if stride is None or stride >= 1:
        # Original path: shared helper + integer modulo subsample.
        w = f_terrain_world_coords(terrain_data, chunk_pitch=chunk_pitch,
                                    chunk_centered=chunk_centered,
                                    flip_x=flip_x, flip_z=flip_z, swap_xz=swap_xz,
                                    cell_size=cell_size)
        world_x_all = w['tx'].astype(np.float32)
        world_z_all = w['tz'].astype(np.float32)
        world_y_all = w['ty'].astype(np.float32)
        if stride and stride > 1:
            cx = terrain_data['x'].values
            cz = terrain_data['z'].values
            sub = (cx % int(stride) == 0) & (cz % int(stride) == 0)
            world_x_all = world_x_all[sub]
            world_y_all = world_y_all[sub]
            world_z_all = world_z_all[sub]
    else:
        # stride < 1: oversample each chunk's heightmap via bilinear interp.
        # Replaces the f_terrain_world_coords call — we generate the upsampled
        # grid and apply the same orientation conventions directly.
        cell_max_x = int(terrain_data['x'].max())
        cell_max_z = int(terrain_data['z'].max())
        zoom_factor = 1.0 / float(stride)
        # World span of a chunk = cell_max * cell_size. After upsampling the
        # chunk's heightmap to (new_cell_max + 1) cells, the new cell spacing
        # is (chunk world span) / new_cell_max = cell_max * cell_size / new_cell_max.
        all_tx, all_ty, all_tz = [], [], []
        for (cpx, cpz), grp in terrain_data.groupby(['ChunkPosX', 'ChunkPosZ']):
            hm = grp.pivot(index='x', columns='z', values='height').values    # (n_x_orig, n_z_orig)
            hm_up = nd_zoom(hm, zoom_factor, order=1)                          # bilinear
            n_x_new, n_z_new = hm_up.shape
            new_cell_max_x = n_x_new - 1
            new_cell_max_z = n_z_new - 1
            new_cs_x = (cell_max_x * float(cell_size)) / new_cell_max_x
            new_cs_z = (cell_max_z * float(cell_size)) / new_cell_max_z
            ix = np.arange(n_x_new, dtype=np.float32)
            iz = np.arange(n_z_new, dtype=np.float32)
            ix2d, iz2d = np.meshgrid(ix, iz, indexing='ij')
            ax_for_x = iz2d if swap_xz else ix2d
            ax_for_z = ix2d if swap_xz else iz2d
            cx_w = (new_cell_max_x - ax_for_x) if flip_x else ax_for_x
            cz_w = (new_cell_max_z - ax_for_z) if flip_z else ax_for_z
            if chunk_centered:
                cx_w = cx_w - new_cell_max_x / 2.0
                cz_w = cz_w - new_cell_max_z / 2.0
            all_tx.append((cpx + cx_w * new_cs_x).ravel())
            all_tz.append((cpz + cz_w * new_cs_z).ravel())
            all_ty.append(hm_up.ravel())
        world_x_all = np.concatenate(all_tx).astype(np.float32)
        world_z_all = np.concatenate(all_tz).astype(np.float32)
        world_y_all = np.concatenate(all_ty).astype(np.float32)

    terrain_world = np.stack([world_x_all, world_y_all, world_z_all], axis=1)  # (N, 3)
    N = terrain_world.shape[0]

    mouse_xyz = np.asarray(mouse_xyz, dtype=np.float32)
    mon_phi   = np.asarray(mon_phi,   dtype=np.float32)
    mon_theta = np.asarray(mon_theta, dtype=np.float32)
    # max_frames: render only the first N frames (default None = full session).
    T = mouse_xyz.shape[0]
    if max_frames is not None:
        T = min(T, int(max_frames))
        mouse_xyz = mouse_xyz[:T]
        mon_phi   = mon_phi[:T]
        mon_theta = mon_theta[:T]

    out = np.zeros((T, vert_samp, lat_samp), dtype=np.uint8)
    if return_depth:
        out_depth = np.full((T, vert_samp, lat_samp), np.inf, dtype=np.float32)

    hFOV     = cam_params['hFOV_rad']
    vFOV     = cam_params['vFOV_rad']
    clip_len = float(cam_params['clip_len'] if clip_len is None else clip_len)

    # eye height adds upward to mouse y so the camera sits above the body
    eye_off = np.array([0.0, eye_height, 0.0], dtype=np.float32)

    if verbose:
        print(f'f_render_terrain: N_terrain={N} samples (after stride={stride}), '
              f'T={T} frames, chunk_t={chunk_t}')

    t_start = 0
    ci = 0
    while t_start < T:
        # Adaptive chunk sizing: shrink the time slice until the (ch, N_local, 3)
        # diff array fits the memory budget. Each halving also tightens the
        # bbox (mouse moves less in fewer frames) → N_local drops too.
        ch_try = min(chunk_t, T - t_start)
        while True:
            t_end = t_start + ch_try
            ch = ch_try
            mouse_c = mouse_xyz[t_start:t_end] + eye_off              # (ch, 3)
            x_min = mouse_c[:, 0].min() - clip_len
            x_max = mouse_c[:, 0].max() + clip_len
            z_min = mouse_c[:, 2].min() - clip_len
            z_max = mouse_c[:, 2].max() + clip_len
            keep = (terrain_world[:, 0] >= x_min) & (terrain_world[:, 0] <= x_max) & \
                   (terrain_world[:, 2] >= z_min) & (terrain_world[:, 2] <= z_max)
            n_local = int(keep.sum())
            diff_bytes = ch * n_local * 3 * 4
            if diff_bytes <= max_diff_bytes or ch_try == 1:
                break
            ch_try = max(1, ch_try // 2)
        terrain_local = terrain_world[keep]                            # (N_local, 3)
        if verbose:
            print(f'  chunk {ci+1}  frames [{t_start}:{t_end}]  '
                  f'N_local={n_local} ({100*n_local/max(N,1):.1f}% of N)  '
                  f'ch={ch}  diff~{diff_bytes/1e9:.2f} GB', flush=True)
        ci += 1
        if n_local == 0:
            t_start = t_end
            continue

        # Per-coord (ch, N) diffs. Three separate allocations instead of one
        # (ch, N, 3) so dx, dy, dz are contiguous in memory — better cache
        # behavior than strided views into the 3D diff.
        dx = terrain_local[None, :, 0] - mouse_c[:, None, 0]
        dz = terrain_local[None, :, 2] - mouse_c[:, None, 2]
        dy = terrain_local[None, :, 1] - mouse_c[:, None, 1]
        r_h_sq = dx*dx + dz*dz                    # horizontal distance²
        r_sq   = r_h_sq + dy*dy                    # full distance² — sqrt deferred

        hFOV_half = hFOV * 0.5
        vFOV_half = vFOV * 0.5
        clip_sq = clip_len * clip_len

        # Cheap horizontal FOV pre-test via cosine cone. Mathematically
        # equivalent to |lat_angle| < hFOV/2 when fwd_proj > 0:
        #   cos(lat_angle) = cos(mon_phi - phi)
        #                  = (cos(mon_phi)*dz + sin(mon_phi)*dx) / r_h
        # FOV test:  cos(lat_angle) > cos(hFOV/2)
        # Square it (after the sign check): fwd_proj² > cos²(hFOV/2) * r_h_sq.
        # Avoids the (ch, N) arctan2 + modulo-wrap that the old code did,
        # which dominated runtime.
        sin_phi_m = np.sin(mon_phi[t_start:t_end]).astype(np.float32)
        cos_phi_m = np.cos(mon_phi[t_start:t_end]).astype(np.float32)
        fwd_proj_h = sin_phi_m[:, None] * dx + cos_phi_m[:, None] * dz   # (ch, N)
        cos_h_half_sq = float(np.cos(hFOV_half) ** 2)

        horiz_pre = (
            (fwd_proj_h > 0) &
            (fwd_proj_h * fwd_proj_h > cos_h_half_sq * r_h_sq) &
            (r_sq < clip_sq) &
            (r_sq > 1e-12)
        )

        if horiz_pre.any():
            # Compact to the surviving subset (n_pre,) — typically ~25% of
            # (ch, N_local) since the horizontal FOV cone covers ~25% of the
            # search bbox. arctan2 / sqrt / modulo wrap now run on a much
            # smaller array.
            ts_pre, _ = np.where(horiz_pre)
            dx_p = dx[horiz_pre]
            dz_p = dz[horiz_pre]
            dy_p = dy[horiz_pre]
            r_h_sq_p = r_h_sq[horiz_pre]
            r_sq_p   = r_sq  [horiz_pre]

            phi_p   = np.arctan2(dx_p, dz_p)
            r_h_p   = np.sqrt(r_h_sq_p)
            theta_p = np.arctan2(r_h_p, dy_p)

            mon_phi_p   = mon_phi  [t_start + ts_pre]
            mon_theta_p = mon_theta[t_start + ts_pre]

            TWO_PI = 2.0 * np.pi
            PI = np.pi
            lat_angle_p  = (mon_phi_p   - phi_p   + PI) % TWO_PI - PI
            vert_angle_p = (mon_theta_p - theta_p + PI) % TWO_PI - PI

            in_fov_p = (np.abs(lat_angle_p) < hFOV_half) & (np.abs(vert_angle_p) < vFOV_half)

            if in_fov_p.any():
                # match f_angles_to_movie_v2 pixel mapping.
                lat_in  = lat_angle_p[in_fov_p]
                vert_in = vert_angle_p[in_fov_p]
                lat_pix_flat  = np.round(((-lat_in  + hFOV_half) / hFOV) * (lat_samp  - 1)).astype(np.int64)
                vert_pix_flat = np.round(((-vert_in + vFOV_half) / vFOV) * (vert_samp - 1)).astype(np.int64)

                # depth-shaded intensity, brighter for nearby terrain.
                r_vis = np.sqrt(r_sq_p[in_fov_p])
                intensity_flat = (max_intensity * (1.0 - r_vis / clip_len)).clip(0, max_intensity).astype(np.uint8)

                # which (frame_idx_in_chunk) does each visible sample belong to?
                ts_flat = ts_pre[in_fov_p]
                base_t  = (t_start + ts_flat).astype(np.int64)

                # scatter via sort-then-reduce.
                if base_t.size > 0:
                    base_v   = vert_pix_flat
                    base_l   = lat_pix_flat
                    base_vals = intensity_flat
                    base_depth = r_vis if return_depth else None
                    # point_size > 1: paint a (2*ps+1)x(2*ps+1) patch around each
                    # sample. Expansion is a small Python loop over patch offsets,
                    # each iteration fully vectorized across all in-FOV samples.
                    ps = max(0, int(point_size) - 1)
                    flat = None
                    if ps == 0:
                        flat = (base_t * vert_samp + base_v) * lat_samp + base_l
                        vals = base_vals
                        depths = base_depth
                    else:
                        flats = []
                        valss = []
                        depthss = [] if return_depth else None
                        for dv in range(-ps, ps + 1):
                            v_pix = base_v + dv
                            v_ok = (v_pix >= 0) & (v_pix < vert_samp)
                            for dl in range(-ps, ps + 1):
                                l_pix = base_l + dl
                                ok = v_ok & (l_pix >= 0) & (l_pix < lat_samp)
                                if not ok.any():
                                    continue
                                flats.append((base_t[ok] * vert_samp + v_pix[ok]) * lat_samp + l_pix[ok])
                                valss.append(base_vals[ok])
                                if return_depth:
                                    depthss.append(base_depth[ok])
                        if flats:
                            flat = np.concatenate(flats)
                            vals = np.concatenate(valss)
                            depths = np.concatenate(depthss) if return_depth else None
                    if flat is not None:
                        order = np.argsort(flat, kind='stable')
                        flat_s = flat[order]
                        vals_s = vals[order]
                        starts = np.concatenate(([0], np.flatnonzero(np.diff(flat_s)) + 1))
                        max_per_group = np.maximum.reduceat(vals_s, starts)
                        unique_idx = flat_s[starts]
                        cur = out.reshape(-1)[unique_idx]
                        new = np.maximum(cur, max_per_group)
                        out.reshape(-1)[unique_idx] = new
                        if return_depth:
                            depths_s = depths[order]
                            min_per_group = np.minimum.reduceat(depths_s, starts)
                            cur_d = out_depth.reshape(-1)[unique_idx]
                            out_depth.reshape(-1)[unique_idx] = np.minimum(cur_d, min_per_group)

        t_start = t_end

    if return_depth:
        return out, out_depth
    return out


def f_composite_with_depth(layers):
    # Z-buffer composite of multiple (intensity, depth) layers. Per pixel,
    # pick the intensity from the layer with the smallest depth; np.inf in a
    # layer's depth means "transparent here" so the next layer shows through.
    #
    # layers : list of (intensity, depth) tuples. intensity is uint8
    #   (T, vert, lat); depth is float32 (T, vert, lat) with np.inf where
    #   nothing was drawn. All layers must share the same shape.
    # Returns: uint8 (T, vert, lat) composite.
    #
    # Notes
    #  - "Smallest depth wins" — terrain in front of an object hides the
    #    object; object in front of terrain hides the terrain pixel.
    #  - Where every layer has np.inf, output is 0 (background).
    #  - For ties (equal depth), the LATER layer in the list wins. Put
    #    terrain first, objects second if you'd rather show objects on ties.
    if not layers:
        raise ValueError('layers must be non-empty')
    shape = layers[0][0].shape
    out = np.zeros(shape, dtype=np.uint8)
    best_depth = np.full(shape, np.inf, dtype=np.float32)
    for intensity, depth in layers:
        if intensity.shape != shape or depth.shape != shape:
            raise ValueError(f'layer shape mismatch: {intensity.shape} / {depth.shape} vs {shape}')
        win = depth <= best_depth
        # also require this layer is actually drawn here (depth < inf), else
        # ties at inf would let an all-empty layer "win" and write its 0s.
        win &= np.isfinite(depth)
        out = np.where(win, intensity, out)
        best_depth = np.where(win, depth, best_depth)
    return out


def f_add_terrain_to_monitor(terrain_data, mouse_xyz, mon_phi, theta, cam_params,
                              mon_frames, mon_depth,
                              lat_samp=51, vert_samp=51,
                              chunk_pitch=122.0, chunk_centered=True, cell_size=1.0,
                              flip_x=False, flip_z=True, swap_xz=False,
                              eye_height=0.0, stride=0.1, max_intensity=200, chunk_t=2000,
                              point_size=1, clip_len=None, max_frames=None,
                              verbose=True):
    # Render the terrain heightmap onto a monitor and Z-buffer composite with
    # an existing object render. Convenience wrapper around f_render_terrain
    # + f_composite_with_depth — use to add the terrain layer behind/around
    # objects on a monitor previously produced by
    # f_angles_to_movie_v2(..., return_depth=True).
    #
    # Parameters
    # ----------
    # terrain_data : DataFrame
    #     bh_data[n_dset]['terrainData'] — chunked heightmap.
    # mouse_xyz : (T, 3) ndarray
    #     Camera position per frame. Pass the eye-shifted xyz
    #     (mouse_y + cam_height) so the renderer's camera matches
    #     f_get_monitor_coords.
    # mon_phi, theta : (T,) arrays
    #     Per-frame yaw + pitch for THIS monitor (already-offset by
    #     ±cam_rotation_rad for L/R).
    # cam_params : dict
    #     Camera params (hFOV_rad, vFOV_rad, clip_len, ...).
    # mon_frames : (T, vert, lat) uint8
    #     Existing monitor render — output of f_angles_to_movie_v2.
    # mon_depth : (T, vert, lat) float32
    #     Companion depth buffer — also from f_angles_to_movie_v2 with
    #     return_depth=True. np.inf where no object was drawn.
    #
    # Remaining kwargs pass through to f_render_terrain (chunk-coord
    # conventions, stride, point_size, clip_len, etc.). Defaults match the
    # rig (chunk_pitch=122, flip_z=True, eye_height=0 since mouse_xyz is
    # already eye-shifted).
    #
    # Returns
    # -------
    # composited : (T, vert, lat) uint8
    #     Z-buffer composite. Terrain pixels in front of an object are
    #     visible; objects in front of terrain occlude it. Terrain is
    #     drawn first in the layer list so objects win ties (rare with
    #     finite depths).
    #
    # Usage in a script (toggle via a bool):
    #     if render_terrain:
    #         left_mon_frames = f_add_terrain_to_monitor(
    #             bh_data[n_dset]['terrainData'], mouse_xyz_eye, mon_l_phi, theta,
    #             cam_params, left_mon_frames, left_mon_depth,
    #             lat_samp=num_samp, vert_samp=num_samp,
    #             stride=terrain_stride, point_size=terrain_point_size,
    #             clip_len=terrain_clip_len, max_frames=max_frames)
    #         right_mon_frames = f_add_terrain_to_monitor(... mon_r_phi ... right_mon_frames, right_mon_depth ...)
    terrain, terrain_depth = f_render_terrain(
        terrain_data, mouse_xyz, mon_phi, theta, cam_params,
        lat_samp=lat_samp, vert_samp=vert_samp,
        chunk_pitch=chunk_pitch, chunk_centered=chunk_centered, cell_size=cell_size,
        flip_x=flip_x, flip_z=flip_z, swap_xz=swap_xz,
        eye_height=eye_height, stride=stride, max_intensity=max_intensity,
        chunk_t=chunk_t, clip_len=clip_len, point_size=point_size,
        return_depth=True, max_frames=max_frames, verbose=verbose,
    )
    return f_composite_with_depth([(terrain, terrain_depth),
                                    (mon_frames, mon_depth)])


def f_terrain_world_coords(terrain_data, chunk_pitch=122.0, chunk_centered=True,
                            flip_x=False, flip_z=True, swap_xz=False,
                            cell_size=1.0):
    # Project a Unity-style chunked heightmap to world coords.
    #
    # Conventions for this rig:
    #  - cells per chunk: 125 (indices 0..124, so cell_max = 124).
    #  - chunk_pitch (world distance between adjacent chunk CENTERS): 122.
    #  - cell_size (world units per cell-to-cell step): 1.0.
    #  - chunk total width: cell_max * cell_size = 124 world units.
    #  - overlap with neighbor: 3 cells = 2 world units per side.
    #    A[122, *] = B[0, *], A[123, *] = B[1, *], A[124, *] = B[2, *]
    #    (verifiable via the "Chunk-overlap diagnostic" cell in
    #    VR_ca_cebra.py — rms-diff zeros on the anti-diagonal of a 3x3
    #    boundary comparison).
    #
    # Parameters
    # ----------
    # chunk_pitch : world distance between adjacent chunk centers. Kept
    #    for reference / downstream callers; the projection itself only
    #    needs cell_size.
    # chunk_centered : if True, ChunkPosX/Z are chunk CENTERS (cell index
    #    cell_max/2 = 62 for this rig). If False, treated as chunk corners.
    # flip_x / flip_z : invert local axis direction within a chunk.
    # swap_xz : transpose the local x and z axes.
    # cell_size : world units per cell. Default 1.0 for this rig.
    #
    # Returns a dict {tx, tz, ty, cs_x, cs_z, cell_max_x, cell_max_z}.
    cpx = terrain_data['ChunkPosX'].values
    cpz = terrain_data['ChunkPosZ'].values
    cx  = terrain_data['x'].values
    cz  = terrain_data['z'].values
    h   = terrain_data['height'].values
    cell_max_x = int(terrain_data['x'].max())
    cell_max_z = int(terrain_data['z'].max())
    cs_x = float(cell_size)
    cs_z = float(cell_size)

    ax_for_x = cz if swap_xz else cx
    ax_for_z = cx if swap_xz else cz
    cx_w = (cell_max_x - ax_for_x) if flip_x else ax_for_x
    cz_w = (cell_max_z - ax_for_z) if flip_z else ax_for_z
    if chunk_centered:
        cx_w = cx_w - cell_max_x / 2.0
        cz_w = cz_w - cell_max_z / 2.0

    return {'tx': cpx + cx_w * cs_x,
            'tz': cpz + cz_w * cs_z,
            'ty': h,
            'cs_x': cs_x,
            'cs_z': cs_z,
            'cell_max_x': cell_max_x,
            'cell_max_z': cell_max_z}



def f_save_mon_movie(frames, out_path, frame_range=None, gap_px=2, gap_val=128,
                       overwrite=True):
    # Save a synthetic-monitor stack as a multi-page TIFF (opens in ImageJ
    # as a movie). Mirrors the ad-hoc save used in VR_ca_analysis.py.
    #
    # frames: (T, vert, lat) for one monitor, or (T, vert, 2*lat) for two
    #         monitors concatenated on the lat axis. dtype int / uint8 ok.
    # out_path: destination path; parent dir is created if missing.
    # frame_range: (start, stop) slice into T; None = all frames.
    # gap_px: vertical separator line drawn between the two monitor halves
    #         (only used when width is even). Set 0 to skip.
    # gap_val: intensity of that separator (0-255).
    # overwrite: if True (default), silently overwrite an existing file at
    #     `out_path`. If False, raise FileExistsError if the path is taken.
    if frame_range is not None:
        frames = frames[frame_range[0]:frame_range[1]]
    f = np.asarray(frames).astype(np.uint8)
    if f.shape[2] % 2 == 0 and gap_px > 0:
        mid = f.shape[2] // 2
        f = f.copy()
        f[:, :, mid - gap_px//2 : mid + (gap_px - gap_px//2)] = gap_val
    out_path = str(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            raise FileExistsError(f'{out_path} already exists (pass overwrite=True to replace)')
    tf.imwrite(out_path, f)
    print(f'saved {f.shape[0]} frames -> {out_path}')


    