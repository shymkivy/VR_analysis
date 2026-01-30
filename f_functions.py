# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:47:27 2025

@author: ys2605
"""
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

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
        tag1 = fil1[:idx1+2]
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

#%%
def f_plot_volt(data_in, time):
    cum_d = data_in - data_in[data_in.index[0]]
    cum_ds = sc.ndimage.gaussian_filter1d(cum_d, 10)
    diff_ds = np.diff(cum_ds, prepend=0)

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
def f_plot_session(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel):
    
    sessions_id = np.arange(len(session_lr))+1
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(sessions_id[session_lr], session_lr_dist[session_lr]/np.max(session_lr_dist[session_lr]), 'tab:orange')
    ax1.plot(sessions_id[session_zonal], session_zone_size[session_zonal]/np.max(session_zone_size[session_zonal]), 'tab:green')
    rect_lr = ptc.Rectangle([sessions_id[np.where(session_lr)[0][0]],0], np.sum(session_lr)-1, 1, facecolor='tab:orange', alpha=0.1) # , color='tab:orange'
    rect_zone = ptc.Rectangle([sessions_id[np.where(session_zonal)[0][0]],0], np.sum(session_zonal)-1, 1, facecolor='tab:green', alpha=0.1) # , color='tab:orange'
    rect_fix = ptc.Rectangle([sessions_id[np.where(session_fixed_wheel)[0][0]],0], np.sum(session_fixed_wheel)-1, 1, facecolor='tab:blue', alpha=0.1) # , color='tab:orange'
    ax1.add_patch(rect_lr)
    ax1.add_patch(rect_zone)   
    ax1.add_patch(rect_fix)
    ax1.yaxis.tick_right()
    ax2.yaxis.tick_left()
    
    return ax2

def f_plot_session2(session_lr, session_lr_dist, session_zonal, session_zone_size, session_fixed_wheel, y_size=1):
    
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


