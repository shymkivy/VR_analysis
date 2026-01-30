# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:39:19 2026

@author: ys2605
"""
import os
import sys
import tifffile as tf
import pickle
import pandas as pd

import numpy as np
#import scipy as sc
import matplotlib.pyplot as plt

sys.path.append('C:/Users/ys2605/Desktop/stuff/VR/VR_analysis/')
from f_function_retmap import f_get_sig_thresh, f_get_pulse_times, f_plot_pulse_times
#import NeuroAnalysisTools.RetinotopicMapping as rm

#%% need to generate trial data from recordings 

data_path = 'D:/VR/mapping/L/'
save_dir = 'F:/VR/data_proc/mapping/'

mov_fname = 'L_VC_mapping00001.tif'

cam_data = tf.imread(os.path.join(data_path, mov_fname))
cam_frametimes = pd.read_csv(os.path.join(data_path, 'L_VC_mapping00001_.csv'))
cam_time = np.array(cam_frametimes['Time_From_Start'])

with open(os.path.join(data_path, '251218220608-KSstimSeqDir-ML_gcamp1-Yuriy-VC_mapping-notTriggered-complete.pkl'), 'rb') as file:
        # Load the data from the file
        stim_data = pickle.load(file)
        
volt_data = pd.read_csv(os.path.join(data_path, 'L_VC_mapping-001/L_VC_mapping-001_Cycle00001_VoltageRecording_001.csv'))
volt_time = np.array(volt_data['Time(ms)'])
volt_frames = np.array(volt_data[' Frames'])
volt_LED = np.array(volt_data[' LED'])
volt_phd = np.array(volt_data[' Photodiode'])

#%%

stim_data['stimulation']['direction']

# stim_data['stimulation']['frame_config']

mon_frame_time = stim_data['presentation']['frame_ts_start']
frame_data = stim_data['stimulation']['frames']

num_frames = len(mon_frame_time)

#%%
is_disp = np.zeros(num_frames, dtype=bool)
for n_fr in range(len(frame_data)):
    is_disp[n_fr] = frame_data[n_fr][0]


(T, d1, d2) = cam_data.shape

cam_ave_ca = np.mean(np.mean(cam_data, axis=2), axis=1)
mean_frame = np.mean(cam_data, axis=0)

#%%

if 0:
    plt.close('all')
    
    plt.figure()
    plt.plot(cam_time, cam_ave_ca)
    
    plt.figure()
    plt.imshow(mean_frame, cmap='gray', interpolation='nearest')
    
    num_t_plot = int(1e10)
    plt.figure()
    plt.plot(volt_time[:num_t_plot], volt_frames[:num_t_plot])
    plt.plot(volt_time[:num_t_plot], volt_LED[:num_t_plot])
    plt.plot(volt_time[:num_t_plot], volt_phd[:num_t_plot]*100)
    plt.legend(['Frames', 'LED', 'Photodiode'])
    plt.title('Voltage data')
    
    plt.figure()
    plt.plot(np.diff(mon_frame_time))
    
    plt.figure()
    plt.plot(mon_frame_time, is_disp)


    
#%% voltage data
# photodiode
pulse_phd = f_get_pulse_times(volt_phd, median_filt_size=501, levels_use=[1,2], thresh_low_frac=0.1, thresh_high_frac=0.9, corr_on_off_shifts=[200, 1000])

# led pulse 
pulse_led = f_get_pulse_times(volt_LED, median_filt_size=31, levels_use=[0,1], thresh_low_frac=0.1, thresh_high_frac=0.9, corr_on_off_shifts=[50, 50])

# cam led on period
pulse_ca = f_get_pulse_times(cam_ave_ca, median_filt_size=31, levels_use=[0,1], thresh_low_frac=0.1, thresh_high_frac=0.5, corr_on_off_shifts=[50, 50])

if 0:
    ax1 = f_plot_pulse_times(pulse_phd, volt_time)
    ax1.set_title('photodiode')
    
    ax1 = f_plot_pulse_times(pulse_led, volt_time)
    ax1.set_title('LED')
    
    ax1 = f_plot_pulse_times(pulse_ca, cam_time)
    ax1.set_title('Calcium')
    
#%%
led_on_volt = volt_time[pulse_led['on_times']]
led_off_volt = volt_time[pulse_led['off_times']]
led_on_ca = cam_time[pulse_ca['on_times']]
led_off_ca = cam_time[pulse_ca['off_times']]

trial_on_times_volt = volt_time[pulse_phd['on_times']] - led_on_volt
trial_off_times_volt = volt_time[pulse_phd['off_times']] - led_on_volt
cam_time_exp = cam_time - led_on_ca

num_trials = len(trial_on_times_volt)

trial_on_idx_ca = np.zeros(num_trials, dtype=int)
trial_off_idx_ca = np.zeros(num_trials, dtype=int)
for n_tr in range(len(trial_on_times_volt)):
    trial_on_idx_ca[n_tr] = np.where(cam_time_exp > trial_on_times_volt[n_tr]/1000)[0][0]
    trial_off_idx_ca[n_tr] = np.where(cam_time_exp > trial_off_times_volt[n_tr]/1000)[0][0]

trial_on_times_ca = cam_time_exp[trial_on_idx_ca]
trial_off_times_ca = cam_time_exp[trial_off_idx_ca]

if 0:
    plt.figure()
    plt.plot(trial_on_times_volt/1000, trial_on_times_ca, '.')
    plt.plot(trial_off_times_volt/1000, trial_off_times_ca, '.')
    plt.legend(['on times', 'off times'])
    plt.title('trial on and off times')
    plt.xlabel('voltage time (s)')
    plt.ylabel('calcium time (s)')
    
    plt.figure()
    plt.plot((trial_off_times_volt - trial_on_times_volt)/1000, trial_off_times_ca-trial_on_times_ca, '.')
    plt.title('trial duration')
    plt.xlabel('voltage time (s)')
    plt.ylabel('calcium time (s)')

#%%

trial_type = np.array(stim_data['stimulation']['direction'])
trial_type_uq = np.unique(stim_data['stimulation']['direction'])

trial_data_2d = []
trial_data_mov = []

for n_tt in range(len(trial_type_uq)):
    trial_data2 = []
    trial_idx = trial_type_uq[n_tt] == trial_type
    
    trial_on_idx = trial_on_idx_ca[trial_idx]
    trial_off_idx = trial_off_idx_ca[trial_idx]
    
    max_len = np.min(trial_off_idx - trial_on_idx)
    
    trial_data2_2d = np.zeros((len(trial_on_idx), max_len))
    trial_data2_mov = np.zeros((len(trial_on_idx), max_len, d1, d2))
    for n_tr in range(len(trial_on_idx)):
        trial_data2_2d[n_tr,:] = cam_ave_ca[trial_on_idx[n_tr]:trial_on_idx[n_tr]+max_len]
        trial_data2_mov[n_tr,:,:,:] = cam_data[trial_on_idx[n_tr]:trial_on_idx[n_tr]+max_len,:,:]
    
    trial_data_2d.append(trial_data2_2d)
    trial_data_mov.append(trial_data2_mov)


trial_ave_mov = []

save_mov = 1
for n_tt in range(len(trial_type_uq)):
    num_t = trial_data_mov[n_tt].shape[1]
    
    trial_ave_mov2 = np.mean(trial_data_mov[n_tt], axis=0)
    trial_ave_mov.append(trial_ave_mov2)
    
    trial_ave_mov2_bs = trial_ave_mov2 - np.mean(trial_ave_mov2, axis=0)
    
    save_fname = 'test_%s_trial_ave_%s.tif' % (mov_fname[0], trial_type_uq[n_tt])
    
    save_fname_bs = 'test_%s_bs_trial_ave_%s.tif' % (mov_fname[0], trial_type_uq[n_tt])
    
    if save_mov:
        tf.imsave(os.path.join(save_dir, save_fname), trial_ave_mov2)
        tf.imsave(os.path.join(save_dir, save_fname_bs), trial_ave_mov2_bs)
    


if 0:
    plt.figure()
    plt.plot(trial_data_2d[3].T, color='gray')



#%% find starting frame

ca_thresh = f_get_sig_thresh(cam_ave_ca)
led_thresh = f_get_sig_thresh(volt_LED)
volt_frame_thresh = f_get_sig_thresh(volt_frames)

idx_vframe = volt_frames > volt_frame_thresh

frame_on_frames = (np.diff(idx_vframe.astype(int),prepend=0)>0.5)

np.sum(frame_on_frames)

frame_times_volt = volt_time[frame_on_frames]

plt.figure()
plt.plot(np.diff(frame_times_volt))


num_frames_volt = np.sum(frame_on_frames)

plt.figure()
plt.plot(np.diff(idx_vframe.astype(int))*0.8)
plt.plot(np.diff(idx_vframe.astype(int))>0.5)

if 0:
    ca_sort = np.sort(cam_ave_ca)
    plt.figure()
    plt.plot(ca_sort)
    plt.title('ecdf')

    plt.figure()
    plt.plot(cam_time, cam_ave_ca)
    plt.plot(cam_time, np.ones(len(cam_ave_ca))*ca_thresh)
    plt.title('cam ca signal thresh')
    
    plt.figure()
    plt.plot(volt_time, volt_LED)
    plt.plot(volt_time, np.ones(len(volt_time))*led_thresh)
    plt.title('voltage LED thresh')
    
    plt.figure()
    plt.plot(idx_vframe)
    


#%%

params = {
          'phaseMapFilterSigma': 0.5,
          'signMapFilterSigma': 8.,
          'signMapThr': 0.4,
          'eccMapFilterSigma': 15.0,
          'splitLocalMinCutStep': 5.,
          'closeIter': 3,
          'openIter': 3,
          'dilationIter': 15,
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1,
          'mergeOverlapThr': 0.1
          }



