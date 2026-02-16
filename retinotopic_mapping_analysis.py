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
from skimage.restoration import unwrap_phase
import math

for user1 in ['ys2605', 'shymk']:
    if os.path.isdir('C:/Users/' + user1):
        path1 = 'C:/Users/' + user1 + '/Desktop/stuff/VR/VR_analysis/'

sys.path.append(path1)
sys.path.append(path1 + '/functions')

from f_function_retmap import f_get_sig_thresh, f_get_pulse_times, f_plot_pulse_times, f_gauss_smooth_mov
import NeuroAnalysisTools.RetinotopicMapping as rm

#%% need to generate trial data from recordings 

data_path = 'D:/VR/mapping/L/'
save_dir = 'F:/VR/data_proc/mapping/'

#mov_fname = 'L_VC_mapping00001_256.tif'
mov_fname = 'L_VC_mapping00001.tif'

pix_use = np.array([[45, -30], [35, -50]])/256      # for L

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


d0 = cam_data.shape[1]
if 1:
    cam_data = cam_data[:,round(pix_use[0][0]*d0):round(pix_use[0][1]*d0),round(pix_use[1][0]*d0):round(pix_use[1][1]*d0)]
    
if 0:
    plt.figure()
    plt.imshow(cam_data[1000,round(pix_use[0][0]*d0):round(pix_use[0][1]*d0),round(pix_use[1][0]*d0):round(pix_use[1][1]*d0)])

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

base_duration = 1  #sec
save_mov = 1

trial_type = np.array(stim_data['stimulation']['direction'])
trial_type_uq = np.unique(stim_data['stimulation']['direction'])

ca_dt = np.median(np.diff(cam_time_exp))
num_base_frames = np.floor(base_duration / ca_dt).astype(int)

trial_data_1d = []
trial_data_3d = []
if base_duration>0:
    trial_baseline_3d = []

for n_tt in range(len(trial_type_uq)):
    trial_data2 = []
    trial_idx = trial_type_uq[n_tt] == trial_type
    
    trial_on_idx = trial_on_idx_ca[trial_idx]
    trial_off_idx = trial_off_idx_ca[trial_idx]
    
    max_len = np.min(trial_off_idx - trial_on_idx)
    
    trial_data2_1d = np.zeros((len(trial_on_idx), max_len))
    trial_data2_3d = np.zeros((len(trial_on_idx), max_len, d1, d2))
    if base_duration>0:
        trial_baseline2_3d = np.zeros((len(trial_on_idx), num_base_frames, d1, d2))
    for n_tr in range(len(trial_on_idx)):
        trial_data2_1d[n_tr,:] = cam_ave_ca[trial_on_idx[n_tr]:trial_on_idx[n_tr]+max_len]
        trial_data2_3d[n_tr,:,:,:] = cam_data[trial_on_idx[n_tr]:trial_on_idx[n_tr]+max_len,:,:]
        if base_duration>0:
            trial_baseline2_3d[n_tr,:,:,:] = cam_data[trial_on_idx[n_tr]-num_base_frames:trial_on_idx[n_tr],:,:]
    
    trial_data_1d.append(trial_data2_1d)
    trial_data_3d.append(trial_data2_3d)
    if base_duration>0:
        trial_baseline_3d.append(trial_baseline2_3d)

trial_ave_mov = []
trial_ave_mov_bs = []
trial_ave_mov_bs_sm = []

for n_tt in range(len(trial_type_uq)):
    num_t = trial_data_3d[n_tt].shape[1]
    
    trial_ave_mov2 = np.mean(trial_data_3d[n_tt], axis=0)
    trial_ave_mov.append(trial_ave_mov2)
    trial_ave_baseline = np.mean(trial_baseline_3d[n_tt], axis=0)
    
    if num_base_frames>0:
        base1 = np.mean(trial_ave_baseline, axis=0)
    else:
        base1 = np.mean(trial_ave_mov2, axis=0)
        
    trial_ave_mov2_bs = trial_ave_mov2 - base1
    
    trial_ave_mov_bs.append(trial_ave_mov2_bs)
    
    trial_ave_mov2_bs_sm = f_gauss_smooth_mov(trial_ave_mov2_bs, sigma=[0,10,10])
    
    trial_ave_mov_bs_sm.append(trial_ave_mov2_bs_sm)
    
    ftag = '%s_trial_ave_%s_%dp' % (mov_fname[0], trial_type_uq[n_tt], d0)
    save_fname = '%s.tif' % (ftag)
    save_fname_bs = '%s_bs.tif' % (ftag)
    save_fname_bs_sm = '%s_bs_sm.tif' % (ftag)
    
    if save_mov:
        tf.imwrite(os.path.join(save_dir, save_fname), trial_ave_mov2)
        tf.imwrite(os.path.join(save_dir, save_fname_bs), trial_ave_mov2_bs)
        tf.imwrite(os.path.join(save_dir, save_fname_bs_sm), trial_ave_mov2_bs_sm)
    
if 0:
    plt.figure()
    plt.plot(trial_data_1d[3].T, color='gray')


#%% fft
plt.close('all')

amp_im = []
ang_im = []

amp_adj = [7/8*np.pi, np.pi, np.pi, 10/8*np.pi]

for n_tt in range(len(trial_type_uq)):
    #T = round(ca_dt,4)
    T = 1/trial_ave_mov_bs_sm[n_tt].shape[0]
    Fs = 1/T
    L = trial_ave_mov_bs_sm[n_tt].shape[0]
    
    xf = np.fft.fftfreq(L, T)
    
    yf = np.fft.fft(trial_ave_mov_bs_sm[n_tt], axis=0)
    yf_sh = np.fft.fftshift(yf, axes=0)
    
    yf2 = np.abs(yf)*np.exp(1j * (np.angle(yf)))
    yf2_sh = np.fft.fftshift(yf2, axes=0)

    for freq in [1]:

        yf2_freq_ang_uw = np.angle(yf2[freq,:,:])
        if 0:
            yf2_freq_ang_uw = unwrap_phase(yf2_freq_ang_uw)
        if 0:
            for n1 in range(d1):
                yf2_freq_ang_uw[n1,:] = np.unwrap(yf2_freq_ang_uw[n1,:])
        
        temp_amp = np.abs(yf[freq,:,:])
        temp_amp = temp_amp - np.min(temp_amp)
        temp_amp = temp_amp/np.max(temp_amp)
        
        temp_yf = yf[freq,:,:]
        #ang_mean = np.mean(np.angle(temp_yf)[temp_amp>0.5])
        ang_mean = 0
        temp_yf = np.abs(temp_yf)*np.exp(1j * (np.angle(temp_yf) - ang_mean + amp_adj[n_tt]))
        temp_yf_ang = np.angle(temp_yf)
        
        amp_im.append(temp_amp)    
        ang_im.append(temp_yf_ang)
        
        fig, (ax1, ax2) = plt.subplots(1,2,sharex=True, sharey=True)
        im1 = ax1.imshow(temp_amp)
        ax1.set_title('trial %s; amplitude at %.2f Hz' % (trial_type_uq[n_tt], xf[freq]))
        ax1.set_xlabel('clim %.2f to %.2f' % (im1.get_clim()[0], im1.get_clim()[1]))
        
        im2 = ax2.imshow(temp_yf_ang)
        ax2.set_title('trial %s; phase at %.2f Hz; ' % (trial_type_uq[n_tt], xf[freq]))
        ax2.set_xlabel('clim %.2f to %.2f pi' % (im2.get_clim()[0], im2.get_clim()[1]))
        
    if 0:
        
        yf1_sh = yf_sh[:,125,125]
        xf_sh = np.fft.fftshift(xf)
        
        plt.figure()
        plt.plot(xf_sh, np.abs(yf1_sh))
        
        plt.figure()
        plt.plot(np.angle(yf1_sh))
        
        plt.figure()
        plt.plot((np.angle(yf[1,125,:])))
        
        n_x = 238
        plt.figure()
        plt.plot(np.unwrap(np.angle(yf[1,n_x,:])))
        
        plt.figure()
        plt.plot(np.angle(yf2[1,n_x,:]))

#%% clean up images with 2d fft (not working so well)

if 0:
    im_in = ang_im[0]
    
    freq_thresh = 0.01
    
    pix_include = np.zeros((d1,d2), dtype=bool)
    
    f_transform = np.fft.fft2(im_in)
    
    freq_1 = np.fft.fftfreq(d1, d=256/d1)
    freq_2 = np.fft.fftfreq(d2, d=256/d2)
    
    num_vals = 0
    f_transform2 = f_transform.copy()
    for n_d1 in range(d1):
        for n_d2 in range(d2):
            if not np.abs(freq_1[n_d1]) < freq_thresh or not np.abs(freq_2[n_d2]) < freq_thresh:
                f_transform2[n_d1, n_d2] = 0
            else:
                pix_include[n_d1, n_d2] = 1
                num_vals+=1
            
    
    
    im_back = np.fft.ifft2(f_transform2)
    
    if 0:
        plt.figure()
        plt.imshow(amp_im[0])
        
        plt.figure()
        plt.imshow(np.abs(im_back))
        
        plt.figure()
        plt.imshow(np.abs(f_transform))
        
        plt.figure()
        plt.imshow(pix_include)
    
        plt.figure()
        plt.plot(np.abs(f_transform)[0,:])


#%% try removing structure

n_tt = 0

base1 = np.mean(np.mean(trial_baseline_3d[n_tt], axis=0), axis=0)

trial_ave1 = np.mean(trial_data_3d[n_tt], axis=0)



trial_ave1

gradmap1 = np.gradient(trial_ave1[0])



temp_amp = ang_im[n_tt]



if 0:
    plt.figure()
    plt.imshow(base1)
    
    plt.figure()
    plt.imshow(temp_amp)
    
    frs = [10, 50, 100, 150]
    pix = 60
    
    plt.figure()
    for fr1 in frs:
        plt.plot(trial_ave1[fr1,pix,:])
    plt.plot(base1[pix,:])
    plt.legend(['frame %d' % frame2 for frame2 in frs] + ['base'])
    
    plt.figure()
    for fr1 in frs:
        plt.plot(trial_ave1[fr1,pix,:] - base1[pix,:])
    plt.legend(['frame %d' % frame2 for frame2 in frs])
    
    plt.figure()
    plt.imshow(trial_ave1[0])
    
    plt.figure()
    plt.imshow(gradmap1[0])
    plt.figure()
    plt.imshow(gradmap1[1])


plt.figure()
plt.imshow(np.diff(trial_ave1[0]))

plt.figure()
plt.imshow(np.cumsum(np.diff(trial_ave1[0]), axis=1))

plt.figure()
plt.imshow(np.cumsum(gradmap1[1], axis=1))


y = np.array([1, 2, 4, 7, 11, 16], dtype=float)
j = np.gradient(y)



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
    


#%% demo part

if 1:
    # trial_type_uq
    vasculature_map = np.mean(cam_data[1000:-1000,:,:], axis=0)

    altitude_map = ang_im[3]
    azimuth_map = ang_im[2]
    altitude_power_map = amp_im[3]
    azimuth_power_map = amp_im[2]
else:
    demo_path = 'C:/Users/ys2605/Desktop/stuff/python_dependencies/NeuroAnalysisTools-master/NeuroAnalysisTools/test/data/'
    
    vasculature_map = tf.imread(os.path.join(demo_path, 'example_vasculature_map.tif'))
    
    altitude_map = tf.imread(os.path.join(demo_path, 'example_altitude_map.tif'))
    azimuth_map = tf.imread(os.path.join(demo_path, 'example_azimuth_map.tif'))
    altitude_power_map = tf.imread(os.path.join(demo_path, 'example_altitude_power_map.tif'))
    azimuth_power_map = tf.imread(os.path.join(demo_path, 'example_azimuth_power_map.tif'))
    
    
if 0:
    plt.figure()
    plt.imshow(vasculature_map)
    plt.title('vasculature_map')
    
    fig, axs  = plt.subplots(2,2)
    
    axs[0,0].imshow(altitude_map)
    axs[0,0].set_title('altitude_map')
    
    axs[0,1].imshow(azimuth_map)
    axs[0,1].set_title('altitude_map')
    
    axs[1,0].imshow(altitude_power_map)
    axs[1,0].set_title('altitude_power_map')

    axs[1,1].imshow(azimuth_power_map)
    axs[1,1].set_title('azimuth_power_map')

#%%

if 0:
    gradmap1 = np.gradient(altitude_map)
    gradmap2 = np.gradient(azimuth_map)

    # gradmap1 = ni.filters.median_filter(gradmap1,100.)
    # gradmap2 = ni.filters.median_filter(gradmap2,100.)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    # gradmag1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))
    # gradmag2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(altitude_map.shape[0]):
        for j in range(azimuth_map.shape[1]):
            graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])

            # gradmag1[i,j] = np.sqrt((gradmap1[1][i,j]**2)+(gradmap1[0][i,j]**2))
            # gradmag2[i,j] = np.sqrt((gradmap2[1][i,j]**2)+(gradmap2[0][i,j]**2))

    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))
    
    plt.figure()
    plt.imshow(gradmap1[1])
    
    plt.figure()
    plt.imshow(areamap)


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

#%%



trial = rm.RetinotopicMappingTrial(altPosMap=altitude_map,
                                   aziPosMap=azimuth_map,
                                   altPowerMap=altitude_power_map,
                                   aziPowerMap=azimuth_power_map,
                                   vasculatureMap=vasculature_map,
                                   mouseID='test',
                                   dateRecorded='160612',
                                   comments='This is an example.',
                                   params=params)


_ = trial._getSignMap(isPlot=True)
plt.show()

