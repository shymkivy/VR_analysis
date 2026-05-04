# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:32:23 2025

@author: ys2605
"""

import os
import h5py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio
import pandas as pd

from scipy.sparse import csr_matrix

#%%
def f_get_fnames_from_dir(dir_path, ext_list = [], tags = None):

    f_list = os.listdir(dir_path)
    f_list2 = []
    for fil1 in f_list:
        if len(ext_list):
            for ext1 in ext_list:
                if fil1.endswith(ext1):
                    f_list2.append(fil1)
        else:
            f_list2.append(fil1)
    
    if tags is not None:
        if type(tags) is str:
            tags = [tags]
        
        f_list_out = []
        for fil1 in f_list2:
            has_tag = True
            for tag in tags:
                if tag not in fil1:
                    has_tag = False
            if has_tag:
                f_list_out.append(fil1)
            
    else:
        f_list_out = f_list2
    
    return f_list_out


#%%

def f_load_caim_data_mat(data_dir, flist, data_tag = 'results_cnmf_sort.mat', proc_tag = 'processed_data.mat', deconvolution='oasis', smooth_std_duration=0.1):
    
    # deconvolution methods are either oasis (caiman default) or smoothdftd - smoothed, rectified first derivative
    
    num_files = len(flist)
    data_out = []
    
    for n_fl in range(num_files):

        fname_core = flist[n_fl]
        if data_tag in fname_core:
            fname_core = fname_core.removesuffix(data_tag)
        if proc_tag in fname_core:
            fname_core = fname_core.removesuffix(proc_tag)
        
        flist_data = f_get_fnames_from_dir(data_dir, ext_list = ['.mat'], tags = [fname_core, data_tag])
        flist_proc = f_get_fnames_from_dir(data_dir, ext_list = ['.mat'], tags = [fname_core, proc_tag])
        
        do_load = False
        if len(flist_proc):
            if len(flist_data):
                do_load = True
            else:
                print(fname_core + " data file with " + data_tag +  " tag not found, skipping")
        else:
            print(fname_core + " proc file with " + proc_tag +  " tag not found, skipping")
        
        if do_load:
            
            data_slice = {'flist_data':       flist_data,
                          'flist_proc':       flist_proc}
            
            f_proc = h5py.File(data_dir + '/' + flist_proc[0], 'r')
            vid_cuts_trace = f_proc[f_proc['data']['file_cuts_params'][0][0]]['vid_cuts_trace'][()].flatten().astype(bool)
            trial_types = f_proc['data']['trial_types'][()].flatten().astype(int)
            stim_times = f_proc[f_proc['data']['stim_times_frame'][0][0]][()].flatten().astype(int)
            
            if 'volume_period' in f_proc['data']['frame_data'].keys():
                data_slice['volume_period'] = f_proc['data']['frame_data']['volume_period'][()].flatten()[0]
            if 'isi' in f_proc['data']['stim_params'].keys():
                data_slice['isi'] = f_proc['data']['stim_params']['isi'][()].flatten()[0]
            if 'MMN_orientations' in f_proc['data'].keys():
                data_slice['MMN_ori'] = f_proc['data']['MMN_orientations'][()].flatten().astype(int)
            if 'MMN_freq' in f_proc['data']['stim_params'].keys():
                data_slice['MMN_ori'] = f_proc['data']['stim_params']['MMN_freq'][()].flatten().astype(int)
    
            f_proc.close()
            
            firing_rates_all = []
            dset_idx_all = []
            
            for n_fl in range(len(flist_data)):
                fname_data = flist_data[n_fl]
                
                f = h5py.File(os.path.join(data_dir, fname_data), 'r')
            
                d_est = f['est']
                d_proc = f['proc']
                
                comp_acc = d_proc['comp_accepted'][()].flatten().astype(bool)
                
                if deconvolution == 'oasis':
                    firing_rates_cut = d_est['S'][()][:,comp_acc].T
    
                elif deconvolution == 'smoothdfdt':
                    C = d_est['C'][()]
                    YrA = d_est['YrA'][()]
                    ca_traces_cut = (C + YrA)[:,comp_acc].T
                    firing_rates_cut = f_smooth_dfdt(ca_traces_cut, sigma_frames=1000/data_slice['volume_period']*0.1, do_smooth=True)
                
                firing_rates_cut = f_gauss_smooth(firing_rates_cut, sigma_frames=1000/data_slice['volume_period']*smooth_std_duration)
                
                peak_rate = np.max(firing_rates_cut, axis=1)[:,None]
                firing_rates_cutn = firing_rates_cut/peak_rate
                
                firing_rates = np.zeros((firing_rates_cutn.shape[0], vid_cuts_trace.shape[0]))
                firing_rates[:, vid_cuts_trace] = firing_rates_cutn
                
                firing_rates_all.append(firing_rates)
                dset_idx_all.append(np.ones(firing_rates.shape[0], dtype=int)*(n_fl))
                f.close()
            
            firing_rates = np.vstack(firing_rates_all)
            dset_idx = np.hstack(dset_idx_all)
            
            data_slice['firing_rates'] = firing_rates
            data_slice['trial_types'] = trial_types
            data_slice['stim_times'] = stim_times
            data_slice['vid_cuts_trace'] = vid_cuts_trace
            data_slice['files_loaded'] = do_load
            data_slice['dset_idx'] = dset_idx
        
            data_out.append(data_slice)
        
    return data_out

def f_load_caim_data(data_dir, dset_names, caiman_tag = 'results_cnmf', cuts_tag=None, frame_data_tag = None, r_values_min = 0.5, min_SNR=1.5, thresh_cnn_min=0.8):
    num_files = len(dset_names)
    data_out = []
    
    make_cuts = 1
    
    for n_fl in range(num_files):    # num_files
        
        flist = f_get_fnames_from_dir(data_dir, tags=[dset_names[n_fl], caiman_tag])
        f = h5py.File(os.path.join(data_dir, flist[0]), 'r')
    
        dims = f['dims'][()]
    
        est1 = f_h5_load_group(f['estimates'], keys=['C', 'S', 'YrA', 'SNR_comp', 'r_values', 'cnn_preds', 'idx_components', 'idx_components_bad'])
        
        A_data = f_h5_load_group(f['estimates']['A'])
        est1['A'] = csr_matrix((A_data['data'], A_data['indices'], A_data['indptr']), shape=(A_data['shape'][1], A_data['shape'][0]))
        est1['dims'] = dims
        
        bool_components = np.ones(est1['C'].shape[0], dtype=bool)
        
        bool_components_r = est1['r_values'] >= r_values_min
        bool_components_snr = est1['SNR_comp'] > min_SNR
        bool_components_cnn = est1['cnn_preds'] >= thresh_cnn_min
        
        bool_components = np.logical_and(bool_components, bool_components_r)
        bool_components = np.logical_and(bool_components, bool_components_snr)
        bool_components = np.logical_and(bool_components, bool_components_cnn)
        
        idx_components = np.where(bool_components)[0]
        idx_components_bad = np.where(~bool_components)[0]
        
        # select good components
        est2 = est1.copy()
        for key1 in ['C', 'S', 'YrA', 'SNR_comp', 'r_values', 'cnn_preds', 'A']:
            if len(est1[key1].shape) > 1:
                est2[key1] = est1[key1][idx_components,:]
            else:
                est2[key1] = est1[key1][idx_components]
        
        est2['idx_components'] = idx_components
        est2['idx_components_bad'] = idx_components_bad
        
        #A2 = est1['A'].toarray()
        
        if frame_data_tag is not None:
            flist_frame_data = f_get_fnames_from_dir(data_dir, tags = [dset_names[n_fl], frame_data_tag])
            frame_data = pd.read_csv(os.path.join(data_dir, flist_frame_data[0]))
            frame_times = frame_data['relativeTimes'].to_numpy()
        else:
            frame_times = np.arange(est2['C'].shape[1])
        
        est2['frame_times'] = frame_times
        
        use_cuts = False
        if cuts_tag is not None:
            flist_cuts = f_get_fnames_from_dir(data_dir, tags = [dset_names[n_fl], cuts_tag])
            if len(flist_cuts):
                fpath = os.path.join(data_dir, flist_cuts[0])
                major_version, minor_version = sio.matlab.matfile_version(fpath)
                
                try:
                    if major_version == 2:
                        f_cuts = h5py.File(fpath, 'r')
                        if not make_cuts:
                            vid_cuts_trace = f_cuts['params']['cuts_data']['vid_cuts_trace'][()].flatten().astype(bool)
                        vid_cuts = f_cuts['params']['cuts_data']['vid_cuts'][()]
                        use_cuts = True
                    else:
                        data = sio.loadmat(fpath)
                        if not make_cuts:
                            vid_cuts_trace = data['params'][0,0]['cuts_data']['vid_cuts_trace'][0,0].flatten().astype(bool)
                        vid_cuts = data['params'][0,0]['cuts_data']['vid_cuts'][0][0]
                        use_cuts = True
                except:
                    print('error trying to load cuts trace, ignored')
        
        if use_cuts:
            if make_cuts:
                vid_cuts_trace = np.zeros(vid_cuts[-1][-1], dtype=bool)
                for cut in vid_cuts:
                    vid_cuts_trace[cut[0]-1:cut[1]] = True
            for key1 in ['C', 'S', 'YrA']:
                    trace_full = np.zeros((est2[key1].shape[0], vid_cuts_trace.shape[0]))
                    trace_full[:,vid_cuts_trace] = est2[key1]
                    est2[key1] = trace_full
                        
            est2['vid_cuts_trace'] = vid_cuts_trace
            est2['vid_cuts'] = vid_cuts
            
            
        est2['dset_name'] = dset_names[n_fl]
         
        data_out.append(est2)
            
    return data_out

def f_h5_load_group(group, keys=None):
    if keys is None:
        keys = group.keys()
        
    data = {}
    for key1 in keys:
        data[key1] = group[key1][()]
    
    return data

def f_get_values(data_out, key):
    
    values = []
    
    for data_slice in data_out:
        if key in data_slice.keys():
            values.append(data_slice[key])
            
    return values

def f_smooth_dfdt(data, do_smooth=True, sigma_frames=1, rectify=True, normalize=True):
    
    num_cells, num_frames = data.shape
    
    firing_rates = np.zeros((num_cells, num_frames));
    
    if sigma_frames == 0:
        do_smooth=False
    
    if do_smooth:
        s_fr = np.ceil(sigma_frames).astype(int)
        x = np.linspace(-3*s_fr, 3*s_fr, s_fr*6+1)
        gauss_kernel = np.exp(-x**2 / (2 * sigma_frames**2))
    
    for n_cell in range(num_cells):
        temp_data = np.diff(data[n_cell,:], prepend=0)
        
        if do_smooth:
            temp_data = np.convolve(temp_data, gauss_kernel, mode='same')
        
        if rectify:
            temp_data = np.maximum(temp_data, 0)
        
        if normalize:
            temp_data = temp_data - np.mean(temp_data)
            temp_data = temp_data/np.max(temp_data)
            
        firing_rates[n_cell,:] = temp_data;
    
    return firing_rates

def f_gauss_smooth(firing_rates, sigma_frames=1):
    
    if sigma_frames:
        num_cells, num_frames = firing_rates.shape
        
        s_fr = np.ceil(sigma_frames).astype(int)
        x = np.linspace(-3*s_fr, 3*s_fr, s_fr*6+1)
        gauss_kernel = np.exp(-x**2 / (2 * sigma_frames**2))
        
        firing_rates_sm = np.zeros((num_cells, num_frames));
        for n_cell in range(num_cells):
            firing_rates_sm[n_cell,:] = np.convolve(firing_rates[n_cell,:], gauss_kernel, mode='same')
    else:
        firing_rates_sm = firing_rates
    
    return firing_rates_sm


def f_get_frames(trial_win = [-0.05, .95], frame_rate = 30):
    # anchor at 0
    frame_start = np.ceil(trial_win[0] * frame_rate)
    frame_end = np.ceil(trial_win[1] * frame_rate)
    trial_frames = [int(frame_start), int(frame_end)]
    plot_t = np.round(np.arange(frame_start/frame_rate, frame_end/frame_rate, 1/frame_rate), decimals=4)
    return trial_frames, plot_t

def f_get_stim_trig_resp(firing_rates, stim_times, trial_frames = [-29, 85]):
    # input: cells x time
    
    num_cells = firing_rates.shape[0]
    num_trials = len(stim_times)
    
    win_size = trial_frames[1] - trial_frames[0]
    
    stim_trig_resp = np.zeros((num_cells, win_size, num_trials))
    
    for n_tr in range(num_trials):
        cur_frame = round(stim_times[n_tr]-1) # correct for matlab to python 
        stim_trig_resp[:,:,n_tr] = firing_rates[:,(cur_frame+trial_frames[0]):(cur_frame+trial_frames[1])]
    
    return stim_trig_resp
    
def f_save_fig(fig, path='/', name_tag=''):
    
    plt.rcParams['svg.fonttype'] = 'none'
    name1 = fig.axes[0].title.get_text()
    now1 = datetime.now()
    
    date_tag = '%d_%d_%d_%dh_%dm' % (now1.year, now1.month, now1.day, now1.hour, now1.minute)
    
    fig.savefig('%s/%s_%s%s.svg' % (path, name1, date_tag, name_tag))
    fig.savefig('%s/%s_%s%s.png' % (path, name1, date_tag, name_tag), dpi=1200)


def f_get_trial_peak(trial_ave, peak_size=3):
    num_cells, num_bins = trial_ave.shape
    
    pad_left = np.floor((peak_size-1)/2).astype(int)
    pad_right = np.ceil((peak_size-1)/2).astype(int)
    
    peak_locs = np.argmax(trial_ave,axis=1)
    
    peak_start = peak_locs - pad_left
    peak_end = peak_locs + pad_right + 1
    
    idx_fix_start = peak_start < 0
    if sum(idx_fix_start):
        peak_start_to_fix = peak_start[idx_fix_start]
        peak_start[idx_fix_start] = peak_start[idx_fix_start] - peak_start_to_fix
        peak_end[idx_fix_start] = peak_end[idx_fix_start] - peak_start_to_fix
    
    idx_fix_end = peak_end > num_bins
    if sum(idx_fix_end):
        peak_end_to_fix = peak_end[idx_fix_end]
        peak_end[idx_fix_end] = peak_end[idx_fix_end] - peak_end_to_fix + num_bins
        peak_start[idx_fix_end] = peak_start[idx_fix_end] - peak_end_to_fix + num_bins
    
    peak_vals = np.zeros(num_cells)
    for n_cell in range(num_cells):
        peak_vals[n_cell] = np.mean(trial_ave[n_cell,peak_start[n_cell]:peak_end[n_cell]])
    
    return peak_vals, peak_locs


def f_compute_tuning(stim_trig_resp, trial_types, trials_analyze, plot_t, num_samp=2000, z_thresh = 3, sig_resp_win = [0, 1.5]):
    
    tt_use_idx = np.sum(trial_types == trials_analyze[:,None],axis=0).astype(bool)
    trial_types_use = trial_types[tt_use_idx]
    stim_trig_resp_use = stim_trig_resp[:,:,tt_use_idx]
    
    num_cells, _, num_trials = stim_trig_resp_use.shape
    num_tt = len(trials_analyze)
    
    # get data
    peak_vals = np.full((num_cells, num_tt), np.nan)
    peak_locs = np.full((num_cells, num_tt), np.nan)
    for n_tt in range(num_tt):
        tt1_idx = trial_types_use == trials_analyze[n_tt]
        if len(tt1_idx):
            trial_ave1 = np.mean(stim_trig_resp_use[:,:,tt1_idx], axis=2)
            peak_vals[:,n_tt], peak_locs[:,n_tt] = f_get_trial_peak(trial_ave1, peak_size=3)
    
    # make shuffled dist
    trials_per_stim = np.zeros(num_tt, dtype=int)
    for n_tt in range(num_tt):
        trials_per_stim[n_tt] = np.sum(trial_types_use == trials_analyze[n_tt])
    trials_per_stim_ave = np.round(np.mean(trials_per_stim)).astype(int)
    
    samp_peak_vals = np.full((num_cells, num_samp), np.nan)
    samp_peak_locs = np.full((num_cells, num_samp), np.nan)
    rng = np.random.default_rng()
    for n_cell in range(num_cells):
        random_integers = rng.integers(low=0, high=num_trials, size=(trials_per_stim_ave, num_samp))
        samp_trial_ave = np.mean(stim_trig_resp_use[n_cell,:,random_integers], axis=0)
        samp_peak_vals[n_cell,:], samp_peak_locs[n_cell,:] = f_get_trial_peak(samp_trial_ave, peak_size=3)

    idx1 = ~np.isnan(peak_locs[0,:])
    peak_locs_t = np.full((num_cells, num_tt), np.nan)
    peak_locs_t[:,idx1] = plot_t[peak_locs[:,idx1].astype(int)]
    
    peak_in_resp_win = np.logical_and(peak_locs_t >= sig_resp_win[0], peak_locs_t <= sig_resp_win[1])
    
    peak_prcntle = norm.cdf(z_thresh)*100
    prc_thresh = np.percentile(samp_peak_vals, peak_prcntle, axis=1)
    resp_cells_peak = np.zeros((num_cells, num_tt), dtype=bool)
    
    resp_cells_peak[:,idx1] = np.logical_and(peak_vals[:,idx1] > prc_thresh[:,None], peak_in_resp_win[:,idx1])
    
    return resp_cells_peak

