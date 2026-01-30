# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:13:48 2026

@author: ys2605
"""

import numpy as np
from scipy.cluster.vq import kmeans
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

#%%
def f_normalize(trace):
    trace = trace - np.min(trace)
    trace = trace/np.max(trace)
    return trace

def f_gauss_smooth(trace, sigma=1):
    radius = int(np.ceil(sigma * 4)) # Truncate at 4*sigma
    ax = np.arange(-radius, radius + 1., dtype=np.float32)
    kernel = np.exp(-(ax**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    trace_sm = np.convolve(trace, kernel, mode='same')
    
    return trace_sm

def f_get_sig_thresh(signal, thresh_prc = .2, off_signal_frac = 0.01):
    # uses percentile to get thresh
    base_ca = np.percentile(signal, off_signal_frac)
    sig_ca = np.median(signal)
    thresh = base_ca + (sig_ca - base_ca)*thresh_prc
    return thresh

def f_get_levels(trace, num_levels = 2):
    range1 = np.max(trace) - np.min(trace)
    levels = []
    trace2 = trace.copy()
    for n_lv in range(num_levels):
        med1 = np.median(trace2)
        levels.append(med1)
        idx_med1 = np.abs(trace2 - med1) < range1/10
        trace2 = trace2[~idx_med1]
    return levels

def f_get_levels_kmeans(trace, num_levels=2):
    km = kmeans(trace, num_levels)
    return km[0]

def f_rolling_correlation(trace1, trace2, shifts=200, return_full_corr = False):
    sh_vals = np.arange(-shifts, shifts+1, 1)
    corr_out = np.zeros(shifts*2+1)
    for n_t in range(len(sh_vals)):
        trace2_sh = np.roll(trace2, sh_vals[n_t])
        corr_out[n_t] = np.correlate(trace1, trace2_sh)
        
    idx_corr = np.argmax(corr_out)
    if return_full_corr:
        return sh_vals[idx_corr], corr_out
    else:
        return sh_vals[idx_corr]

def f_clean_stim_times(low_crossing, high_crossing, correlation_shifts=200, remove_outlier=True, outlier_thresh_factor=10):
    # when getting stim onset pulses, use low thresh (10%) and high thresh
    # (90%) crossing to verify stim on times, and then clean up low crossings
    
    # claning up on times with low thresh by making sure it crossed high thresh within given delay
    shift, corr_trace = f_rolling_correlation(low_crossing, high_crossing, shifts=correlation_shifts, return_full_corr = True)
    if sum(corr_trace):
    
        if remove_outlier:
            corr_locs = np.where(corr_trace>0)[0]
            corr_cent = np.median(corr_locs)
            corr_disp = np.abs(corr_locs - corr_cent)
            med_disp = np.median(corr_disp)
            bad_locs = corr_locs[corr_disp > outlier_thresh_factor*med_disp]
            corr_trace[bad_locs] = 0
            
        high_crossing_conv = np.convolve(high_crossing, corr_trace, mode='same')
        low_crossing_clean = (low_crossing*high_crossing_conv).astype(bool)
        
        return low_crossing_clean
    else:
        print('no corralation within given range, try higher range')
        return 0


def f_get_pulse_times(trace, median_filt_size=31, levels_use=[0,1], thresh_low_frac=0.1, thresh_high_frac=0.9, corr_on_off_shifts=[200, 200]):
    # levels are from lowest to hights magnitude
    
    trace_n = f_normalize(trace)
    trace_n2 = median_filter(trace_n, size=median_filt_size)
    # geting 3 different levels pf photodiode intensity (off, stim off, stim on) and set thresholds between middle and top
    levels = np.sort(np.array(f_get_levels(trace_n2, num_levels = np.max(levels_use)+1)))
    #levels2 = f_get_levels_kmeans(volt_phd2, num_levels = 3)
    
    levels2 = levels[np.array(levels_use)]

    thresh_low = np.min(levels2) + (np.max(levels2) - np.min(levels2))*thresh_low_frac
    thresh_high = np.min(levels2) + (np.max(levels2) - np.min(levels2))*thresh_high_frac
    
    on_times_low = np.hstack((0, np.diff((trace_n2 > thresh_low).astype(int))>0))
    on_times_high = np.hstack((0, np.diff((trace_n2 > thresh_high).astype(int))>0))
    on_times_low_clean = f_clean_stim_times(on_times_low, on_times_high, correlation_shifts=corr_on_off_shifts[0], remove_outlier=True, outlier_thresh_factor=10)
    
    off_times_low = np.hstack((0, np.diff((trace_n2 < thresh_low).astype(int))>0))
    off_times_high = np.hstack((0, np.diff((trace_n2 < thresh_high).astype(int))>0))
    off_times_high_clean = f_clean_stim_times(off_times_high, off_times_low, correlation_shifts=corr_on_off_shifts[1], remove_outlier=True, outlier_thresh_factor=10)

    data_out = {'trace_n':              trace_n,
                'on_times':             on_times_low_clean,
                'off_times':            off_times_high_clean,
                'thresh_low_frac':      thresh_low_frac,
                'thresh_high_frac':     thresh_high_frac,
                'thresh_low':           thresh_low,
                'thresh_high':          thresh_high,
                'levels_use':           levels_use,
                'levels':               levels,
                'median_filt_size':     median_filt_size,
                }
    return data_out

def f_plot_pulse_times(pulse_data, trace_time):

    len_trace = len(pulse_data['trace_n'])
    
    fig, ax1 = plt.subplots(1,1)
    pl1 = ax1.plot(trace_time, pulse_data['trace_n'])
    for nc in range(len(pulse_data['levels'])):
        pl2 = ax1.plot(trace_time, np.ones(len_trace)*pulse_data['levels'][nc], color='k')
    pl3 = ax1.plot(trace_time, np.ones(len_trace)*pulse_data['thresh_low'], color='r')
    ax1.plot(trace_time, np.ones(len_trace)*pulse_data['thresh_high'], color='r')
    pl4 = ax1.plot(trace_time, pulse_data['on_times'])
    pl5 = ax1.plot(trace_time, pulse_data['off_times'])
    ax1.legend(pl1+ pl2+ pl3+ pl4+ pl5, ['trace', 'levels', 'thresh', 'on times', 'off times'])
    
    return ax1

