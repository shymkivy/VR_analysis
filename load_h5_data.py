# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:12:10 2026

@author: ys2605
"""
import os
import h5py
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

#%%

file_path = 'D:\VR\data_proc\L\movies\\'.replace("\\", "/")

fname = 'L_10_21_25_im1.h5'

num_frames = 1000
    
with h5py.File(file_path + fname, 'r') as f:
    # Access only the first frame (fast)
    data = f['mov'][:num_frames, :, :]
    
    
if 0:
    tf.imwrite(os.path.join(file_path, fname[:-3] + '.tif'), np.mean(data, axis=0).T)    

#%%

plt.figure()
plt.imshow()







