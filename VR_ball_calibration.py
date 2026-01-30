# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:26:21 2025

@author: ys2605
"""

import sys
import os

path2 = ['C:/Users/ys2605',
         'C:/Users/shymk']

for path3 in path2:
    if os.path.isdir(path3):
        path1 = path3 + '/Desktop/stuff/VR';

sys.path.append(path1 + '/VR_analysis')

import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

#%%
dpath = path1 + '/UNITY/data_sens_calib'
#dpath = path1 + '/UNITY/data_out'

#ftag = '2025_9_23_11h_17m_40s_'
ftag = '2025_9_30_23h_32m_52s_'
#ftag = '2025_9_30_21h_34m_33s_'

fpath_sens_data = dpath + '/' + ftag + 'SensorData.csv'
fpath_sens_data2 = dpath + '/' + ftag + 'SensorData2.csv'
fpath_mov = dpath + '/' + ftag + 'MovementData.csv'

#%%

df_sens = pd.read_csv(fpath_sens_data)
df_sens2 = pd.read_csv(fpath_sens_data2)
df_mov = pd.read_csv(fpath_mov)

#%%
# plt.close('all')
sens_use = df_sens2

S00n = sens_use.S00[1:] - df_sens.S00[1]
S01n = sens_use.S01[1:] - df_sens.S01[1]
S10n = sens_use.S10[1:] - df_sens.S10[1]
S11n = sens_use.S11[1:] - df_sens.S11[1]



ds00 = np.diff(sens_use.S00)[1:]/1000
ds01 = np.diff(sens_use.S01)[1:]/1000
ds10 = np.diff(sens_use.S10)[1:]/1000
ds11 = np.diff(sens_use.S11)[1:]/1000

plt.figure()
plt.plot(ds00)
plt.plot(ds01)
plt.plot(ds10)
plt.plot(ds11)
plt.legend(['S00', 'S01', 'S10', 'S11'])

#%% computing motion

s00x = 0.1849
s00y = -0.8151
s00z = -0.5959

s01x = -0.2414
s01y = -0.2414
s01z = -0.7779

s10x = -0.8151
s10y = 0.1849
s10z = 0.5959

s11x = 0.2414
s11y = 0.2414
s11z = -0.7779

x_ball = ds00*s00x + ds01*s01x + ds10*s10x + ds11*s11x
y_ball = ds00*s00y + ds01*s01y + ds10*s10y + ds11*s11y
z_ball = ds00*s00z + ds01*s01z + ds10*s10z + ds11*s11z


plt.figure()
plt.plot(x_ball)
plt.plot(y_ball)
plt.plot(z_ball)
plt.legend(['x', 'y', 'z'])

plt.figure()
plt.plot(np.cumsum(x_ball))
plt.plot(np.cumsum(y_ball))
plt.plot(np.cumsum(z_ball))

plt.figure()
plt.plot(sens_use.convX[1:])
plt.plot(sens_use.convY[1:])
plt.plot(sens_use.convZ[1:])
plt.legend(['x', 'y', 'z'])


plt.figure()
plt.plot(-sens_use.dS00[1:])
plt.plot(sens_use.convY[1:])
plt.legend(['S00', 'y'])

#%%
per_list = [[6, 85],
            [87, 175],
            [182,214],
            [231, 371],
            [411,517],
            [553, 684],
            [718, 839]]

per_list = [[6, 839],
            [2680, 3127]]

plt.figure()
for n_per in range(len(per_list)):
    per1 = per_list[n_per]

    S00n2 = np.asarray(S00n[per1[0]:per1[1]] - S00n[per1[0]])
    S11n2 = np.asarray(S11n[per1[0]:per1[1]] - S11n[per1[0]])/0.766

    #plt.plot(S00n2)
    #plt.plot(S11n2)
    
    plt.plot(-S00n2/S11n2)

#%%
per_list = [[3988, 4087]]

plt.figure()
for n_per in range(len(per_list)):
    per1 = per_list[n_per]

    
    S01n2 = np.asarray(S01n[per1[0]:per1[1]] - S01n[per1[0]])
    S11n2 = np.asarray(S11n[per1[0]:per1[1]] - S11n[per1[0]])

    #plt.plot(S00n2)
    #plt.plot(S11n2)
    
    plt.plot(S01n2/S11n2)

#%%

plt.figure()
plt.plot(df_sens2['S00n'][1:])
plt.plot(df_sens2['S00'][1:])

plt.figure()
plt.plot(np.diff(df_sens2['S00n'][1:]))
plt.plot(df_sens2['dS00'][2:]*1000)

plt.figure()
plt.plot(np.diff(df_sens2.Time))

plt.figure()
plt.plot(np.diff(df_sens.Time))


plt.figure()
plt.plot(np.diff(df_mov.Time))


plt.figure()
plt.plot(df_mov.dT)


plt.figure()
plt.plot(np.cumsum(np.diff(df_sens2['S00n'][1:])))


plt.figure()
plt.plot(np.diff(df_sens2.Time))
plt.figure()
plt.plot(df_sens2.rowsPerRead)





