# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:51:24 2025

@author: ys2605
"""

import sys
import os

path2 = ['C:/Users/ys2605/Desktop/stuff/VR/VR_analysis']

for path3 in path2:
    if os.path.isdir(path3):
        path1 = path3;

sys.path.append(path1);

import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt



from f_functions import load_XML, if_spheric_to_cart, if_cart_to_spheric #, f_pickle_load

#%%
fpath = 'C:\\Users\\ys2605\\Desktop\\stuff\\VR\\UNITY\\data_out2'


fname_xml = 'Terrain3data_2025_3_17_11h_57m_58s.xml'

fname_loc = 'tracking_2025_3_13_11h_46m_0s.csv'
fname_vert = 'MeshVerticesTerrain3_2025_3_13_11h_45m_52s.csv'
fname_obj = 'MeshVerticesTerrain3_objectList_2025_3_13_11h_45m_52s.csv'

mesh_y_offset = -39.6

df = pd.read_csv(fpath + '\\' + fname_loc)

#df_vert = pd.read_csv(fpath + '\\' + fname_vert)
#df_obj = pd.read_csv(fpath + '\\' + fname_obj)

#df2 = df.loc[~pd.isnull(df.data_in0)]
df2 = df

#df2.y_pos = df2.y_pos - mesh_y_offset

xml_data = load_XML(fpath + '\\' + fname_xml)


#%%

cum_d = df2.data_in0 - df2.data_in0[sum(pd.isnull(df.data_in0))]
cum_ds = sc.ndimage.gaussian_filter1d(cum_d, 10)
diff_ds= np.diff(cum_ds, prepend=0)

vel = df2.data_in1 - df2.data_in1[sum(pd.isnull(df.data_in0))]
cumvel = np.cumsum(vel)
vel_s = sc.ndimage.gaussian_filter1d(vel, 5)
cumvel_s = np.cumsum(vel_s)

plt.close('all')

plt.figure()
plt.plot(df2.Time, cum_d/np.std(cum_d))
plt.plot(df2.Time, cum_ds/np.std(cum_d))
plt.plot(df2.Time, cumvel/np.std(cumvel))
#plt.plot(df2.Time, cumvel_s/np.std(cumvel_s))
plt.title('DAQ distance data')
plt.legend(['dist volt', 'dist volt sm', 'rec dist'])

plt.figure()
plt.plot(df2.Time, vel/np.std(vel))
plt.plot(df2.Time, vel_s/np.std(vel_s))
plt.plot(df2.Time, diff_ds/np.std(diff_ds))
plt.title('DAQ velocity data')
plt.legend(['vel volt', 'vel volt sm', 'rec vel'])

plt.figure()
plt.plot(df2.Time, df2.x_pos - df.x_pos[0])
plt.plot(df2.Time, df2.y_pos - df.y_pos[0])
plt.plot(df2.Time, df2.z_pos - df.z_pos[0])
plt.title('xyz positions')

plt.figure()
plt.plot(df2.Time, df2.x_rot_q - df.x_rot_q[0])
plt.plot(df2.Time, df2.y_rot_q - df.y_rot_q[0])
plt.plot(df2.Time, df2.z_rot_q - df.z_rot_q[0])
plt.plot(df2.Time, df2.w_rot_q - df.w_rot_q[0])
plt.title('xyzw rotations')


#%% plotting land
# rotation coords quaternion and degrees (0-360)
# y++ clockwise turn (yaw) phi  ***
# z++ counterclockwise roll
# x++ facing down (pitch) theta 

mesh_x = np.asarray(xml_data['meshVert']['x'])
mesh_y = np.asarray(xml_data['meshVert']['y']) + mesh_y_offset
mesh_z = np.asarray(xml_data['meshVert']['z'])


plt.close('all')

cam_aspect = 16/9
cam_clip_len = 50
FOV_deg = 60

max_x = max(mesh_x)+1
max_z = max(mesh_z)+1
FOV_rad = FOV_deg/360*2*np.pi
num_pts = len(df)


d = cam_clip_len
h = d/np.cos(FOV_rad/2)
w = h*np.sin(FOV_rad/2)
h_adj = np.sqrt((w*cam_aspect)**2 + d**2)
FOV_rad_adj = np.asin((w*cam_aspect)/(h_adj))*2

z = cam_clip_len
h = z/np.cos(FOV_rad/2)
x = np.sin(FOV_rad/2)*h

#rho = pd.Series(np.ones(num_pts), name='rho')
phi = df.y_rot_eu/360*2*np.pi   # direction mouse is facing
theta = df.x_rot_eu/360*2*np.pi+np.pi/2  

rot_vec = if_spheric_to_cart(phi, theta)
rot_vec_r = if_spheric_to_cart(phi+FOV_rad_adj/2, theta)
rot_vec_l = if_spheric_to_cart(phi-FOV_rad_adj/2, theta)

spher_vec = if_cart_to_spheric(rot_vec.x, rot_vec.y, rot_vec.z)

mouse_pos_xyz = (pd.concat([df.x_pos, df.y_pos, df.z_pos], axis=1))     # df.x_pos.iloc[n_pt]


surf = np.reshape(mesh_y, shape=[round(max_x), round(max_z)])

fig1 = plt.figure()
plt.imshow(surf)
ax1 = fig1.gca()
ax1.invert_xaxis()
x_pt = 0
z_pt = 2


n_pt = 0
mouse_xyz = np.array(mouse_pos_xyz.iloc[n_pt])
mouse_dir_xyz = np.array(rot_vec.iloc[n_pt])
mouse_dir_xyz_r = np.array(rot_vec_r.iloc[n_pt])
mouse_dir_xyz_l = np.array(rot_vec_l.iloc[n_pt])

# mouse
ax1.plot(mouse_xyz[x_pt], mouse_xyz[z_pt], 'o', color='lightgreen')
ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mouse_dir_xyz[x_pt]*d], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mouse_dir_xyz[z_pt]*d], 'o-', color='green')
ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mouse_dir_xyz_r[x_pt]*h_adj], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mouse_dir_xyz_r[z_pt]*h_adj], 'o-', color='darkgreen')
ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mouse_dir_xyz_l[x_pt]*h_adj], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mouse_dir_xyz_l[z_pt]*h_adj], 'o-', color='darkgreen')

n_pt = -50
mouse_xyz = np.array(mouse_pos_xyz.iloc[n_pt])
mouse_dir_xyz = np.array(rot_vec.iloc[n_pt])
mouse_dir_xyz_r = np.array(rot_vec_r.iloc[n_pt])
mouse_dir_xyz_l = np.array(rot_vec_l.iloc[n_pt])

ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mouse_dir_xyz[x_pt]*d], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mouse_dir_xyz[z_pt]*d], 'o-', color='green')
ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mouse_dir_xyz_r[x_pt]*h_adj], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mouse_dir_xyz_r[z_pt]*h_adj], 'o-', color='darkgreen')
ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mouse_dir_xyz_l[x_pt]*h_adj], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mouse_dir_xyz_l[z_pt]*h_adj], 'o-', color='darkgreen')


obj_x = np.asarray(xml_data['objList']['x'])
obj_y = np.asarray(xml_data['objList']['y'])
obj_z = np.asarray(xml_data['objList']['z'])

# objects
ax1.plot(obj_x, obj_z, 'o', color='red')
# path
ax1.plot(df.x_pos, df.z_pos, color='pink')


#%% projecting object onto cam FOV

obj_pos_xyz = np.concat([obj_x[:,None], obj_y[:,None], obj_z[:,None]], axis=1)     # df.x_pos.iloc[n_pt] np.array

n_pt = 2
obj1_xyz = obj_pos_xyz[n_pt,:]

obj_dir = obj1_xyz - mouse_pos_xyz

#x = np.exp(1j*phi)
#phi2 = pd.Series(np.angle(np.exp(1j*phi)), name='phi')

obj_dir_sph = if_cart_to_spheric(obj_dir.x_pos, obj_dir.y_pos, obj_dir.z_pos)
obj_fov_phi = pd.Series(np.angle(np.exp(1j*(obj_dir_sph.phi-phi))), name='phi')
in_fov = np.logical_and(obj_fov_phi<(FOV_rad_adj/2), obj_fov_phi>(-FOV_rad_adj/2))

obj_fov_theta = pd.Series(np.angle(np.exp(1j*(obj_dir_sph.theta-theta))), name='theta')


ax1.plot(df.x_pos.loc[in_fov], df.z_pos.loc[in_fov], color='red')


fov_w_pos = np.tan(obj_fov_phi.loc[in_fov])
fov_h_pos = np.tan(obj_fov_theta.loc[in_fov])

plt.figure()
ax2 = plt.subplot()
ax2.set_xlim([np.tan(-FOV_rad_adj/2), np.tan(FOV_rad_adj/2)])
ax2.set_ylim([np.tan(-FOV_rad/2), np.tan(FOV_rad/2)])
ax2.plot(fov_w_pos, fov_h_pos)



# convert to FOV locations
# FOV_rad_adj wide and FOV_rad high


#%%


plt.figure()
plt.plot(phi)




#%%


