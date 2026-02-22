# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:58:50 2026

@author: ys2605
"""
import os
import sys

for user1 in ['ys2605', 'shymk']:
    if os.path.isdir('C:/Users/' + user1):
        path1 = 'C:/Users/' + user1 + '/Desktop/stuff/'
        
sys.path.append(path1 + '/VR/VR_analysis/')
sys.path.append(path1 + '/VR/VR_analysis/functions')
sys.path.append(path1 + '/slow_dynamics_analysis/functions')

import xml.etree.ElementTree as ET

from f_sd_utils import f_get_fnames_from_dir
import pandas as pd

#%%

mouse_id = 'L'

data_dir = 'F:/VR/data_proc/' + mouse_id + '/preprocessing/'    # edit this  

flist = f_get_fnames_from_dir(data_dir, ext_list = ['xml'], tags=[mouse_id])  # 'results_cnmf_sort'

save_tag = '_framedata'

#%%
overwrite = False

for file in flist:
    print('processing %s' % file[:-4] + save_tag + '.csv')
    save_fname = data_dir + file[:-4] + save_tag + '.csv'
    
    if not overwrite and not os.path.exists(save_fname):
    
        xml_tree =  ET.parse(data_dir + file)
        root = xml_tree.getroot()
         
        sequence = root.find('Sequence')
        
        frame_index = []
        rel_times = []
        abs_times = []
        channel = []
        page = []
        
        for child in sequence:
            if child.tag == 'Frame':
                frame_index.append(int(child.get('index')))
                rel_times.append(float(child.get('relativeTime')))
                abs_times.append(float(child.get('absoluteTime')))
                channel.append(int(child[0].get('channel')))
                page.append(int(child[0].get('page')))
                
        
        data = {'index':            frame_index,
                'relativeTimes':    rel_times,
                'absoluteTimes':    abs_times,
                'channel':          channel,
                'page':             page}
        
        df = pd.DataFrame(data)
        
        
        df.to_csv(save_fname, index=False, encoding='utf-8')
    else:
        print('file already exists')
    
    
print('Done')
    
    
    
    
        
        
        