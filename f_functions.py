# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:47:27 2025

@author: ys2605
"""
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

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