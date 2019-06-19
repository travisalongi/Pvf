#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:49:52 2019
adjust depths to account for deviation
@author: talongi
"""

import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.rcParams["font.family"] = "serif"
plt.close('all')

def true_depth(MD, thetas):
    d_0 = MD[0]
    dz_measured = np.diff(MD)
    theta = np.deg2rad(thetas[1:])
    
    dz = dz_measured * np.cos(theta) 
    Z_true = np.cumsum(dz) + d_0
    Z_true = np.append(d_0, Z_true)
    
    return(Z_true)
    
def smoothing(array, win_size):
    df = pd.DataFrame(array)
    
    df_smooth = df.rolling(window = win_size,
                           min_periods = int(win_size/2),
                            win_type = 'hamming',
                            center = True).mean()
    
    return(df_smooth)

#def tol_check() 
    
# Files    
LAS_dir = '/auto/home/talongi/Pvf/FOIA_request_1/Beta_Exploratory_Logs/Beta_Exploratory_Logs'
dev_dir = '/auto/home/talongi/Pvf/Beta_Deviation/All_deviations'
#work_dir = 'D:\\Palos_Verdes_Fault_TA\\Beta_logs\\request_1'

#~~LAS files
os.chdir(LAS_dir)
files = sorted(os.listdir(LAS_dir))

las_dic = {}
for ind,item in enumerate(files):
    las_file = lasio.read(item)
    las_df = las_file.df() #make dataframe // indexed by depth [ft]
    las_api = '0' + str(las_file.well.api.value) #api number
    las_df['Vel_smooth'] = smoothing(las_df['DTWS'],100)
    
    las_dic[las_api] = las_df #store each las df by api
    


#%%

#~~Deviation files    
os.chdir(dev_dir)
dev_file_names = np.asarray(sorted(os.listdir(dev_dir)))  # get files
# format file names
dev_api_list = np.zeros_like(dev_file_names)    
for ind,item in enumerate(dev_file_names):
    dev_api = item[:-4] #rm file extn. for matching
    dev_api_list[ind] = dev_api

    #record matches & indicies
    match = np.array([]) 
    indicies = []
    for ind,item in enumerate(las_dic.keys()):
        i = str(item)
        mask = dev_api_list == i
        idx = np.where(mask)
       
        if sum(mask) > 0: #if match add to array/list
            indicies.append(int(idx[0]))
            match = np.append(match, dev_api_list[idx])
                

dev_dic = {}
for i in dev_file_names[indicies]:
    # get api number from file
    api =  open(i).readlines()[0].strip(),
    api = api[0] # b/c tuple
    
    df = pd.read_csv(i,  skiprows = 2, delim_whitespace = True)
    
    #calc true depth
    z_true = true_depth(df['DEPTH'], df['DEV'])
    df['TVD'] = z_true #make new entry for true vert. depth
    
    #update dictionary - key is api
    dev_dic[api] = df
        
  
#%%
plt.close('all')
make_plots = True

interp_dic = {}
for i in match:
    print(i)
    if i in dev_dic and las_dic:
        dev = dev_dic[i]
        las = las_dic[i]
        
        #interpolate angles - int_dev is function
        int_dev = interpolate.interp1d(dev['DEPTH'], dev['DEV'], kind = 'cubic')
        
        # only want LAS entries that contain logs for Delta-time/Velocity
        mask = ~np.isnan(las['DTWS']) 
        las = las[mask]
        V = las['DTWS']
        MD = las.index
        
        #calc true depth
        Z_new = true_depth(MD, int_dev(MD))
            
        #interp velocity evenly spaced
        int_vel = interpolate.interp1d(Z_new, V)
        Z_even_sp = np.arange(round(Z_new[1]), round(Z_new[-1]), 1)
        V_even_sp = int_vel(Z_even_sp)
        V_smooth = smoothing(V_even_sp, 60)
        
        df_even = pd.DataFrame(data = {'DEPTH': Z_even_sp, 
                                       'VEL' : V_even_sp,
                                       'V_smooth' : V_smooth.values.flatten()})
                                                                              
        interp_dic[i] = df_even
                        
        if make_plots == True:
            plt.plot(MD,1/V, c = 'gray')
            plt.plot(Z_new, 1/V,'r')
            plt.plot(Z_even_sp,1/V_even_sp,'b')
            plt.plot(Z_even_sp, 1/V_smooth, 'k')
            
            