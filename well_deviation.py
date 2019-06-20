#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:49:52 2019
improved on 'make_time_depth_model.py'
Matches LAS files to Deviation Files.
Interpolate deviation angles, then apply interpolation to LAS measured depths
to calculate true vertical depth.
Interpolate true vertical depths to a standard 1ft spacing.
Smooth velocity with running mean, then stack all wells.
Calculate two way travel time, then write to table
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
    """Converts Deviated well depth (MD) to true vertical depth (TVD)."""
    d_0 = MD[0]
    dz_measured = np.diff(MD)
    theta = np.deg2rad(thetas[1:])
    
    dz = dz_measured * np.cos(theta) 
    Z_true = np.cumsum(dz) + d_0
    Z_true = np.append(d_0, Z_true)
    
    return(Z_true)
    
def smoothing(array, win_size):
    """Smoothes array or list using pandas rolling function"""
    df = pd.DataFrame(array)    
    df_smooth = df.rolling(window = win_size,
                           min_periods = int(win_size/2),
                            win_type = 'hamming',
                            center = True).mean()
    
    return(df_smooth)
 
    
# Files    
LAS_dir = '/auto/home/talongi/Pvf/FOIA_request_1/Beta_Exploratory_Logs/Beta_Exploratory_Logs'
dev_dir = '/auto/home/talongi/Pvf/Beta_Deviation/All_deviations'
#work_dir = 'D:\\Palos_Verdes_Fault_TA\\Beta_logs\\request_1'

# =============================================================================
# ~LAS files:
# =============================================================================
os.chdir(LAS_dir)
files = sorted(os.listdir(LAS_dir))

las_dic = {}
for ind,item in enumerate(files):
    las_file = lasio.read(item)
    las_df = las_file.df() #make dataframe // indexed by depth [ft]
    las_api = '0' + str(las_file.well.api.value) #api number
    las_df['Vel_smooth'] = smoothing(las_df['DTWS'],100) #smooth w/ funciton
    
    las_dic[las_api] = las_df #store each las df by api
    


#%%
# =============================================================================
# ~Deviation files:
# =============================================================================  
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
                
# create deviation dictionary; key = api
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
        
# =============================================================================
# Perform interpolation, then stack results.    
# =============================================================================   
#plt.close('all')
make_plots = False
if make_plots == True:
    plt.figure(figsize = (20,6))

# make new dic of interp data
interp_dic = {}

# set arr for stacking
depth_1d_arr = np.array([])
vel_1d_arr = np.array([])
for i in match:
    if i in dev_dic and las_dic:
        print('Match deviation file to LAS file for ', i, ' API')
        dev = dev_dic[i]
        las = las_dic[i]
        
        #interpolate angles - int_dev is function
        int_dev = interpolate.interp1d(dev['DEPTH'], dev['DEV'], kind = 'cubic')
        
        # only want LAS entries that contain logs for Delta-time/Velocity
        mask = ~np.isnan(las['DTWS']) # ~ opposite of isnan
        las = las[mask]
        V = las['DTWS']
        MD = las.index #LAS df are indexed by depth (measured)
        
        #calc true depth with interpolated measured depth
        Z_new = true_depth(MD, int_dev(MD))
            
        #interp velocity evenly spaced
        int_vel = interpolate.interp1d(Z_new, V)
        Z_even_sp = np.arange(round(Z_new[1]), round(Z_new[-1]), 1)
        V_even_sp = int_vel(Z_even_sp)
        V_smooth = smoothing(V_even_sp, 60)
        
        df_even = pd.DataFrame(data = {'TVD': Z_even_sp, 
                                       'VEL' : V_even_sp,
                                       'V_smooth' : V_smooth.values.flatten()})
        
        # add dataframe to dictionary - key = api                                                  
        interp_dic[i] = df_even
        
        # build 1D array
        depth_1d_arr = np.append(depth_1d_arr, Z_even_sp)
        vel_1d_arr = np.append(vel_1d_arr, V_smooth)
                        
        if make_plots == True:
            plt.plot(MD,1/V)
            plt.plot(Z_new, 1/V)
            plt.plot(Z_even_sp,1/V_even_sp, label = i)
            plt.plot(Z_even_sp, 1/V_smooth, label = i)

# stack velocities
depths_unique_arr = np.unique(depth_1d_arr)

stacked_velocity = []  
stacked_depths = []
n_logs = np.zeros_like(depths_unique_arr)            
for ind, dep in enumerate(depths_unique_arr):
    mask = (depth_1d_arr == dep)
    n_logs[ind] = np.count_nonzero(mask) #count number of logs used in calc
    
    if n_logs[ind] >= 2:
        stacked_velocity.append(np.nanmean(vel_1d_arr[mask]))
        stacked_depths.append(dep)

#list to arrays 
stacked_depths = np.asarray(stacked_depths)
stacked_velocity = np.asarray(stacked_velocity)

# set velocities for unsampled shallow depths
my_depth = np.arange(0, stacked_depths[0], np.diff(stacked_depths).mean())   
v_min = 4900 # [ft/s] cannot be slower than water
v_min = 1/(v_min / 1e6) #us/ft conversion
v_max = stacked_velocity[0]
# linear increase w/ depth
my_v = np.linspace(v_min, v_max, len(my_depth))

# append arrays with shallow data - converted to metric
depths_to_surface = np.append(my_depth, stacked_depths) * 0.3048 #[m]
stacked_velocity_to_surface = 1/(np.append(my_v, stacked_velocity)) #[ft/s]
stacked_velocity_to_surface = stacked_velocity_to_surface * 1e6 * 0.3048 #[m/s]

# calc two way travel time.
time = 2e3 * np.diff(depths_to_surface).mean() / stacked_velocity_to_surface #[ms]
travel_time = np.cumsum(time)    

# =============================================================================
# Write data to text file.
# =============================================================================
write_2_file = True

if write_2_file == True:
    # format data
    data_dic = {'Depth[m]': depths_to_surface, 
                'Velocity[m/s]': stacked_velocity_to_surface,
                'TWT[ms]': travel_time }
                
    # use pandas to write            
    depth_time = pd.DataFrame(data = data_dic) 
    depth_time.to_csv('/auto/home/talongi/Pvf/Data_tables/depth_time_mod.txt', 
                      sep = ' ',
                      index = False)
    