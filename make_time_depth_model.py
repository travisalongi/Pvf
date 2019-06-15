
 # -*- coding: utf-8 -*-
"""
read in las files -- develop time depth model
author: talongi
"""

import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.close('all')

work_dir = '/auto/home/talongi/Pvf/FOIA_request_1/Beta_Exploratory_Logs/Beta_Exploratory_Logs'

os.chdir(work_dir)
files = sorted(os.listdir(work_dir))


fig99 = plt.figure(f99, figsize = (10,10))
make_plots = False
counter = 1
dic = {} #empty dictionary to fill in loop
for i in files:
    las_file = lasio.read(i)
    las_file_name = i[3:13]
    print(las_file_name)
    
    depth = las_file.depth_m
    delta_time = las_file['DTWS'] # in [us/ft] micro seconds / foot
    dt_s = delta_time / 1e6 / 0.3048 #convert to [s/m]
    V = 1/dt_s
    
    #make dictionary of dictionaries
    dic_in_loop = {'vel' : V, 'dep' : depth}
    dic[las_file_name] = dic_in_loop        
    
    if make_plots == True:
        #initiate subplt
        sp = plt.subplot(1, len(files), counter)
        sp.plot(V,depth, linewidth = 0.2, color = 'k')
        
        #hardwired...
        if counter == 1:
            plt.ylabel('Depth [m]', fontsize = 20)
        plt.ylim([0,3e3])    
    
        fig.gca().invert_yaxis()        
        counter += 1
    
if make_plots == True:
    fig2 = plt.figure(2, figsize = (10,10))
    counter = 1
    for i in files:
        las_file = lasio.read(i)
        las_file_name = i[3:-13]
        
        depth = las_file.depth_m
        delta_time = las_file['DTWS'] # in [us/ft] micro seconds / foot
        dt_s = delta_time / 1e6 / 0.3048 #convert to [s/m]
        V = 1/dt_s
    
        plt.plot(V,depth, 
                 linewidth = 0.2,
                 linestyle = ':')
    
    fig2.gca().invert_yaxis()
    plt.ylabel('Depth [m]')
    plt.xlabel('V [m/s]')
    
#%%
#plt.close('all')
for i in dic.keys():
    well_id = int(i[:3])
    v = dic[i]['vel']
    d = dic[i]['dep'] #spacing is 0.15 m
    
    # apply smoothing
    v_df = pd.DataFrame(v)    
    window_size = 100
    smooth_dist = window_size * np.diff(d).mean()
    
    # rolling smoothing window applying median filtering
    v_smooth = v_df.rolling(window = window_size).median()
    dic[i]['vel_smooth'] = v_smooth.values #add smooth data to dictionary
    dic[i]['dep_min'] = d.min()
    dic[i]['dep_max'] = d.max()
    
    fig99 = plt.figure(99, figsize = (6,16))
    al = 0.5
    lw = 1.5
    if well_id == 296:
        c = 'seagreen'
        print(well_id, c)    
        plt.plot(v_smooth,d,    
                 color = c,
                 alpha = al,
                 linewidth = lw,
                 label = str(i))

    if well_id == 300:
        c = 'red'
        print(well_id, c)
        plt.plot(v_smooth,d,                  
                 color = c,
                 alpha = al,
                 linewidth = lw,
                 label = str(i))   
    
    if well_id == 301:
        c = 'navy'
        print(well_id,c)
        plt.plot(v_smooth,d,                 
                 color = c,
                 alpha = al,
                 linewidth = lw,
                 label = str(i))
                
    if well_id == 306:
        c = 'grey'
        print(well_id,c)
        plt.plot(v_smooth,d,                
                 color = c,
                 alpha = al,
                 linewidth = lw,
                 label = str(i))        
    

    
#% Make long 1D arrays of all velocities and depths            
arr_depths = np.array([])
arr_vel = np.array([])
for key in dic.keys():
    arr_depths = np.append(arr_depths, dic[key]['dep'])
    arr_vel = np.append(arr_vel, dic[key]['vel_smooth'])

arr_depths_unique = np.unique(arr_depths)            

# stack velocities
stacked_velocity = N_wells = np.zeros_like(arr_depths_unique) 
for i,x in enumerate(arr_depths_unique):
    mask = (arr_depths == x)
    
    stacked_velocity[i] = np.nanmean(arr_vel[mask])

# plot stacked data
plt.plot(stacked_velocity,arr_depths_unique,
         'k',
         label = 'Stacked Data')

plt.legend()
plt.ylabel('Depth [m]', fontsize = 18)
plt.xlabel('Velocity [m/s]', fontsize = 18)
plt.title('Wells Median Filtered\n Window Size = ' + str(window_size) + '\n Smoothed over ' + str(smooth_dist) + 'meters',
          fontsize = 21)
    
fig99.gca().invert_yaxis()    
    
#%% Write data to text file

#calc twtt
time = 2 * arr_depths_unique / stacked_velocity *1e3
travel_time = time.cumsum()

# format data
data_arr = {'Depth[m]': arr_depths_unique, 
            'Velocity[m/s]': np.nan_to_num(stacked_velocity),
            'TWT[ms]': travel_time }
            
# use pandas to write            
depth_time = pd.DataFrame(data = data_arr) 
depth_time.to_csv('/auto/home/talongi/Pvf/Data_tables/depth_time_mod.txt', sep = ' ', index = False)

    
        
   