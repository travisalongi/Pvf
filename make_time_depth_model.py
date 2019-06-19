 # -*- coding: utf-8 -*-
"""
read in las files -- develop time depth model
this is spaghetti code
author: talongi
"""

import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams["font.family"] = "serif"
plt.close('all')

work_dir = '/auto/home/talongi/Pvf/FOIA_request_1/Beta_Exploratory_Logs/Beta_Exploratory_Logs'
#work_dir = 'D:\\Palos_Verdes_Fault_TA\\Beta_logs\\request_1'

os.chdir(work_dir)
files = sorted(os.listdir(work_dir))
#~~~~~~~~~~~~~~~~~
make_plots = False

if make_plots == True:
    fig99 = plt.figure(99, figsize = (10,10))

dic = {} #empty dictionary to fill in loop
for j,i in enumerate(files):
    las_file = lasio.read(i)
    las_file_name = i[3:13]
    print(las_file_name)
    
    # assign vars for LAS logs
    depth = las_file.depth_m #depth [m]
    bulk_density = las_file['DENWS']
    gamma = las_file['GRWS']
    med_resistivity = las_file['MRESWS']
    sp_potential = las_file['SPWS']
    
    # convert sonic log to velocity [m/s]
    delta_time = las_file['DTWS'] # in [us/ft] micro seconds / foot
    dt_s = delta_time / 1e6 / 0.3048 #convert to [s/m]
    V = 1/dt_s # [m/s]
    
    
    #make dictionary of dictionaries
    dic_in_loop = {'vel' : V, 
                   'dep' : depth,
                   'dens' : bulk_density,
                   'gamma' : gamma,
                   'resis' : med_resistivity,
                   'potent' : sp_potential}
    dic[las_file_name] = dic_in_loop        
    
    if make_plots == True: #plots each log separately
        #initiate subplt
        sp = plt.subplot(1, len(files), j)
        sp.plot(V,depth, linewidth = 0.2, color = 'k')
        
        #hardwired...
        if j == 1:
            plt.ylabel('Depth [m]', fontsize = 20)
        plt.ylim([0,3e3])    
    
        fig99.gca().invert_yaxis()        


    
if make_plots == True: # plots all logs on same figure
    fig2 = plt.figure(2, figsize = (10,10))
    
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
    d = dic[i]['dep'] #spacing is 0.15 m
    v = dic[i]['vel']
    r = dic[i]['resis']
    g = dic[i]['gamma']
    p = dic[i]['dens']
    sp = dic[i]['potent']
    d_dist = window_size * np.diff(d).mean()
    
    # convert to dataframe to use pandas smoothing method
    v_df = pd.DataFrame(v)
    r_df = pd.DataFrame(r)
    g_df = pd.DataFrame(g)
    p_df = pd.DataFrame(p)   
    sp_df = pd.DataFrame(sp)
    
    
    # rolling smoothing window applying median filtering
    window_size = 60
    v_smooth = v_df.rolling(window = window_size,
                            min_periods = int(window_size/2),
                            win_type = 'hamming',
                            center = True).mean()
    r_smooth = r_df.rolling(window = window_size,
                            min_periods = int(window_size/2),
                            win_type = 'hamming',
                            center = True).mean()
    g_smooth = g_df.rolling(window = window_size,
                            min_periods = int(window_size/2),
                            win_type = 'hamming',
                            center = True).mean()
    p_smooth = p_df.rolling(window = window_size,
                            min_periods = int(window_size/2),
                            win_type = 'hamming',
                            center = True).mean()
    
    sp_smooth = sp_df.rolling(window = window_size,
                              min_periods = int(window_size/2),
                              win_type = 'hamming',
                              center = True).mean()
    
    #add smooth data to dictionary
    dic[i]['vel_smooth'] = v_smooth.values 
    dic[i]['res_smooth'] = r_smooth.values
    dic[i]['gam_smooth'] = g_smooth.values
    dic[i]['den_smooth'] = p_smooth.values
    dic[i]['sp_smooth'] = sp_smooth.values

    if make_plots == True:
        fig99 = plt.figure(99, figsize = (6,16))
        al = 0.5
        lw = 1.5
        if well_id == 296:
            c = 'seagreen'
            print(well_id, c)    
            plt.plot(v_smooth,d,    
    #                 color = c,
                     alpha = al,
                     linewidth = lw,
                     label = str(i))
    
        if well_id == 300:
            c = 'grey'
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
    
    
#% Make long 1D arrays of all log values and depths            
arr_depths = np.array([])
arr_vel = np.array([])
arr_res = np.array([])
arr_gam = np.array([])
arr_den = np.array([])
arr_sp = np.array([])
for key in dic.keys():
    arr_depths = np.append(arr_depths, dic[key]['dep'])
    arr_vel = np.append(arr_vel, dic[key]['vel_smooth'])
    arr_res = np.append(arr_res, dic[key]['res_smooth'])
    arr_gam = np.append(arr_gam, dic[key]['gam_smooth'])
    arr_den = np.append(arr_den, dic[key]['den_smooth'])
    arr_sp = np.append(arr_sp, dic[key]['sp_smooth'])

arr_depths_unique = np.unique(arr_depths[arr_depths >= 100])            

# stack data
stacked_velocity = np.zeros_like(arr_depths_unique)
stacked_resistivity = np.zeros_like(arr_depths_unique)
stacked_gamma = np.zeros_like(arr_depths_unique)
stacked_density = np.zeros_like(arr_depths_unique)
stacked_sp = np.zeros_like(arr_depths_unique) 
for i,x in enumerate(arr_depths_unique):
    mask = (arr_depths == x)
    
    stacked_velocity[i] = np.nanmean(arr_vel[mask])         
    stacked_resistivity[i] = np.nanmean(arr_res[mask])     
    stacked_gamma[i] = np.nanmean(arr_gam[mask])    
    stacked_density[i] = np.nanmean(arr_den[mask])
    stacked_sp[i] = np.nanmean(arr_sp[mask])
    
    
# set velocities for unsampled shallow depths
my_depth = np.arange(0, arr_depths_unique[0], np.diff(d).mean())    
v_min = 1550 # [m/s] cannot be slower than water
v_max = stacked_velocity[0]
my_v = np.linspace(v_min, v_max, len(my_depth))

# append arrays with shallow data
depths_to_surface = np.append(my_depth, arr_depths_unique)
stacked_velocity_to_surface = np.append(my_v, stacked_velocity)

if make_plots == True:
    plt.plot(stacked_velocity_to_surface,depths_to_surface,
             'k',
             label = 'Stacked Data')
    
    plt.legend()
    plt.ylabel('Depth [m]', fontsize = 18)
    plt.xlabel('Velocity [m/s]', fontsize = 18)
    plt.title('Wells Median Filtered\n Window Size = ' + str(window_size) + '\n Smoothed over ' + str(d_dist) + 'meters',
              fontsize = 21)
        
    fig99.gca().invert_yaxis()  

#%% make plot of all stacked values
fig50 = plt.figure(50, figsize = (8.5,11))
ylimits = [0,3200]

plt.subplot(151)
plt.plot(stacked_velocity, arr_depths_unique,
         label = 'Velocity')
plt.title('Velocity')
plt.xlabel('[m/s]')
plt.ylabel('Depth [mbsf]')
plt.ylim(ylimits)
fig50.gca().invert_yaxis()


plt.subplot(152)
plt.plot(stacked_resistivity, arr_depths_unique,
         label = 'Resistivity')
plt.title('Resistivity')
plt.xlabel('[ohms]')
plt.yticks([])
plt.ylim(ylimits)
fig50.gca().invert_yaxis()


plt.subplot(153)
plt.plot(stacked_gamma, arr_depths_unique,
         label = 'Gamma')

plt.title('Gamma')
plt.xlabel('[gapi]')
plt.yticks([])
plt.ylim(ylimits)
fig50.gca().invert_yaxis()


plt.subplot(154)
plt.plot(stacked_density, arr_depths_unique,
         label = 'Density')
plt.title('Density')
plt.xlabel('[g/cm^3]')
plt.yticks([])
plt.ylim(ylimits)
fig50.gca().invert_yaxis()

plt.subplot(155)
plt.plot(stacked_sp, arr_depths_unique,
         label = 'SP')
plt.title('Spon. Pot.')
plt.xlabel('[mV]')
plt.yticks([])
plt.ylim(ylimits)
fig50.gca().invert_yaxis()


    
#%% Write data to text file
write_2_file = False

if write_2_file == True:
    #calc twtt
    time = 2 * np.diff(arr_depths_unique)[0] / stacked_velocity *1e3
    time = np.nan_to_num(time)
    travel_time = time.cumsum()
    
    # format data
    data_arr = {'Depth[m]': arr_depths_unique, 
                'Velocity[m/s]': np.nan_to_num(stacked_velocity),
                'TWT[ms]': travel_time }
                
    # use pandas to write            
    depth_time = pd.DataFrame(data = data_arr) 
    depth_time.to_csv('/auto/home/talongi/Pvf/Data_tables/depth_time_mod.txt', sep = ' ', index = False)