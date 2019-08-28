# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:58:25 2019
Build upon mini tfl scaling
Check scaling relationships for large Chevron Volume
@author: talongi
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import definitions as my
from scipy.stats import mode

### set work directory
#workdir = '/Volumes/gypsy/Palos_Verdes_Fault_TA/TFL_volume/' #macbook
#workdir = 'E:\Palos_Verdes_Fault_TA\TFL_volume' #windows
workdir = '/run/media/talongi/gypsy/Palos_Verdes_Fault_TA/TFL_volume' #linux
os.chdir(workdir)

### files to be loaded
tfl_file = 'chev_2019-07-30.dat'
fault_file = 'c_pvf.dat'
dist_file = 'chev_dist_pts_to_fault.txt'
easting_file = 'chev_easting_of_fault.txt'
tops_file = 'formation_tops.txt'

tops_df = pd.read_csv(tops_file, sep = ' ') #load formation tops

try: #check if data is loaded
    values
except NameError:
    values, xy, z = my.load_odt_att(tfl_file) #tfl data
    print('tfl data loaded')
    Xf, Yf, Zf, df_f = my.load_odt_fault(fault_file, n_grid_points=1000, int_type='linear') #central fault
    print('fault data loaded')
    
    fault_points = np.column_stack((Xf.flat, Yf.flat, Zf.flat)) # set fault points from interpolated data  
    fault_points = fault_points[~np.isnan(fault_points).any(axis = 1)] #filter out rows with nan in z

calc_dist = 'y'
if calc_dist == 'y':
    print('calculating min distances, this may take a while.')
    distances, easting = my.calc_min_dist(values, xy, fault_points)

    save_text = input('would you like to save the result? (y/n)  ')
    if save_text == 'y':
        print('results saved')
        np.savetxt(dist_file, distances, delimiter = ' ')
        np.savetxt(easting_file, easting, delimiter = ' ')

# load data if not already loaded
try:
    distances
except NameError:
    distances = pd.read_csv(dist_file, sep = '\s+', header = None)
    print('distances data loaded')    
try:
    easting
except NameError:
    easting = pd.read_csv(easting_file, sep = '\s+', header = None)
    print('distances data loaded')

#%%
    
modes = mode(distances, axis = 0)
fig1 = plt.figure(1, figsize = (18,10))
plt.plot(modes[0].flat) #shows that samples 50 - 610 are valid ~> (150 - 3200)m
#%%
# =============================================================================
# Background plot
# =============================================================================
min_distance_around_fault = 0
max_distance_around_fault = 800
tfl_new = my.trim_tfl_distance(values, distances, 
                               min_dist = min_distance_around_fault, 
                               max_dist = max_distance_around_fault)



plt.close('all')
smoothing = 5
minD = 250
maxD = 2500
thresholds = np.flip(np.arange(0.7,0.96,0.1))
cmap = plt.cm.viridis_r(np.linspace(0,1,len(thresholds)),)
slopes_arr = np.zeros_like(cmap)

fig = plt.figure(1)
for i,thresh in enumerate(thresholds):
    zed, fd, fd_fit, slope = my.fracture_density_depth(tfl_new,thresh, smoothing, 
                                                     depth_arr = z,
                                                    min_depth= minD, max_depth= maxD,
                                                    bootstrap_iterations=1,
                                                    apply_bool_data = east_bool)
    slopes_arr[i] = slope
    
    # data
    plt.plot(fd, zed, '-',
               c = cmap[i],
                label = 'TFL > %2.2f, Slope = %2.2f' % (thresh, fd[3]))
    # fit
#    plt.plot(fd_fit, zed,  ':', c = cmap[i])

xlims = fig.gca().get_xlim()     
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    x = np.linspace(xlims[0], xlims[1], 3)
    plt.plot(x,np.ones_like(y) * dep, label = tops_df.columns[j])
    
plt.ylabel('Depth Below Seafloor [m]')
plt.xlabel('Fracture Density')
plt.legend(frameon = False, fontsize = 8, ncol = 1)
#plt.axis('equal')
fig.gca().invert_yaxis()
plt.title('Chevron Vol. Fracture Density \n %i m From Fault Sampling %i m Width' 
          % (min_distance_around_fault, max_distance_around_fault - min_distance_around_fault))

print('Median of the slopes calculated ',np.median(slopes_arr))

#%%
# =============================================================================
# Normalized plot
# =============================================================================
tfl_fault = my.trim_tfl_distance(values, distances, max_dist = 700)
tfl_background = my.trim_tfl_distance(values, distances, 
                                      min_dist = 1500, max_dist = 3500)

east_bool = easting > 0 # apply in frac dens calc

plt.close('all')
smoothing = 2
minD = 250
maxD = 2500
thresholds = np.flip(np.arange(0.7,0.96,0.15))
thresholds = [0.7, 0.8, 0.9]

cmap = plt.cm.viridis_r(np.linspace(0,1,len(thresholds)),)
slopes_arr = np.zeros_like(cmap)
t0 = time.time()
fig = plt.figure(figsize = (7,11))
for i,thresh in enumerate(thresholds):
    fd_fault = my.fracture_density_depth(tfl_fault, thresh, smoothing, depth_arr = z,
                                min_depth= minD, max_depth= maxD,
                                apply_bool_data = east_bool)
    fd_background = my.fracture_density_depth(tfl_background, thresh, smoothing, depth_arr= z,
                                              min_depth = minD, max_depth = maxD,
                                              bootstrap_iterations = 10,
                                              apply_bool_data = east_bool)
    fd_normalized = fd_fault[1] / fd_background[1]
    
    # data
    plt.plot(fd_normalized, fd_fault[0], '-',
               c = cmap[i],
                label = 'TFL > %2.2f' % (thresh))
print('loop took %f s' % (time.time() - t0))

xlims = fig.gca().get_xlim() 
# plot formations tops   
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    x = np.linspace(xlims[0], xlims[1], 3)
    plt.plot(x, np.ones_like(x) * dep, '--', label = tops_df.columns[j])
    
plt.ylabel('Depth Below Seafloor [m]')
plt.xlabel('Fracture Density (normalized to background)')
plt.legend(frameon = False, fontsize = 8, ncol = 1)
#plt.axis('equal')
plt.title(' ')
plt.tight_layout()
fig.gca().invert_yaxis()

print('Median of the slopes calculated ',np.median(slopes_arr))


