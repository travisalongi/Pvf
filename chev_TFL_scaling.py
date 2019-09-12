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

calc_dist = 'n'
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
    distances = distances.values
    print('distances data loaded')    
try:
    easting
except NameError:
    easting = pd.read_csv(easting_file, sep = '\s+', header = None)
    easting = easting.values
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
max_distance_around_fault = 1000
tfl_new = my.trim_tfl_distance(values, distances, 
                               min_dist = min_distance_around_fault, 
                               max_dist = max_distance_around_fault)
east_bool = easting > 0 # apply in frac dens calc only values east of c-fault


plt.close('all')
smoothing = 3
minD = 250
maxD = 2300
thresholds = np.flip(np.arange(0.7,0.94,0.1))
cmap = plt.cm.viridis_r(np.linspace(0,1,len(thresholds)),)
slopes_arr = np.zeros_like(cmap)

fig = plt.figure(1, figsize = (7,11))
for i,thresh in enumerate(thresholds):
    fd, zed = my.fracture_density_depth(tfl_new,thresh, smoothing, 
                                                     depth_arr = z,
                                                    min_depth= minD, max_depth= maxD,
                                                    bootstrap_iterations=1,
                                                    apply_bool_data = east_bool)
    # data
    plt.plot(fd, zed, '-',
               c = cmap[i],
                label = 'Thinned Fault Likelihood > %2.2f' % (thresh))

# plot formations tops 
xlims = fig.gca().get_xlim()   
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    x = np.linspace(xlims[0], xlims[1], 3)
    plt.plot(x, np.ones_like(x) * dep, ':',c = 'black') 
    plt.text(xlims[1] - 0.013, dep + 30, tops_df.columns[j], 
             color = 'black', fontsize = 10)
    
    
plt.ylabel('Depth Below Seafloor [m]', fontsize = 20)
plt.xlabel('Fracture Density', fontsize = 20)
plt.title('Background', fontsize = 26)
#plt.legend(frameon = False, fontsize = 10, ncol = 1)
#plt.axis('equal')

plt.tick_params(labelsize = 14.0)
fig.gca().invert_yaxis()
fig.gca().set_facecolor('lightgray')
plt.tight_layout()

#plt.title('Chevron Vol. Fracture Density \n %i m From Fault Sampling %i m Width' 
#          % (min_distance_around_fault, max_distance_around_fault - min_distance_around_fault))

#%%
# =============================================================================
# Normalized plot
# =============================================================================
tfl_fault = my.trim_tfl_distance(values, distances, max_dist = 700)
tfl_background = my.trim_tfl_distance(values, distances, 
                                      min_dist = 1500, max_dist = 3000)

east_bool = easting > 0 # only values east of fault

plt.close('all')
smoothing = 3
minD = 250
maxD = 2300
thresholds = np.flip(np.arange(0.70,0.96,0.10))
#thresholds = [0.80]

cmap = plt.cm.viridis_r(np.linspace(0,1,len(thresholds)),)

slopes_arr = np.zeros_like(cmap)
t0 = time.time()
fig = plt.figure(2, figsize = (7,11))
for i,thresh in enumerate(thresholds):
    fd_fault = my.fracture_density_depth(tfl_fault, thresh, smoothing, depth_arr = z,
                                min_depth= minD, max_depth= maxD)#,
#                                apply_bool_data = east_bool)
    fd_background = my.fracture_density_depth(tfl_background, thresh, smoothing, depth_arr= z,
                                              min_depth = minD, max_depth = maxD,
                                              bootstrap_iterations = 10,
                                              apply_bool_data = east_bool)
    fd_normalized = fd_fault[0] / fd_background[0]
    
    # data
    plt.plot(fd_normalized, fd_fault[1], '-',
               c = cmap[i],
                label = 'Thinned Fault Likelihood > %2.2f' % (thresh))
print('loop took %f s' % (time.time() - t0))

# plot formations tops 
xlims = fig.gca().get_xlim()   
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    x = np.linspace(xlims[0], xlims[1], 3)
    plt.plot(x, np.ones_like(x) * dep, '--',c = 'black') 
    plt.text((xlims[1] - xlims[0]) * .72, dep + 30, tops_df.columns[j], 
             color = 'black', fontsize = 9)
    
plt.ylabel('Depth Below Seafloor [m]', fontsize = 20)
plt.xlabel('Fracture Density (Normalized)', fontsize = 20)
plt.legend(facecolor = 'white', fontsize = 10, ncol = 1)
#plt.axis('equal')
plt.title(' ')
plt.tight_layout()
fig.gca().invert_yaxis()
plt.tick_params(labelsize = 14.0)
fig.gca().set_facecolor('lightgray')

#%%
# =============================================================================
# Bin by distance from fault
# =============================================================================
plt.close('all')
# set parameters
east_bool = easting > 0 # apply in frac dens calc, only values east of c-fault
smoothing = 4
minD = 250
maxD = 2300
thresh = 0.85

cmap = plt.cm.viridis_r(np.linspace(0,1,len(thresholds)),)

# set up binned distances
min_distance_around_fault = 0
max_distance_around_fault = 2101
bin_size = 300
fault_distances = np.arange(min_distance_around_fault, max_distance_around_fault, bin_size)

fract_dens_list = []

fig = plt.figure(3, figsize = (7,11))
# plot of background data
fd_background = my.fracture_density_depth(tfl_background, thresh, smoothing, depth_arr = z,
                                          min_depth= minD, max_depth= maxD,
                                          apply_bool_data= east_bool)
plt.plot(fd_background[0], fd_background[1], 'k', label = 'Background')

# plot binned by distance data
for mini,maxi in zip(fault_distances[:-1], fault_distances[1:]):
    tfl_dist = my.trim_tfl_distance(values, distances, mini, maxi)
    
    fract_dens_depth, zed = my.fracture_density_depth(tfl_dist, thresh, smoothing,depth_arr=z, 
                              min_depth=minD, max_depth = maxD, 
                              apply_bool_data= east_bool)
    
    fract_dens_list.append(fract_dens_depth.flatten())
    
    plt.plot(fract_dens_depth, zed, 
             label = '%i - %i m away from fault' % (mini, maxi))

fract_dens_arr = np.vstack((fract_dens_list)) #add all results to an array

# plot formations tops 
xlims = fig.gca().get_xlim()   
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    x = np.linspace(xlims[0], xlims[1], 3)
    plt.plot(x, np.ones_like(x) * dep, 
             ':',c = 'black') 
    plt.text((xlims[1] - xlims[0]) * 0.7, dep + 30, tops_df.columns[j], 
             color = 'black', fontsize = 8)
   
plt.ylabel('Depth Below Seafloor [m]')
plt.xlabel('Fracture Density')
plt.legend(frameon = True, fontsize = 6, ncol = 1, loc = 'lower left')

plt.title(' ')
plt.tight_layout()
fig.gca().invert_yaxis()

#%%
# =============================================================================
# fracture density as a function of distance from fault 
# =============================================================================
plt.close('all')
# set parameters
east_bool = easting > 0 # apply in frac dens calc, only values east of c-fault
smoothing = 4
minD = 270
maxD = 2150
thresh = 0.75

# set up binned distances
min_distance_around_fault = 0
max_distance_around_fault = 2101
bin_size = 100
fault_distances = np.arange(min_distance_around_fault, max_distance_around_fault, bin_size)

fract_dens_list = []

# plot binned by distance data
for mini,maxi in zip(fault_distances[:-1], fault_distances[1:]):
    tfl_dist = my.trim_tfl_distance(values, distances, mini, maxi)
    
    fract_dens_depth, zed = my.fracture_density_depth(tfl_dist, thresh, smoothing,depth_arr=z, 
                              min_depth=minD, max_depth = maxD, 
                              apply_bool_data= east_bool)
    
    fract_dens_list.append(fract_dens_depth.flatten())
    
fract_dens_arr = np.vstack((fract_dens_list)) #add all results to an array
medians = np.nanmedian(fract_dens_arr, axis = 1)

# fitting
fit_dist = fault_distances[1:]
fit_dist, fit, slope, error = my.semilogy_fit(fit_dist, medians)

fault_width = np.abs((1/slope)/np.log(10))

plt.close('all')
fig = plt.figure(4, figsize = (7,11))
plt.semilogy(fit_dist, fit, ':r', label = 'Fit, slope = %2.4f +/- %2.5f, fault width = %i m' % (slope, error, fault_width))
plt.semilogy(fit_dist, medians,'ko', label = 'data')


plt.xlabel('Distance From Center of Fault [m]', fontsize = 20)
plt.ylabel('Fracture Density', fontsize = 20)
plt.legend(facecolor = 'white', fontsize = 10)  
plt.gca().set_facecolor('lightgray')
plt.tick_params(labelsize = 14.0)
plt.tight_layout()
fig.gca().set_facecolor('lightgray')



#%%
# =============================================================================
# fracture density as a function of distance from fault per unit
# =============================================================================
# set parameters
east_bool = easting > 0 # apply in frac dens calc, only values east of c-fault
smoothing = 4
thresh = 0.75

# depths
formations_arr = np.array([tops_df['Pico-upper'],
                           tops_df['Repetto-upper'],
                           tops_df['Monterey-delmontian'],
                           tops_df['Basement']])
formation_names = ['Pico', 'Repetto', 'Monterey']

# set up binned distances
min_distance_around_fault = 0
max_distance_around_fault = 2101
bin_size = 100
fault_distances = np.arange(min_distance_around_fault, max_distance_around_fault, bin_size)

cmap = plt.cm.plasma(np.linspace(0,1,len(formation_names)),)

# plot binned by distance data
plt.close('all')
fig = plt.figure(4, figsize = (7,11))
i = 0
for minD, maxD in zip(formations_arr[:-1], formations_arr[1:]):
    fract_dens_list = []
    
    for mini,maxi in zip(fault_distances[:-1], fault_distances[1:]):
        tfl_dist = my.trim_tfl_distance(values, distances, mini, maxi)
        
        fract_dens_depth, zed = my.fracture_density_depth(tfl_dist, thresh, smoothing,depth_arr=z, 
                                  min_depth=minD, max_depth = maxD, 
                                  apply_bool_data= east_bool)
                
        fract_dens_list.append(fract_dens_depth.flatten())
      
    fract_dens_arr = np.vstack((fract_dens_list)) #add all results to an array
    medians = np.nanmedian(fract_dens_arr, axis = 1)

    # fitting
    median_threshold = medians >= 0.01
    fitting_dist = fault_distances[1:]
    fit_dist, fit, slope, error = my.semilogy_fit(fitting_dist[median_threshold], 
                                                  medians[median_threshold])
    
#    #bootstrap slope error
#    bootstrap_dist, bootstrap_meds = my.bootstrap(fit_dist[median_threshold], 100, 
#                                                  medians[median_threshold])
#    slopes = []
#    for d, m in zip(bootstrap_dist, bootstrap_meds):
#        slope1 = my.semilogy_fit(d, m)[2] #only want slopes
#        slopes.append(slope1)
#    std_slope = np.std(slopes)

    plt.semilogy(fitting_dist, medians,'o', color = cmap[i],
                 label = formation_names[i])
#    plt.errorbar(fit_dist, medians, yerr = bootstrap_std_list,
#                 color = cmap[i], fmt = '',
#                 alpha = 0.6)
    plt.semilogy(fit_dist, fit, ':', color = cmap[i], 
                 label = 'Least Squares Fit, slope = %2.4f +/- %2.5f' % (slope, error))   
    i += 1


plt.xlabel('Distance From Center of Fault [m]',fontsize = 20)
plt.ylabel('Fracture Density', fontsize = 20)
plt.legend(frameon = False)  

#%%
#%%
# =============================================================================
# fracture density as a function of distance from fault per depth range 
# =============================================================================
# set parameters
east_bool = easting > 0 # apply in frac dens calc, only values east of c-fault
smoothing = 4
thresh = 0.75

# depths not formations
formations_arr = np.arange(300, 2000, 200)

# set up binned distances
min_distance_around_fault = 0
max_distance_around_fault = 2101
bin_size = 100
fault_distances = np.arange(min_distance_around_fault, max_distance_around_fault, bin_size)

cmap = plt.cm.plasma(np.linspace(0,1,len(formations_arr)),)
slopes_list = []  #empty list for calced slopes
width_list = []

# plot binned by distance data
plt.close('all')
fig = plt.figure(4, figsize = (7,11))
i = 0
for minD, maxD in zip(formations_arr[:-1], formations_arr[1:]):
    fract_dens_list = []
      
    
    for mini,maxi in zip(fault_distances[:-1], fault_distances[1:]):
        tfl_dist = my.trim_tfl_distance(values, distances, mini, maxi)
        
        fract_dens_depth, zed = my.fracture_density_depth(tfl_dist, thresh, smoothing,depth_arr=z, 
                                  min_depth=minD, max_depth = maxD, 
                                  apply_bool_data= east_bool)
                
        fract_dens_list.append(fract_dens_depth.flatten())
      
    fract_dens_arr = np.vstack((fract_dens_list)) #add all results to an array
    medians = np.nanmedian(fract_dens_arr, axis = 1)

    # fitting
    median_threshold = medians >= 0.01
    fitting_dist = fault_distances[1:]
    fit_dist, fit, slope, error = my.semilogy_fit(fitting_dist[median_threshold], 
                                                  medians[median_threshold])
    
    # calc fault width with exponential relation
    fault_width = np.abs((1/slope)/np.log(10))
    width_list.append(fault_width)
    
    # plotting each depth
    plt.semilogy(fitting_dist, medians,'o', color = cmap[i],
                 label = 'Depth %i to %i m' % (minD, maxD))
    plt.semilogy(fit_dist, fit, ':', color = cmap[i], 
                 label = 'Fit, slope = %2.4f +/- %2.5f, width = %i m' % (slope, error,fault_width))   
    i += 1

plt.xlabel('Distance From Center of Fault [m]',fontsize = 20)
plt.ylabel('Fracture Density', fontsize = 20)
plt.legend(frameon = False, fontsize = 8) 
fig.gca().set_facecolor('lightgray') 

#%%
width_list = np.asarray(width_list/2)
midpoints = (formations_arr[:-1] + 100)


fig = plt.figure(5, figsize = (7,11))
plt.plot(width_list, midpoints,'ko', markersize = 10, label = 'Damage Zone Width')
slope, intercept = np.polyfit(width_list, midpoints,1)
plt.plot(width_list, width_list * slope + intercept, 'r:' , linewidth = 4, label = 'Fit, slope = -%2.2f' % slope)
plt.xlabel('Damage Zone Width [m]', fontsize = 20)
plt.ylabel('Depth Below Seafloor [m]', fontsize= 20)
plt.axis('equal')
fig.gca().invert_yaxis()
fig.gca().set_facecolor('lightgray')
plt.tick_params(labelsize = 14.0)
plt.legend(fontsize = 10, frameon = False)
plt.tight_layout()