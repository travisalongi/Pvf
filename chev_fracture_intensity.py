#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:38:12 2021

Calculate fracture intensity - sum of TFL / N voxels.

@author: talongi
"""

import time, h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import definitions as my
from scipy.stats import mode
plt.rcParams['font.size'] = 14

w_fault_file = 'w_pvf.dat'
c_fault_file = 'c_pvf.dat'
e_fault_file = 'e_pvf.dat'
all_data_file = 'all_data.h5'
tops_file = 'formation_tops.txt'

c_pvf = my.load_odt_fault(c_fault_file)
e_pvf = my.load_odt_fault(e_fault_file)
tops_df = pd.read_csv(tops_file, sep = ' ') #load formation tops

# import all data that was calculated in chev tfl scaling.py
values, xy, z, fault_points, distances, easting = my.load_h5(all_data_file, ['values', 'xy', 'z', 'fault_points', 'distances', 'easting'])
del fault_points

#%% Compare the median distance between fault strands 
# get the average value of x/y per stick then smooth results for ploting
c_x, c_y = my.fault_avg_trace(my.load_odt_fault(c_fault_file))
e_x, e_y = my.fault_avg_trace(my.load_odt_fault(e_fault_file))
w_x, w_y = my.fault_avg_trace(my.load_odt_fault(w_fault_file))

# fault trace plot
f,ax = plt.subplots()
ax.plot(c_x, c_y, 'bo')
ax.plot(e_x, e_y, 'ro')
ax.plot(w_x, w_y, 'go')

# rotated traces
f,ax = plt.subplots()
cxy = my.rotate_data(-59, np.array([c_x, c_y]).T)
exy = my.rotate_data(-58.5, np.array([e_x, e_y]).T)
wxy = my.rotate_data(-59, np.array([w_x, w_y]).T)
ybar = cxy[:,1].mean()
ax.plot(cxy[:,0], cxy[:,1] - ybar, 'bo')
ax.plot(exy[:,0], exy[:,1] - ybar, 'ro')
ax.plot(wxy[:,0], wxy[:,1] - ybar, 'go')

# ax.axis('equal')
ax.set_ylim([-1000, 1500])
"""
Rotate!!!

# calculate distance between lines
c = np.polyfit(c_x, c_y, 1)
e = np.polyfit(e_x, e_y, 1)

c_y_detrend = c_y - (c_x * c[0])
e_y_detrend = e_y - (e_x * e[0])


# sanity check
f, ax = plt.subplots()
ax.plot(c_x, c_y_detrend, 'ro')
ax.plot(e_x, e_y_detrend, 'bo')

#%%
from scipy.signal import detrend
from scipy.interpolate import interp1d

fc = interp1d(c_x, c_y, fill_value = 'extrapolate')
fe = interp1d(e_x, e_y, fill_value = 'extrapolate')

x = np.arange(390500, 396800)
plt.plot(x, fc(x), 'g:')
plt.plot( x, fe(x), 'r:', alpha = 1)
# plt.plot(x, np.abs(fc(x) - fe(x)))
dist = np.abs(fc(x) - fe(x))
dist_median = np.median(dist)
print(dist_median)
"""

#%% fracture intensity w/ distance from fault
minD = 250
maxD = 2000
min_ind = my.depth2sample(minD, 4)
max_ind = my.depth2sample(maxD, 4)

# trim data
tfl = my.trim_tfl_depth(values, minD, maxD, 4, easting = easting)
dist = my.trim_tfl_depth(distances, minD, maxD, 4, easting = easting)

# adjust distance bins
min_dist = 0
max_dist = 2500
bin_size = 25
fault_distances = np.arange(min_dist, max_dist, bin_size)

intensities = []
# calc binned by distance
for mini,maxi in zip(fault_distances[:-1], fault_distances[1:]):
    print(mini,maxi)
    m = (dist < maxi) & (dist > mini) # mask of voxels within bounds
    T = np.nansum(m)
    tfl_sum = np.nansum(tfl[m])
    print('number of points in this distance bin = {}; TFL total = {}'.format(np.sum(m), tfl_sum))
    
    intensity = tfl_sum / T    
    intensities.append(intensity)
    # print(intensity)

#%% plot Fracture intensity w/ distance from fault
f, ax = plt.subplots(figsize = (12,12))
ax.plot(my.midpoint(fault_distances), intensities, 'ko:')
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Fracture Intensity')
ax.grid('all')
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
ax.xaxis.set_major_locator(plt.MaxNLocator(18))

#%% some z-slices for comparison
n_slices = 4
depths = np.linspace(minD, maxD, n_slices)
slice_ind = [my.depth2sample(i, 4) for i in depths] # these are the indicies of z-slices to plot

for i in slice_ind:
    plt.figure(figsize = (14,21))        
    my.plot_slice(values, xy, i, threshold = 0.0)
    
    # faults
    plt.plot(c_x, c_y, 'r')
    plt.plot(e_x, e_y,  linestyle = '--', c = 'limegreen', alpha = 0.8)
    plt.plot(w_x, w_y, linestyle = '--', c ='limegreen', alpha = 0.8)
    
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Z-slice {} km Depth'.format(int(my.sample2depth(i, 4))))
    
    
#%% Background
dist_range = [2000,4000] # clearly in the background 
tfl_background = my.trim_tfl_distance(tfl,dist, dist_range[0], dist_range[1])

b_intensities = np.zeros(tfl_background.shape[1]) # intensities array to fill
for i,col in enumerate(tfl_background.T):
    
    T = np.nansum(~np.isnan(col)) # total number of voxels that are not nan
    tfl_sum = np.nansum(col)
    
    intensity = tfl_sum / T    
    b_intensities[i] = intensity
del tfl_background

#%% Fault
dist_range = [0, 200]
tfl_fault = my.trim_tfl_distance(tfl,dist, dist_range[0], dist_range[1]) # within 0 - 200 meters of fault

f_intensities = np.zeros(tfl_fault.shape[1]) # intensities array to fill
for i,col in enumerate(tfl_fault.T):
    
    T = np.nansum(~np.isnan(col)) # total number of voxels that are not nan
    tfl_sum = np.nansum(col)
    
    intensity = tfl_sum / T    
    f_intensities[i] = intensity
del tfl_fault

#%% Normalized plot  
f, ax = plt.subplots(figsize = (8,12))
ax.plot(f_intensities - b_intensities, z[min_ind : max_ind], 'k')


# plot formations tops 
xlims = ax.get_xlim()
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    ax.axhline(dep, 0, 1, color = 'blue') 
    plt.text(np.abs(xlims[1] - xlims[0]) * .05 + xlims[0], dep + 30, tops_df.columns[j], 
              color = 'black', fontsize = 12)

ax.invert_yaxis()
ax.set_ylabel('Depth [km]')
ax.set_xlabel('Normalized Intensity')
ax.set_facecolor('lightgray')
ax.set_title('{} - {} Fault Distance'.format(dist_range[0], dist_range[1]))


#%%
#%% fault plot  
f, ax = plt.subplots(figsize = (8,12))
ax.plot(f_intensities,  z[min_ind : max_ind], 'k')


# plot formations tops 
xlims = ax.get_xlim()
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    ax.axhline(dep, 0, 1, color = 'blue') 
    plt.text(np.abs(xlims[1] - xlims[0]) * .05 + xlims[0], dep + 30, tops_df.columns[j], 
              color = 'black', fontsize = 12)

ax.invert_yaxis()
ax.set_ylabel('Depth [km]')
ax.set_xlabel('Normalized Intensity')
ax.set_facecolor('lightgray')
ax.set_title('{} - {} Fault Distance'.format(dist_range[0], dist_range[1]))

#%% background plot  
f, ax = plt.subplots(figsize = (8,12))
ax.plot( b_intensities, z[min_ind : max_ind], 'k')


# plot formations tops 
xlims = ax.get_xlim()
for j, dep in enumerate(tops_df.iloc[0]): #row of depths
    ax.axhline(dep, 0, 1, color = 'blue') 
    plt.text(np.abs(xlims[1] - xlims[0]) * .05 + xlims[0], dep + 30, tops_df.columns[j], 
              color = 'black', fontsize = 12)

ax.invert_yaxis()
ax.set_ylabel('Depth [km]')
ax.set_xlabel('Normalized Intensity')
ax.set_facecolor('lightgray')
ax.set_title('{} - {} Fault Distance'.format(dist_range[0], dist_range[1]))
