#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 17:20:58 2019

@author: travisalongi
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import definitions as my

#workdir = '/Volumes/gypsy/Palos_Verdes_Fault_TA/TFL_volume/' #macbook
workdir = 'E:\Palos_Verdes_Fault_TA\TFL_volume'
os.chdir(workdir)
TFL_shell_file = 'shell_2019-07-08.dat'
tfl_arr, xy_arr, z_arr = my.load_odt_att(TFL_shell_file)

# modify matrix dimensions
dimensions = (730,172)
new_len = 730*172
tfl = tfl_arr[:,2:]
tfl_new = tfl[:new_len,:]

#%%
#formation tops
y = np.linspace(0.006,0.10, 20)
sea_floor = np.ones_like(y)
pico = np.ones_like(y) * 400 #mbsf
repetto = np.ones_like(y) * 820
monterey = np.ones_like(y) * 1000
basement = np.ones_like(y) * 2150

fm_arr = np.array([sea_floor, pico, repetto, monterey, basement])[:,0]
fm_samp_arr = np.zeros_like(fm_arr)
for i, dep in enumerate(fm_arr):
    fm_samp_arr[i] = my.depth2sample(dep) 
                
#%% single plot
n_divisions = 7 # times to divide up data
min_depth = 300 #meters
max_depth = 2000 #meters
min_sample = my.depth2sample(min_depth) #convert to twt to sample
max_sample = my.depth2sample(max_depth)
sample_arr = np.linspace(min_sample, max_sample, n_divisions + 1)
#sample_arr = fm_samp_arr
#cmap = plt.cm.Spectral_r(np.linspace(0.05,0.95,n_divisions))
cmap = plt.cm.Spectral_r(np.linspace(0,1, len(sample_arr)))
plt.close('all')
smoothing = 80
fig = plt.figure(10, figsize = (8.5,11))
for i,x in enumerate(sample_arr[:-1]):
    if i == n_divisions:
        print('skip')
    else:
        plt.plot(np.arange(730), 
                 my.collapse_smooth_data(tfl_new, int(sample_arr[i]), int(sample_arr[i+1]), smoothing), 
                 color = cmap[i],
                 label = ('Depth = %d - %d [m]' % (z_arr[int(sample_arr[i])], z_arr[int(sample_arr[i+1])] )))
plt.xlabel('Distance in Vol. West to East (p[, which  [m]')
plt.ylabel('Mean Fracture Density')
plt.legend(frameon = False, fontsize = 8)

#%% calc fracture density(depth)
plt.close('all')
smoothing = 5
# center is 220
#ll = 150
#hl = 310
ll = 400
hl = 560
minD = 500
maxD = 2100
thresholds = np.flip(np.arange(0.35, 0.66, 0.05))
cmap = plt.cm.cividis_r(np.linspace(0,1,len(thresholds)),)
slopes_arr = np.zeros_like(cmap)
fig = plt.figure(11, figsize = (8.5,11))
for i, thresh in enumerate(thresholds):
    fd = my.fracture_density_depth(tfl_new,thresh, smoothing, depth_arr=z_arr, 
                                low_lim = ll, high_lim = hl,
                                min_depth= minD, max_depth= maxD)
    slopes_arr[i] = fd[4]
    plt.loglog(fd[0], 
             fd[1],
             '-',
             c = cmap[i],
             label = 'TFL > %2.2f, Slope = %2.2f' % (thresh, fd[4]))
    plt.loglog(fd[0], fd[2], ':',
               c = cmap[i])
    print('background = %f2.2 at threshold = %2.2f' % (fd[3], thresh))

# add formations
plt.loglog(pico, y, 'r', label = 'Pico Top')
plt.loglog(repetto, y, 'g', label = 'Repetto Top')
plt.loglog(monterey, y, 'b', label = 'Monterey Top')
plt.loglog(basement, y, 'k', label = 'Basement Top')

plt.xlabel('Depth Below Seafloor [m]')
plt.ylabel('Fracture Density')
plt.legend(frameon = False, fontsize = 8, ncol = 2)
plt.axis('equal')
plt.ylim([0.005, 0.105])
plt.title('Shell Vol. Fracture Density %i m Wide Region Surrounding Fault' % ((hl-ll)*13.1))
print('Median of the slopes calculated ',np.median(slopes_arr))

   
