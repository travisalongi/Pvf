# -*- coding: utf-8 -*- python3
"""
Created on Mon Apr  8 10:27:49 2019
adaptation of faultsNN.py
@author: T. Alongi
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as ml
from scipy.interpolate import griddata, interp1d
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic
from sklearn.neighbors import NearestNeighbors

workdir =  '/auto/home/talongi/Pvf/TFL_volume'
#workdir = 'F:\\Palos_Verdes_Fault_TA\\NN_fault_distance\\Odt_export'
os.chdir(workdir)
print('Files in this directory:', os.listdir(workdir))

vmod_file = '/auto/home/talongi/Pvf/Data_tables/depth_time_mod.txt'
v_mod = pd.read_csv(vmod_file, sep = '\s+')
#interpolation function -- twt to depth
f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
               bounds_error = False, fill_value = 'extrapolate') 
#interpolation function -- depth to twt
f_d = interp1d(v_mod['Depth[m]'], v_mod['TWT[ms]'], 
               bounds_error = False, fill_value = 'extrapolate') 

# =============================================================================
# Faults
# =============================================================================
# Files
east_file = 'e_pvf.dat'
central_file = 'c_pvf.dat'
west_file = 'w_pvf.dat'

# set header names
hdr_names = pd.array(['x', 'y', 'z', 'stick_ind', 'node_ind'])

# interpolation type
int_type = 'linear'
n_points = int(1e3)

east = pd.read_csv(east_file, sep = '\t', names = hdr_names)
#convert twt to depth
east.z = f_t(east.z)
xi_e = np.linspace(min(east.x), max(east.x), n_points)
yi_e = np.linspace(min(east.y), max(east.y), n_points)
#make grid
X_e, Y_e = np.meshgrid(xi_e,yi_e)
# interpolate the vlaues of z for all points in rectangular grid
Z_e = griddata((east.x, east.y), east.z, (X_e, Y_e), method = int_type)


central = pd.read_csv(central_file, sep = '\t', names = hdr_names)
central.z = f_t(central.z)
xi_c = np.linspace(min(central.x), max(central.x), n_points)
yi_c = np.linspace(min(central.y), max(central.y), n_points)
X_c, Y_c = np.meshgrid(xi_c, yi_c)
Z_c = griddata((central.x, central.y), central.z, (X_c, Y_c), method = int_type)


west = pd.read_csv(west_file, sep = '\t', names = hdr_names)
west.z = f_t(west.z)
xi_w = np.linspace(min(west.x), max(west.x), n_points)
yi_w = np.linspace(min(west.y), max(west.y), n_points)
X_w, Y_w = np.meshgrid(xi_w, yi_w)
Z_w = griddata((west.x, west.y), west.z, (X_w, Y_w), method = int_type)


# =============================================================================
# Plot the Faults
# =============================================================================
plot_faults = False
if plot_faults == True:
    rs = 5
    cs = 5
    lw = 2
    
    fig = plt.figure(1, figsize= (18,10))
    ax = fig.gca(projection = '3d')
    
    # plot east fault
    ax.plot_surface(X_e, Y_e, -Z_e, 
                    rstride = rs, cstride = cs, linewidth = lw, 
                    color = 'blue',
                    shade = True,
                    antialiased = True)
    
    # plot central fault
    ax.plot_surface(X_c, Y_c, -Z_c, 
                    rstride = rs, cstride = cs, linewidth = lw, 
                    color = 'green',
                    shade = True,
                    antialiased = True)
    
    # plot west fault
    ax.plot_surface(X_w, Y_w, -Z_w, 
                    rstride = rs, cstride = cs, linewidth = lw, 
                    color = 'pink',
                    shade = True,
                    antialiased = True)
    ax.set_zlabel('Z [ms]')
    plt.show()


# =============================================================================
#  TFL data
# =============================================================================
# UTM-11 # coords inline/crsline format
TFL_chev_file = 'TFL_chev.dat'
TFL_shell_file = 'TFL_shell.dat'

# formating for sensible headers
coords = np.array([998, 999]) #998 = x (inlines) :: 999 = Y (crslines)
samp_depth = np.arange(0,987,1)
col_nums = np.concatenate((coords,samp_depth))
# load data & preprocess
tfl_shell = pd.read_csv(TFL_shell_file, sep = '\s+', header = None,names = col_nums) 
tfl_shell = tfl_shell.replace(1e30, np.nan) #1e30 = null values
print(tfl_shell.describe()) #check standard statistics
#convert data to an array
tfl_arr = tfl_shell.values 


# set fault points from interpolated data - filter out rows with nan in z
e_fault_points = np.column_stack((X_e.flat, Y_e.flat, Z_e.flat))
e_fault_points = e_fault_points[~np.isnan(e_fault_points).any(axis = 1)]

c_fault_points = np.column_stack((X_c.flat, Y_c.flat, Z_c.flat))
c_fault_points = c_fault_points[~np.isnan(c_fault_points).any(axis = 1)]

w_fault_points = np.column_stack((X_w.flat, Y_w.flat, Z_w.flat))
w_fault_points = w_fault_points[~np.isnan(w_fault_points).any(axis = 1)]

# combine all fault points into single array
all_points = np.vstack([w_fault_points, c_fault_points, e_fault_points])

# xyz arrays
xy_arr = tfl_shell.values[:,:2] # x/y pairs
z_arr = np.arange(len(tfl_arr[0,2:])) * 4 # z values to twt [ms]
z_arr = f_t(z_arr) # convert z to depth [m]
z_arr = z_arr.reshape(len(z_arr),1)
n_z = len(z_arr)

calc_distances = False
if calc_distances == True:
    t0 = time.time()
    dist_arr = np.zeros_like(tfl_arr[:,2:])    
    model = NearestNeighbors(n_neighbors = 1, algorithm = 'ball_tree').fit(all_points)    
    # calculate distances either with scipy cdist or knn method    
    use_scipy = False
    use_kn = True
    for i, xy in enumerate(xy_arr):
        xyz_tile = np.concatenate((np.tile(xy, (n_z,1)), z_arr),1)
        
        if use_scipy == True:
            w_min_dist = cdist(xyz_tile, w_fault_points).min(axis = 1)
            c_min_dist = cdist(xyz_tile, c_fault_points).min(axis = 1)
            e_min_dist = cdist(xyz_tile, e_fault_points).min(axis = 1)
        
        if use_kn == True:        
            d_min_dist = model.kneighbors(xyz_tile)[0].min(axis = 1)
        
        dist_arr[i,:] = d_min_dist        
    print(time.time() - t0)
    np.savetxt('calc_dist_pts_to_fault.txt', dist_arr, delimiter = ' ')

  
try:
    dist_arr
except NameError:
    dist_arr = pd.read_csv('calc_dist_pts_to_fault.txt', sep = '\s+')
    dist_arr = dist_arr.values
    print('Data loaded from text file')
else:
    print('Distance data array loaded')
      
 #%%  
plt.close('all')
tfl_thresh = 0.85
n_divisions = 9
cmap = plt.cm.jet(np.linspace(0,1,n_divisions))
sample_arr = np.linspace(0, len(z_arr), n_divisions)

plt.figure(1, figsize = (8.5,11))
for i in np.arange(len(sample_arr[:-1])):        
    samples = np.arange(int(sample_arr[i]), int(sample_arr[i +1]), 1)    
    d_min = dist_arr[:,samples]
    tfl = tfl_arr[:,2:][:,samples]
    
    mask = (tfl > tfl_thresh)
    tfl_threshold = tfl[mask]
    d_min_threshold = d_min[mask]
    
    # bin data by distance and take median of the the tfl value
    n_bins = np.arange(50, 5000, 25)
    #tfl_avg = binned_statistic(d_min.flatten(), tfl.flatten(), bins = n_bins, statistic = 'mean')
    #tfl_median = binned_statistic(d_min.flatten(), tfl.flatten(), bins = n_bins, statistic = 'median')
    tfl_threshold_count= binned_statistic(d_min_threshold.flatten(), tfl_threshold.flatten(), 
                                          bins = n_bins, 
                                          statistic = 'count')
    tfl_count = binned_statistic(d_min.flatten(), tfl.flatten(), 
                                 bins = n_bins, 
                                 statistic = 'count')
    tfl_count_normalized = tfl_count.statistic/tfl_count.statistic.max()    
    L_bar = tfl_threshold_count.statistic / tfl_count.statistic
    
    
    top = plt.subplot(2,1,1)
    plt.plot(tfl_count.bin_edges[:-1], L_bar, '-o',
               color = cmap[i], alpha = 0.5,
               label = ('Sample range: %d - %d [m]' % (z_arr[samples.min()], z_arr[samples.max()] )))

    bottom = plt.subplot(2,1,2)
    plt.plot(tfl_count.bin_edges[:-1], (1/2) * (tfl_count.statistic + tfl_threshold_count.statistic), '-o',
             color = cmap[i], alpha = 0.5)
 
top.legend(frameon = False)
top.set_title('Shell Volume: Thinned Fault Likelihood > %2.2f' % (tfl_thresh))
top.set_ylabel('Fault Density -- N > %2.2f / N' % (tfl_thresh)) 
bottom.set_ylabel('N in Bin')
bottom.set_xlabel('Distance from fault [m]')

#%% look at points east of fault strands similar to above
calc_distances = False
if calc_distances == True:
    t0 = time.time()
    dist_east = np.full_like(tfl_arr[:,2:], np.nan)        
    model = NearestNeighbors(n_neighbors = 1, algorithm = 'ball_tree').fit(e_fault_points)   
    # calculate distances knn method
    for i, xy in enumerate(xy_arr):
        xyz_tile = np.concatenate((np.tile(xy, (n_z,1)), z_arr),1) # make xyz array
        distances = model.kneighbors(xyz_tile, return_distance = True) # calc dist w/ knn
        closest_fault_points = e_fault_points[distances[1]] #1 is index of e_fault w/ min dist
        closest_fault_points = closest_fault_points.reshape(987,3)
        
        # points east of east fault
        bool_arr = (xyz_tile[:,0] - closest_fault_points[:,0]) > 0             
        index_bool = np.where(bool_arr) # return index of true values   
        # handle empty array
        if distances[0][index_bool].shape[0] == 0:
            continue
        else:
            dist_east[i,index_bool] = np.transpose(distances[0][index_bool])           
    print(time.time() - t0)
    np.savetxt('calc_dist_east_of_fault.txt', dist_east, delimiter = ' ')

#plt.close('all')

tfl_thresh= 0.70
n_divisions = 1 # times to divide up data
n_minimum = 10 # minimum number of traces containing tfl above threshold
min_depth = 200 #meters
max_depth = 2200 #meters
#div_size = 200 # meters division bins

min_sample = f_d(min_depth) / 4 #convert to twt to sample
max_sample = f_d(max_depth) / 4 
#div_sample = f_d(div_size) / 4
sample_arr = np.linspace(min_sample, max_sample, n_divisions + 1)
#sample_arr = np.arange(min_sample, max_sample, div_sample)
cmap = plt.cm.copper(np.linspace(0,1,n_divisions))

plt.figure(1, figsize = (8.5,11))
for i in np.arange(len(sample_arr[:-1])):        
    samples = np.arange(int(sample_arr[i]), int(sample_arr[i +1]))    
    d_min = dist_east[:,samples]
    tfl = tfl_arr[:,2:][:,samples]
    
    mask = (tfl > tfl_thresh)
    tfl_threshold = tfl[mask]
    d_min_threshold = d_min[mask]
    
    # bin data by distance and take median of the the tfl value
    n_bins = np.arange(50, 5000, 40)
    tfl_threshold_count= binned_statistic(d_min_threshold.flatten(), tfl_threshold.flatten(), bins = n_bins, statistic = 'count' )
    n_min_bool = tfl_threshold_count.statistic > n_minimum
    plot_dist = tfl_threshold_count.bin_edges[:-1][n_min_bool] #for distances in plot
    
    tfl_threshold_count = tfl_threshold_count.statistic[n_min_bool] #use data with enough points
    
    
    tfl_count = binned_statistic(d_min.flatten(), tfl.flatten(), bins = n_bins, statistic = 'count')  
    tfl_count = tfl_count.statistic[n_min_bool]
    
    # calc fault density
    rho_F = tfl_threshold_count / tfl_count
    
    
    top = bottom =  plt.subplot(1,1,1)
    plt.plot(plot_dist, rho_F, '-o',
             linewidth = 3, markersize = 3,
               color = cmap[i], alpha = 0.6,
               label = ('E %d - %d [m]' % (z_arr[samples.min()], z_arr[samples.max()] )))
 
top.legend(frameon = False, ncol = 2, fontsize = 8)
top.set_xbound( upper = 3000)
top.set_ybound( upper = 0.2)
top.set_title('Shell Volume :East of East strand: Thinned Fault Likelihood > %2.2f' % (tfl_thresh))
top.set_ylabel('Fault Density -- (N > %2.2f ) / N' % (tfl_thresh)) 
bottom.set_xlabel('Distance from fault [m]')


#%% Look at points east of central fault
calc_distances = False
if calc_distances == True:
    t0 = time.time()
    dist_cent = np.full_like(tfl_arr[:,2:], np.nan)        
    model = NearestNeighbors(n_neighbors = 1, algorithm = 'ball_tree').fit(c_fault_points)   
    # calculate distances knn method
    for i, xy in enumerate(xy_arr):
        xyz_tile = np.concatenate((np.tile(xy, (n_z,1)), z_arr),1) # make xyz array
        distances = model.kneighbors(xyz_tile, return_distance = True) # calc dist w/ knn
        closest_fault_points = e_fault_points[distances[1]] #1 is index of e_fault w/ min dist
        closest_fault_points = closest_fault_points.reshape(987,3)
        
        # points east of central fault
        bool_arr = (xyz_tile[:,0] - closest_fault_points[:,0]) > 0             
        index_bool = np.where(bool_arr) # return index of true values   
        # handle empty array
        if distances[0][index_bool].shape[0] == 0:
            continue
        else:
            dist_cent[i,index_bool] = np.transpose(distances[0][index_bool])           
    print(time.time() - t0)
    np.savetxt('calc_dist_east_of_C-fault.txt', dist_cent, delimiter = ' ')

plt.close('all')
tfl_thresh= 0.70
n_divisions = 1 # times to divide up data
n_minimum = 10 # minimum number of traces containing tfl above threshold
min_depth = 200 #meters
max_depth = 2200 #meters
#div_size = 200 # meters division bins

min_sample = f_d(min_depth) / 4 #convert to twt to sample
max_sample = f_d(max_depth) / 4 
#div_sample = f_d(div_size) / 4
sample_arr = np.linspace(min_sample, max_sample, n_divisions + 1)
#sample_arr = np.arange(min_sample, max_sample, div_sample)
cmap = plt.cm.winter(np.linspace(0,1,n_divisions))

plt.figure(2, figsize = (8.5,11))
for i in np.arange(len(sample_arr[:-1])):        
    samples = np.arange(int(sample_arr[i]), int(sample_arr[i +1]))    
    d_min = dist_cent[:,samples]
    tfl = tfl_arr[:,2:][:,samples]
    
    mask = (tfl > tfl_thresh)
    tfl_threshold = tfl[mask]
    d_min_threshold = d_min[mask]
    
    # bin data by distance and take median of the the tfl value
    n_bins = np.arange(50, 5000, 50)
    tfl_threshold_count= binned_statistic(d_min_threshold.flatten(), tfl_threshold.flatten(), bins = n_bins, statistic = 'count' )
    n_min_bool = tfl_threshold_count.statistic > n_minimum
    plot_dist = tfl_threshold_count.bin_edges[:-1][n_min_bool] #for distances in plot
    
    tfl_threshold_count = tfl_threshold_count.statistic[n_min_bool] #use data with enough points
    
    
    tfl_count = binned_statistic(d_min.flatten(), tfl.flatten(), bins = n_bins, statistic = 'count')  
    tfl_count = tfl_count.statistic[n_min_bool]
    
    # calc fault density
    rho_F = tfl_threshold_count / tfl_count
    
    
    top = bottom =  plt.subplot(1,1,1)
    plt.plot(plot_dist, rho_F, '-o',
             linewidth = 3, markersize = 3,
               color = cmap[i], alpha = 0.6,
               label = ('C %d - %d [m]' % (z_arr[samples.min()], z_arr[samples.max()] )))
 
top.legend(frameon = False, ncol = 2, fontsize = 8)
top.set_xbound( upper = 3000)
top.set_ybound( upper = 0.2 )
top.set_title('Shell Volume- East of Central Strand: Thinned Fault Likelihood > %2.2f' % (tfl_thresh))
top.set_ylabel('Fault Density -- N > %2.2f / N' % (tfl_thresh)) 
bottom.set_xlabel('Distance from fault [m]')


#%%
#fig = plt.figure(4, figsize = (18,10) )
#plt.plot(dist_arr[::500,2:300][mask],nn_arr[::500,2:300][mask],',', alpha = 0.3, color = 'black', markerfacecolor = 'gray')    

fig86, axs = plt.subplots(3,1)
# make arrays correct size
nn_sliced = nn_arr[:,2:]
nn_flat = nn_sliced.flatten()

dist_sliced = dist_arr[:,2:]
dist_flat = dist_sliced.flatten()

#axs[0].hist(nn_flat, bins = 1000, alpha = 0.7)
#axs[0].set_xlim(-0.5, 1.5)
#axs[0].set_xlabel('Pr')
#axs[0].set_ylabel('Counts')

axs[0].plot(dist_flat[::50], nn_flat[::50],',', alpha = 0.02, color = 'darkred')
axs[0].set_ylim(0,1.5)
axs[0].set_xlabel('Distance from fault')
axs[0].set_ylabel('Pr')

axs[1].semilogx(dist_flat[::50], nn_flat[::50],',', alpha = 0.02, color = 'sienna')
axs[1].set_ylim(0,1.5)
#axs[1].set_xlabel(r'$\log(Distance from fault)$')
axs[1].set_ylabel('Pr')

axs[2].loglog(dist_flat[::50], nn_flat[::50],',', alpha = 0.02, color = 'goldenrod')
axs[2].set_ylim(0,1.5)
axs[2].set_xlabel(r'$\log(Distance)$')
axs[2].set_ylabel(r'$\log(Pr)$')

#%%
prob_tol = 0.9
prob_mask = nn_sliced >= prob_tol
 
plt.figure(20, figsize = (18,10))
plt.hist(dist_sliced[prob_mask],bins = 1000)


#%%

""" 
want to calculate the azimuths between points of the fault data
"""
az_e = []
dip_e = []
d_e = []
for i in range(0,  east.shape[0] -1):
    p1 = (east.x[i], east.y[i], east.z[i])
    p2 = (east.x[i+1], east.y[i+1], east.z[i+1])
    p3 = np.asarray(p2) - np.asarray(p1)
    

    dy = east.y[i] - east.y[i+1]
    dx = east.x[i] - east.x[i+1]
    dz = east.z[i] - east.z[i+1]

    dip = (np.rad2deg(np.arctan(dz/dy)))
    az = (np.rad2deg(np.arctan(dy/dx)))
    d = np.linalg.norm(np.asarray(p1) - np.asarray(p2))

    dip_e.append(dip)
    az_e.append(az)
    d_e.append(d)


#%%
plt.close('all')

fig5 = plt.figure(5, figsize = (18,10)) 
plt.scatter(east.x[:100], east.y[:100], s = 10*east.stick_ind[:100], 
            c = east.stick_ind[:100], cmap = 'bwr',
            alpha = 0.5)       
plt.colorbar()
plt.axis('equal')


#%% looking at stick indexes
fig8 = plt.figure(9, figsize = (18,10))
for i in west.stick_ind.unique():
     loop_mask_e = east.stick_ind == i
     loop_mask_c = central.stick_ind == i
     loop_mask_w = west.stick_ind == i
     
     plt.plot(east.x[loop_mask_e], east.y[loop_mask_e])
     plt.plot(central.x[loop_mask_c], central.y[loop_mask_c])
     plt.plot(west.x[loop_mask_w], west.y[loop_mask_w])    
plt.axis('equal')     

#%% Plotting NN data
plt.close()
# n-th layer
sample_depth = 150

figgy = plt.figure(3, figsize = (18,10))
scatter = plt.scatter(dist_arr[:,0], dist_arr[:,1], 
            s = None, 
            c = dist_arr[:,sample_depth], 
            cmap = 'YlGnBu',
            alpha = 0.9)

prob_tol = 0.98
nn_z = nn_sliced[:,sample_depth]
prob_mask = nn_z >= prob_tol
nn_x = nn_arr[:,0]
nn_y = nn_arr[:,1]

plot = plt.plot(nn_x[prob_mask], nn_y[prob_mask], 'r.', markersize = 1.5, alpha = 0.5)
plt.plot(east.x,east.y,'b:', markersize = 1)
plt.plot(central.x,central.y,'b:', markersize = 1)
plt.plot(west.x,west.y,'b:', markersize = 1)

plt.ylim( 3710500,3717500)
plt.colorbar()

#%%
plt.figure(21, figsize = (18,10))
plt.loglog(dist_flat[::20], nn_flat[::20],
           '.',markersize = 1.5,
           alpha = 0.1, 
           color = 'goldenrod')
plt.xlim(10,5e3)
plt.ylim(1e-3,2)
#plt.axis('equal')
plt.xlabel('log(Distance From Fault)')
plt.ylabel('log(Probability')

#%%

bin_size = 50
bins_arr = np.arange(10,5000, bin_size) #50 km bins
n_in_bin = np.zeros_like(bins_arr)
mean_prob = []
mid_dist = []

cum_sum = 0
count = 0
for i in range(len(bins_arr)-1):
    dist_min = dist_sliced >= bins_arr[i]
    dist_max = dist_sliced < bins_arr[i + 1]

    dist_mask = np.logical_and(dist_min, dist_max)
       
    n_in_bin[count] = dist_mask.sum()
    mean_val = np.nanmean(nn_sliced[dist_mask])
    mid_val = (bins_arr[i] + bins_arr[ i + 1])/2

    mean_prob.append(mean_val)
    mid_dist.append(mid_val)
    
    cum_sum += np.sum(dist_mask)
    count += 1
    
print(cum_sum)
#%%
fs = 18
pf_mask = np.logical_and(np.asarray(mid_dist) > 1e2, np.asarray(mid_dist) < 3e3)
pf = np.polyfit(mid_dist, mean_prob,2)# m = pf[0], b = pf[1]
f = np.poly1d(pf)

x = np.linspace(min(mid_dist),max(mid_dist), len(mid_dist))
y = f(x)



plt.figure(22, figsize = (18,10))
plt.plot(mid_dist, mean_prob, ':', color = 'indigo')
plt.scatter(mid_dist, mean_prob, s = (n_in_bin[:-1])**(0.29))
plt.plot(x,y)

plt.figure(23, figsize = (18,10))
ax = plt.gca()

ax.plot(mid_dist, mean_prob, ':', color = 'teal')
ax.scatter(mid_dist, mean_prob, 
           s = (n_in_bin[:-1])**(0.25),
           color = 'lightblue',
           edgecolor = 'navy')
#y = x**pf[0] + 10**pf[1]
#ax.plot(x,y)


ax.set_xscale('log')
ax.set_xlabel(r'$log(Dist)$', fontsize = fs)

ax.set_yscale('log')
ax.set_ylabel(r'$log(Pr)$', fontsize = fs)

#%% old way of calc distances brut force
"""
# set threshold for calculation
threshold = 0.3

start_time = time.time()
d_min = np.full_like(tfl_arr[:,2:], np.nan, dtype = np.double) # no coordinates array of nan
d_min = []
tfl = []
for i in i_th[::25]:
    # set xy coord
    xy = tfl_0[i,:2]
    
    for j in j_th[::5]:
        # set values        
        z = sample_times[j] #z depth in time
        tfl_point = np.append(xy,z) 
        
        # tfl probability
        tfl_value = tfl_arr[i,j+2]
        
        if tfl_value > threshold:
#            print(i,j, tfl_value)
                       
            # calculate distances to each strand
            dist_e = np.linalg.norm(tfl_point - e_fault_points, axis = 1)
            dist_c = np.linalg.norm(tfl_point - c_fault_points, axis = 1)
            dist_w = np.linalg.norm(tfl_point - w_fault_points, axis = 1)
        
            # determine min dist
            minimum_distances = ([dist_w.min(), dist_c.min(), dist_e.min()])
            
            # only want points on the outside of the fault
            if min(minimum_distances) == dist_c.min():
                continue
            if (dist_e.min() < dist_c.min() < dist_w.min()) or (dist_e.min() > dist_c.min() > dist_w.min()):
                d_min.append(min(minimum_distances))
                tfl.append(tfl_value)
                
#            save min value in matrix in same order as orig nn_arr
#            d_min[i,j] = min((d_e_min, d_c_min, d_w_min))
                
end_time = time.time() - start_time
print(end_time)
"""
