# -*- coding: utf-8 -*- python3
"""
Created on Mon Apr  8 10:27:49 2019
adaptation slight update from TFL_scaling
@author: T. Alongi
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic
from sklearn.neighbors import NearestNeighbors

workdir =  '/Volumes/gypsy/Palos_Verdes_Fault_TA/TFL_volume'
#workdir =  '/auto/home/talongi/Pvf/TFL_volume'
#workdir = 'F:\\Palos_Verdes_Fault_TA\\NN_fault_distance\\Odt_export'
os.chdir(workdir)
print('Files in this directory:', os.listdir(workdir))

vmod_file = 'depth_time_mod.txt'
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
hdr_names = np.array(['x', 'y', 'z', 'stick_ind', 'node_ind'])

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

# check that data is loaded  
try:
    dist_arr
except NameError:
    dist_arr = pd.read_csv('calc_dist_pts_to_fault.txt', 
                           header = None, sep = '\s+')
    dist_arr = dist_arr.values
    print('Data loaded from text file')
else:
    print('Distance data array loaded')
      
 #%% All data 
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

try:
    dist_east
except NameError:
    dist_east = pd.read_csv('calc_dist_east_of_fault.txt', sep = '\s+')
    dist_east = dist_east.values
    print('Data loaded from text file')
else:
    print('Distance data array loaded')

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
n_divisions = 7 # times to divide up data
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


#%%rotate data to fault strike
def rotate_data(theta, data):
    """ rotates points in xy plane counter-clock wise 
    by angle theta (degrees)"""
    #rotation matrix
    theta = np.radians(theta)
    c,s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s,c)))
    
    #move data to origin
    center_point = data[int(len(data)/2)]
    data_origin = data - center_point
    data_rotated = np.matmul(data_origin, R)
    data_transformed = data_rotated + center_point
    return data_transformed

def collapse_smooth_data(data, min_samp, max_samp, winsize):
    """takes tfl data, slice by depth in samples
    calculate averages in z-dir, then averages in the crsline dir
    smooth the result and return"""
    d_trim = data[:,min_samp:max_samp]    
    z_mean = np.nanmean(d_trim, axis = 1)
    z_shape = z_mean.reshape((172,730))
    d_mean = np.nanmean(z_shape, axis = 0) 
    df = pd.DataFrame(d_mean)
    d_smooth = df.rolling(window = winsize, min_periods = int(winsize/4), win_type = None, center = True).mean()
    return d_smooth

def fracture_density_depth(data, threshold, winsize = 2, low_lim = [], high_lim =[]):
    """takes data and calculates fracture density 
    N above a threshold / N total in trace"""
    shp = data.shape
    densities = np.full(shp[1], np.nan)
    for i in np.arange(shp[1]):
        col = data[:,i]
        col = np.where(np.isnan(col), -1, col) #replace nan with -1
        trim_col = col.reshape(172,730)[low_lim:high_lim,:]
        count_above_threshld = np.sum(trim_col > threshold)

        if count_above_threshld <= 0:
            print('no values above threshold')
        n_trimed = trim_col.size
        fracture_density = count_above_threshld / n_trimed
        densities[i] = fracture_density
        # calculate mode
        hist = np.histogram(fracture_density, bins = 'auto')
        mode = hist[1][np.argmax(hist[0])]               
    df = pd.DataFrame(densities)
    d_smooth = df.rolling(window = winsize, center = True).mean()
    return d_smooth.values, mode

def plot_slice(data, xy_data, sample, threshold):
    xy_coords = xy_data[:len(data),:]
    z_slice = data[:,sample]
    plt.scatter(xy_coords[:,0], xy_coords[:,1], s = [], c = z_slice, vmin = threshold) with 

xy_rot = rotate_data(10, xy_arr)

# modify matrix dimensions
dimensions = (730,172)
new_len = 730*172
tfl = tfl_arr[:,2:]
tfl_new = tfl[:new_len,:]

plt.close('all')
n_divisions = 5 # times to divide up data
min_depth = 200 #meters
max_depth = 1800 #meters
min_sample = f_d(min_depth) / 4 #convert to twt to sample
max_sample = f_d(max_depth) / 4 
sample_arr = np.linspace(min_sample, max_sample, n_divisions + 1)

smoothing = 40
xmax = 500
fig = plt.figure(10, figsize = (8.5,11))
for i,x in enumerate(sample_arr):
    if i == n_divisions:
        ax.set_ylim([0.04, 0.11])
        ax.set_xlim([0, xmax])

    else:
        ax = plt.subplot(n_divisions,1,i+1)
        ax.plot(collapse_smooth_data(tfl_new, int(sample_arr[i]), int(sample_arr[i+1]), smoothing))
        ax.set_ylim([0.04, 0.11])
        ax.set_xlim([0, 450])
    
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([])
ax.spines['bottom'].set_visible(True)
#ax.set_xticks(np.arange(0,xmax,50))
                
#%% single plot
n_divisions = 7 # times to divide up data
min_depth = 300 #meters
max_depth = 2000 #meters
min_sample = f_d(min_depth) / 4 #convert to twt to sample
max_sample = f_d(max_depth) / 4 
sample_arr = np.linspace(min_sample, max_sample, n_divisions + 1)
cmap = plt.cm.Spectral(np.linspace(0.05,0.95,n_divisions))
plt.close('all')
smoothing = 80
fig = plt.figure(10, figsize = (8.5,11))
for i,x in enumerate(sample_arr):
    if i == n_divisions:
        print('skip')
    else:
        plt.plot(np.arange(730) * 13.1, 
                 collapse_smooth_data(tfl_new, int(sample_arr[i]), int(sample_arr[i+1]), smoothing), 
                 color = cmap[i],
                 label = ('Depth = %d - %d [m]' % (z_arr[int(sample_arr[i])], z_arr[int(sample_arr[i+1])] )))
plt.xlabel('Distance in Vol. West to East (p[, which  [m]')
plt.ylabel('Mean Fracture Density')
plt.legend(frameon = False, fontsize = 8)

#%% calc fracture density(depth)
plt.close('all')
smoothing = 5
thresholds = np.arange(0.45, 0.96, 0.1)
cmap = plt.cm.cividis(np.linspace(0,1,len(thresholds)))
plt.figure(11, figsize = (8.5,11))
for i, thresh in enumerate(thresholds):
    plt.loglog(z_arr, 
             fracture_density_depth(tfl_new,thresh, smoothing, low_lim = 60, high_lim = 420)[0],
             '-',
             c = cmap[i],
             label = 'TFL > %2.2f' % thresh)
plt.xlabel('Depth Below Seafloor [m]')
plt.ylabel('Fracture Density')
plt.legend(frameon = False, fontsize = 8)
plt.xlim([300,2600])

#%% make some plots of tfl along various z-slices
plt.close('all')
sampling = np.arange(150,500,80)
for i in sampling:
    plt.figure()
    plot_slice(tfl_new, xy_arr, i, 0.1)
    plt.title('Depth = ' + str(int(z_arr[i])))

