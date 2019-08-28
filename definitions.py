# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:14:19 2019

@author: USGS Kluesner
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp

def bootstrap(array, iterations):
    """takes a 1d array as input, bootstraps the array (randomly sample array
    with replacement) and returns a randomly sampled array, where each row is 
    an iterations """
    boot_arr = np.zeros((iterations, array.size))
    for i in np.arange(iterations):
#    for i in boot_arr:
        random_index = np.random.randint(0, 
                                        high = array.size, 
                                        size = array.size)
        
        boot_arr[i,:] = array.flatten()[random_index] #each row is a single iterations
    return boot_arr

def bootstrap_parallel(array):
    """takes a 1d array as input, bootstraps the array (randomly sample array
    with replacement) and returns a randomly sampled array, this actually runs
    slower than the above bootstrapping method"""

    random_index = np.random.randint(0, 
                                        high = array.shape[0], 
                                        size = array.shape[0])
    boot_arr = array[random_index]
    return boot_arr

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
    smooth the result and return -- !!only works for shell data!!"""
    d_trim = data[:,min_samp:max_samp]    
    z_mean = np.nanmean(d_trim, axis = 1)
    z_shape = z_mean.reshape((172,730))
    d_mean = np.nanmean(z_shape, axis = 0) 
    df = pd.DataFrame(d_mean)
    d_smooth = df.rolling(window = winsize, min_periods = int(winsize/4), win_type = None, center = True).mean()
    return d_smooth

def fracture_density_depth(data, threshold,  winsize = 2, depth_arr = [], low_lim = [], high_lim =[],
                           min_depth = [], max_depth = [], 
                           bootstrap_iterations = None,
                           apply_bool_data = []):
    """takes data and calculates fracture density
    N above a threshold / N total in z-slice
    winsize is the range which the data will be smoothed
    depths entered in meters
    bootstrap iterations (integer)"""
    shp = data.shape
    densities = np.full(shp[1], np.nan)
    
    if type(max_depth) == list:
        min_depth = 0
        max_depth = sample2depth(shp[1])
    min_ind = depth2sample(min_depth)
    max_ind = depth2sample(max_depth)  
        
    # calc fracture density for each depth sample in range
    for i in np.arange(min_ind, max_ind, 1):
        if type(apply_bool_data) == list:                  
            col = data[:,i]
            n_total = col.size
            col = col[~np.isnan(col)]
        else:
            boolean = apply_bool_data
            col = data[:,i][boolean[:,i]]
            n_total = col.size
            col = col[~np.isnan(col)]
        
        if len(low_lim) > 0 or len(high_lim) > 0:
            trim_col = col.reshape(172,730)[low_lim:high_lim,:]
            count_above_threshld = np.sum(trim_col > threshold)
            n_total = trim_col.size
        
        if bootstrap_iterations != None:            
            bs_col = bootstrap(col.flatten(), bootstrap_iterations)
            
            count_above_threshld = np.sum(bs_col > threshold)
            
            fracture_density = count_above_threshld / n_total
            densities[i] = fracture_density                        
        else:
            count_above_threshld = np.sum(col > threshold)
            
            fracture_density = count_above_threshld / n_total
            densities[i] = fracture_density

        if count_above_threshld <= 0:
            print(i, ' no values above threshold')

    df = pd.DataFrame(densities)
    d_smooth = df.rolling(window = winsize, center = True).mean()
    d_smooth = d_smooth.values
    
    # fit data    
    Z = depth_arr[min_ind:max_ind]
    Z = Z.reshape(len(Z),1)
    D = d_smooth[min_ind:max_ind]
    print(nans.shape, Z.shape, D.shape, np.sum(~nans), type(nans))
    nans = np.isnan(D) #boolean to remove nans from calculation
    
    Z = Z[~nans]
    D = D[~nans]

    pf = np.polyfit(np.log10(Z.flatten()),np.log10(D.flatten()),1)
    slope = pf[0]
    inter = pf[1]
    D_fit =  Z**slope * 10**inter
    
    return Z, D, D_fit, slope
    
#def background_density(data, min_depth, max_depth, threshold = 0):
#    """ calculate background fracture density of subvolume using
#    histogram binning to find the mode
#    >> Not doing what I want !! Do not USE"""
#    min_ind = depth2sample(min_depth)
#    max_ind = depth2sample(max_depth)
#    d = data[:,min_ind:max_ind]
#    d = d.flatten()
#    d = d[~np.isnan(d)]
#    d = d[d != 0]
#    N = d.size
#    print(N)
#    median = np.median(d)
#    hist = np.histogram(d, bins = 'auto')
#    mode = hist[1][np.argmax(hist[0])]
#    bin_edges = hist[1][1:]
#    counts = hist[0]
#    bg_density = np.sum(d > median) / N
#    return median, bg_density, bin_edges, counts

def plot_slice(data, xy_data, sample, threshold):
    xy_coords = xy_data[:len(data),:]
    z_slice = data[:,sample]
    plt.scatter(xy_coords[:,0], xy_coords[:,1], s = [], c = z_slice, vmin = threshold) 


def depth_time_conversion(vmod_file, a, depth_or_time):
    """ import velocity model file, 
    a = values to convert
    enter whether to convert a to depth or time [str]"""    
    # load in velocity model to make conversions
    vmod_file = 'depth_time_mod.txt'    
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    
    #interpolation function -- twt to depth
    if depth_or_time == 'depth':
        f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
                       bounds_error = False, fill_value = 'extrapolate')
        return f_t(a)
    
    #interpolation function -- depth to twt
    if depth_or_time == 'time':
        f_d = interp1d(v_mod['Depth[m]'], v_mod['TWT[ms]'], 
                       bounds_error = False, fill_value = 'extrapolate')
        return f_d(a)

def depth2sample(depth):
    """converts depths entered in meters to sample (column) in volume"""
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_d = interp1d(v_mod['Depth[m]'], v_mod['TWT[ms]'], bounds_error = False, fill_value = 'extrapolate')
    samp = int(f_d(depth) / 4) # 4 b/c of sampling rate
    return samp

def sample2depth(sample):
    """converts depth entered in meters to sample (column) in volume"""
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
                   bounds_error = False, fill_value = 'extrapolate')
    depth = f_t(sample * 4) # 4b/c of sampling rate
    return depth
    
def load_odt_att(textfile):
    """ takes odt text file and formats data
    returns the values, xy_coords for each row, z in depth for each column"""
    arr = pd.read_csv(textfile, sep = '\s+', header = None).values
    xy_coords = arr[:,:2]
    values = arr[:,2:]
    n_col = values.shape[1]    
    z = sample2depth(np.arange(n_col))
    return values, xy_coords, z

def load_odt_fault(textfile, n_grid_points = 1000, int_type = 'linear'):
    """ takes odt file in xyz format (z is in twt) and interpolates a 3D surface
    the resolution is limited by number of grid points.
    returns surface data in xyz (z in meters format"""
    hdr_names = np.array(['x', 'y', 'z', 'stick_ind', 'node_ind'])
    data = pd.read_csv(textfile, sep = '\t', names = hdr_names)
    #convert twt to depth
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
                   bounds_error = False, fill_value = 'extrapolate')        
    data.z = f_t(data.z) #depth now in depth
    xi = np.linspace(min(data.x), max(data.x), n_grid_points)
    yi = np.linspace(min(data.y), max(data.y), n_grid_points)    
    X, Y = np.meshgrid(xi,yi) #make grid
    # interpolate the vlaues of z for all points in rectangular grid
    Z = griddata((data.x, data.y), data.z, (X, Y), method = int_type)
    return X, Y, Z, data

def calc_min_dist(data_arr, data_xy, fault_points, algorithm = 'ball_tree'):
    """calculate min distance from point in volume to point along interpolated fault
    min. dist. calculated w/ knn
    data MxN
    data_xy Mx2 - xy coordinates of data points 
    fault cords Px3 - in xyz format
    returns minimum distances and easting distance from fault
    """
    t0 = time.time()
    dist_arr = np.full_like(data_arr, np.nan) # arr to fill
    east_arr = np.full_like(dist_arr, np.nan) # measure easting of fault
    
    z_arr = np.arange(data_arr.shape[1])
    z_arr = sample2depth( z_arr.reshape(len(z_arr),1)) # other function that converts sample to depth w/ vel. model.
    fault_points = fault_points[~np.isnan(fault_points).any(axis = 1)] # filter out rows with nan in z
    n_z = len(z_arr)
      
    model = NearestNeighbors(n_neighbors = 1, algorithm = algorithm).fit(fault_points)   
    
    # calculate distances knn method
    for i, xy in enumerate(data_xy):
        xyz_tile = np.concatenate((np.tile(xy, (n_z,1)), z_arr),1) # make xyz array of data points
        distances, indicies = model.kneighbors(xyz_tile)
        d_min_dist = distances.min(axis = 1)
        fault_points_min_dist = fault_points[indicies].reshape(987,3)
        difference = xyz_tile[:] - fault_points_min_dist[:]
        
        dist_arr[i,:] = d_min_dist
        east_arr[i,:] = difference[:,0] #x coords are east/west
    print(time.time() - t0)    
    return dist_arr, east_arr
    
def trim_tfl_distance(data, distance_data, min_dist = 0, max_dist= 6000, replace_with = np.nan):
    """ takes attribute data, calculated min distances, and masks off data 
    larger than the max_distance, and returns a new attribute array in same shape
    with nans in place of data beyond max distance"""
    mask = np.logical_and(distance_data < max_dist, distance_data > min_dist)
    tfl_trimmed = np.where(mask, data, replace_with)  
    return tfl_trimmed

    
    