# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:14:19 2019

@author: USGS Kluesner
"""

import os
import time
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp

def bootstrap(array, iterations, array2 = []):
    """takes a 1d array as input, bootstraps the array (randomly sample array
    with replacement) and returns a randomly sampled array, where each row is 
    an iterations """
    array = array.flatten()

    if type(array2) == list:
        boot_arr = np.zeros((iterations, array.size))

        for i in np.arange(iterations):
    #    for i in boot_arr:
            random_index = np.random.randint(0, 
                                            high = array.size, 
                                            size = array.size)
            
            boot_arr[i,:] = array.flatten()[random_index] #each row is a single iterations
        return boot_arr
    elif len(array) == len(array2):
        boot_arr = np.zeros((iterations, array.size))
        boot_arr2 = np.zeros_like(boot_arr)
        for i in np.arange(iterations):
    #    for i in boot_arr:
            random_index = np.random.randint(0, 
                                            high = array.size, 
                                            size = array.size)
            
            boot_arr[i,:] = array.flatten()[random_index] #each row is a single iterations
            boot_arr2[i,:] = array2.flatten()[random_index]
        return boot_arr, boot_arr2
    else:
        print('two arrays with different lengths entered')
        
        
def midpoint(array):
    """returns mid point off adjacent array values"""
    x = array
    y = (x[1:] + x[:-1]) / 2
    return y

def rotate_data(theta, data):
    """ rotates points in xy plane clock wise 
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

def fracture_density_depth(data, threshold, sample_rate, 
                           winsize = 2, depth_arr = [], low_lim = [], high_lim =[],
                           min_depth = [], max_depth = [], 
                           fit_data = False,
                           bootstrap_iterations = None,
                           apply_bool_data = False):
    """takes data and calculates fracture density
    N above a threshold / N total in z-slice
    winsize is the range which the data will be smoothed
    depths entered in meters
    bootstrap iterations (integer)"""
    shp = data.shape
    densities = np.full(shp[1], np.nan)
    
    if type(max_depth) == list:
        min_depth = 0
        max_depth = sample2depth(shp[1], sample_rate)
    min_ind = depth2sample(min_depth, sample_rate)
    max_ind = depth2sample(max_depth, sample_rate)  
        
    # calc fracture density for each depth sample in range
    for i in np.arange(min_ind, max_ind, 1):
        # print('min index = {}; max_index = {}'.format(min_ind, max_ind))
        if type(apply_bool_data) == False:                  
            col = data[:,i]            
            col = col[~np.isnan(col)]
            n_total = col.size
            
        else:
            boolean = apply_bool_data
            col = data[:,i][boolean[:,i]]            
            col = col[~np.isnan(col)]
            n_total = col.size
        
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

#        if count_above_threshld <= 0:
#            print(i, ' no values above threshold') #useful for debugging
    print('    N > threshold = %i --- N total = %i' % (count_above_threshld, n_total))
    df = pd.DataFrame(densities)
    d_smooth = df.rolling(window = winsize, center = True).mean()
    d_smooth = d_smooth.values
    D = d_smooth[min_ind:max_ind]
    
    Z = depth_arr[min_ind:max_ind]
    Z = Z.reshape(len(Z),1)
    
    if fit_data == True:
        # fit data            
        D = d_smooth[min_ind:max_ind]
        
        nans = np.isnan(D) #boolean to remove nans from calculation
        print(nans.shape, Z.shape, D.shape, np.sum(~nans), type(nans))
        Z = Z[~nans]
        D = D[~nans]
    
        pf = np.polyfit(np.log10(Z.flatten()),np.log10(D.flatten()),1)
        slope = pf[0]
        inter = pf[1]
        D_fit =  Z**slope * 10**inter
        
        return Z, D, D_fit, slope
    
    else:
        return D, Z
    
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

def plot_slice(data, xy_data, sample, threshold = 0.8, scale_bar_length = 1000):
    """ plot a single depth slice example """
    xy_coords = xy_data[:len(data),:]
    z_slice = data[:,sample]
    fig, ax = plt.subplots(figsize = (14,14))
    plt.scatter(xy_coords[:,0], xy_coords[:,1], s = 0.5, c = z_slice, vmin = threshold)
    
    # scale_bar = [x_scale,y_scale]
    lims = (xy_coords[:,0].max(), xy_coords[:,1].max())
    length = scale_bar_length
    x_scale = [lims[0], lims[0] + length]
    y_scale = [lims[1], lims[1]]
    
    plt.plot(x_scale, y_scale, 'k')
    plt.text(x_scale[0], y_scale[0] + length/2, str(length))
    plt.axis('equal')
    cb = plt.colorbar()
    cb.ax.set_ylabel('Thinned Fault Likelihood')


def depth_time_conversion(array, depth_or_time):
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
        dep = f_t(array)
        return dep
    
    #interpolation function -- depth to twt
    if depth_or_time == 'time':
        f_d = interp1d(v_mod['Depth[m]'], v_mod['TWT[ms]'], 
                       bounds_error = False, fill_value = 'extrapolate')
        t = f_d(array)
        return t


def depth2sample(depth, sample_rate):
    """converts depths entered in meters to sample (column) in volume"""
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_d = interp1d(v_mod['Depth[m]'], v_mod['TWT[ms]'], bounds_error = False, fill_value = 'extrapolate')
#    samp = int(f_d(depth) / 4) # 4 b/c of sampling rate
    samp = int(f_d(depth) / sample_rate )
    return samp


def sample2depth(sample, sample_rate):
    """converts sample number to depth using time depth model"""
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
                   bounds_error = False, fill_value = 'extrapolate')
    depth = f_t(sample * sample_rate) 
    return depth


def load_h5(filename, data_str):
    """returns a list of all arrays"""
    h5f = h5py.File(filename, 'r')
    arrays = []
    for name in data_str:
        var = h5f[name][:]
        arrays.append(var)
    h5f.close()
    return arrays

    
def load_odt_att(textfile, sample_rate):
    """ takes odt text file and formats data
    returns the values, xy_coords for each row, z in depth for each column"""
    arr = pd.read_csv(textfile, sep = '\s+', header = None).values
    xy_coords = arr[:,:2]
    values = arr[:,2:]
    n_col = values.shape[1]    
    z = sample2depth(np.arange(n_col) + 1, sample_rate)
    z = z.reshape(len(z), 1)
    return values, xy_coords, z


def load_odt_fault_interpolate(textfile, n_grid_points = 1000, int_type = 'linear'):
    """ takes odt file in xyz format (z is in twt) and interpolates a 3D surface
    the resolution is limited by number of grid points.
    returns surface data in xyz w/ z in meters format"""
    hdr_names = np.array(['x', 'y', 'z', 'stick_ind', 'node_ind'])
    data = pd.read_csv(textfile, sep = '\t', names = hdr_names)
    #convert twt to depth
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
                   bounds_error = False, fill_value = 'extrapolate')        
    data.z = f_t(data.z) #z now in depth
    xi = np.linspace(min(data.x), max(data.x), n_grid_points)
    yi = np.linspace(min(data.y), max(data.y), n_grid_points)    
    X, Y = np.meshgrid(xi,yi) #make grid
    # interpolate the vlaues of z for all points in rectangular grid
    Z = griddata((data.x, data.y), data.z, (X, Y), method = int_type)
    return X, Y, Z, data

def load_odt_fault(textfile):
    """ loads text file in xyz format from OpenDTect returns pd.Dataframe where
    where all coords (xyz) are in meters"""
    hdr_names = np.array(['x', 'y', 'z', 'stick_ind', 'node_ind'])
    data = pd.read_csv(textfile, sep = '\t', names = hdr_names)
    #convert twt to depth
    vmod_file = 'depth_time_mod.txt'
    v_mod = pd.read_csv(vmod_file, sep = '\s+')
    f_t = interp1d(v_mod['TWT[ms]'], v_mod['Depth[m]'],
                   bounds_error = False, fill_value = 'extrapolate')        
    data.z = f_t(data.z) #z now in depth
    return data


def interpolate_3d(data_to_grid, data_to_interpolate, n_points = 1000):
    """
    data_to_grid: list or tuple of array like data
    data_to_interpolate: array
    """
    x1 = data_to_grid[0]
    x2 = data_to_grid[1]
    
    x1i = np.linspace(min(x1), max(x1), n_points)
    x2i = np.linspace(min(x2), max(x2), n_points)
    
    X1, X2 = np.meshgrid(x1i, x2i)
    
    X3 = griddata((x1, x2), data_to_interpolate, (X1, X2))
    
    return X1, X2, X3

def vertical_fault(fault_file, n_horizontal, z_arr):
    """creates a vertical fault based on the inputs from odt fault file
    this defintion takes the median point on a fault stick, then interpolates
    based on those x,y coordinates, the data within the limits of the 
    fault file for the chosen number of horizontal points, then duplicates
    based on the input of depth array. 
    
    returns a new array that is [(n_horizontal * depth_arr) X 3] dimensions
    where the columns are x,y,z for the vertical version of the fault.
    
    This method is klunky and not really correct. do not use"""
    arr = np.genfromtxt(fault_file)    
    x = arr[:,0]
    y = arr[:,1]
    stick_ind = arr[:,3] 
    unique_sticks = np.unique(stick_ind)
    
    #empty array to fill
    x_bar = np.zeros_like(unique_sticks)
    y_bar = np.zeros_like(unique_sticks)
    for i in unique_sticks:
        mask = [stick_ind == i]
        x_bar[int(i)] = np.median(x[mask])
        y_bar[int(i)] = np.median(y[mask])
    
    #do interpolation    
    f_northing = interp1d(x_bar, y_bar, kind = 'linear',
                                      fill_value = 'extrapolate')
    
    #set up fault points using interpolation
    n_points = n_horizontal
    xnew = np.linspace(x.min(), x.max(), n_points)
    ynew = f_northing(xnew)
        
    z_arr = np.reshape(z_arr, (len(z_arr),1)) #to concat with the xy data

    new_arr = np.empty((0,3)) #set up new array to fill
    
    for i, xy_pair in enumerate(zip(xnew, ynew)):
        x = xy_pair[0]
        y = xy_pair[1]
        xy_tile = np.tile((x,y), (len(z_arr),1))
        xyz = np.concatenate((xy_tile, z_arr), axis = 1)
        
        new_arr = np.append(new_arr, xyz, axis = 0)
        
     #plots for debugging   
#    fig = plt.figure(1, figsize= (18,10))
#    ax = fig.gca(projection = '3d')
#    ax.scatter3D(new_arr[:,0], new_arr[:,1], new_arr[:,2])
#    ax.scatter3D(arr[:,0], arr[:,1], arr[:,2])
    return new_arr

def calc_min_dist(data_arr, data_xy, fault_points, survey_sample_rate, algorithm = 'ball_tree'):
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
    z_arr = sample2depth(z_arr.reshape(len(z_arr),1), survey_sample_rate) # other function that converts sample to depth w/ vel. model.
    fault_points = fault_points[~np.isnan(fault_points).any(axis = 1)] # filter out rows with nan in z
    n_z = len(z_arr)
      
    model = NearestNeighbors(n_neighbors = 1, algorithm = algorithm).fit(fault_points)   
    
    # calculate distances knn method
    for i, xy in enumerate(data_xy):
        tile = np.tile(xy, (n_z,1))
        
        
#        xyz_tile = np.concatenate((np.tile(xy, (n_z,1)), z_arr),1) # make xyz array of data points
        xyz_tile = np.concatenate((tile, z_arr), axis = 1)
        distances, indicies = model.kneighbors(xyz_tile)
        d_min_dist = distances.min(axis = 1)
        fault_points_min_dist = fault_points[indicies].reshape(n_z,3)
        difference = xyz_tile[:] - fault_points_min_dist[:]
        
        dist_arr[i,:] = d_min_dist
        east_arr[i,:] = difference[:,0] #x coords are east/west ... x dist from fault
    print(time.time() - t0)    
    return dist_arr, east_arr
    
def trim_tfl_distance(data, distance_data, min_dist = 0, max_dist= 6000, replace_with = np.nan):
    """ takes attribute data, calculated min distances, and masks off data 
    larger than the max_distance, and returns a new attribute array in same shape
    with nans in place of data beyond max distance"""
    mask = np.logical_and(distance_data < max_dist, distance_data > min_dist)
    tfl_trimmed = np.where(mask, data, replace_with)  
    return tfl_trimmed

def semilogx_fit(x,y,deg=1):
    """ produces y fit vector for semilogx graph & returns slope"""
    fit_dist = x
    pf = np.polyfit(np.log10(x.flatten()), (y), deg)
    slope = pf[0]
    inter = pf[1]
    fit =  np.log10(fit_dist**(slope)) + inter
    return x, fit

def semilogy_fit(x,y,deg=1):
    fit_dist = x
    pf, cov = np.polyfit((x.flatten()), np.log10(y), 1, cov = True)
    slope = pf[0]
    inter = pf[1]
    slope_std_dev = np.sqrt(cov[0,0])
    fit = 10 ** (slope * fit_dist + inter)
    return x, fit, slope, slope_std_dev

def trim_tfl_depth(data, min_depth, max_depth, sample_rate, replace_with = np.nan, easting = None):
    """ takes attribute data and trims off depths that are not of importance, returns matrix with shape
    if east bool is used this selects data that is east of the fault only, if west of fault it will be replaced_with"""
    min_ind = depth2sample(min_depth, sample_rate)
    max_ind = depth2sample(max_depth, sample_rate)
    
    v = data[:, min_ind : max_ind]
    
    if type(easting) == type(None):        
        return v
    
    else:
        v = np.where(easting[:, min_ind : max_ind] > 0, v, replace_with)
        return v

def fault_avg_trace(fault_df):
    """ 
    This takes a fault data frame returned by load_odt_fault.
    This averages the x,y per fault stick, then smooths the data using a rolling window
    Beware that this averages through any dip a fault has and is only for plotting purposes
    """
    stick_ind = fault_df.stick_ind.unique()
    pvf_x = []
    pvf_y = []
    for ind in stick_ind:
        sub = fault_df[fault_df.stick_ind == ind] # sub dataframe
        # median x,y per stick
        x_bar = np.median(sub.x)
        y_bar = np.median(sub.y)
        pvf_x.append(x_bar)
        pvf_y.append(y_bar)
        
    # smooth results
    pvf_x = pd.Series(pvf_x).rolling(window = 1, min_periods=1, win_type='hamming').mean()
    pvf_y = pd.Series(pvf_y).rolling(window = 1, min_periods=1, win_type='hamming').mean()
    return pvf_x, pvf_y
    
    