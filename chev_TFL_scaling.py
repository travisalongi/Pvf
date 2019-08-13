# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:58:25 2019
Build upon mini tfl scaling
Check scaling relationships for large Chevron Volume
@author: talongi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import definitions


#workdir = '/Volumes/gypsy/Palos_Verdes_Fault_TA/TFL_volume/' #macbook
workdir = 'E:\Palos_Verdes_Fault_TA\TFL_volume'
os.chdir(workdir)
tfl_file = 'chev_2019-07-30.dat'
fault_file = 'c_pvf.dat'
dist_file = 'chev_dist_pts_to_fault.txt'

values, xy, z = definitions.load_odt_att(tfl_file) #tfl data
Xf, Yf, Zf, df_f = definitions.load_odt_fault(fault_file,n_grid_points=2000, int_type='linear') #central fault

# set fault points from interpolated data - filter out rows with nan in z
fault_points = np.column_stack((Xf.flat, Yf.flat, Zf.flat))
fault_points = fault_points[~np.isnan(fault_points).any(axis = 1)]

calc_dist = input('calculate distances? (y/n)')
if calc_dist == 'y':
    print('calculating min distances, this may take a while.')
    distances = definitions.calc_min_dist(values, xy, fault_points)

    save_text = input('would you like to save the result? (y/n)  ')
    if save_text == 'y':
        print('results saved')
        np.savetxt(dist_file, distances, delimiter = ' ')

else:
    distances = pd.read_csv(dist_file, sep = '\s+', header = None)         
    