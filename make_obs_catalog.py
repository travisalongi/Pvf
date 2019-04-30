#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:06:43 2018
Take Hauksoon and Shearer catalog and make a new catalog with events in time 
period of the OBS deployment
2018-12-21 :: change bounds for offshore events only.


@author: talongi
"""

import shapefile
from astropy.io import ascii
from astropy.table import Table
import datetime as dt
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os

#%%
cat_file = '/auto/home/talongi/Pvf/Data_tables/Eq_cat/SCSN_alt_cleaned.txt'
d = ascii.read(cat_file)
print(d.colnames)

#%% Import shapefile and format it for plotting
shp_file = '/auto/home/talongi/Cascadia/Data_tables/USGS_Base_Layer/NOS80k.shp'
sf = shapefile.Reader(shp_file)

shape_ex = sf.shape(0)

# set empty array
x_lon = np.zeros((len(shape_ex.points),1))
y_lat = np.zeros((len(shape_ex.points),1))
for ip in range(len(shape_ex.points)):
    x_lon[ip] = shape_ex.points[ip][0]
    y_lat[ip] = shape_ex.points[ip][1]

#%% import oilplatform survey perimeter and fault location
rig_file = '/auto/home/talongi/Pvf/Data_tables/oil_rig_loc.txt' 
rig = ascii.read(rig_file)
r_lat = rig['col7']
r_lon = rig['col8']

surv_file = '/auto/home/talongi/Pvf/Data_tables/chev_survey_coords.txt'
surv = ascii.read(surv_file)
surv_lat = surv['col2']
surv_lon = surv['col1']

flt_file = '/auto/home/talongi/Pvf/Data_tables/fault_location/pvf_lat_lon_ordered.txt'
flt = ascii.read(flt_file)
flt_lon = flt['col1']
flt_lat = flt['col2']
       
obs_loc = np.matrix('33.717 -118.8729; 33.5503 -118.4177; 33.2085 -118.4802; 33.1329 -118.1514')
#%%
yr = np.asarray(d['col1'])
mo = np.asarray(d['col2'])
dy = np.asarray(d['col3'])
hr = np.asarray(d['col4'])
mn = np.asarray(d['col5'])
sec = np.asarray(d['col6'])
lat = d['col7']
lon = d['col8']
depth = d['col9']
mag = d['col10']

N = len(yr)

# format dates
count = 0
date_vec = []
for i in d:
    # deals with second format for datetime   
    b = math.modf(i[5]) # splits into # and decimal
    b_sec = int(b[1])
    b_micro_sec = int(b[0] * 1e6) # must convert to microsec
    
    b_min = i[4]
    
    if b_sec == 60 or b_min == 60 or b_min <= 0: # some errors in the times 
        do = dt.datetime(1970,1,1) # this date indicates timing error
        date_vec.append(do)
            
    else:
        do = dt.datetime(i[0],i[1],i[2],i[3],b_min,b_sec,b_micro_sec)
        date_vec.append(do)
    
    count += 1

date_vec = np.asarray(date_vec) # convert to numpy array

#%% #mask off data of interest

# OBS deployment
start = dt.datetime(2010,8,25)
end = dt.datetime(2011,9,9)

# region of interest
n_lat = 33.75
s_lat = 33.0
w_lon = -118.7
e_lon = -117.7
#n_lat = 34
#s_lat = 33.0
#w_lon = -118.7
#e_lon = -117.7

date_vec = np.asarray(date_vec)
mask = (date_vec > start) & (date_vec < end) & (lat < n_lat) & (lat > s_lat) & (lon > w_lon) & (lon < e_lon)
print('number of events in time period', sum(mask))

#%% plot data // look at it

# change global font
matplotlib.rcParams['font.family'] = 'serif'
coast_mask = (y_lat < n_lat) & (y_lat > s_lat)

# plot coastline
plt.plot(x_lon[coast_mask], y_lat[coast_mask], 'k.',markersize = 0.6, alpha = 0.7)

# plot rig locs
plt.plot(r_lon,r_lat, 's', 
         markersize = 6,
         color = 'lawngreen',
         markeredgecolor = 'black',
         linewidth = 0.5)

# plot obs locaitons
plt.plot(obs_loc[:,1], obs_loc[:,0], 'v',
         color = 'navy',
         markersize = 10,
         markeredgecolor = 'silver',
         linewidth = 0.5)

# plot survey bounds
plt.plot(surv_lon,surv_lat, dashes = [6, 2])

# plot fault location
plt.plot(flt_lon, flt_lat, dashes  = [4, 1], color = 'crimson')

# plot events
plt.scatter(lon[mask], lat[mask], 7*mag[mask]**2.5, depth[mask],
                  alpha = 0.75,
                  cmap = 'RdYlBu', 
                  edgecolors = 'black',
                  linewidth = 0.5)



plt.axis('equal')
plt.xlim([w_lon,e_lon])
plt.ylim([33.0, 34.0])
cbar = plt.colorbar()

plt.title('Seismicity Aug.2010 - Sept.2011', fontsize = 16)
plt.xlabel('Longitude',fontsize = 14)
plt.ylabel("Lattitude", fontsize = 14)

cbar.set_label('Depth', fontsize = 13)

plt.show()

#%% Make Data table of results
d_masked = d[mask]
date_masked = date_vec[mask]

t = Table([date_masked, d_masked['col7'], d_masked['col8'], d_masked['col9'], d_masked['col10']],
           names=['Date/Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude'])
    

t_filename = 'h&s_cat_obs_deploy_smaller.txt'
os.chdir('/auto/home/talongi/Pvf/Data_tables')
t.write(t_filename, format='ascii.fixed_width_two_line',overwrite=True)

