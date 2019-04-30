#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:42:23 2019
Want to load in LAS files and begin to make some sense of them
@author: talongi
"""

import lasio
from matplotlib.pyplot import *
import numpy as np
import glob
import os

%matplotlib qt5

workdir = '/auto/home/talongi/Pvf/FOIA_request/Beta_Exploratory_Logs/Beta_Exploratory_Logs'
os.chdir(workdir)
pwd = os.getcwd()

las_files = os.listdir(pwd)


fig = figure(figsize = (18,10))
counter = 1
for file in sorted(las_files):
    las = lasio.read(file)
#    print file, 'min depth', las['DEPT'][0], 'max depth', las['DEPT'][-1]
    
    depth = las.depth_m
    gamma = las['GRWS']
    

    # initiate subplt
    sp = subplot(1, len(las_files), counter)
    sp.plot(gamma,depth, linewidth=0.2, color = 'k')

    # hardwired...
    if counter == 1:
        ylabel('Depth [m]', fontsize = 20)
    ylim([0, 3e3 + 500])
    
    
    fig.gca().invert_yaxis()
    
    counter += 1    

#%%
# for smoothing
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


fig = figure(figsize = (18,10))
counter = 1
for file in sorted(las_files):
    las = lasio.read(file)
#    print file, 'min depth', las['DEPT'][0], 'max depth', las['DEPT'][-1]
    
    depth = las.depth_m
    gamma = las['GRWS']
    

    # initiate subplt
    sp = subplot(1, len(las_files), counter)
    
    sp.plot(gamma, smooth(depth,100) , linewidth=0.2, color = 'r')

    # hardwired...
    if counter == 1:
        ylabel('Depth [m]', fontsize = 20)
    ylim([0, 3e3 + 500])
    
    
    fig.gca().invert_yaxis()
    
    counter += 1    
