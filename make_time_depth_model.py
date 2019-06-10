# -*- coding: utf-8 -*-
"""
read in las files -- develop time depth model

author: talongi
"""

import os
import lasio
import pandas as pd
import matplotlib.pyplot as plt

work_dir = 'D:\\Palos_Verdes_Fault_TA\\Beta_logs\\request_1'

os.chdir(work_dir)
files = os.listdir(work_dir)


fig = plt.figure(figsize = (10,10))
counter = 1
for i in files:
    las_file = lasio.read(i)
    las_file_name = i[3:-13]
    
    depth = las_file.depth_m
    delta_time = las_file['DTWS'] # in [us/ft] micro seconds / foot
    dt_s = delta_time / 1e6 / 0.3048 #convert to [s/m]
    V = 1/dt_s
    
    #initiate subplt
    sp = plt.subplot(1, len(files), counter)
#    sp.plot(delta_time,depth, linewidth = 0.2, color = 'k')
    sp.plot(V,depth, linewidth = 0.2, color = 'k')
    
    #hardwired...
    if counter == 1:
        plt.ylabel('Depth [m]', fontsize = 20)
    plt.ylim([0,3e3])    

    fig.gca().invert_yaxis()
    
    counter += 1
    
#%%