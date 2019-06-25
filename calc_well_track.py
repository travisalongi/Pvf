#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:05:10 2019
Makes well track file (.ZPN) in x,y,tvd,md format from oddly formatted deviation file
@author: talongi
"""
#from geo.sphere import destination
import os
import glob
import pandas as pd
import numpy as np

write_2_file = True

work_dir = '/auto/home/talongi/Pvf/Beta_logs/Beta_Deviation'
os.chdir(work_dir)

well_folders = sorted(glob.glob('P*'))

count = 0
for folder in well_folders:
    os.chdir(work_dir + '/' + folder)
    
    directories = os.listdir()
    
    for directory in directories:
        os.chdir(work_dir + '/' + folder + '/' + directory)
        
        # identify different file types - sort for uniformity
        DTA_files = sorted(glob.glob('P*.DTA'), reverse = True) 
        ZPN_file = glob.glob('*.ZPN')
        FND_file = sorted(glob.glob('*.FND'))
                
        if len(DTA_files) > 0 and len(FND_file) > 0: #calc requires both files
            pwd = str(os.getcwd()) + '/'
            # take first file in list to be uniform
            dta_file = DTA_files[0]
            fnd_file = FND_file[0]
            
            # ~Get data from files
            # DTA file - 3 bits of info in the header
            tvd_0 = float(open(pwd + dta_file).readlines()[4].strip())
            x_0 = float(open(pwd + dta_file).readlines()[5].strip())
            y_0 = float(open(pwd + dta_file).readlines()[6].strip())  
            
            # FND file
            df = pd.read_csv(pwd + fnd_file,  skiprows = 2, delim_whitespace = True)
            MD = df.DEPTH
            dMD = np.diff(df.DEPTH)
            dev = np.deg2rad(df.DEV[1:])
            az = np.deg2rad(df.AZI[1:])
            
            # calculate distance & X/Y coordinates
            distance = dMD * np.sin(dev)
            
            dx = distance * np.sin(az)
            dy = distance * np.cos(az)
                       
            x = x_0 + np.cumsum(dx).values
            y = y_0 + np.cumsum(dy).values
            
            X = np.append(x_0, x)
            Y = np.append(y_0, y)
            
            # calculate TVD
            z_0 = np.array(-tvd_0) 
            dz = dMD * np.cos(dev)
            Z = z_0 + np.cumsum(dz)         
            TVD = np.append(z_0, np.cumsum(dz))            
            
            if write_2_file == True:
                # format data
                data_dic = {'x': X, 
                            'y': Y,
                            'tvd': TVD,
                            'md' : MD}
                
                # format file name
                file_name = dta_file[:-4] + '_TA.ZPN'
                            
                # use pandas to write            
                track_data = pd.DataFrame(data = data_dic) 
                track_data.to_csv(pwd + file_name, 
                                  sep = '\t',
                                  float_format = '%.2f',
                                  index = False,
                                  header = False,
                                  encoding = 'utf-8')
            print('Created track file (ZPN) from:' ,dta_file, FND_file)
            count += 1