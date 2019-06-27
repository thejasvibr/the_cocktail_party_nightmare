# -*- coding: utf-8 -*-
"""Plotting the raw data : spatial unmasking function and temporal masking
function
Created on Thu Jan 11 14:17:50 2018

@author: tbeleyur
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000

spatial_releasefn = pd.read_csv('spatial_release_fn.csv').iloc[:,1:]

temporalmasking_fn = pd.read_csv('temporal_masking_fn.csv').iloc[:,1:]
#plt.figure(1)
#plt.plot(spatial_releasefn['deltatheta'],spatial_releasefn['dB_release'],'-',
#         )
#plt.xlabel('Echo-masker angular separation, $^{\circ}$', fontsize=15);
#plt.ylabel('Spatial unmasking, dB', fontsize=15);
#plt.yticks(np.arange(-30,6,6), fontsize=12) ;plt.grid();plt.xticks(np.arange(0,30,5), fontsize=12)

# load the temporal masking function:
with open('temporal_masking_function.pkl', 'rb') as tempmasking:
    fwd, simult, bkwd = pickle.load(tempmasking)
temp_fn = np.concatenate((fwd,[simult]*3000,bkwd))
temp_resolution = 10**-6
plt.figure(2, figsize=(12,9))
plt.plot(np.linspace(0, temp_fn.size*temp_resolution, temp_fn.size), temp_fn)
plt.xticks(np.array([0,fwd.size,(fwd.size+3000), temp_fn.size])*temp_resolution,
            np.round([fwd.size*temp_resolution,0,0,-(bkwd.size*temp_resolution)],3),
            rotation=-60, fontsize=12)
plt.grid();plt.ylim(-36,-6);plt.yticks(np.arange(-35,-6,3),fontsize=12)
plt.ylabel('Echo-masker relative intensities, \n dB ', fontsize=15)
plt.xlabel('Temporal masking time window, milliseconds', labelpad=-40,fontsize=15)
#forward masking
plt.text(fwd.size*temp_resolution*0.5,-23, 'Forward masking region', rotation=30)
#simultaneous masking
plt.text((fwd.size-1500)*temp_resolution,-7.5, 'Simultaneous masking region')
# backward masking 
plt.text((fwd.size+4000)*temp_resolution,-12, 'Backward masking region', rotation=-80)