# -*- coding: utf-8 -*-
"""Plotting the raw data : spatial unmasking function and temporal masking
function
Created on Thu Jan 11 14:17:50 2018

@author: tbeleyur
"""
import pandas as pd
import matplotlib.pyplot as plt

spatial_releasefn = pd.read_csv('spatial_release_fn.csv').iloc[:,1:]

temporalmasking_fn = pd.read_csv('temporal_masking_fn.csv').iloc[:,1:]

plt.plot(spatial_releasefn['deltatheta'],spatial_releasefn['dB_release'],'-',
                                             )
plt.grid()

plt.figure(2)
plt.plot(temporalmasking_fn['timegap_ms']*10**3,temporalmasking_fn['dB_leveldiff'],
         '-')
plt.xlim(30,-3)
plt.grid()