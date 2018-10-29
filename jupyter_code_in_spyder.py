# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:52:58 2018

@author: tbeleyur
"""
import multiprocessing as mp
from multiprocessing import Pool
import os
import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
import pandas as pd
from the_cocktail_party_nightmare_MC import run_multiple_trials, calc_echoes2Pgeqechoes

# load the temporal-masking function
tempmasking_fn = pd.read_csv('data//temporal_masking_fn.csv').iloc[:,1:]
# load the spatial-release function
spatialrelease_fn = pd.read_csv('data//spatial_release_fn.csv').iloc[:,1:]

# set up simulation parameters
numtrials = 10
#call_densities = 5*np.arange(1,8)
call_densities = np.arange(1,7)*5
echorange = (60,82)
callrange = (100,106)

asym_param = 7
call_directionality = {'A':asym_param}
start  = time.time()

np.random.seed(1)
print('#### temp masking only #################')
only_temp_masking = run_multiple_trials(numtrials, call_densities, tempmasking_fn, spatialrelease_fn,
                                        spatial_unmasking=False,echo_level_range = echorange,
                                       call_level_range = callrange)
print('#############################')


np.random.seed(1)

with_SpatialUnmasking = run_multiple_trials(numtrials, call_densities, tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=True,echo_level_range = echorange,
                                           call_level_range = callrange)
print('#############################')

np.random.seed(1)

with_CallDirectionality = run_multiple_trials(numtrials, call_densities, tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=False,echo_level_range = echorange,
                                             with_dirnlcall = call_directionality,
                                             call_level_range = callrange)
print('#############################')

np.random.seed(1)

with_SUm_CallDirectionality = run_multiple_trials(numtrials, call_densities, tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=True,echo_level_range = echorange,
                                             with_dirnlcall = call_directionality,
                                                 call_level_range = callrange)
print('#############################')


pgeq3_temp_masking = calc_echoes2Pgeqechoes(only_temp_masking, 5, 3)
pgeq3_SUm = calc_echoes2Pgeqechoes(with_SpatialUnmasking, 5, 3)
pgeq3_CD = calc_echoes2Pgeqechoes(with_CallDirectionality, 5, 3)
pgeq3_SUm_CallDirn = calc_echoes2Pgeqechoes(with_SUm_CallDirectionality, 5, 3)


plt.figure(1,figsize=(12,8))
plt.plot(call_densities,pgeq3_temp_masking,'r*-',linewidth=5,label='no SUm or CD') ;
plt.plot(call_densities,pgeq3_SUm,'g*-',label='SUm')
plt.plot(call_densities,pgeq3_CD,'b*-',label='CD')
plt.plot(call_densities,pgeq3_SUm_CallDirn,'y*-',label='SUm+CD')

plt.legend()
plt.ylim(0,1.25)
plt.show()
