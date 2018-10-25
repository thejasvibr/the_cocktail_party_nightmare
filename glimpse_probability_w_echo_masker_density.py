# -*- coding: utf-8 -*-
""" Quantifying glimpse probability as number of echoes and 
number of calls increase. 

This script calculates the P(median glimpse) quantifies how 
this changes w reference to the number of echoes and number of maskers 
present. 

Created on Thu Oct 25 18:00:17 2018

@author: tbeleyur
"""


import multiprocessing as mp
from multiprocessing import Pool
import time
import copy
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
import pandas as pd
from the_cocktail_party_nightmare_MC import run_multiple_trials, calc_echoes2Pgeqechoes

### Set the biologically relevant parameters #######

# load the temporal-masking function
tempmasking_fn = pd.read_csv('data//temporal_masking_fn.csv').iloc[:,1:]
print('loaded tempmasking')
# load the spatial-release function
spatialrelease_fn = pd.read_csv('data//spatial_release_fn.csv').iloc[:,1:]

# set up simulation parameters
numtrials = 10**4
#call_densities = 5*np.arange(1,8)
call_densities = np.insert( np.arange(1,7)*5,0,1)
echorange = (60,82)
callrange = (100,106)

asym_param = 7
call_directionality = {'A':asym_param}

# parameters required to implement group geometry - a roughly equidistantly
# placed group of echolocating bats emerging together :
nbr_distance = 0.5
call_level = {'intensity':100,'ref_distance':1.0}
poisdisk_params = {  'source_level' : call_level, 'min_nbrdist': nbr_distance  }

# set number of echoes and calls:
num_target_echoes = np.arange(1,15,2)
num_masker_calls = np.arange(1,15,2)



### Create one simulation per echo and masker number combination:
sim_scenarios = {}
for echo_num in num_target_echoes:
    for masker_num in num_masker_calls:
        scenario_tuple = (echo_num, masker_num)
        sim_scenarios[scenario_tuple] = lambda : run_multiple_trials(numtrials,
                                             masker_num,
                                             tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=True,
                                             echo_level_range = echorange,
                                        with_dirnlcall = call_directionality,
                                                call_level_range = None,
                                                poisson_disk = poisdisk_params,
                                                num_echoes=echo_num)
