# -*- coding: utf-8 -*-
""" Quantifying glimpse probability as number of echoes and 
number of calls increase. 

This script calculates the P(median glimpse) quantifies how 
this changes w reference to the number of echoes and number of maskers 
present. 

Created on Thu Oct 25 18:00:17 2018

@author: tbeleyur
"""

import copy
import datetime as dt
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle
import time

from the_cocktail_party_nightmare_MC import run_multiple_trials, calc_echoes2Pgeqechoes

np.random.seed(1)

### Set the biologically relevant parameters #######

# load the temporal-masking function
tempmasking_fn = pd.read_csv('data//temporal_masking_fn.csv').iloc[:,1:]
print('loaded tempmasking')
# load the spatial-release function
spatialrelease_fn = pd.read_csv('data//spatial_release_fn.csv').iloc[:,1:]

# set up simulation parameters
numtrials = 5
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
num_target_echoes = np.arange(29,33,2)
num_masker_calls = np.arange(1,33,2)





### Create one simulation per echo and masker number combination:

#wrapper function that accepts number of target echoes as only input.
def num_echoes_wrapper(num_echoes):
    '''Wrapper function that runs run_multiple_trials
    so that you can use Pool.map to run parallelised calculations.s
    '''
    print(num_echoes,' being calculated now')

    output = run_multiple_trials(numtrials, num_masker_calls,
                                             tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=True,
                                             echo_level_range = echorange,
                                             with_dirnlcall = call_directionality,
                                                  call_level_range = callrange,
                                                 num_echoes = num_echoes)
    return(num_echoes, output)

parallel=False
print('Starting time ..')
start_pll = time.time()
if parallel:
    print('Serial starting:')
    pool = Pool(processes=4)
    all_echoes = pool.map(num_echoes_wrapper, num_target_echoes.tolist())
    
else:
    print('parallel starting:')
    all_echoes = map(num_echoes_wrapper, num_target_echoes.tolist())

print('parallel time: ',time.time()-start_pll)



# calculate the fraction of simulations where there was a glimpse :
glimpse_prob = {}
for i, num_echo in enumerate(num_target_echoes):
    glimpse_echoes = int(np.median(range(1,all_echoes[i][0]+1)))
    echoes_heard = all_echoes[i][1]
    peq_numglimpse = calc_echoes2Pgeqechoes(echoes_heard, all_echoes[i][0],
                                            glimpse_echoes)
    glimpse_prob[int(all_echoes[i][0])] = peq_numglimpse

## Save the glimpse probability dictionary :
#with open('glimpse_probs.pkl', 'wb') as probs_file:
#    pickle.dump(glimpse_prob, probs_file)



plt.figure()
for num_echoes, probs in glimpse_prob.iteritems():
    plt.scatter(num_masker_calls, np.tile(num_echoes, num_masker_calls.size),
                s=(1-probs) *5000)
    
    





