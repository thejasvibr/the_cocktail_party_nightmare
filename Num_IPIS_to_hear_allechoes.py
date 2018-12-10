# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 10:49:26 2018

@author: tbeleyur
"""


import copy
import datetime as dt
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import sys,os
sys.path.append(os.path.realpath('..'))
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
numtrials = 10**4
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
num_target_echoes = np.arange(11,17,2)
num_masker_calls = np.arange(1,33,2)





### Create one simulation per echo and masker number combination:

#wrapper function that accepts number of target echoes as only input.
def num_echoes_wrapper(num_echoes):
    '''Wrapper function that runs run_multiple_trials
    so that you can use Pool.map to run parallelised calculations.s
    '''
    print(num_echoes,' being calculated now')

    echoesheard, echoes_id = run_multiple_trials(numtrials, num_masker_calls,
                                             tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=True,
                                             echo_level_range = echorange,
                                             with_dirnlcall = call_directionality,
                                                  call_level_range = callrange,
                                                 num_echoes = num_echoes,
                                                 one_hot=True)
    folder = 'results//the_CPN_nIPIs_to_hear_all_echoes//'
    computer_name = '_mytable_ubuntu_'
    with open(folder+'N_IPIS_to_hear_allechoes'+computer_name+str(num_echoes)+'echoes_.pkl',
              'wb') as nipis_file:
        result_dict ={'num_echoes': num_echoes,
                      'echoes_heard':echoesheard,
                      'echo_ids' : echoes_id}
        pickle.dump(result_dict, nipis_file)


    output = {'num_targetechoes':num_echoes, 'num_echoesheard':echoesheard, 'echo_id':echoes_id}

    return(output)

parallel=True
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
  
    





