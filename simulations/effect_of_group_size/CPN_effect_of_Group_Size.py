# -*- coding: utf-8 -*-
"""Running the Sonar Cocktail Party Nightmare:
    Checking the effect of Group Size. 

Created on Sat Jun 08 16:59:26 2019

@author: tbeleyur
"""
import multiprocessing as mp
from multiprocessing import Pool
import time
import os
import sys
sys.path.append('..//..//poisson-disc-master//')
sys.path.append('..//')
from the_cocktail_party_nightmare import run_CPN
from commong_hearing_calling_directionality import hearing_directionality_fn, call_directionality_fn
import pickle 

import numpy as np 
np.random.seed(82319)



# load the common simulation parameters 

common_paramsfile = '..//' + 'commonsim_params.pkl' 
with open(common_paramsfile, 'rb') as commonfile:
    common_kwargs = pickle.load(commonfile)

common_kwargs.keys()
common_kwargs['call_directionality'] = call_directionality_fn
common_kwargs['hearing_directionality'] = hearing_directionality_fn

def run_each_groupsize(group_size,  num_replicates = 10, kwargs=common_kwargs):
    '''
    
    '''
    start = time.time()

    simoutput_container = {(group_size,i) : None for i in range(num_replicates)}
    
    # set the varying factor - group size 
    common_kwargs['Nbats'] = group_size
    print('RUNNING '+str(group_size)+' bat simulations now')
    for replicate_run in xrange(num_replicates):
        num_echoes, sim_output = run_CPN(**kwargs)
        simoutput_container[(group_size, replicate_run)] = sim_output
    picklefilename = 'results//' +'group_size_effect_'+str(group_size)+'bats_CPN.pkl'
    print('did ' + str(group_size)+ ' in ' + str(time.time()-start))
    try:
        with open(picklefilename, 'wb') as picklefile:
            pickle.dump(simoutput_container, picklefile)
        return(True)
    except:
        raise IOError('UNABLE TO SAVE PICKLE FILE!!')
        return(False)



def wrapper_each_group_size(groupsize):
    output =  run_each_groupsize(groupsize)
    return(output)


# run simulations for all group sizes of interest
group_sizes = [5,10,15,20,25]
start = time.time()
all_outputs = map(wrapper_each_group_size, group_sizes)
print('OVERALL SIMS TOOK', time.time()-start )

    