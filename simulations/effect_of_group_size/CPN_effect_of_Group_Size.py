# -*- coding: utf-8 -*-
"""Running the Sonar Cocktail Party Nightmare:
    Checking the effect of Group Size. 

Created on Sat Jun 08 16:59:26 2019

@author: tbeleyur
"""
import hashlib
import multiprocessing as mp
from multiprocessing import Pool
import time
import uuid
import os
import sys
sys.path.append('..//..//poisson-disc-master//')
sys.path.append('..//..//')
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
    
    thanks to Raymond Hettinger for the integer hashing comment
    https://stackoverflow.com/a/16008760/4955732
    '''
    start = time.time()
    # a little bit of circus to ensure this code can still be run parallely 
    # and not result in repeated seeds!! 
    file_uuid = str(uuid.uuid4())
    unique_seed = int(hashlib.sha1(file_uuid).hexdigest(), 16) % (10 ** 8)
    np.random.seed(unique_seed)
    unique_name = 'uuid_'+file_uuid+'_numpyseed_'+str(unique_seed)
    simoutput_container = {(group_size,i) : None for i in range(num_replicates)}

    # set the varying factor - group size 
    common_kwargs['Nbats'] = group_size
    print('RUNNING '+str(group_size)+' bat simulations now')
    for replicate_run in xrange(num_replicates):
        num_echoes, sim_output = run_CPN(**kwargs)
        simoutput_container[(group_size, replicate_run)] = sim_output
    picklefilename = 'results//'+str(group_size)+'bats_CPN_'+unique_name+'.pkl'
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
group_sizes = [5,10]
start = time.time()
#pool = Pool(4)
#all_outputs = pool.map(wrapper_each_group_size, group_sizes)
all_outputs = map(wrapper_each_group_size, group_sizes)
print('OVERALL SIMS TOOK', time.time()-start )


# check 