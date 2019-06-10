
# -*- coding: utf-8 -*-
"""Running the Sonar Cocktail Party Nightmare:
    Checking the effect of changing different variables
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

def simulate_each_variable(variable_and_value,  num_replicates = 1, kwargs=common_kwargs):
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
    
    variable_name, variable_value = variable_and_value
    simoutput_container = {(variable_name,i) : None for i in range(num_replicates)}

    # set the varying factor - group size 
    common_kwargs[variable_name] = variable_value
    print('RUNNING ' + variable_name + str(variable_value)+' bat simulations now')
    for replicate_run in xrange(num_replicates):
        num_echoes, sim_output = run_CPN(**kwargs)
        simoutput_container[(variable_value, replicate_run)] = sim_output
    picklefilename = 'results//'+str(variable_value)+'bats_CPN_'+unique_name+'.pkl'
    print('did ' + variable_name+' '+str(variable_value)+ ' in ' + str(time.time()-start))
    try:
        with open(picklefilename, 'wb') as picklefile:
            pickle.dump(simoutput_container, picklefile)
        return(True)
    except:
        raise IOError('UNABLE TO SAVE PICKLE FILE!!')
        return(False)



def wrapper_each_group_size(groupsize):
    output =  simulate_each_variable(groupsize)
    return(output)

if __name__ == '__main__':
	common_kwargs['Nbats'] = 25
    # run simulations for all group sizes of interest
	group_sizes = [('heading_variation',5)] *4
	start = time.time()
	pool = Pool(4)
	#all_outputs = pool.map(wrapper_each_group_size, group_sizes)
	all_outputs = map(wrapper_each_group_size, group_sizes)
	print('OVERALL SIMS TOOK', time.time()-start )


# check 