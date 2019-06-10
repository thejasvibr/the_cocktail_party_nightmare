# -*- coding: utf-8 -*-
""" Changes a single variable and runs 10,000 runs of it. 
Created on Mon Jun 10 13:55:43 2019

@author: tbeleyur
"""

import sys 
sys.path.append('..//..//poisson-disc-master//')
sys.path.append('..//..//')
sys.path.append('..//')
from multiprocessing import Pool
import pickle 
import time
from commong_hearing_calling_directionality import hearing_directionality_fn, call_directionality_fn

from simulate_effect_of import wrapper_each_variable


if __name__ == '__main__':
    start = time.time()
    # load the common simulation parameters 
    common_paramsfile = '..//commonsim_params.pkl' 
    with open(common_paramsfile, 'rb') as commonfile:
        common_kwargs = pickle.load(commonfile)

    common_kwargs.keys()
    common_kwargs['call_directionality'] = call_directionality_fn
    common_kwargs['hearing_directionality'] = hearing_directionality_fn
    num_bats = [20,25,30]    
    # run simulations for all group sizes of interest
    var_and_value = []
    for each_nbats in num_bats:
        for i in range(10): # make sure overall 10,000 sims are run 
            var_and_value.append((('Nbats',each_nbats), common_kwargs)) 
    start = time.time()
    #pool = Pool(4)
    #all_outputs = pool.map(wrapper_each_group_size, group_sizes)
    all_outputs = map(wrapper_each_variable, var_and_value)
    print('OVERALL SIMS TOOK', time.time()-start )
