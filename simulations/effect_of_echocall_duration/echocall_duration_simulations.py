# -*- coding: utf-8 -*-
""" Effect of source level on the CPN
Created on Mon Jun 10 15:04:23 2019

@author: tbeleyur
"""

import sys 
sys.path.append('..//..//poisson-disc-master//')
sys.path.append('..//..//')
sys.path.append('..//')
import multiprocessing
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

    
    # run simulations for all durations
    echocall_durations = np.array([2.5, 1.25, 1]) *10**-3
    var_and_value = []
    for durn in echocall_duration:
        for i in range(1000): # make sure overall 10,000 sims are run 
            var_and_value.append((('echocall_duration', durn), common_kwargs)) 
    start = time.time()
    num_cores = multiprocessing.cpu_count()
    pool = Pool(num_cores);
	all_outputs = pool.map(wrapper_each_variable, var_and_value)
    #all_outputs = map(wrapper_each_variable, var_and_value)
    print('OVERALL SIMS TOOK', time.time()-start )
