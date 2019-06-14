# -*- coding: utf-8 -*-
""" Effect of minimum spacing 
Created on Mon Jun 10 14:40:33 2019

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
    common_kwargs['Nbats'] = 25 # fix group size and vary min spacing 
    
    # run simulations for all group sizes of interest
    r_min_values = [0.5, 0.75, 1.0, 1.5]
    var_and_value = []
    for each_rmin in r_min_values:
        for i in range(10): # make sure overall 10,000 sims are run 
            var_and_value.append((('min_spacing',each_rmin), common_kwargs)) 
    start = time.time()

    pool = Pool(multiprocessing.cpu_count());
	all_outputs = pool.map(wrapper_each_variable, var_and_value)
    #all_outputs = map(wrapper_each_variable, var_and_value)
    print('OVERALL SIMS TOOK', time.time()-start )
