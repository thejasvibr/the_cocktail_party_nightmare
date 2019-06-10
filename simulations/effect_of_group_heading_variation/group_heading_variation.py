# -*- coding: utf-8 -*-
""" effect of heading variation
Created on Mon Jun 10 15:23:55 2019

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
    common_kwargs['Nbats'] = 25 # fix group size and vary min spacing 
    
     # run simulations for all group sizes of interest
    theta_variation = [10, 20, 40, 80]
    var_and_value = []
    for theta_var in theta_variation:
        for i in range(10): # make sure overall 10,000 sims are run 
            var_and_value.append((('heading_variation',theta_var), common_kwargs)) 
    start = time.time()
    #pool = Pool(4)
    pool = Pool(4);all_outputs = pool.map(wrapper_each_variable, var_and_value)
    #all_outputs = map(wrapper_each_variable, var_and_value)
    print('OVERALL SIMS TOOK', time.time()-start )
