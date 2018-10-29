# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:06:44 2017

@author: tbeleyur
"""
import os
import multiprocessing as mp
from multiprocessing import Pool
import time
import pandas as pd
from the_cocktail_party_nightmare_MC import *

call_ds = [1,10,20,40]

# WORKS ONLY WITH IPYTHON i THINK



script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

temporal_masking_fn = pd.read_csv(script_dir+'data\\temporal_masking_fn.csv')
spatial_release_fn = pd.read_csv(script_dir+'data\\spatial_release_fn.csv')

def calc_w_and_wo_sum(calldens):
    num_replicates = 10
    p_leq3= {'wsum':[],'wosum':[]}

    w_sum = [ run_one_trial(calldens,temporal_masking_fn,spatial_release_fn,
                            True,echo_level_range=(82,90)) for k in range(num_replicates)]
    wo_sum = [ run_one_trial(calldens,temporal_masking_fn,spatial_release_fn,False,
                             echo_level_range=(82,90)) for k in range(num_replicates)]
    probs,cum_probs = calc_pechoesheard(w_sum,5)
    probs_wo, cum_probs_wo = calc_pechoesheard(wo_sum,5)
    print(probs,probs_wo)

    p_leq3['wsum'].append(sum(probs[1:]))
    p_leq3['wosum'].append(sum(probs_wo[1:]))

    return(p_leq3)

def calc_parallel(all_densities):
    pool = Pool(processes=4)
    results = pool.map(calc_w_and_wo_sum,all_densities)
    return(results)

#all_Results = calc_parallel(call_ds)

start = time.time()
calc_w_and_wo_sum(10)
print(time.time()-start)


start = time.time()
calc_parallel(call_ds)
print(time.time()-start)