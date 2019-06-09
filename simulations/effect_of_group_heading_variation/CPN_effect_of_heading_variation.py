# -*- coding: utf-8 -*-
"""Running the Sonar Cocktail Party Nightmare:
    Checking the effect of Group Size. 

Created on Sat Jun 08 16:59:26 2019

@author: tbeleyur
"""
import multiprocessing as mp
from multiprocessing import Pool
import os
import sys
import time
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
common_kwargs['Nbats'] = 10


def run_each_headingvariation(headingvariation, num_replicates = 100, kwargs=common_kwargs):
    '''
    '''
    start = time.time()
    simoutput_container = {(headingvariation,i) : None for i in range(num_replicates)}
    
    kwargs['heading_variation'] = headingvariation
    
    print('RUNNING '+str(headingvariation)+' degrees heading variation')
    for replicate_run in xrange(num_replicates):
        num_echoes, sim_output = run_CPN(**kwargs)
        simoutput_container[(headingvariation, replicate_run)] = sim_output
    group_size = 'Nbats_'+str(kwargs['Nbats'])
    picklefilename = 'results//' +group_size+'_headingvariation_'+str(headingvariation)+'bats_CPN.pkl'
    try:
        with open(picklefilename, 'wb') as picklefile:
            pickle.dump(simoutput_container, picklefile)
        print('succesful run and saving took', time.time()-start)
        return(True)
    except:
        raise IOError('UNABLE TO SAVE PICKLE FILE!!')
        return(False)



def wrapper_each_headingvariation(headingvariation):
    output =  run_each_headingvariation(headingvariation)
    return(output)


# run simulations for all group sizes of interest
heading_range = [0, 45,90,135]
all_outputs = map(wrapper_each_headingvariation, heading_range)


    