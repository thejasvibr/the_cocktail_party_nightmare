#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Create simulation parameters and run to study the effect of group size 

Created on Sat Aug 24 22:10:10 2019

@author: tbeleyur
"""
import dill as pickle
import pandas  as pd
import numpy as np 
import statsmodels.api as sm

with open('../commonsim_params_class.pkl','rb') as pklfile:
    simparams_class = pickle.load(pklfile)

group_sizes = [5, 10, 20, 50]


for groupsize in group_sizes:
        
    groupsize_params = simparams_class()
    groupsize_params.bistatic_TS_file = '..//..//data//bistatic_TS_bat.csv'
    groupsize_params.shadowing_model_file = '..//..//data//acoustic_shadowing_model.pkl'
    groupsize_params.tempmasking_file = '..//..//data//temporal_masking_function.pkl'
    groupsize_params.spatial_unmasking_file = '..//..//data//spatial_release_fn.csv'
    
    groupsize_params.load_parameters()
    
    
    groupsize_params.kwargs['Nbats']  = groupsize
    groupsize_params.kwargs['Nruns']  = 10
    groupsize_params.kwargs['detailed description'] = '1000 runs across'+str(groupsize)
    
    param_filename = 'groupsize_simparams_' + str(groupsize)+'_.paramset'
    with open(param_filename,'wb') as pklfile:
        pickle.dump(groupsize_params, pklfile)
    del groupsize_params



# load a file and check what the Nbstas is :
        
