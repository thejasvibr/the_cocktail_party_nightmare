#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Create simulation parameters and run to study the effect of group size 

Created on Sat Aug 24 22:10:10 2019

@author: tbeleyur
"""
import dill as pickle

with open('../commonsim_params_v11.pkl', 'rb') as pkl:
    current_sim_params = pickle.load(pkl)

group_sizes = [5, 10, 20, 50, 100, 200]

for groupsize in group_sizes:
    current_sim_params.kwargs['Nbats']  = groupsize
    current_sim_params.kwargs['Nruns']  = 1000
    current_sim_params.kwargs['detailed description'] = '1000 runs across'+str(groupsize)
    
    param_filename = 'groupsize_simparams_' + str(groupsize)+'_.paramset'
    with open(param_filename,'wb') as pklfile:
        pickle.dump(current_sim_params, pklfile)
