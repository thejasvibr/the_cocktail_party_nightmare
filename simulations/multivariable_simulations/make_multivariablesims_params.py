#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Explores the effect of multiple variables on the CPN all for 200 bats -
which is the a group size at which a focal  bat in the centre can just
about detect one neighbour per call.

Created on Sun Sep  1 20:22:09 2019

@author: tbeleyur
"""
import dill as pickle
import pandas  as pd
import numpy as np 
import statsmodels.api as sm


join_into_string = lambda Y: '*'.join(map(lambda X:str(X), Y))

group_size = 200
interpulse_duration = [0.05, 0.1]
call_duration = [0.001, 0.0025]
shadowing = [True, False]
source_level = [94, 100, 106]
spacing = [0.5, 1.0]
group_heading_variation = [10, 90]
atmospheric_absorption = [-1.0,-1.5,-2.0]


with open('../commonsim_params_class.pkl','rb') as pklfile:
    simparams_class = pickle.load(pklfile)

for ipi in interpulse_duration:
    for call_durn in call_duration:
        for w_shadowing in shadowing:
            for SL in source_level:
                for interbat_spacing in spacing:
                    for heading_variation in group_heading_variation:
                        for atm_abs in atmospheric_absorption:
        
                            multivariable_params = simparams_class()
                            multivariable_params.bistatic_TS_file = '..//..//data//bistatic_TS_bat.csv'
                            multivariable_params.shadowing_model_file = '..//..//data//acoustic_shadowing_model.pkl'
                            multivariable_params.tempmasking_file = '..//..//data//temporal_masking_function.pkl'
                            multivariable_params.spatial_unmasking_file = '..//..//data//spatial_release_fn.csv'
                            
                            multivariable_params.load_parameters()
    
                            multivariable_params.kwargs['Nbats']  = group_size
                            multivariable_params.kwargs['Nruns']  = 200
                            msg = str(multivariable_params.kwargs['Nruns']) + 'runs across'+str(group_size)
                            multivariable_params.kwargs['detailed description'] = msg
                            multivariable_params.kwargs['interpulse_interval'] = ipi
                            multivariable_params.kwargs['echocall_duration'] = call_durn
                            multivariable_params.kwargs['implement_shadowing'] = w_shadowing
                            multivariable_params.kwargs['source_level'] = {'dBSPL' : SL, 
                                                                               'ref_distance':1.0}
                            multivariable_params.kwargs['min_spacing'] = 0.5
                            multivariable_params.kwargs['heading_variation'] = heading_variation
                            multivariable_params.kwargs['atmospheric_attenuation'] = atm_abs
                            
                            all_params = [ipi,call_durn,w_shadowing,SL,interbat_spacing,heading_variation,atm_abs]
                            changed_variables = join_into_string(all_params)
                                            
                            param_filename = 'multivariable_params_' + changed_variables+'_.paramset'
                            with open(param_filename,'wb') as pklfile:
                                pickle.dump(multivariable_params, pklfile)
                            del multivariable_params



