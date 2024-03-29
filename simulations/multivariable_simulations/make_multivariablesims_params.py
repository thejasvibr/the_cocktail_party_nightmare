#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Explores the effect of multiple variables on the CPN all for 200 bats -
which is the a group size at which a focal  bat in the centre can just
about detect one neighbour per call.

Created on Sun Sep  1 20:22:09 2019

@author: tbeleyur
"""
import glob
import os
import sys

import dill 
import pandas  as pd
import numpy as np 
import statsmodels.api as sm


join_into_string = lambda Y: '*'.join(map(lambda X:str(X), Y))

group_size = 100
interpulse_duration = [0.2, 0.3]
call_duration = [0.001, 0.0025]
shadowing = [True, False]
source_level = [94, 100, 106]
spacing = [0.5, 1.0]
group_heading_variation = [10, 90]
atmospheric_absorption = [0, -1, -2]
number_of_simulation_runs =  200

with open('../common_simulation_parameters.paramset','rb') as pklfile:
    simulation_parameters = dill.load(pklfile)
    
simulation_parameters['echoes_beyond_ipi'] = True

i = 0
for ipi in interpulse_duration:
    for call_durn in call_duration:
        for w_shadowing in shadowing:
            for SL in source_level:
                for interbat_spacing in spacing:
                    for heading_variation in group_heading_variation:
                        for atm_abs in atmospheric_absorption:
                            i += 1
                            simulation_parameters['Nbats']  = group_size
                            simulation_parameters['Nruns']  = number_of_simulation_runs
                            description = str(simulation_parameters['Nruns']) + 'runs across'+str(group_size)
                            simulation_parameters['detailed description'] = description
                            simulation_parameters['interpulse_interval'] = ipi
                            simulation_parameters['echocall_duration'] = call_durn
                            simulation_parameters['implement_shadowing'] = w_shadowing
                            simulation_parameters['source_level'] = {'dBSPL' : SL, 
                                                                               'ref_distance':1.0}
                            simulation_parameters['min_spacing'] = interbat_spacing
                            simulation_parameters['heading_variation'] = heading_variation
                            simulation_parameters['atmospheric_attenuation'] = atm_abs

                            all_params = [ipi,call_durn,w_shadowing,SL,interbat_spacing,heading_variation,atm_abs]
                            variables_as_string = join_into_string(all_params)

                            param_filename = 'simulation_parameters_' + variables_as_string+'_.paramset'
                            with open(param_filename,'wb') as pklfile:
                                dill.dump(simulation_parameters, pklfile)
