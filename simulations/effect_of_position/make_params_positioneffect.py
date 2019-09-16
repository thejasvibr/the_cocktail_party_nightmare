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
interpulse_duration =  0.1
call_duration =  0.0025
shadowing = True
source_level =  100
spacing = 0.5
group_heading_variation = 10
atmospheric_absorption =  -1
number_of_simulation_runs =  2
radial_distance = [0.25, 0.5, 0.75] 
azimuth_location = [np.pi/4, 3*np.pi/4, 3*np.pi/2, 5*np.pi/4] 

with open('../common_simulation_parameters.paramset','rb') as pklfile:
    simulation_parameters = dill.load(pklfile)
    
simulation_parameters['echoes_beyond_ipi'] = True

i = 0

for r in radial_distance:
    for theta in azimuth_location:
        i += 1
        focal_bat_position = (r,theta)
        simulation_parameters['Nbats']  = group_size
        simulation_parameters['Nruns']  = number_of_simulation_runs
        description = str(simulation_parameters['Nruns']) + 'runs across'+str(group_size)
        simulation_parameters['detailed description'] = description
        simulation_parameters['interpulse_interval'] = interpulse_duration
        simulation_parameters['echocall_duration'] = call_duration
        simulation_parameters['implement_shadowing'] = shadowing
        simulation_parameters['source_level'] = {'dBSPL' : source_level, 
                                                           'ref_distance':1.0}
        simulation_parameters['min_spacing'] = spacing
        simulation_parameters['heading_variation'] = group_heading_variation
        simulation_parameters['atmospheric_attenuation'] = atmospheric_absorption
        simulation_parameters['noncentral_bat'] = focal_bat_position

        all_params = [interpulse_duration, call_duration, shadowing,
                          source_level, spacing,
                          group_heading_variation,
                          atmospheric_absorption,
                          focal_bat_position]
        variables_as_string = join_into_string(all_params)

        param_filename = 'simulation_parameters_' + variables_as_string+'_.paramset'
        with open(param_filename,'wb') as pklfile:
            dill.dump(simulation_parameters, pklfile)