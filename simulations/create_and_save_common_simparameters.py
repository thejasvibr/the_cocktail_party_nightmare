# -*- coding: utf-8 -*-
"""Create a common parameter set to initialise all simulations and 
change only the parameters of interest for each simulation type.

Created on Sat Jun 08 18:59:27 2019

@author: tbeleyur
"""

import dill as pickle
import numpy as np 
import pandas as pd
import statsmodels.api as sm

class parameter_container():
    
    def __init__(self):
        self.bistatic_TS_file = '..//data//bistatic_TS_bat.csv'
        self.shadowing_model_file = '..//data//acoustic_shadowing_model.pkl'
        self.tempmasking_file = '..//data//temporal_masking_function.pkl'
        self.spatial_unmasking_file = '..//data//spatial_release_fn.csv'

        
    def load_parameters(self):
        self.kwargs={}
        self.kwargs['interpulse_interval'] = 0.1
        self.kwargs['v_sound'] = 330.0
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['echocall_duration'] = 0.0025        
        reflection_func = pd.read_csv(self.bistatic_TS_file)
        self.kwargs['reflection_function'] = reflection_func
        self.kwargs['heading_variation'] = 10.0
        self.kwargs['min_spacing'] = 0.5
        self.kwargs['Nbats'] = None
        self.kwargs['source_level'] = {'dBSPL' : 100, 'ref_distance':1.0}
        self.kwargs['hearing_threshold'] = 20
        self.kwargs['rectangle_width'] = 0.25
        self.kwargs['implement_shadowing'] = True
        
        # load the model to predict shadowing 
        
        self.kwargs['acoustic_shadowing_model']=sm.load(self.shadowing_model_file)
        
        
        with open(self.tempmasking_file, 'rb') as pklfile:
            temporal_masking_fn = pickle.load(pklfile)
        
        spatial_unmasking_fn = pd.read_csv(self.spatial_unmasking_file)
        
        self.kwargs['temporal_masking_thresholds'] = temporal_masking_fn
        self.kwargs['spatial_release_fn'] = np.array(spatial_unmasking_fn)[:,1:]
           
        self.kwargs['call_directionality'] = lambda  X, A=7 : A*(np.cos(np.deg2rad(X))-1)
        
        self.kwargs['hearing_directionality'] = lambda X, B=2 : B*(np.cos(np.deg2rad(X))-1)


common_params_class = 'commonsim_params_class.pkl'
with open(common_params_class,'wb') as paramsclass:
    pickle.dump(parameter_container, paramsclass)


common_params_file = 'commonsim_params_v11.pkl'
parameter_instance = parameter_container()
with open(common_params_file,'wb') as commonparamsfile:
    pickle.dump(parameter_instance, commonparamsfile)
