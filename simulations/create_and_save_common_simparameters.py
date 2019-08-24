# -*- coding: utf-8 -*-
"""Create a common parameter set to initialise all simulations and 
change only the parameters of interest for each simulation type.

Created on Sat Jun 08 18:59:27 2019

@author: tbeleyur
"""


# the functions must be lambdas because of the limitations of 
# the current implementation of Pool.map which uses pickle to serialise
# and de-serialise data onto processes ! 

call_directionality_fn = lambda  X, A=7 : A*(np.cos(np.deg2rad(X))-1)
hearing_directionality_fn = lambda X, B=2 : B*(np.cos(np.deg2rad(X))-1)



import dill as pickle
import numpy as np 
import pandas as pd
import statsmodels.api as sm

class parameter_container():
    
    def __init__(self):
        self.load_parameters()
        
        
    def load_parameters(self):
        self.kwargs={}
        self.kwargs['interpulse_interval'] = 0.1
        self.kwargs['v_sound'] = 330.0
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['echocall_duration'] = 0.0025
        self.kwargs['call_directionality'] = None
        self.kwargs['hearing_directionality'] = None
        reflection_func = pd.read_csv('..//data//bistatic_TS_bat.csv')
        self.kwargs['reflection_function'] = reflection_func
        self.kwargs['heading_variation'] = 10.0
        self.kwargs['min_spacing'] = 0.5
        self.kwargs['Nbats'] = 50
        self.kwargs['source_level'] = {'dBSPL' : 100, 'ref_distance':1.0}
        self.kwargs['hearing_threshold'] = 20
        self.kwargs['rectangle_width'] = 0.25
        self.kwargs['implement_shadowing'] = True
        
        # load the model to predict shadowing 
        self.kwargs['acoustic_shadowing_model']=sm.load('..//data//acoustic_shadowing_model.pkl')
        
        tempmasking_file = '..//data//temporal_masking_function.pkl'
        with open(tempmasking_file, 'rb') as pklfile:
            temporal_masking_fn = pickle.load(pklfile)
            
        spatial_unmasking_fn = pd.read_csv('..//data//spatial_release_fn.csv')
        
        self.kwargs['temporal_masking_thresholds'] = temporal_masking_fn
        self.kwargs['spatial_release_fn'] = np.array(spatial_unmasking_fn)[:,1:]
           
        self.kwargs['call_directionality'] = call_directionality_fn
        
        self.kwargs['hearing_directionality'] = hearing_directionality_fn



common_params_file = 'commonsim_params_v11.pkl'
parameter_instance = parameter_container()
with open(common_params_file,'wb') as commonparamsfile:
    pickle.dump(parameter_instance, commonparamsfile)
