# -*- coding: utf-8 -*-
"""Create a common parameter set to initialise all simulations and 
change only the parameters of interest for each simulation type.

Created on Sat Jun 08 18:59:27 2019

@author: tbeleyur
"""
import pickle
import numpy as np 
import pandas as pd

kwargs={}
kwargs['interpulse_interval'] = 0.1
kwargs['v_sound'] = 330.0
kwargs['simtime_resolution'] = 10**-6
kwargs['echocall_duration'] = 0.0025
kwargs['call_directionality'] = None
kwargs['hearing_directionality'] = None
reflection_func = pd.read_csv('..//data//bistatic_TS_bat.csv')
kwargs['reflection_function'] = reflection_func
kwargs['heading_variation'] = 10.0
kwargs['min_spacing'] = 0.5
kwargs['Nbats'] = 50
kwargs['source_level'] = {'dBSPL' : 120, 'ref_distance':0.1}
kwargs['hearing_threshold'] = 20
kwargs['rectangle_width'] = 0.5
kwargs['implement_shadowing'] = True


# extract the bistatic TS of all incoming-outgoing angle separations of 
# 180 degrees
kwargs['shadow_TS'] = reflection_func[reflection_func['delta_theta']==180]['reflection_strength'].tolist()

tempmasking_file = '..//data//temporal_masking_function.pkl'
with open(tempmasking_file, 'rb') as pklfile:
    temporal_masking_fn = pickle.load(pklfile)
    
spatial_unmasking_fn = pd.read_csv('..//data//spatial_release_fn.csv')

kwargs['temporal_masking_thresholds'] = temporal_masking_fn
kwargs['spatial_release_fn'] = np.array(spatial_unmasking_fn)[:,1:]

common_params_file = 'commonsim_params.pkl'
with open(common_params_file,'wb') as commonparamsfile:
    pickle.dump(kwargs, commonparamsfile)
