# -*- coding: utf-8 -*-
"""
Created on Wed May 08 15:37:10 2019

@author: tbeleyur
"""
import time
import numpy as np 
import pandas as pd

from the_cocktail_party_nightmare import run_CPN


    
A = 7
B = 2 

kwargs={}
kwargs['interpulse_interval'] = 0.1
kwargs['v_sound'] = 330.0
kwargs['simtime_resolution'] = 10**-6
kwargs['echocall_duration'] = 0.003
kwargs['call_directionality'] = lambda X : 7*(np.cos(np.deg2rad(X))-1)
kwargs['hearing_directionality'] = lambda X : 2*(np.cos(np.deg2rad(X))-1)
reflectionfunc = pd.DataFrame(data=[], columns=[], index=range(144))
thetas = np.linspace(-180,180,12)
input_output_angles = np.array(np.meshgrid(thetas,thetas)).T.reshape(-1,2)
reflectionfunc['reflection_strength'] = np.random.normal(-40,5,
                                          input_output_angles.shape[0])
reflectionfunc['incoming_theta'] = input_output_angles[:,0]
reflectionfunc['outgoing_theta'] = input_output_angles[:,1]
reflectionfunc['ref_distance'] = 0.1
kwargs['reflection_function'] = reflectionfunc
kwargs['heading_variation'] = 0
kwargs['min_spacing'] = 0.5
kwargs['source_level'] = {'dBSPL' : 120, 'ref_distance':0.1}
kwargs['hearing_threshold'] = 10

fwd_masking_region = np.linspace(-27, -7, 20000)        
bkwd_masking_region = np.linspace(-10, -24, 3000)
simult_masking_regio = np.array([-8])
temporal_masking_fn = (fwd_masking_region,simult_masking_regio,
                                        bkwd_masking_region)

spatial_unmasking_fn = pd.read_csv('data/spatial_release_fn.csv')
kwargs['temporal_masking_thresholds'] = temporal_masking_fn
kwargs['spatial_release_fn'] = spatial_unmasking_fn

numbats = np.array([2, 4, 8, 16,32, 64])
time_taken = {each : [] for each in numbats}

for each_num in numbats:
    kwargs['Nbats'] = each_num
    for k in range(3):
        start = time.time()
        a, b = run_CPN(**kwargs)
        run_time = time.time() - start
        print(str(each_num)+' bats, in:' +str(run_time)+' seconds')
        time_taken[each_num].append(run_time)

avg_time = [np.mean(timetaken) for _,timetaken in sorted(time_taken.iteritems())]
numrows = (numbats-1)**2

dB = lambda X : 20*np.log10(X)

plt.figure()
plt.plot(dB(numbats/numbats[0]), dB(avg_time/avg_time[0]),'*-', label='time taken')
plt.plot(dB(numbats/numbats[0]), dB(numrows/numrows[0]), '^-',label='num rows')
plt.grid()
plt.legend()

        
    
    

