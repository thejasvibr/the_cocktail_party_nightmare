# -*- coding: utf-8 -*-
""" RE-formatting the temporal masking function to 
suit the requirement of the ROUND2 submission model 


In Round2 version the temporal masking function is a tuple 
with the masking envelopes of fwd, simultaneous and backward functions.
These envelopes are placed over the simulation timeline to decide if the
echo is masked or not. 

Created on Sat Jun 08 17:56:26 2019

@author: tbeleyur
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

tm_fn = pd.read_csv('temporal_masking_fn_orig.csv')
# rename the teimgap_ms to timegap_sec
tm_fn = tm_fn.loc[:,'timegap_ms':]
tm_fn.loc[:,'timegap_ms'] *= 10**-3
tm_fn.columns = ['timegap_sec','dB_leveldiff']
tm_fn.head()

time_resolution = 10**-6

### get simultaneous masking value 
closest_to_zero = np.argmin(abs(tm_fn['timegap_sec']-0.0))
time_closest_to_zero = tm_fn['timegap_sec'][closest_to_zero]
simult_masking = tm_fn.loc[closest_to_zero,'dB_leveldiff']
########

########## get forward masking curve 
fwd_masking = tm_fn[tm_fn['timegap_sec']>time_closest_to_zero]

# now resample the fwd_masking region to show a timestep level resolution
fwdmasking_tnew = np.arange(np.max(fwd_masking['timegap_sec']),
                            time_resolution, 
                            -time_resolution)

interp_fwdmasking = interpolate.interp1d(fwd_masking['timegap_sec'],
                                         fwd_masking['dB_leveldiff'],
                                         'cubic',
                                         bounds_error=False,
                                         fill_value='extrapolate')

fwd_dBleveldiffs = interp_fwdmasking(fwdmasking_tnew)

plt.figure()
plt.plot(fwd_masking['timegap_sec'],fwd_masking['dB_leveldiff'],'.', markersize=5,
         label='original')
plt.plot(fwdmasking_tnew, fwd_dBleveldiffs,'-', alpha=0.5, label='extra and interpolated')
plt.title('Overview - forward masking function inter and extrapolation')
plt.legend()
plt.grid()
print(fwd_dBleveldiffs[-1], fwdmasking_tnew[-1])

fwdmasking_curve = np.array(fwd_dBleveldiffs)

###################

####### get backward masking curve 
bkwd_masking = tm_fn[tm_fn['timegap_sec']<0]
bkwdmasking_tnew = np.arange(-time_resolution, np.min(bkwd_masking['timegap_sec']),
                             -time_resolution)
interp_bkwdmasking = interpolate.interp1d(bkwd_masking['timegap_sec'],
                                         bkwd_masking['dB_leveldiff'],
                                         'slinear',
                                         bounds_error=False,
                                         fill_value='extrapolate')

bkwd_dBleveldiffs = interp_bkwdmasking(bkwdmasking_tnew)

plt.figure()
plt.plot(bkwd_masking['timegap_sec'],bkwd_masking['dB_leveldiff'],'.', markersize=5,
         label='original')
plt.plot(bkwdmasking_tnew, bkwd_dBleveldiffs,'-', alpha=0.5, label='extra and interpolated')
plt.title('Overview - backward masking function inter and extrapolation')
plt.legend()
plt.grid()

bkwdmasking_curve = np.array(bkwd_dBleveldiffs)


temporal_masking_function = (fwdmasking_curve, simult_masking, bkwdmasking_curve)

#plt.plot(np.hstack((fwdmasking_curve, np.array(simult_masking), bkwdmasking_curve)))

file_name = 'temporal_masking_function.pkl'

with open(file_name,'wb') as outfile:
    pickle.dump(temporal_masking_function,outfile)
#
## load and check it's the same ...
#openfile = open('temporal_masking_function.pkl','rb')
#
#loaded_tempmasking = pickle.load(open(file_name,'rb'))


