# -*- coding: utf-8 -*-
"""
trying out the acoustics package
Created on Tue Nov 21 15:07:53 2017

@author: tbeleyur
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import acoustics as ac

conv_to_kelvin = lambda celsius: celsius + 273.15
signal_rms = lambda in_signal: np.sqrt( np.sum(in_signal**2 )/in_signal.size)

linearisefromdB = lambda x_db : 10**(x_db/20.0)

takedB = lambda value : 20*np.log10(value)

atmos = ac.atmosphere.Atmosphere()

atmos.temperature = conv_to_kelvin(20)
atmos.relative_humidity = 30



# now let's just get the impulse response of the atmosphere:
fs = 192000
atm_IR = atmos.impulse_response(1,fs,512)



# let's create a signal with two pure frequencies, and calculate the
# expected attenuation vs the one obtained by the filter:
durn = 0.003
t = np.linspace(0,durn,int(fs*durn))

f1 = 40*10**3
f2 = 80*10**3



attn_f1 = atmos.attenuation_coefficient(f1)
attn_f2 = atmos.attenuation_coefficient(f2)

sig1 = np.sin(2*np.pi*f1*t) *0.5
sig2 =  np.sin(2*np.pi*f2*t) *0.5
sig12 = sig1 + sig2

# append with some zeros on either side :
pbk_12 = np.concatenate((np.zeros(2000),sig12,np.zeros(2000),np.zeros(2000)))
dbrms_pbk12 = takedB(signal_rms(pbk_12))
print('original signal dBrms',)

# now filter with the atmospheric impulse response :
post_attn = np.convolve(pbk_12,atm_IR,'same')
dbrms_postir = takedB(signal_rms(post_attn))
print('filtered signal dBrms',dbrms_postir)

# expected dB_rms change :

plt.plot(pbk_12,label='original');plt.plot(post_attn,label='post atm IR')
plt.legend()
print('Attenuation dbrms ' , dbrms_pbk12- dbrms_postir)
print(attn_f1,attn_f2)

## and now let's test it by attenuating and adding the signals independently:
sig1_atn = np.copy(sig1)/linearisefromdB(attn_f1)
sig2_atn = np.copy(sig2)/linearisefromdB(attn_f2)

expected_sig12 = sig1_atn + sig2_atn
expected_pbk12 = np.concatenate((np.zeros(2000),expected_sig12,np.zeros(2000),np.zeros(2000)))

dbrms_expected_sig12 = takedB(signal_rms(expected_pbk12))

print('The difference between expected and obtained dBrms of the signals with the filtering is: \n')
print(dbrms_expected_sig12 - dbrms_postir)

print('So, it is safe to conclude that the atmospheric IR will do a good job of simulating atmospheric attenuation for us')

## Now let's check if the rms drop of the signal is dependent on the
# signal itself, or is independent of the signal - is it linear in the dB scale ?

print('\n Let us now check if the rms drop across frequencies will be uniform in the dB scale, or,whether it will depend on the signal itself')

pbk12_w_noise = np.copy(pbk_12) + np.random.normal(0,linearisefromdB(-10),pbk_12.size)
postir_pbk12wnoise = np.convolve(pbk12_w_noise,atm_IR,'same')

dBrms_pbk12wnoise, dBrms_postirpbk12wnoise = takedB(signal_rms(pbk12_w_noise)),takedB(signal_rms(postir_pbk12wnoise))

print('difference in attenuation for noisy signal: ',dBrms_pbk12wnoise-dBrms_postirpbk12wnoise)

print('\n ...apparently not -the total amount of rms drop will OF COURSE depend on the frequency content of each signal..this means we need to calculate it separately for each signal')








