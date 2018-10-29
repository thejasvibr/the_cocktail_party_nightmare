# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:19:20 2017

@author: tbeleyur
"""
from __future__ import division
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000

fs = 192000

beep_durn = 0.04
freq = 100
t = np.linspace(0,beep_durn,int(beep_durn*fs))

#beep = np.sin(2*np.pi*freq*t + 0.03)
beep = signal.chirp(t,5000,np.max(t),1000,)
beep *= signal.tukey(beep.size,alpha=0.6)

beep *= 0.5

pbk_rate = 5
pbk_durn = 10

durn_onecycle = 1/pbk_rate
silence_durn = durn_onecycle - beep_durn

silence_pbk = np.zeros(int(silence_durn*fs/2))

one_beep_w_silence = np.concatenate((silence_pbk,beep,silence_pbk))

plt.plot(one_beep_w_silence)

one_pbk_cycle = np.tile(one_beep_w_silence,pbk_rate*pbk_durn)

sd.play(one_pbk_cycle,samplerate=fs,mapping=[1,2])