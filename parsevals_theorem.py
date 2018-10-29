# -*- coding: utf-8 -*-
"""
Checking Pareseval's theorem for the DFT case :
https://en.wikipedia.org/wiki/Parseval%27s_theorem

We will check if the LHS and RHS are equal to each other in the case of the
discrete Fourier transform

Created on Tue Nov 21 14:26:31 2017

@author: tbeleyur
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000


def check_Parsevals_theorem(input_signal):

    # sum of X**2
    LHS = np.sum(input_signal**2)

    # the RHS
    dft = np.fft.fft(input_signal,input_signal.size)
    RHS = np.sum(abs(dft)**2)/input_signal.size

    error = LHS - RHS

    return(error)

linearisefromdB = lambda x_db : 10**(x_db/20.0)

takedB = lambda value : 20*np.log10(value)

def calc_rmsfromspectrum(in_signal):

    dft_sqsum = np.sum( np.abs(np.fft.fft(in_signal))**2   )

    rms_from_spectrum  = (1.0/in_signal.size)*np.sqrt(dft_sqsum)

    signal_rms = np.sqrt( np.sum(in_signal**2 )/in_signal.size)

    dB_rmserror = takedB(signal_rms) - takedB(rms_from_spectrum)

    return(dB_rmserror,rms_from_spectrum)


if __name__ == '__main__':

    # create a chirp :
    fs = 192000
    call_durn = 0.003
    t = np.linspace(0,call_durn,int(call_durn*fs))
    f_start,f_end = 96000,20000

    sweep = signal.chirp(t,f_start,t[-1],f_end,method='hyperbolic')
    sweep *= signal.tukey(sweep.size,0.8)
    sweep_signal = np.concatenate((np.zeros(fs),sweep,np.zeros(fs)))

    print('difference between LHS and RHS is: ',check_Parsevals_theorem(sweep_signal))

    print('Now let us check the theorem for a signal with some more noise in it')

    noisy_signal = np.copy(sweep_signal);
    noisy_signal += np.random.normal(0,linearisefromdB(-20),noisy_signal.size)

    print('LHS and RHS difference for noisy signal is : ',check_Parsevals_theorem(sweep))


    # if this is true, then we can basically calculate the rms straight from the signal spectrum
    # itself :

    print(calc_rmsfromspectrum(sweep_signal))




