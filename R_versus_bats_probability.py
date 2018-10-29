# -*- coding: utf-8 -*-
"""Calculates the probability of bats being present at R radial distance
away from a focal bat
Created on Sun Jan 07 12:22:06 2018

@author: tbeleyur
"""

import numpy as np
import matplotlib.pyplot as plt


R = np.linspace(0.5,2.93,100)
D = 3 # bats per sq meter

N_R = D*np.pi*R**2

# let's see how many bats there are per 'ring':

dNR = np.diff(N_R)


P_nr = dNR/sum(dNR)

plt.figure(1,figsize=(12,8))
plt.subplot(311)
plt.plot(R[:-1],dNR,'*-')
plt.subplot(312)
plt.plot(R,N_R,'*-')
plt.subplot(313)
plt.plot(R[:-1], P_nr,'*-')

# calculate call intensities over this range - without
# atmospheric absorption - and only spherical spreading :
SL = 100 # dB SPL at 1m , re 20 microPa
call_level = SL - 20*np.log10(R)
plt.plot(call_level[:-1],P_nr,'*-')

