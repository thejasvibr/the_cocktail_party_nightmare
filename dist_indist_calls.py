# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 19:30:21 2017

@author: tbeleyur
"""
from __future__ import division
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt


# let's count the number of ways to split Ncalls into b Bins.

Ncalls = 16
Bins = 33

bins_full = np.arange(0,Bins+1,1)
num_ways = np.array( [ misc.comb( Ncalls+num_bins-1,Ncalls)*misc.comb(Bins,num_bins) for num_bins in bins_full] )

ways2fill_nboxes = np.diff(num_ways)
prob_nboxes_full = ways2fill_nboxes/np.sum(ways2fill_nboxes)
#
#plt.plot(np.arange(Bins,0,-1),  prob_nboxes_full,'*-')
#plt.xticks(np.arange(Bins,0,-1))
#plt.grid()
#
#
## and now taking on the probability of one echo being heard :
## let's do the math and see how often the bat would be able to perceive
## this echo multiple times in a row:
#
#
#plt.figure(2)
#for trials in [7,10,15,20]:
#    #trials = 20
#    # in this case we sum up a bunch of probabilities because I'm interested
#    # in looking at how much of the time the bat can hear  echo !
#    p_oneecho = np.sum(prob_nboxes_full[-2])
#    q_oneecho = 1 - p_oneecho
#
#    p_nsuccess = [ misc.comb(trials,success)*(p_oneecho**success)*(q_oneecho**(trials-success)) for success in range(trials) ]
#
#
#    plt.plot(p_nsuccess,'*-',label=str(trials))


### 3-10-2017:
B = 4
Ncalls = 2

nways_formula = misc.comb(Ncalls+B-1,Ncalls)
nways_myform = [  misc.comb(Ncalls+i-1,Ncalls)*misc.comb(B,i)  for i in range(1,Ncalls+1)]




## round 2 :
p = 0.03 # duty cycle of the call
q = 1- p #

prob_nocolln = 1 - p*q

pulse_per_sec = 30

prob_num_Calls = [ misc.comb(pulse_per_sec,i)* (p**(i))*q**(pulse_per_sec-i) for i in range(0,15)]
plt.plot(prob_num_Calls)
