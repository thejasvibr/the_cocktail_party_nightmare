# -*- coding: utf-8 -*-
""" playing with Bridson package : https://pypi.org/project/bridson/#description
    
Created on Fri Jun 14 14:11:02 2019

@author: tbeleyur
"""

import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
from bridson import poisson_disc_samples

import scipy.spatial as spl
import numpy as np
import random
random.seed(82319)

above_minR = []
numpoints = []
rmin=0.5
numreplicates = 1000
sidelengths = range(2,9)
npoints_to_length = np.zeros((len(sidelengths), numreplicates))

for row,length in enumerate(sidelengths):
    for J in range(numreplicates):
        xy = np.array(poisson_disc_samples(length,length, rmin))
        
        if xy.shape[0] < 2:
            raise ValueError('whats going on ')
        npoints_to_length[row,J]  = xy.shape[0]
        #dist_matrix = spl.distance_matrix(xy,xy)
        #ptptdist = np.unique(np.triu(dist_matrix))
        #above_minR.append(np.min(ptptdist[1:]))

plt.figure()
plt.boxplot(npoints_to_length.T)
plt.xticks(range(1,len(sidelengths)+1), sidelengths)
