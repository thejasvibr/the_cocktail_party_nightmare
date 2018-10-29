# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:08:50 2018

@author: tbeleyur
"""

import sys
folder = 'C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\2016_jamming response modelling\\analytical_modelling\\poisson-disc-master\\poisson-disc-master'
sys.path.append(folder)
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix as dm

from poisson_disc import Grid

r = 0.5
square_length = 3.0
length, width = square_length, square_length
grid = Grid(r, length, width)


def unzip(items):
    return ([item[i] for item in items] for i in range(len(items[0])))

data = grid.poisson((length,width))
data_np = np.asanyarray(data)

print(data_np.shape)
max_points, columns = data_np.shape
num_points = max_points


# thanks Retozi, https://tinyurl.com/ybumhquf
def centeroidpython(data):
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l

def calc_centroid(data):
    '''
    '''
    centroids = np.apply_along_axis(np.mean,0,data)
    return(centroids)

xy = calc_centroid(data_np)


plt.plot(data_np[:num_points,0],data_np[:num_points,1],'*')
plt.plot(xy[0],xy[1],'r*')

# generate points for a grid with given dimensions and point-separation
# if not enough points are there, then increase the grid size till at least
