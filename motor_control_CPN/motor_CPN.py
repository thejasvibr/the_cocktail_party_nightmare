# -*- coding: utf-8 -*-
""" Module with functions that implement an agent based model to
test if glimpses in the CPN are sufficient for navigation
Created on Thu Mar 08 11:14:35 2018

@author: tbeleyur
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial




def plan_movement_direction(object_locns,current_heading,max_turn=20,**kwargs):
    '''Calculate the movement of the agent towards the closest object,
    and output the final change in heading angle constrained by the max_turn.


    Parameters:
        object_locns : Nobjects x 2 np.array. R and relative angular coordinates
                        of the detected  objects. Nobjects can also be an empty
                        np.array if no objects are detected. A relative angle
                        means that the values lie between 0 and 180.

                        If objects are on the left of the agent, they are between
                        0 >= theta >= -180. If objects are on the right theya
                        are 0 <= theta <= 180.

        current_heading : float. Heading of the agent in degrees. 3o'clock is
                        set as the zero point, and the angles increase to 360
                        degrees in a counter-clockwise fashion.


        max_turn : float >0. Maximum possible turn possible within one
                   inter-pulse interval in degrees.

        **kwargs :

        turning_noise : float. The motor error in turning modelled
                        as N(0,turning_noise).

    Returns:

        heading_direction : float. The new heading direction.

    '''
    
    closest_object = np.argmin(object_locns[:,0])
    
    required_turn = current_heading - object_locns[closest_object,1]
    
    
    
    
    


