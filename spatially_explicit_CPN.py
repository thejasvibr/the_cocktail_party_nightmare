# -*- coding: utf-8 -*-
"""
Spatially explicit Monte-carlo Simulations of the
Cocktail party nightmare

To Do:
1) for each simulation run:
    > setup Nbats over a space with sigma_x and sigma_y spacing
    > choose three target bats in front of the focal bat - these will be the
        target echoes
    > calculate the echo level of the returning echoes
    > simulate N-1 calls from the Nbats-1, and assign the calls to the Nbats-1
        > calculate the relative angles of the echoes and the calls
        > calculate the emission angle and the distance of the emitting agents

    > check if each echo can be heard or not
    > return the number of echoes that can be heard

2) repeat steps 1) for 10^6 simulations and then get the underlying probability
    distribution of the P(1 echo heard) at MULTIPLE CALL DENSITIES


Created on Thu Nov 16 13:03:57 2017

@author: tbeleyur
"""

import numpy as np
import matplotlib.pyplot as plt


def place_bats(Nbats,mean_deltaxy=(0.5,0.5)):
    '''
    Nbats: integer. Number of bats to be placed in space
    mean_deltaxy: tuple. mean spacing between bats in x and y dimensions in meters

    '''
    if not Nbats>1:
        raise ValueError('Number of bats in the group must be >1')

    if not isinstance(sigma_xy,tuple):
        raise TypeError('igma_xy must be a tuple')

    all_bat_positions = np.zeros((Nbats,2))

    all_bat_positions[0,:] = np.array((0,0)) # explicit statement that bat 0 is,
                            # in the centre

    random_xs =
    random_ys =

def calculate_echo_level(bat_positions,source_level,TS,focal_bat_position=np.array([0,0])):
    '''
    Calculate echo level distance of the echoes from conspecifics.
    A simple calculation based on the sonar equation is used to calculate
    the received level.

    Inputs:
        bat_positions: (Nbats -1) x 2 np.array. with x and y coordinates
        SL
        TS: negative float. target strength of the conspecific bats
        focal_bat_position: 1x2 np.array with x and y coordinates of bat

    Outputs:
        echo_RL : float. received level of the returning echoes.


    '''






