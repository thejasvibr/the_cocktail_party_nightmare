 -*- coding: utf-8 -*-
""" Simulation of what many bats flying in a milling type
configuration may be experiencing. 

The model of collective behaviour implemented here is taken from:
Couzin et al. 2002, J. theor. Biol, 218, 1-11 
'Collective Memory and Spatial Sorting in Animal Groups'

Created on Wed Jun 26 13:15:52 2019

@author: tbeleyur
"""
import sys
sys.path.append('..//..//')
import numpy as np 
from the_cocktail_party_nightmare import  place_bats_inspace



def simulate_collective_movement(**kwargs):
    '''
    '''
    
    xy_positions, directions = initiate_simulations(**kwargs)

    for each_timestep in xrange(kwargs['timesteps']):
        kwargs['current_timestep'] = each_timestep

        for individual in kwargs['individuals']:
            desired_direction = calculate_desired_direction_of_movement(individual, **kwargs)
            
            # effective direction is limited by the max turning angle of the individual 
            effective_direction = calculate_effective_direction(desired_direction,
                                                                **kwargs)
            new_position = calculate_new_positions(effective_direction, 
                                                   **kwargs)
            
            assign_new_direction_and_position(individual, **kwargs)



def initiate_simulations(**kwargs):
    '''Creates the starting xy positions and direction vectors of 
    individuals. 

    Keyword Arguments
    -----------------
    Nbats : int >2.

    timesteps : int >0. 
              Number of timesteps to simulate. 

    min_spacing : float>0.
                Bats are initiated with a Poisson disk arrangement.
                All individuals are more than the min_spacing apart from 
                their nearest neighbour.

    heading_variation : float>=0. 
                       The range of direction of flight of individuals.
                       eg. The default heading direction is 90 degrees. 
                       a heading_variaiton of 10 degrees implies 
                       a range of 80-100 degrees. 
                       Zero degrees is set to be 3 oçlock and 
                       90  degrees is 12 o'clock. 

    Returns
    -------
    all_positions, all_directions : Nbats x 2 x timesteps np.arrays.
                                    all_positions has the xy coordinates
                                    all_directions has the unitvectors of
                                    travel. 
    
    '''
    (otherbats_xy, focal_xy), headings = place_bats_inspace(**kwargs)
    bats_xy = np.row_stack((focal_xy, otherbats_xy))
    bats_theta = headings
    bats_unitvectors = convert_to_unitvector(headings)
    
    # create array that holds XY positions and direction vectors 
    all_positions = np.zeros((kwargs['Nbats'],2,kwargs['timesteps']))
    all_directions = np.zeros((kwargs['Nbats'],2,kwargs['timesteps']))

    all_positions[:,:,0] = bats_xy
    all_directions[:,:,0] = bats_unitvectors
    return(all_positions, all_directions)


def convert_to_unitvector(headings):
    '''
    '''
    in_radians = np.radians(headings)
    unit_vectors = np.column_stack((np.cos(in_radians),
                                    np.sin(in_radians)))
    return(unit_vectors)





def calculate_desired_direction_of_movement(individual, **kwargs):
    '''Calculates the next desired direction of movement given the equations 1-3
    in the paper on page 3

    Parameters
    ----------
    
    individual : int >=0. 
                 Identifying index number which corresponds to the 
                 row in a 2D array

    Keyword Arguments
    -----------------
    speed
    
    turning_rate
    
    r_repulsion
    r_orientation
    r_attraction

    field_perception
    
    error 

    timestep_increment 
    '''
    nbrs_in_zor, d_repulsion = calc_direction_repulsion(individual, **kwargs) # eqn. 1
    
    if nbrs_in_zor:
        desired_direction = d_repulsion + kwargs['sensory_error']
    else:
        zoo_direction = calc_direction_orientation(individual, **kwargs) # eqn 2
        zoa_direction = calc_direction_attraction(individual, **kwargs) # eqn 3 
        
        desired_direction = combine_zoa_and_zor_directions(zoo_direction, 
                                                           zoa_direction, 
                                                           **kwargs)

    return(desired_direction)








if __name__ == '__main__':
    kwgs = {'Nbats':5, 'min_spacing':0.5, 'heading_variation':10, 
            'timesteps':10}
    xy, direction = initiate_simulations(**kwgs)
    
