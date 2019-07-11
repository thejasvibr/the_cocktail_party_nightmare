 #-*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
import numpy as np 
import scipy.spatial as spatial
from the_cocktail_party_nightmare import  place_bats_inspace



def simulate_collective_movement(**kwargs):
    '''
    '''
    
    xy_positions, directions = initiate_simulations(**kwargs)
    kwargs['xy'] = xy_positions
    kwargs['directions'] = directions

    for each_timestep in xrange(1, kwargs['timesteps']):
        kwargs['current_timestep'] = each_timestep

        for individual in xrange(kwargs['Nbats']):
            desired_direction = calculate_desired_direction_of_movement(individual, **kwargs)
            
            # effective direction is limited by the max turning angle of the individual 
            effective_direction = calculate_effective_direction(individual, 
                                                                desired_direction,
                                                                **kwargs)

            new_position = calculate_new_positions(effective_direction, 
                                                   **kwargs)

            assign_new_direction_and_position(individual, effective_direction, 
                                              **kwargs)



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
    sensory_error
    speed
    
    turning_rate
    
    r_repulsion
    r_orientation
    r_attraction

    field_perception
    
    error 

    timestep_increment 
    '''
    print('calculating repulsion')
    [nbrs_in_zor, d_repulsion] = calc_direction_repulsion_or_attraction(individual, -1, **kwargs) # eqn. 1
    
    if nbrs_in_zor:
        print('repelled by ',nbrs_in_zor)
        desired_direction = d_repulsion 
    else:
        print('calcualting orientation nbrs')
        zoo_output = calc_direction_orientation(individual, **kwargs) # eqn 2
        zoa_output = calc_direction_repulsion_or_attraction(individual, 1, **kwargs) # eqn 3 
        
        desired_direction = combine_zoa_and_zor_directions(zoa_output, 
                                                           zoo_output)
    return(desired_direction)

def calculate_effective_direction(calculate_effective_direction(individual, 
                                                                desired_direction,
                                                                **kwargs):
            '''
            '''
            


def calc_direction_repulsion_or_attraction(index, to_or_away, **kwargs):
    '''Calculates direction of repulsion or attraction based on summing 
    the unitvectors of the current individual to its neighbours. 
    
    Repulsion ('away'') is implemented by multiplication by -1 of the average vector
    Attaction ('to') is implement by multiplicaiton by +1 of the average vector

    Parameters
    ----------
    index : int >= 0.
            Row index identifier for the current individual being calculated

    to_or_away : integer, either +1 or -1.
                +1 indicates attraction is being calculated
                -1 indicates repulsion is being calculated

    Returns
    -------
    [bats_in_zone, avg_direction] : list with two objects:

        bats_in_zone : Boolean. 
                     True if there are bats in the repulsion/attraction zones

        avg_direction : 1 x 2 np.array 
                        unit vector indicating the overall direction of repulse/attraction
    '''
    current_batxy = kwargs['xy'][:,:,kwargs['current_timestep']]
    if to_or_away == -1:
        zone_range = (0, kwargs['r_repulsion'])
    elif to_or_away == 1:
        zone_range = (kwargs['r_orientation'], kwargs['r_attraction'])
    
    bats_in_zone, bat_ids = check_for_individuals_in_zone(zone_range, index, 
                                                          current_batxy)
    print('bat_ids',bat_ids,'to_or_away',to_or_away)
    if bats_in_zone:
        avg_direction = to_or_away*calculate_rij_sum_direction(current_batxy[index,:],
                                                               current_batxy[bat_ids,:])
    else:
        avg_direction = np.nan

    return([bats_in_zone, avg_direction])
    
def calc_direction_orientation(index, **kwargs):
    '''
    '''
    current_batxy = kwargs['xy'][:,:,kwargs['current_timestep']]
    # first check if any individuals fall in the zone :
    zone_range = (kwargs['r_repulsion'], kwargs['r_attraction'])
    bats_in_zone, bat_ids = check_for_individuals_in_zone(zone_range, index, 
                                                              current_batxy)
    print('bats in orientaiton zone', bat_ids)
    prev_timestep = kwargs['current_timestep']-1

    if prev_timestep < 0:
        prev_timestep = 0

    prev_directions = kwargs['directions'][bat_ids,:,prev_timestep]
    if bats_in_zone:
        avg_direction = calculate_vj_sum_direction(prev_directions)
    else:
        avg_direction = prev_directions[index,:]
    
    return(bats_in_zone, avg_direction)
    
def combine_zoa_and_zor_directions(zoa, zoo):
    '''
    '''
    forces = [zoa, zoo]
    forces_acting = [zoa[0],zoo[0]]
    if np.all(forces_acting):
        print('taking average directions....')
        final_direction = (zoo[1] + zoa[1])*0.5
    else:
        only_one_force = np.argwhere(forces_acting)
        final_direction = forces[only_one_force][1]
    return(final_direction)
        
                                     



def check_for_individuals_in_zone(radial_range, focal_index, xy_posns):
    '''
    Parameters
    ----------
    radial_range : tuple/array-like. 
                   [rmin, rmax] with the minimum and maximum radial distances

    focal_index : int >=0. 
                  Row index of the current bat being dealt with 

    xy_posns : Nbats x 2 np.array 
               current xy coordinates of individuals. 
    '''
    dist_to_focal = spatial.distance_matrix(xy_posns, xy_posns)[:,focal_index]
    within_zone = np.logical_and(dist_to_focal>=radial_range[0],
                                dist_to_focal<=radial_range[1])
    all_indices = np.argwhere(within_zone).flatten()
    valid_indices = all_indices[np.argwhere(all_indices!= focal_index).flatten()]

    if valid_indices.shape[0] >0:
        return( True, valid_indices )
    else:
        return( False, np.array([]) )

def calculate_rij_sum_direction(index_posn, other_posns):
    ''' implement the summation term in eqns1 and 3.

    '''
    rij = other_posns - index_posn
    rij_norm = np.linalg.norm(rij, axis=1)
    unit_vector = np.array([0.0,0.0])
    for each_rij, each_norm in zip(rij, rij_norm):
        unit_vector += each_rij/each_norm
    return(unit_vector)

def calculate_vj_sum_direction(all_vjs):
    '''implements summation in eqn 2. 
    '''
    orientation_vector = np.array([0.0,0.0])
    norm_vjs = np.linalg.norm(all_vjs, axis=1)
    for each_vj, norm_vj in zip(all_vjs, norm_vjs):
        orientation_vector += each_vj/norm_vj
    return(orientation_vector)



if __name__ == '__main__':
    kwargs = {'Nbats':100, 'min_spacing':0.8, 'heading_variation':10, 
            'timesteps':10, 'current_timestep':0, 
            'sensory_error':np.random.choice(np.arange(0,90),1)
            , 'r_repulsion':0.6,
            'r_orientation':1.5,
            'r_attraction':15}
    xy, direction = initiate_simulations(**kwargs)
    kwargs['xy'] = xy
    kwargs['directions'] = direction
    calculate_desired_direction_of_movement(0,**kwargs)
    plt.plot(xy[:,0,0], xy[:,1,0],'*')

    for i, xypos in enumerate(xy[:,:,0]):
        x,y = xypos
        plt.text(x+0.1,y+0.1,str(i))
