# -*- coding: utf-8 -*-
"""script to process and format the outputs of the CPN
Created on Sat Jun 08 10:29:30 2019

@author: tbeleyur
"""

import numpy as np
import scipy.stats 


def gather_results_from_simoutput(sim_output, detected_status=1):
    '''Generates heard/unheard echo properties in terms of 
    R , theta, neighbour rank and echo level. 

    Parameters
    ----------
    simulation_output : tuple/list with the following simulation outputs in them.
                        [echo_ids, sounds_in_ipi, group_geometry].
                        This is the output from one simulation. 
    
    detected_status : binary int. 1/0 where 1 indicates the extraction of 
                      heard echoes and 0 of undetected echoes. Defaults to 1.
    
    R_theta_NN : np.array with the following columns
                R, theta, NN_ranking, received_level

    Returns
    -------
    summary_output : nechoes x 4 np.array. The outputs from one simulation
                     in terms of its R, theta, neighbour rank and level
                     of the arriving echo. 

                     Note : if the simulation resulted in NO echoes being
                     heard/unheard , then nechoes=0, and am empty 
                     np.array will be returned. 
    '''
    echo_ids, sounds_in_ipi, group_geometry = sim_output
    NN_ranks = get_heard_NN(echo_ids, sounds_in_ipi, detected_status)
    R_theta = get_R_theta_of_echoes(echo_ids, sounds_in_ipi,detected_status)
    echo_levels = sounds_in_ipi['target_echoes'].loc[echo_ids==detected_status,'level']

    summary_output = np.column_stack((R_theta, NN_ranks,
                                      echo_levels))

    return(summary_output)




def get_R_theta_of_echoes(echo_ids, sounds_in_ipi,
                                   detected_status=1):
    '''
    simulation_output : [echo_ids, sounds_in_ipi, group_geometry]

    Parameters
    ----------

    echo_ids : 1 x Nbats-1  np.array with 1/0 entries. 1's indicate the echo
               of conspecific with this index was heard. 

    sounds_in_ipi : dictionary with following keys to pd.DataFrames

               target_echoes
               2ndary_echoes
               conspecific_calls
    
    detected_status : binary int. 1/0 where 1 indicates the extraction of 
                      heard echoes and 0 of undetected echoes. Defaults to 1.

    Returns
    -------
    R_theta : nechoes x 2 np.array. Column 0 holds the radial distance of 
              of the echo's source. Column 1 holds the angle of arrival. 
   
    '''
    echoes = sounds_in_ipi['target_echoes'].loc[echo_ids==detected_status,:]
    R_echoes = echoes['R_incoming']
    theta_echoes = echoes['theta']
    R_theta = np.column_stack((R_echoes, theta_echoes))
    return(R_theta)
    

def get_heard_NN(echo_ids, sounds_in_ipi, detected_status=1):
    '''Gives nearest neighbour ranking of the heard conspecifics

    Parameters
    ----------
    echo_ids : 1 x Nbats-1  np.array with 1/0 entries. 1's indicate the echo
               of conspecific with this index was heard. 

    sounds_in_ipi : dictionary with following keys to pd.DataFrames

               target_echoes
               2ndary_echoes
               conspecific_calls
    
    detected_status : binary int. 1/0 where 1 indicates the extraction of 
                      heard echoes and 0 of undetected echoes. Defaults to 1.
    Returns
    -------
    heard_neighbours_rank : necheos x np.array
                            Each row contains the rank that may be between
                            1 to Nbats-1. 
                           
        
    '''
    distance_conspecifics = sounds_in_ipi['target_echoes'].R_incoming
    rank_distances = scipy.stats.rankdata(distance_conspecifics)
    heard_neigbours_rank = rank_distances[echo_ids==detected_status]
    return(heard_neigbours_rank)

