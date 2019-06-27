# -*- coding: utf-8 -*-
"""script to process and format the outputs of the CPN
Created on Sat Jun 08 10:29:30 2019

@author: tbeleyur
"""
import pickle
import numpy as np
import scipy.stats 

import re
import glob
import multiprocessing
import sys

import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 1000


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


def get_summary_for_each_variablevalue(variable_value,all_files, **kwargs):
    '''relies on a file called result_files 

    Parameters
    ----------
    variable_value : variable value

    all_files : list with filepaths. Each filepath is of a pickle file that has
                multiple simulation runs in it. 

    Returns
    -------
    
    '''
    print(variable_value, ' starting')
    group_results = []
    numechoes_heard = []
    for each_simrun in all_files[variable_value]:
        with open(each_simrun, 'rb') as results:
            simoutputs = pickle.load(results)
            for _, outputs in simoutputs.iteritems():
                processed_results = gather_results_from_simoutput(outputs,
                                                                      kwargs['detected_status'])
                num_echoes = processed_results.shape[0]
                group_results.append(processed_results)
                numechoes_heard.append(num_echoes)
    r, theta, nn, level = np.split(np.concatenate(group_results).reshape(-1,4), 4, axis=1)
    print(variable_value, 'group size Done')
    return(numechoes_heard, r, theta, nn, level)

def process_simulation_outputs(results_folder, variable_values, **kwargs):
    '''
    '''
    result_files = {value: glob.glob(results_folder+str(value)+'bats_CPN*.pkl') for value in variable_values}
    if len(kwargs.keys())>0:
        summary_for_variablevalue = lambda  X: get_summary_for_each_variablevalue( X, result_files,**kwargs)
    else:
        summary_for_variablevalue = lambda  X: get_summary_for_each_variablevalue( X, result_files)

    outputs = map(summary_for_variablevalue, variable_values)
    numechoes, r, theta, nn, level = {}, {}, {}, {}, {}

    for value_output, value in zip(outputs, variable_values):
        numechoes[value] = value_output[0]
        r[value] = value_output[1]
        theta[value] = value_output[2]
        nn[value] = value_output[3]
        level[value] = value_output[4]
    return(numechoes, r, theta, nn, level)
        
        
        
        


if __name__ == '__main__':
    results_folder = 'D:\\the_cocktail_party_nightmare\\simulations\\effect_of_group_size_noshadowing\\'
    group_size = [5]
    result_files = {Nbats : glob.glob(results_folder+str(Nbats)+'bats_CPN*.pkl') for Nbats in group_size}

    nechoes, r, theta, nn, level = process_simulation_outputs(results_folder,[20],
                                                              **{'detected_status':1})

    nechoes, r_not, theta_not, nn_not, level_not = process_simulation_outputs(results_folder,[20],
                                                              **{'detected_status':0})

