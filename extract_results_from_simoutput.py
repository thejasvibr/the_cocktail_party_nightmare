# -*- coding: utf-8 -*-
"""script to run and test the outputs of the CPN
Created on Sat Jun 08 10:29:30 2019

@author: tbeleyur
"""
import pickle
import sys
import time

folder = '.\\poisson-disc-master\\'

sys.path.append(folder)

import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000
import numpy as np
np.random.seed(82319)
import pandas as pd
import scipy.spatial as spl
import scipy.stats 

from the_cocktail_party_nightmare import run_CPN


def get_R_theta_of_detected_echoes(echo_ids, sounds_in_ipi):
    '''
    simulation_output : [echo_ids, sounds_in_ipi, group_geometry]

    echo_ids : 1 x Nbats-1  np.array with 1/0 entries. 1's indicate the echo
               of conspecific with this index was heard. 

    sounds_in_ipi : dictionary with following keys to pd.DataFrames

               target_echoes
               2ndary_echoes
               conspecific_calls

    
    '''
    detected_echoes = sounds_in_ipi['target_echoes'].loc[echo_ids==1,:]
    R_echoes = detected_echoes['R_incoming']
    theta_echoes = detected_echoes['theta']
    R_theta = np.column_stack((R_echoes, theta_echoes))
    return(R_theta)
    

def get_heard_NN(echo_ids, sounds_in_ipi):
    '''Gives nearest neighbour ranking of the heard conspecifics
    '''
    distance_conspecifics = sounds_in_ipi['target_echoes'].R_incoming
    rank_distances = scipy.stats.rankdata(distance_conspecifics)
    heard_neigbours_rank = rank_distances[echo_ids==1]
    return(heard_neigbours_rank)


def gather_results_from_simoutput(sim_output):
    '''Gives 
    
    R_theta_NN : np.array with the following columns
                R, theta, NN_ranking, received_level
    
    '''
    echo_ids, sounds_in_ipi, group_geometry = sim_output
    NN_ranks = get_heard_NN(echo_ids, sounds_in_ipi)
    R_theta = get_R_theta_of_detected_echoes(echo_ids, sounds_in_ipi)
    echo_levels = sounds_in_ipi['target_echoes'].loc[echo_ids==1,'level']
    
    summary_output = np.column_stack((R_theta, NN_ranks,
                                      echo_levels))

    return(summary_output)

def make_R_theta_data(results, group_size, num_replicates):
    '''Generates summary data on the R and theta plot
    '''
    R_theta = []
    for i in range(num_replicates):
        R_theta.append(results[(group_size, i)][:,[0,1]])
    r_theta_output = np.concatenate(R_theta).reshape(-1,2)
    return(r_theta_output)
        
def make_NNrank_data(results, group_size, num_replicates):
    '''
    '''
    NN_ranks = np.concatenate(map(lambda i : results[(group_size,i)][:,2]  ,
                   range(num_replicates)))
    return(NN_ranks)

def make_received_level_data(results, group_size, num_replicates):
    '''
    '''
    echo_level = np.concatenate(map(lambda i : results[(group_size,i)][:,3]  ,
                   range(num_replicates)))
    return(echo_level)
    





if __name__== '__main__':

    A = 7
    B = 2 
    
    kwargs={}
    kwargs['interpulse_interval'] = 0.1
    kwargs['v_sound'] = 330.0
    kwargs['simtime_resolution'] = 10**-6
    kwargs['echocall_duration'] = 0.003
    kwargs['call_directionality'] = lambda X : A*(np.cos(np.deg2rad(X))-1)
    kwargs['hearing_directionality'] = lambda X : B*(np.cos(np.deg2rad(X))-1)
    reflection_func = pd.read_csv('data/bistatic_TS_bat.csv')
    kwargs['reflection_function'] = reflection_func
    kwargs['heading_variation'] = 90
    kwargs['min_spacing'] = 0.5
    kwargs['Nbats'] = 25
    kwargs['source_level'] = {'dBSPL' : 120, 'ref_distance':0.1}
    kwargs['hearing_threshold'] = 20
    
    fwd_masking_region = np.linspace(-27, -7, 20000)        
    bkwd_masking_region = np.linspace(-10, -24, 3000)
    simult_masking_regio = np.array([-8])
    
    temporal_masking_fn = (fwd_masking_region,simult_masking_regio,
                                            bkwd_masking_region)
#   
#    temporal_masking_fn = pd.read_csv('data/temporal_masking_fn.csv')
    spatial_unmasking_fn = pd.read_csv('data/spatial_release_fn.csv')
    kwargs['temporal_masking_thresholds'] = temporal_masking_fn
    kwargs['spatial_release_fn'] = spatial_unmasking_fn
    
    start = time.time()
    all_num_echoes = {}
    for group_size in [2,4,8]:
        sim_results = {}
        print(group_size)
        kwargs['Nbats'] = group_size
        group_run = time.time()
        file_name = 'CPN_'+str(group_size)
        outfile = open(file_name,'wb')
        for run in range(10):
            num_echoes, sim_output = run_CPN(**kwargs)
            run_id = (group_size,run)
            all_num_echoes[run_id] = num_echoes
            sim_results[run_id] = sim_output
        pickle.dump(sim_results, outfile)
        outfile.close()
            
        print('Groupsize of 1000X runs took',time.time()-group_run)
    print(time.time()-start)