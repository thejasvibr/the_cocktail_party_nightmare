# -*- coding: utf-8 -*-
""" Given a set of echoes that are heard in multiple trials
calculates how many trials it will take a bat to hear m of N echoes.

Created on Tue Nov 20 14:48:17 2018

@author: tbeleyur
"""

import numpy as np 

def calc_num_IPIs_to_hear_echoes(echoes_heard, minheard_echoes,
                                 n_echoes):
    '''Calculates the number of consecutive trials it takes to hear 
    at least m echoes of the total N echoes available in an interpulse interval.
    
    This function mimics how often a bat will gain a 'glimpse' of its surrounding
    eg, will it see 3/5 echoes every 10 interpulse intervals, or every 2 interpulse
    intervals ?

    Parameters:
        
        echoes_heard : Ntrials x N echoes np.array with one-hot coded entries.
                     Each trial is a row, and each column is an array. 
                     Eg. In a trial with 5 echoes, if the 2nd and 3rd echoes
                     are heard the row will be :
                         np.array([0,1,1,0,0])
        
        minheard_echoes : integer >0. Number of echoes that need to be heard
                          for one particular series of trials to be counted.
        
        n_echoes : integer>0. Number of echoes present in the interpulse interval


    Returns:

        numIPIs_to_hear_mechoes_quartiles : 1 x 3 np.array. The 25, 50 and 75 %ile
                                        of the number of trials it requires 
                                        to hear >= m echoes.
    '''
    # check if the input matrix is one-hot coded:
    binary_vals = set([0,1])
    input_vals = set(np.unique(echoes_heard))
    common_values =  binary_vals.union(input_vals)
    if len(common_values) >2 :
        raise ValueError('Not a binary array, please check input array...')

    rows_still_left = True
    trials_to_hear_mechoes = []

    while rows_still_left:
        num_trials = count_numrows_to_m_echoes(echoes_heard, minheard_echoes)

        if num_trials is None:
            trials_to_hear_mechoes.append(num_trials)
            break
        else:
            trials_to_hear_mechoes.append(int(num_trials))

        rows_still_left, echoes_heard = remove_previous_rows(echoes_heard, 
                                                             num_trials)
    return(trials_to_hear_mechoes)  



def remove_previous_rows(input_array, row_index):
    '''Tries to remove all rows above and including row_index.
    If it fails, then returns a Boolean, else another array.
    Parameters:
        input_array : Nrows x Ncols np.array 

        row_index : integer. The row index <=
                    which will be removed from input_array.
    
    Returns : 

        success :         
    
    '''
    
    
    try:
        sub_array = input_array[row_index:,:]
    except TypeError:
        success = False
        return(success, None)
        
    if sub_array.size >0:
        success = True
        return(success, sub_array)
    else:
        success = False
        return(success, None)


def count_numrows_to_m_echoes(echoes_heard, m_echoes):
    '''Get the number of rows it takes for m echoes
    to be heard from the top.
    
    '''
    
    cum_echoes_heard = np.cumsum(echoes_heard, 0)
    # get first row at which all echoes are heard :
    echoes_heard_mask = cum_echoes_heard>0
    echoes_heard_pertrial = np.sum(echoes_heard_mask, 1)
    trials_mechoes_heard = np.where(echoes_heard_pertrial >= m_echoes)[0]
    
    
    # if there are any trials at all : 
    if trials_mechoes_heard.size >0:
        first_trial = np.min(trials_mechoes_heard) 
        return(first_trial+1)
    else : 
        return(None)


    
    