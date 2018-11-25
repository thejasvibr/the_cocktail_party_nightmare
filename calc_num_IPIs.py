# -*- coding: utf-8 -*-
""" Given a set of echoes that are heard in multiple trials
calculates how many trials it will take a bat to hear m of N echoes.

Created on Tue Nov 20 14:48:17 2018

@author: tbeleyur
"""

import numpy as np 

def calc_num_IPIs_to_hear_echoes(echoes_heard, minheard_echoes,
                                 n_echoes, **kwargs):
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

    Keyword Arguments:
        
        numtimes_allechoes : integer. Number of times that *all* echoes are detected.
                             Defaults to 1. For example, if numtimes_allechoes = 2 
                             and then the number of rows till 3 echoes are seen twice
                             is counted. 

    Returns:

        trials_to_hear_mechoes : list with integers. The number of consecutive
                                 trials it takes to hear m echoes of N present ones.

                                 
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
        num_trials = count_numrows_to_m_echoes(echoes_heard, minheard_echoes,
                                               **kwargs)

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


def count_numrows_to_m_echoes(echoes_heard, m_echoes, **kwargs):
    '''Get the number of rows it takes for m echoes
    to be heard from the top.
    
    Parameters:
        
        echoes_heard : Ntrials x N echoes np.array with one-hot coded entries.
                     Each trial is a row, and each column is an array. 
                     Eg. In a trial with 5 echoes, if the 2nd and 3rd echoes
                     are heard the row will be :
                         np.array([0,1,1,0,0])

        m_echoes : integer. Minimum number of echoes to be detected.

    Keyword Arguments:
        
        numtimes_allechoes : integer. Number of times that all echoes are visible.
                             Defaults to 1. For example, if numtimes_allechoes = 2 
                             then the number of rows till all echoes are seen twice
                             is counted. 
    
    Returns : 
        first_trial : integer or None. Number of rows till the m_echoes criterion
                      is satisfied.

    Note: If numtimes_allechoes is >1 , then m_echoes must be equal to the 
    number of echoes in the interpulse interval. An error will be thrown otherwise
    as this switch was specifically written to count the number of trials till 
    *all echoes* are detected.
    '''
    
    if 'numtimes_allechoes' in  kwargs.keys():
        numtimes_allechoes = int(kwargs['numtimes_allechoes'])
        # check that the minimum number of echoes to be detected 
        # is equal to the full number of echoes in the IPI
        check_if_mechoes_equals_allechoes(echoes_heard, m_echoes)
    else:
        numtimes_allechoes = int(1)

    cum_echoes_heard = np.cumsum(echoes_heard, 0)
    # get first row at which all echoes are heard at least numtimes:
    echoes_heard_mask = cum_echoes_heard >= numtimes_allechoes
    echoes_heard_pertrial = np.sum(echoes_heard_mask, 1)

    trials_mechoes_heard = np.where(echoes_heard_pertrial >= m_echoes)[0]

    # if there are any trials at all : 
    if trials_mechoes_heard.size >0:
        first_trial = np.min(trials_mechoes_heard) 
        return(first_trial+1)
    else : 
        return(None)

def check_if_mechoes_equals_allechoes(echoes_heard, m_echoes):
    if m_echoes != echoes_heard.shape[1]:
        raise ValueError('m_echoes != number of echoes, this case is not yet implemented')
    else:
        pass
   

def calc_numtrials_per_echonumber(numechoes_results, **kwargs):
    '''Calculate the number of trials required to hear m echoes of N target echoes
    across all call densities. 
    
    Parameters:
    
        numechoes_results : dictionary with 
                        multiple entries, including a key with
                        'echo_ids' and 'num_targetechoes.
                Key-wise dictinoary contents :
                'echo_ids' : Calldensities x numtrials x numechoes np.array with one hot coding if an echo is heard or not.
                'num_targetechoes' : integer. number of echoes present in an interpulse interval.

        Keyword Arguments:
        min_echoes : integer. Number of echoes to be heard. Defaults to numechoes. 
    
    Returns:
    
        numtrials : list with integers and Nones. The number of trials required to hear all echoes (90%ile)      

    '''
    numtrials = {}
    num_echoes = numechoes_results['num_targetechoes']
    
    if 'min_echoes' in kwargs.keys():
        min_echoes = kwargs['min_echoes']
    else:
        min_echoes = num_echoes
    
    if min_echoes > num_echoes:
        raise ValueError('The number of echoes to be detected is more than number of echoes present')

    for j,each_calldensity in enumerate(np.arange(1,33,2)):
        numtrials[each_calldensity] = calc_num_IPIs_to_hear_echoes(numechoes_results['echo_id'][j,:,:],
                                                             min_echoes, num_echoes,**kwargs)
    
    return(numtrials)

    
    