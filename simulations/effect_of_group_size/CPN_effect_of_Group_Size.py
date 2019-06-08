# -*- coding: utf-8 -*-
"""Running the Sonar Cocktail Party Nightmare:
    Checking the effect of Group Size. 

Created on Sat Jun 08 16:59:26 2019

@author: tbeleyur
"""

from the_cocktail_party_nightmare import run_CPN
import pickle 

import numpy as np 
np.random.seed(82319)



# load the common simulation parameters 
common_keyword_args = None

# set the varying factor - group size 
group_sizes = [5,10,15,20,25]

def run_each_groupsize(group_size, common_kwargs,num_replicates = 10**4):
    '''
    '''
    simoutput_container = {(group_size,i) : None for i in range(num_replicates)}

    for replicate_run in xrange(num_replicates):
        num_echoes, sim_output = run_CPN(**common_kwargs)
        simoutput_container[(group_size, replicate_run)] = sim_output

    picklefilename = 'simulation_results//effect_of_group_size//' +str(group_size)+'_CPN.pkl'
    try:
        with open(picklefilename, 'wb'):
            pickle.dump(simoutput_container, picklefilename)
    except:
        raise IOError('UNABLE TO SAVE PICKLE FILE!!')
    
    
    
        
        
        
    


    