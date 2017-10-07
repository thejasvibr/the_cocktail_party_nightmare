# -*- coding: utf-8 -*-
"""
Checking if the 'fractional length' of echoes/calls
and their overlap probabilities are actually
as I think they are.
Created on Tue Oct 03 15:27:52 2017

@author: tbeleyur

TO DO:

> make a high resolution time line
> add in one echo and 1,2,3..Ncalls  and
  check how often an overlap is seen

"""
from __future__ import division
import numpy as np
import random
import scipy.misc as misc
import matplotlib.pyplot as plt

random.seed(111)

def check_simultaneous_masking(echo_range,call_range):
    '''

    '''
    simult_ovlps = 0
    for each_call in call_range:

        overlap_occured = each_call == echo_range

        if  overlap_occured :

            simult_ovlps += 1

    return(simult_ovlps)

def check_masking(echo_range,calls,masking_region=[0,0]):
    '''
    Checks if the target echo has been overlapped or forward masked

    echo_range : list with 2 integers. with the indices of the echo's location in the timeline
    calls : list with sublists. container list with multiple calls
    fwd_masking: list with 2 integers. Number of iterations over which forward masking
                and backward masking occurs.
                Default is [0,0], which means no forward or backward masking.
    '''


    ovlps = 0

    echo_indices = range(echo_range[0],echo_range[1]+1)


    for each_call in calls:

        # first check if there is a direct overlap

        call_indices = range(each_call[0],each_call[1]+1)
        overlap_region = set(echo_indices) & set(call_indices)

        if len(overlap_region)>0:
            ovlps += 1

            return(1)


        elif sum(masking_region) > 0 :

            fwd_masker_seprtn = echo_range[0] - each_call[1]
            bkwd_masker_seprtn = each_call[0] - echo_range[1]

            fwd_mask_check = fwd_masker_seprtn <= masking_region[0] and fwd_masker_seprtn >= 0
            bkwd_mask_check = bkwd_masker_seprtn <= masking_region[1] and bkwd_masker_seprtn >= 0

            if fwd_mask_check or bkwd_mask_check :
                ovlps += 1


    if ovlps>0:

        return(1)

    else :

        return(0)



def generate_calls_randomly(timeline,calldurn_steps,Ncalls = 1,replicates=10**5):
    '''
    Function which does a Monte Carlo simulation of call arrival in the pulse
    interval

    Inputs:

    timeline: list. range object with iteration numbers ranging from 0 to the
             the number of iterations the pulse interval is parametrised to
    calldurn_steps : integer. the length of the calls which are arriving in the
              pulse interval
    Ncalls : integer. number of calls to generate per pulse interval.
    replicates: integer. number of times to generate Ncalls in the pulse interval

    Outputs:

    calls : list with sublists. The 1st sublist layer contains calls from multiple
            replicates. The 2nd layer contains the multiple calls within each replicate


    '''

    # Achtung: I actually assume a right truncated timeline here
    # because I want to ensure the full length of the call is always
    # assigned

    actual_timeline = timeline[:-calldurn_steps]

    multi_replicate_calls =[]

    for each_replicate in range(replicates):

        this_replicate = []

        for every_call in range(Ncalls):

            call_start = random.choice(actual_timeline)

            call_end = call_start + calldurn_steps

            if call_end > len(timeline):
                raise Exception('call_end is beyond current timeline')
            else:
               this_replicate.append([call_start,call_end])

        multi_replicate_calls.append(this_replicate)

    return(multi_replicate_calls)





if __name__ == '__main__':
    timeres = 10**-4
    length_timeline = 0.07
    timeline = range(int(length_timeline/timeres))

    # let's place the echo somewhere in the centre of the pulse interval
    echo_durn = 3*10**-3
    echo_durn_steps = int(echo_durn/timeres)

    echo_start = int(0.5*len(timeline))
    echo_end = echo_start + echo_durn_steps
    echo_range = [echo_start,echo_end]

    all_calls_containter = []
    call_densities = np.array([1,2,4,8,16,32])
    num_replicates = 10**2

    for this_calldensity in call_densities:
        print(this_calldensity)
        num_calls = this_calldensity

        all_calls_containter.append( generate_calls_randomly(timeline,echo_durn_steps,num_calls,num_replicates) )

    # and now let's look at the number of simulations with echo overlap :
    num_ovlps = []

    for each_calldensity in all_calls_containter:
        ovlps = []
        for all_replicates in each_calldensity:
            ovlps.append( check_masking(echo_range,all_replicates,[60,0]) )

        num_ovlps.append( sum(ovlps) )

    plt.plot(call_densities,np.array(num_ovlps)/num_replicates,'*-')
    plt.ylim(0,1)



#    nrow = 0
#    plt.figure(1)
#    for each_call in rand_calls[0]:
#        plt.plot(each_call,[nrow]*2)
#        nrow +=1
#
#
#    echo_x = echo_range
#    echo_y = [nrow+1]*2
#    plt.plot(echo_x,echo_y,'k')
