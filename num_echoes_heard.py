# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:13:26 2017

@author: tbeleyur and others
"""

from __future__ import division
import itertools, operator
import numpy as np

def combinations_with_replacement_counts(n, r):
    # code credit: https://stackoverflow.com/questions/37711817/generate-all-possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego
    size = n + r - 1
    for indices in itertools.combinations(range(size), n-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield tuple(map(operator.sub, stops, starts))


def count_num_echoes(all_combinations,nechoes=3,maskingwindow=1):
    '''
    Given a set of combinations of indistinguishable balls in distinguishable boxes
    - counts the number of cases where the 'echoes' could be heard as decided by the maskingwindow

    Input:
    all_combinations: list with sublists. each entry in the sublist has the number of balls in that box
    nechoes : number of target echoes
    maskingwindow: the number of boxes (including one box for the echo) which are to be considered

    Output:
    P_echoes : list with fractions of cases where 0 ,1,2...nechoes are audible.

    The masking window is a composite window treated as a block of time within which a call is not supposed to be registered
    for an echo to be heard
    '''

    # designate the echos with their maskingwindows as being laid out one after the other
    # we are not including any temporal aspects in the hearing/masking in this model - and so it doesn't really matter which
    # set of boxes we assign to the masking windows. This is a choice to make the coding easier.

    echo_start = np.arange(0,maskingwindow*nechoes,maskingwindow)
    echo_end = echo_start + maskingwindow
    if maskingwindow >1:

        echo_ranges = [ [echo_start[each_echo],echo_end[each_echo]]  for each_echo in range(nechoes)   ]
    else:
        echo_ranges = range(nechoes)

    print(echo_ranges)


    num_freeechoes = [ calc_numfreeechoes(each_combination,echo_ranges)  for each_combination in all_combinations ]

    n_freechoes = [ num_freeechoes.count(k) for k in range(nechoes+1) ]

    return(n_freechoes)







def calc_numfreeechoes(one_combination,echo_ranges):
    '''
    '''
    num_freeechoes = 0

    for each_echo in echo_ranges:

        try:
            num_calls = sum( one_combination[ each_echo[0]:each_echo[1] ] )
        except:
            #print(each_echo)
            num_calls =  one_combination[ each_echo ]


        if num_calls == 0:
            num_freeechoes += 1
        #print(num_freeechoes)


    return(num_freeechoes)





if __name__ == '__main__':


    for num_calls in [4,5]:
        print(num_calls)
        y = list(combinations_with_replacement_counts(33,num_calls))
        q = count_num_echoes(y,3,2);
        print( np.array(q)/float(len(y)) )
        #calc_numfreeechoes(y[6500],range(5))
    #z = [ [0,0,0,1,1,1,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,1] ]
    #z_u = calculate_fraction_echoes(z,2,2)

