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

def check_masking(echo_range,calls,masking_region=[0,0]):
    '''
    Checks if the target echo has been overlapped or forward masked

    echo_range : list with 2 integers. with the indices of the echo's location in the timeline
    calls : list with sublists. container list with multiple calls
    masking_region: list with 2 integers. Number of iterations over which forward masking
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


def check_spatial_unmasking(echo_angle, call_arrivalangles,sp_unmask_func):
    '''
    Checks if the echo and a call could have possibly been heard because of
    spatial release from masking , or spatial unmasking.

    It does this by comparing the difference in angles of arrival (deltaAOA) of the
    echo and the masking call. If the deltaAOA is more than what is expected for
    an angle of arrival (AOA) - then the echo could have been heard.

    Inputs:
        echo_angle :0<=float<=2pi. AOA of the incoming focal echo in degrees.
                    Zero-degrees begins at 3'o clock and the angles increase ccw

        call_arrivalangles: list with float values. Container list with AOAs of
                        the masking calls

        sp_unmask_func : dictionary.
                        The spatial unmasking function  describes how deltaAOA
                        varies with the AOA. The top row has the AOA bins and
                        the bottom row is the deltaAOA.


    '''

    for each_call in call_arrivalangles:
        pass




    pass


def generate_calls_randomly(timeline,calldurn_steps,Ncalls = 1,replicates=10**5):
    '''
    Function which does a Monte Carlo simulation of call arrival in the pulse
    interval

    Inputs:

    timeline: list. range object with iteration numbers ranging from 0 to the
             the number of iterations the inter pulse interval consists of

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
    # assigned within the inter-pulse interval

    actual_timeline = timeline[:-calldurn_steps]

    multi_replicate_calls =[]

    for each_replicate in range(replicates):

        this_replicate = []

        for every_call in range(Ncalls):

            call_start = random.choice(actual_timeline)
            call_end = call_start + calldurn_steps -1

            if call_end > len(timeline):

                raise Exception('call_end is beyond current timeline')

            else:

               this_replicate.append([call_start,call_end])

        multi_replicate_calls.append(this_replicate)

    return(multi_replicate_calls)



def generate_arrival_angles(angle_range,angular_resolution,num_calls):
    '''
    Simulate the arrival angles of masking calls

    Inputs:
        angle_range : tuple. Angular range within which the masking calls arrive
                      . Angles are defined in ccw direction, with 3 'o clock being
                      0 degrees.
        angular_resolution: float. The 'fineness' with which the arrival angles
                            are generated
        num_calls : integer. number of calls for which angles need to be assigned

    Outputs:
        call_arrivalangles : 1 x num_calls numpy array

    '''

    if angle_range[0] > angle_range[1]:
        raise ValueError('The first value in angle_range must be lesser than the seconds')


    all_angles = np.arange(angle_range[0],angle_range[1]+angular_resolution,angular_resolution)

    arrival_angles = np.random.choice(all_angles,num_calls)

    return(arrival_angles)






class simulate_jamming_experiment():

    def __init__(self,pi_durn,time_res,echo_durn,num_echoes,call_density,num_replicates):

        self.num_replicates = num_replicates

        self.pi_durn = pi_durn
        self.time_res = time_res # 10**-4 ## based on the minimum double-echo separation bats can detect [Simmons et al. XXXX]
        self.call_densities = call_density

        self.num_bins = int(self.pi_durn/self.time_res)
        self.pi_as_bins = range(self.num_bins)

        self.echo_durn = echo_durn
        self.echo_numbins = int(self.echo_durn/self.time_res)
        self.num_echoes = num_echoes

        self.echo_times = np.linspace(0,self.pi_durn-0.02,self.num_echoes) # the start times of the echos

        if min(self.echo_times) < 0:
            raise ValueError('echoes cannot be placed in negative time')


        self.echo_bins = np.int64(self.echo_times/self.time_res)

        self.echo_positions = [ [thisecho_start, thisecho_start+self.echo_numbins-1] for thisecho_start in self.echo_bins]


        self.fwd_masking_bins = None

        self.bkwd_masking_bins = None



    def compile_all_steps(self):

        self.sim_random_call_arrival()
        self.analyse_masking()



    def sim_random_call_arrival(self):
        self.all_calls_containter = []
        print('here we are')
        for this_calldensity in self.call_densities:
            print('calls being generated at ',this_calldensity,' per pulse interval')
            self.all_calls_containter.append( generate_calls_randomly(self.pi_as_bins,
                                                                      self.echo_numbins,
                                                                      this_calldensity,
                                                                      self.num_replicates) )


    def analyse_masking(self):

        self.heardechoes_calldens = []

        if self.fwd_masking_bins == None or self.bkwd_masking_bins == None:
            raise ValueError('Need to define fwd and/or bkwd masking conditions')


        i = 0
        for each_calldensity in self.all_calls_containter:
                print('analysing masking at:',self.call_densities[i],' density')
                i += 1
                self.unmaskedechoes = []

                for each_replicate in each_calldensity:


                    echoes_masked = [ check_masking( each_echo, each_replicate,[self.fwd_masking_bins,self.bkwd_masking_bins] ) for each_echo in self.echo_positions ]

                    num_masked = sum(echoes_masked)

                    self.num_free = self.num_echoes - num_masked

                    self.unmaskedechoes.append(self.num_free)


                self.heardechoes_calldens.append(self.unmaskedechoes)

        self.p_numechoesheard = [  each_density.count(numberofechoes)  for each_density in self.heardechoes_calldens  for numberofechoes in range(self.num_echoes) ]

        self.num_unmaskedechoes = []

        for each_density in self.heardechoes_calldens:

            self.unmasked_thisdensity = []

            for numberofechoes in range(self.num_echoes+1):

                self.unmasked_thisdensity.append(  each_density.count(numberofechoes)/float(self.num_replicates) )

            self.num_unmaskedechoes.append(self.unmasked_thisdensity)


        self.prob_unmaskedechoes = np.array(self.num_unmaskedechoes)


    def convert_to_P_nechoes(self):
        '''
        Converts the number of echoes based probability array
        to a P( 1,2>= number of echoes heard) value at each call density

        '''
        print('converting...')
        self.geq_1echo = []
        self.geq_2echoes = []
        self.geq_3echoes = []

        if self.num_echoes > 2 :
            morethan2echoes = True


        self.num_rows = self.prob_unmaskedechoes.shape[0]
        for each_row in range(self.num_rows):
            try:
                self.geq_1echo.append( np.sum(self.prob_unmaskedechoes[each_row,1:]))
                self.geq_2echoes.append( np.sum(self.prob_unmaskedechoes[each_row,2:]))

                if morethan2echoes:
                    self.geq_3echoes.append( np.sum(self.prob_unmaskedechoes[each_row,3:]))


            except:
                raise IndexError('P(>=Nechoes) being calculated for more echoes than actually present')




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

#    plt.plot(call_densities,np.array(num_ovlps)/num_replicates,'*-')
#    plt.ylim(0,1)

    b = simulate_jamming_experiment(0.1,0.0001,0.003,3,[10,5,6],10)
    b.fwd_masking_bins = 0;b.bkwd_masking_bins = 0
    b.compile_all_steps()
    b.convert_to_P_nechoes()



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
