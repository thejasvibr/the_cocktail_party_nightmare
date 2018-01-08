# -*- coding: utf-8 -*-
"""Tests for the_cocktail_party_nightmare_MC
Created on Tue Dec 12 22:07:22 2017

@author: tbeleyur
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from the_cocktail_party_nightmare_MC import *

class TestTheCPN(unittest.TestCase):

    def setUp(self):
        temp_mask_colnames = ['timegap_ms','deltadB']
        time_range = np.linspace(10,-1)


        deltadB_vals = np.linspace(-10,0,time_range.size)
        self.temporalmasking_fn = pd.DataFrame(index=range(time_range.size),
                                         columns = temp_mask_colnames)

        self.temporalmasking_fn['deltadB'] = deltadB_vals
        self.temporalmasking_fn['timegap_ms'] = time_range

        deltatheta_vals = np.linspace(0,25,100)
        sp_rel_colnames = ['deltatheta','dB_release']
        self.spatialrelease_fn =  pd.DataFrame(index=range(deltatheta_vals.size),
                                         columns = sp_rel_colnames)
        self.spatialrelease_fn['deltatheta'] = deltatheta_vals
        self.spatialrelease_fn['dB_release'] = np.linspace(0,-25,deltatheta_vals.size)



    def test_populate_sounds(self):

        pi_range = np.arange(0,1000,1)
        call_durn = 3
        intensity_range = (60,80)
        arrival_angles = (75,105)
        numcalls_in_pi = 10

        all_calls = populate_sounds(pi_range,call_durn,intensity_range,
                                    arrival_angles,numcalls_in_pi)

        nrows,ncols = all_calls.shape
        self.assertEqual(nrows,numcalls_in_pi)
        self.assertEqual(ncols,4)

    def test_calc_angularseparation(self):
        a1 = 20
        a2 = 40

        self.assertEqual(calc_angular_separation(a1,a2),20)

        a3 = -90
        self.assertEqual(calc_angular_separation(a1,a3),110)

        a4 = 310
        self.assertEqual(calc_angular_separation(a1,a4),70)


    def test_quantify_temporal_masking(self):
        col_names = ['start','stop','theta','level']
        call = pd.DataFrame(index=[0],columns=col_names)
        echo = pd.DataFrame(index=[0],columns=col_names)

        call['start'] = 0; call['stop'] = 10
        echo['start'] = 1 ; echo['stop'] = 11

        timegap_simult = quantify_temporalmasking(echo,call)

        self.assertEqual(timegap_simult,0)

        # case where there is non-overlapping fwd masking:
        call['start'] = 3; call['stop'] = 13
        echo['start'] = 15; echo['stop'] = 19

        fwdmask_nonovlp = quantify_temporalmasking(echo,call)
        self.assertEqual(fwdmask_nonovlp[0],2)

        call['start'] = 5; call['stop'] = 15
        fwdmask_ovlp = quantify_temporalmasking(echo,call)
        self.assertEqual(fwdmask_ovlp,0)

        # simultaneous masking :
        call['start'] = echo['start']; call['stop'] = echo['stop']
        simult_ovlp = quantify_temporalmasking(echo,call)
        self.assertEqual(simult_ovlp,0)

        # bkwd overlap :
        call['start'] = echo['start']+2 ; call['stop'] = echo['stop']+2
        bkwd_ovlp = quantify_temporalmasking(echo,call)
        self.assertEqual(bkwd_ovlp[0],-2)

        #bkwd no overlap :
        call['start'] = echo['start'] +20; call['stop'] = echo['stop'] + 20
        bkwd_nonoverlap = quantify_temporalmasking(echo,call)
        self.assertEqual(bkwd_nonoverlap[0],-20)


    def test_check_if_echo_heard(self):
        col_names = ['start','stop','theta','level']
        call = pd.DataFrame(index=[0],columns=col_names)
        echo = pd.DataFrame(index=[0],columns=col_names)

        call['start'] = 00; call['stop'] =10
        echo['start'] = 5000000; echo['stop'] = 5000100
        call['theta'] = 60; echo['theta'] = 60
        call['level'] = 80 ; echo['level'] = 60
        # check the case where the echo and call are *very* far away.

        call_far_away = check_if_echo_heard(echo,call,self.temporalmasking_fn,
                                            self.spatialrelease_fn)
        self.assertTrue(call_far_away)



        # not so far away : this will be false

        call['start'] = echo['start']+20 ; call['stop'] = echo['stop']+20
        call_nearby = check_if_echo_heard(echo,call,self.temporalmasking_fn,
                                            self.spatialrelease_fn)
        self.assertFalse(call_nearby)

        # simultaneous masking with spatial unmasking  :
        call['theta'] = 50; echo['theta'] = 60
        call['level'] = 80 ; echo['level'] = 70
        timeres = 10**-4; durn = 0.003;
        call_timesteps = int(durn/timeres)
        call['start'] = 15 ; call['stop'] = call['start']+call_timesteps
        echo['start'] = call['start'] ; echo['stop'] = call['stop']

        angle_diff = abs(call['level'] - echo['level'])

        colloc_echocalldeltadB = get_collocalised_deltadB(0,self.temporalmasking_fn)

        spat_release = calc_spatial_release(angle_diff[0], self.spatialrelease_fn)

        threshold_deltadB_spatrelease = colloc_echocalldeltadB + spat_release


        echocall_deltadB = float(echo['level'] - call['level'])

        heard_ornot = echocall_deltadB > threshold_deltadB_spatrelease


        function_result = check_if_echo_heard(echo,call,self.temporalmasking_fn,
                                                    self.spatialrelease_fn)

        expected_as_calculated = np.all([function_result,heard_ornot])

        self.assertTrue(expected_as_calculated)


        # temporal masking with spatial unmasking:

        call_arrival_delay = 2 # in ms
        delay_timesteps = int((call_arrival_delay*10**-3)/timeres)
        call['start'] = echo['start'] + delay_timesteps
        call['stop'] = echo['stop'] + delay_timesteps
        call['theta'] = 40 ; echo['theta'] = 70
        echo['level'] = 80 ; call['level'] = 107

        angle_diff = calc_angular_separation(call['theta'],echo['theta'])

        # expected deltadB from temporal masking :
        deltadb_tempomasking = get_collocalised_deltadB(call_arrival_delay,
                                                    self.temporalmasking_fn)

        # expected spatial release from spatial unmasking :
        spat_release = calc_spatial_release(angle_diff,self.spatialrelease_fn)

        # threshold deltadB :
        deltadB_threshold = deltadb_tempomasking + spat_release

        # current deltadB :
        deltadB_echocall = float(echo['level'] - call['level'])

        expected_outcome = deltadB_echocall > deltadB_threshold

        #print(deltadB_echocall,deltadB_threshold)

        function_outcome = check_if_echo_heard(echo,call,self.temporalmasking_fn,
                                                    self.spatialrelease_fn)
        self.assertTrue( np.all([expected_outcome, function_outcome]) )

    def test_calc_pechoesheard(self):
        numechoes = [0]*10+[1]*10+[2]*10 + [3]*10

        probs,cumprob = calc_pechoesheard(numechoes,3)

        as_expected = np.all(probs == np.array([0.25]*4))

        self.assertTrue(as_expected)




    def test_get_collocalised_deltadB(self):

        test_timegap = self.temporalmasking_fn['timegap_ms'][3]+0.004

        obtained_deltadB = get_collocalised_deltadB(test_timegap,
                                                    self.temporalmasking_fn)

        self.assertEqual(obtained_deltadB,self.temporalmasking_fn.iloc[3,1])

class TestingNumEchoesHeard(unittest.TestCase):

    def setUp(self):
        print('setting up')
        # temporal masking function -  make everything linear to ease
        # quick calculation of expected results

        timegap_ms = np.arange(10,-3,-1)
        fwd_masking_deltadB = np.linspace(-10,-1,10)
        bkwd_masking_deltadB = np.linspace(0,-2,3)
        deltadB = np.concatenate( (fwd_masking_deltadB,bkwd_masking_deltadB ))
        temp_masking = np.column_stack((timegap_ms,deltadB))
        self.temporalmasking_fn = pd.DataFrame(temp_masking)
        self.temporalmasking_fn.columns = ['timegap_ms','deltadB']

        # spatial release function - make everything linear

        deltatheta = np.linspace(0,25)
        release_dB = np.linspace(0,-25,deltatheta.size)
        self.spatialrelease_fn = pd.DataFrame(index = range(deltatheta.size) )
        self.spatialrelease_fn['deltatheta'] = deltatheta
        self.spatialrelease_fn['dB_release'] = release_dB

        # the calls and echoes objects
        col_names = ['start','stop','theta','level']
        self.calls = pd.DataFrame()
        self.echoes = pd.DataFrame()


    def test_calculate_num_heard_echoes_with_single_call(self):
        #calculate_num_heardechoes(echoes,calls,temporalmasking_fn,spatialrelease_fn)


        self.calls['start'] = [0]; self.calls['stop'] = [10]
        self.echoes['start'] = [20,40]; self.echoes['stop'] = [30,50]
        self.calls['theta'] = [50]; self.echoes['theta'] = [60,90]

        # echo intensities are *very* low in comparison to the

        self.calls['level'] = [90] ; self.echoes['level'] = [0,10]



        num_echoes = calculate_num_heardechoes(self.echoes,self.calls,
                                               self.temporalmasking_fn,
                                               self.spatialrelease_fn)

        self.assertEqual(num_echoes,0)

        # the very obvious case where the self.echoes are equal to and louder than
        # the masking call

        self.echoes['level'] = [90,100]

        num_louder_echoes = calculate_num_heardechoes(self.echoes,self.calls,
                                               self.temporalmasking_fn,
                                               self.spatialrelease_fn)

        self.assertEqual(num_louder_echoes,2)

        # echo intensities are influenced by temporal separation and angle of
        # angle of arrival

        self.echoes['level'] = [70,65]

        expected_outcome = []
        for each_echo in range(2):
            expected_outcome.append(check_if_echo_heard(
                            self.echoes.iloc[each_echo,:], self.calls.iloc[0,:],
                            self.temporalmasking_fn, self.spatialrelease_fn))


        expected_heardechoes = sum(expected_outcome)

        numheardechoes = calculate_num_heardechoes(
                                    self.echoes,self.calls, self.temporalmasking_fn,
                                    self.spatialrelease_fn)



        self.assertEqual(numheardechoes,expected_heardechoes)

    def test_calculate_num_heard_echoes_with_softechoes_and_multicalls(self):

        # all calls overlapping with echoes exactly and echoes very soft

        self.echoes['start'] = [20,40,60]; self.echoes['stop'] = [30,50,70]
        self.calls['start'] = self.echoes['start']
        self.calls['stop'] = self.echoes['stop']
        thetas = [90,100,120];self.calls['theta'] = thetas
        self.echoes['theta'] = thetas
        self.calls['level']=[100,103,102];
        self.echoes['level'] = [0,0,0];

        numheardechoes = calculate_num_heardechoes(
                                    self.echoes,self.calls, self.temporalmasking_fn,
                                    self.spatialrelease_fn)

        self.assertEqual(numheardechoes,0)

        # all calls temporally overlapping exactly and
        # one echo very loud - at 0 deltadB

        self.echoes['level'][0] = self.calls['level'][0]+2

        expect1echo = calculate_num_heardechoes(
                                    self.echoes,self.calls, self.temporalmasking_fn,
                                    self.spatialrelease_fn)

        self.assertEqual(expect1echo,1)

    def test_calculate_num_heard_echoes_singleecho_multicalls(self):
        '''2 calls, one before and one after, with same angle of arrival.
        - both of
        '''
        self.echoes['start'] = [20] ; self.echoes['stop'] = [50]
        # create two calls 3ms b4 and 3ms after the echo edges:
        self.calls['start'] = [-40,80] ; self.calls['stop'] = [-10,110]
        #self.echoes['level'] = [50] ; self.calls['level'] = []




class TestingPopulateSounds(unittest.TestCase):
    '''Checks if the various switches for call directionality
    and spatial arrangement are running correctly at the original function
    and at upper functions like run_one_trial
    '''
    def setUp(self):
        num_sounds = 4
        self.A = 7.3
        self.sound_df = pd.DataFrame()

        self.sound_df['start'] = np.random.random_integers(0,10,num_sounds)
        self.sound_df['stop'] = self.sound_df['start'] + 3
        self.sound_df['level'] = [100,96,90,84]
        self.sound_df['theta'] = [0,90,180,270]

        # temporal masking and spatial unmasking functions :
        timegap_ms = np.arange(10,-3,-1)
        fwd_masking_deltadB = np.linspace(-10,-1,10)
        bkwd_masking_deltadB = np.linspace(0,-2,3)
        deltadB = np.concatenate( (fwd_masking_deltadB,bkwd_masking_deltadB ))
        temp_masking = np.column_stack((timegap_ms,deltadB))
        self.temporalmasking_fn = pd.DataFrame(temp_masking)
        self.temporalmasking_fn.columns = ['timegap_ms','deltadB']

        # spatial release function - make everything linear

        deltatheta = np.linspace(0,25)
        release_dB = np.linspace(0,-25,deltatheta.size)
        self.spatialrelease_fn = pd.DataFrame(index = range(deltatheta.size) )
        self.spatialrelease_fn['deltatheta'] = deltatheta
        self.spatialrelease_fn['dB_release'] = release_dB

    def test_implementcalldirectionality(self):
        '''check fi the call directionality switch works as expected
        '''
        original_levels = np.copy(self.sound_df['level'])
        emsn_angle = np.pi - np.deg2rad(self.sound_df['theta'])
        cd_factor = []

        for angle in emsn_angle:
            cd_factor.append(call_directionality_factor(self.A,angle))

        implement_call_directionality(self.sound_df,self.A)
        expected = original_levels + cd_factor

        self.assertTrue(np.all(expected==self.sound_df['level']))

    def test_populatesounds_withcalldirectionalityswitch(self):
        '''Implement the call directionality switch in the
        test populate sounds
        '''
        common_seednumber = 11
        np.random.seed(common_seednumber)
        timerange = np.arange(200)
        duration = 3
        intensityrange = (90,100)
        arrivalangles = (0,90)
        numsounds = 4

        calldirn = {'A':7.0}

        dirnl_output = populate_sounds(timerange,duration,intensityrange,
                                                    arrivalangles,numsounds,
                                                    with_dirnlcall=calldirn)
        # run populate_sound with the previous parameters without call
        # directionality
        np.random.seed(common_seednumber)

        wodirnl_output = populate_sounds(timerange,duration,intensityrange,
                                             arrivalangles,numsounds)
        # and now implement call directionality
        implement_call_directionality(wodirnl_output,calldirn['A'])

        levels_aresame = wodirnl_output['level'] == dirnl_output['level']
        self.assertTrue(np.all(levels_aresame))

    def test_runonetrialworkswithcalldirectionalityswitch(self):
        '''Run run_one_trial with call directionality and check if expected
        outcome is produced
        '''

        numheard = run_one_trial(2,  self.temporalmasking_fn,
                                 self.spatialrelease_fn,spatial_unmasking=True,
                                 with_dirnlcall={'A':self.A})
        self.assertTrue(numheard<5)















if __name__ == '__main__':

    unittest.main()



