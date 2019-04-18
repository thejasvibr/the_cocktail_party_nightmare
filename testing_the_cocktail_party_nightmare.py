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




class TestingCheckIfEchoHeard(unittest.TestCase)    :
    '''
    '''

    def setUp(self):
        # temporal masking function -  make everything linear to ease
        # quick calculation of expected results

        timegap_ms = np.arange(0.010,-0.003,-0.001)
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

        # dummy spatial release function - with 0 dB release

        self.nomaskingrelease_fn = self.spatialrelease_fn.copy()
        self.nomaskingrelease_fn['dB_release'] = 0

        # the calls and echoes objects
        col_names = ['start','stop','theta','level']
        self.call = pd.DataFrame(index=[0],columns=col_names)
        self.echo = pd.DataFrame(index=[0],columns=col_names)


    def test_echocallveryseparated(self):
        self.call['start'] = 00; self.call['stop'] =10
        self.echo['start'] = 5000000; self.echo['stop'] = 5000100
        self.call['theta'] = 60; self.echo['theta'] = 60
        self.call['level'] = 80 ; self.echo['level'] = 60
        # check the case where the echo and call are *very* far away.

        call_far_away = check_if_echo_heard(self.echo,self.call,self.temporalmasking_fn,
                                            self.spatialrelease_fn)
        self.assertTrue(call_far_away)

    def test_echocallcloseby(self):
        '''No SUm and call is closeby
        '''
        self.echo['start'] = 5000000; self.echo['stop'] = 5000100
        self.call['theta'] = 60; self.echo['theta'] = 60
        self.call['level'] = 80 ; self.echo['level'] = 60
        # not so far away : this will be false

        self.call['start'] = self.echo['start']-20
        self.call['stop'] = self.echo['stop']-20

        call_nearby = check_if_echo_heard(self.echo,self.call,self.temporalmasking_fn,
                                            self.spatialrelease_fn)
        self.assertFalse(call_nearby)

    def test_simultaneousmask_wSUm(self):

        # simultaneous masking with spatial unmasking  :
        timeres = 10**-4; durn = 0.003;
        call_timesteps = int(durn/timeres)

        self.call['theta'] = 50; self.echo['theta'] = 60
        self.call['level'] = 80 ; self.echo['level'] = 70
        self.call['start'] = 15
        self.call['stop'] = self.call['start']+call_timesteps
        self.echo['start'] = self.call['start']
        self.echo['stop'] = self.call['stop']

        angle_diff = abs(self.call['level'] - self.echo['level'])

        colloc_echocalldeltadB = get_collocalised_deltadB(0,self.temporalmasking_fn)

        spat_release = calc_spatial_release(angle_diff[0], self.spatialrelease_fn)

        threshold_deltadB_spatrelease = colloc_echocalldeltadB + spat_release


        echocall_deltadB = float(self.echo['level'] - self.call['level'])

        heard_ornot = echocall_deltadB >= threshold_deltadB_spatrelease


        function_result = check_if_echo_heard(self.echo,self.call,self.temporalmasking_fn,
                                                    self.spatialrelease_fn)

        expected_as_calculated = np.all([function_result,heard_ornot])

        self.assertTrue(expected_as_calculated)

    def test_nonoverlap_wSUm(self):

        # temporal masking with spatial unmasking:
        timeres = 10**-4;
        call_arrival_delay = 5*10**-3 # seconds
        delay_timesteps = int(call_arrival_delay/timeres)

        self.echo['start'] = 100 ; self.echo['stop'] = 129
        self.call['stop'] = self.echo['start'] - delay_timesteps
        self.call['start'] = self.call['stop'] - 29
        self.call['theta'] = 40 ; self.echo['theta'] = 70
        self.echo['level'] = 80 ; self.call['level'] = 107

        angle_diff = calc_angular_separation(self.call['theta'],self.echo['theta'])

        # required deltadB from temporal masking :
        tg = quantify_temporalmasking(self.echo,self.call)

        deltadb_tempomasking = get_collocalised_deltadB(call_arrival_delay,
                                                    self.temporalmasking_fn)

        # drop in deltadB due to spatial release from spatial unmasking :
        spat_release = calc_spatial_release(angle_diff,self.spatialrelease_fn)


        # threshold deltadB :
        deltadB_threshold = deltadb_tempomasking + spat_release

        # current deltadB :
        deltadB_echocall = float(self.echo['level'] - self.call['level'])

        expected_outcome = deltadB_echocall > deltadB_threshold

        function_outcome = check_if_echo_heard(self.echo,self.call,self.temporalmasking_fn,
                                                    self.spatialrelease_fn)

        exp_and_outcome_True = np.all([expected_outcome, function_outcome])
        self.assertTrue( exp_and_outcome_True )




class TestingNumEchoesHeard(unittest.TestCase):

    def setUp(self):

        # temporal masking function -  make everything linear to ease
        # quick calculation of expected results

        timegap_ms = np.arange(0.010,-0.003,-0.001)
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

        # dummy spatial release function - with 0 dB release

        self.nomaskingrelease_fn = self.spatialrelease_fn.copy()
        self.nomaskingrelease_fn['dB_release'] = 0

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

    def test_withandwithoutspatialunmasking(self):
        '''Echo-call with and without spatial unmasking included.
        '''

        # with no difference in angle of arrival :
        self.calls['start'] = [0]; self.calls['stop'] = [29]
        self.echoes['start'] = [60] ; self.echoes['stop'] = [89]

        self.calls['theta'] = [80] ; self.echoes['theta'] = [80]

        self.calls['level'] = [90] ; self.echoes['level'] = [86]

        noechoes = calculate_num_heardechoes(self.echoes, self.calls,
                                             self.temporalmasking_fn,
                                             self.spatialrelease_fn)

        self.assertEqual(noechoes,0)

        # with angular difference but dummy spatial unmasking function :

        self.echoes['theta'] = 90

        noechoes_dummyfn = calculate_num_heardechoes(self.echoes, self.calls,
                                             self.temporalmasking_fn,
                                             self.nomaskingrelease_fn)

        self.assertEqual(noechoes_dummyfn,0)

        # with angular difference and proper spatial unmasking function :

        oneecho_heard = calculate_num_heardechoes(self.echoes, self.calls,
                                             self.temporalmasking_fn,
                                             self.spatialrelease_fn)

        self.assertEqual(oneecho_heard,1)

    def test_withandwithoutspatialunmasking_multiechoes(self):
        self.calls['start'] = [0]  ; self.calls['stop'] = [29]
        self.calls['theta'] = [80] ; self.calls['level'] = [90]

        # with multiple echoes - same angle of arrival :

        self.echoes['start'] = [60, 100] ; self.echoes['stop'] = [89,129]
        self.echoes['theta'] = [80,80]   ; self.echoes['level'] = [86,82 ]

        bothechoes_notheard = calculate_num_heardechoes(self.echoes,self.calls,
                                                self.temporalmasking_fn,
                                                self.spatialrelease_fn)
        self.assertEqual(bothechoes_notheard,0)

        # with one echo having different angle of arrival :
        self.echoes['theta'] = [80, 90]
        oneecho_heard = calculate_num_heardechoes(self.echoes,self.calls,
                                                self.temporalmasking_fn,
                                                self.spatialrelease_fn)
        self.assertEqual(oneecho_heard,1)

        # with one echo having different angle of arrival + dummy spatial unmasking

        noneheard_dummyfn = calculate_num_heardechoes(self.echoes,self.calls,
                                                self.temporalmasking_fn,
                                                self.nomaskingrelease_fn)
        self.assertEqual(noneheard_dummyfn,0)
    
    def test_onehotcoding_switch_positive(self):
        '''
        Checks if the one-hot coding option activated by the keyword argument
        gives the expected output when all echoes are heard
        '''
        # copy the test with spatial unmasking multiechoes
        self.calls['start'] = [0]  ; self.calls['stop'] = [29]
        self.calls['theta'] = [80] ; self.calls['level'] = [90]

        # with multiple echoes - same angle of arrival :

        self.echoes['start'] = [1500, 10000] ; self.echoes['stop'] = [1600,10600]
        self.echoes['theta'] = [0, 0]   ; self.echoes['level'] = [86,86]


        oneecho_heard, obtained_outcome = calculate_num_heardechoes(self.echoes,self.calls,
                                                self.temporalmasking_fn,
                                                self.spatialrelease_fn,
                                                one_hot=True)
        self.assertEqual(oneecho_heard,2)
        expected_outcome = np.array([1,1])
        outcomes_match_positive = np.sum(obtained_outcome - expected_outcome) == 0
        self.assertTrue(outcomes_match_positive)
    
    def test_onehotcoding_switch_negative(self):
        '''
        Checks if the one-hot coding option activated by the keyword argument
        gives the expected output when no echoes are heard
        '''
        # copy the test with spatial unmasking multiechoes
        self.calls['start'] = [0]  ; self.calls['stop'] = [29]
        self.calls['theta'] = [80] ; self.calls['level'] = [90]

        # with multiple echoes - same angle of arrival :

        self.echoes['start'] = [0, 15] ; self.echoes['stop'] = [10,20]
        self.echoes['theta'] = [80, 80]   ; self.echoes['level'] = [5,2]


        noecho_heard, obtained_outcome = calculate_num_heardechoes(self.echoes,self.calls,
                                                self.temporalmasking_fn,
                                                self.spatialrelease_fn,
                                                one_hot=True)
        self.assertEqual(noecho_heard,0)
        expected_outcome = np.array([0,0])
        outcomes_match_negative = np.sum(obtained_outcome - expected_outcome) == 0
        self.assertTrue(outcomes_match_negative)
    
    def test_onehotcoding_switch_secondechoheard(self):
        '''
        Checks if the one-hot coding option activated by the keyword argument
        gives the expected output when the second of two echoes is heard
        '''
        # copy the test with spatial unmasking multiechoes
        self.calls['start'] = [0]  ; self.calls['stop'] = [29]
        self.calls['theta'] = [80] ; self.calls['level'] = [90]

        # with multiple echoes - same angle of arrival :

        self.echoes['start'] = [0, 10000] ; self.echoes['stop'] = [16,10600]
        self.echoes['theta'] = [0, 0]   ; self.echoes['level'] = [1,86]


        oneecho_heard, obtained_outcome = calculate_num_heardechoes(self.echoes,self.calls,
                                                self.temporalmasking_fn,
                                                self.spatialrelease_fn,
                                                one_hot=True)
        self.assertEqual(oneecho_heard,1)
        expected_outcome = np.array([0,1])
        outcomes_match_oneheard = np.sum(obtained_outcome - expected_outcome) == 0
        self.assertTrue(outcomes_match_oneheard)
    
        





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

    def test_populatesoundswithpoissondisk(self):
        '''Spread the emitting bats uniformly according to the poisson disk
        sampling and check that the correct number of points are produced
        '''

        common_seednumber = 11
        np.random.seed(common_seednumber)
        timerange = np.arange(200)
        duration = 3
        intensityrange = (90,100)
        arrivalangles = (0,90)
        numsounds = 100


        sourcelevel = {'ref_distance':0.1,'intensity':120}
        nbr_dist = 0.6
        poisdisk = {'source_level':sourcelevel, 'min_nbrdist': nbr_dist }

        received_sounds = populate_sounds(timerange,duration,intensityrange,
                                             arrivalangles,numsounds,
                                             poisson_disk = poisdisk)

        num_rows,num_cols = received_sounds.shape
        self.assertEqual(num_rows, numsounds)
        self.assertEqual(num_cols, 4)

        # Also just check that the populate sounds with poisson disk accepts
        # None for the other arguments

        received_sounds_nonecheck = populate_sounds(timerange,duration,
                                                    None,
                                             None,numsounds,
                                             poisson_disk = poisdisk)

        num_rowsnonecheck, num_colsnonecheck = received_sounds_nonecheck.shape

        self.assertEqual(num_rowsnonecheck, numsounds)
        self.assertEqual(num_colsnonecheck, 4)



class TestingRunMultipleTrials(unittest.TestCase):

    def setUp(self):
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

        self.calldensities = [1,5,10]

        A_param = 7
        self.calldirectionality = {'A':A_param}
        self.numtrials = 1

    def test_runmultipletrials_onerun(self):

        echoesheard = run_multiple_trials(2, [10], self.temporalmasking_fn,
                        self.spatialrelease_fn,
                        spatial_unmasking=True,
                        echo_level_range=(90,100))
        #print(echoesheard)

    def test_catchinappropriatesizedfunctions(self):
        '''If a temporal masking  or spatial unmasking function
        with != 2 columns - then raise error.
        '''

        self.temporalmasking_fn['nonsensecolumn'] = 0


        with self.assertRaises(IndexError):
            run_multiple_trials(2, [10], self.temporalmasking_fn,
                                          self.spatialrelease_fn,
                                          spatial_unmasking=True,
                                          echo_level_range=(90,100))

    def test_runmultipletrials_onerunwithpoissondisksampling(self):
        '''
        '''
        sourcelevel = {'ref_distance':0.1,'intensity':120}
        nbr_dist = 0.6
        poisdisk_params = {'source_level':sourcelevel, 'min_nbrdist': nbr_dist}

        seed_number = 203
        np.random.seed(seed_number)
        multip_result = run_multiple_trials(2, [10], self.temporalmasking_fn,
                                          self.spatialrelease_fn,
                                          spatial_unmasking=True,
                                          echo_level_range=(90,100),
                                            poisson_disk=poisdisk_params)
        print('multip_result',multip_result)

    def test_runmultipletrials_onehot(self):
        '''Run the simulations with the onehot switch activated 
        '''
        sourcelevel = {'ref_distance':0.1,'intensity':120}
        nbr_dist = 0.6
        poisdisk_params = {'source_level':sourcelevel, 'min_nbrdist': nbr_dist}

        seed_number = 203
        np.random.seed(seed_number)
        multip_result = run_multiple_trials(5, [10], self.temporalmasking_fn,
                                          self.spatialrelease_fn,
                                          spatial_unmasking=True,
                                          echo_level_range=(90,100),
                                            poisson_disk=poisdisk_params,
                                            one_hot=True)
        num_echoes_heard = multip_result[0]
        echoes_id = multip_result[1]
        numechoes_heard_fromid = np.apply_along_axis(np.sum, 2, echoes_id)
        num_heardechoes_match = np.allclose(numechoes_heard_fromid, num_echoes_heard)
        self.assertTrue(num_heardechoes_match)

    def test_runmultipletrials_onehot_multicalldensities(self):
        ''' Run onehot switch at multiple call densities
        '''
        sourcelevel = {'ref_distance':0.1,'intensity':120}
        nbr_dist = 0.6
        poisdisk_params = {'source_level':sourcelevel, 'min_nbrdist': nbr_dist}

        seed_number = 203
        np.random.seed(seed_number)
        multip_result = run_multiple_trials(5, [1,10,20], self.temporalmasking_fn,
                                          self.spatialrelease_fn,
                                          spatial_unmasking=True,
                                          echo_level_range=(90,100),
                                            poisson_disk=poisdisk_params,
                                            one_hot=True)
        num_echoes_heard = multip_result[0]
        echoes_id = multip_result[1]
        num_densities, num_trials, num_echoes = echoes_id.shape
        
        sum_echoes_heard = np.zeros(num_echoes_heard.shape)
        for i, each_density in enumerate(range(num_densities)):
            for j, each_trial in enumerate(range(num_trials)):
                sum_echoes_heard[i,j] = np.sum(echoes_id[i,j,:])
        self.assertTrue(np.allclose(sum_echoes_heard, num_echoes_heard))
        
 
class TestingCalcNumTimes(unittest.TestCase):

    def setUp(self):
        self.p = 0.5
        self.ntrials = 10

    def test_simplecase(self):
        '''Check if 5 trials has the greatest probability
        when p = 0.5 in 10 trials
        '''
        prob_occrnce = calc_num_times(self.ntrials,self.p)
        maxvalue_row = np.argmax(prob_occrnce['probability'])

        max_times = prob_occrnce['num_times'][maxvalue_row]
        expected_times = 5

        self.assertEqual(expected_times, max_times)

    def test_checkallprobssumto1(self):
        '''Add up all probabilities and check that they add upto one.
        '''
        prob_occrnce = calc_num_times(self.ntrials,self.p)

        total_prob = np.sum(prob_occrnce['probability'])

        equalto1 = np.isclose(total_prob,1)
        self.assertTrue(equalto1)

    def test_catchincorrectpvalues(self):
        '''Input p values that are <0 and 1>.
        '''

        self.p = 1.5

        self.assertRaises(ValueError,
                                  lambda : calc_num_times(self.ntrials,self.p))

        self.p = -0.2

        self.assertRaises(ValueError,
                                  lambda : calc_num_times(self.ntrials,self.p))

    def test_catchincorrectNtrials(self):
        '''Input incorrect Ntrials that is <=0
        '''
        self.ntrials = -10
        self.assertRaises(ValueError,
                                  lambda : calc_num_times(self.ntrials,self.p))

        self.ntrials = 0
        self.assertRaises(ValueError,
                                  lambda : calc_num_times(self.ntrials,self.p))

    def test_catchincorrectNtrialstype(self):
        '''Make sure Ntrials is an integer-like entry
        '''

        self.ntrials = 3.4
        self.assertRaises(TypeError,
                                  lambda : calc_num_times(self.ntrials,self.p))

        self.ntrials = np.array([64.0],dtype='float16')
        self.assertRaises(TypeError,
                                  lambda : calc_num_times(self.ntrials,self.p))


class TestingSpatialArrangementHexagonal(unittest.TestCase):
    '''Testing the component functions that implement spatial arrangement
    in the form of a hexagonal array
    '''

    def test_filluphexagonalrings(self):
        nbats = 5
        expected_rings = 1

        ringnums,numbats = fillup_hexagonalrings(nbats)

        self.assertEqual(numbats,nbats)
        self.assertEqual(expected_rings,ringnums.size)

        nbats = 29

        ringnums,numbats = fillup_hexagonalrings(nbats)

        expected_numbats = np.array([6,12,11])
        expected_numrings = np.array([1,2,3])

        values_equal = expected_numbats == numbats
        rings_equal = expected_numrings == ringnums


        self.assertTrue(np.all(values_equal))
        self.assertTrue(np.all(rings_equal))

        self.assertRaises(ValueError,lambda : fillup_hexagonalrings(-1))

    def test_calcRL(self):
        '''
        '''
        SL = 120
        ref_dist = 0.1
        dist = 1.0


        rl_calculated = calc_RL(dist, SL, ref_dist)
        rl_expected = 100.0

        self.assertEqual(rl_expected, rl_calculated)

        self.assertRaises(ValueError,
                          lambda : calc_RL(-0.1, SL, ref_dist))

        self.assertRaises(ValueError,
                          lambda : calc_RL(0.1, SL, -ref_dist))

        self.assertRaises(ValueError,
                          lambda : calc_RL(-0.1, SL, -ref_dist))




    def test_calculate_receivedlevels(self):
        '''
        '''
        source_level = {'intensity':120,'ref_distance':0.1}

        rls = calculate_receivedlevels(1.0,source_level)

        self.assertEqual(rls,100.0)

        distances = np.array([1.0,2.0,0.5])

        rls_distances = calculate_receivedlevels(distances,source_level)

        expected_distances = np.apply_along_axis(calc_RL,0,
                                            distances.reshape(1,-1),
                                            source_level['intensity'],
                                            source_level['ref_distance']
                                            )
        values_equal = np.array_equal(expected_distances, rls_distances)

        self.assertTrue(np.all(values_equal))


    def test_implement_hexagonal_spatial_arrangement(self):
        '''
        '''
        bat_source_level = {'intensity':100, 'ref_distance':1.0}
        nbr_distance = 0.5

        self.assertRaises(ValueError,
                          lambda: implement_hexagonal_spatial_arrangement(-1,
                                                            nbr_distance,
                                                            bat_source_level) )

        self.assertRaises(ValueError,
                          lambda: implement_hexagonal_spatial_arrangement(0,
                                                            nbr_distance,
                                                            bat_source_level) )

        numbats = 5
        received_levels, num_calls = implement_hexagonal_spatial_arrangement(
                                                            numbats,
                                                            nbr_distance,
                                                            bat_source_level)

        exp_levels = calc_RL(nbr_distance,
                             bat_source_level['intensity'],
                            bat_source_level['ref_distance']  ).reshape(1,-1)

        levels_equal = np.array_equal(exp_levels, received_levels)
        numcalls_equal = np.array_equal( np.array(numbats),
                                        num_calls)
        self.assertTrue(levels_equal)
        self.assertTrue(numcalls_equal)


        manybats = 25
        manyreceived_levels, many_calls =implement_hexagonal_spatial_arrangement(
                                                            manybats,
                                                            nbr_distance,
                                                            bat_source_level)

        num_rings, numinrings = fillup_hexagonalrings(manybats)
        distances = num_rings*nbr_distance
        received_levelscalc = calculate_receivedlevels(distances,
                                                       bat_source_level)

        multibatlevels_equal = np.array_equal(received_levelscalc,
                                                          manyreceived_levels)
        multibats_numsequal = np.array_equal(numinrings, many_calls)

        self.assertTrue(multibatlevels_equal)
        self.assertTrue(multibats_numsequal)



class TestingSpatialArrangementPoissondisk(unittest.TestCase):
    '''
    '''

    def test_findclosestpoint(self):

        test_points  = np.array([[0,5],[0,1],[0,2]])
        tgt_point = np.array([0,0])

        calc_point = find_closestpoint(test_points, tgt_point)

        closest_point = test_points[1,:]
        calcandtgtsame =  np.array_equal(calc_point, closest_point)

        self.assertTrue(calcandtgtsame)

    def test_findnearbypoints(self):
        '''Set up a set of points in a horizontal line and choose point 4
        as the focal point. Select 4 points around the focal point and check
        results
        '''

        pts_x = np.arange(0,10)
        pts_y = np.tile(0,10)
        allpts = np.column_stack((pts_x,pts_y))

        focalpt_indx = 3
        numnearbypts = 4
        nearbypts = find_nearbypoints(allpts, focalpt_indx, numnearbypts)

        expected_xs = np.array([2,4,1,5])
        expected_ys = np.tile(0,expected_xs.size)
        expected_pts = np.column_stack((expected_xs,expected_ys))

        outputexpected_match = np.array_equal(expected_pts,nearbypts)
        self.assertTrue(outputexpected_match)

        self.assertRaises(ValueError, lambda : find_nearbypoints(allpts,
                                                    focalpt_indx,
                                                    allpts.shape[0]+20 )
                                                    )

        self.assertRaises(IndexError, lambda : find_nearbypoints(allpts,
                                                    focalpt_indx-50,
                                                    numnearbypts )
                                                    )

    def test_choosecentremostpoint(self):
        '''

        '''

        pts_x = np.array([1,1,-1,-1,0.25])
        pts_y = np.array([1,-1,1,-1,0.3])
        all_points = np.column_stack((pts_x,pts_y))
        expected_point = all_points[-1,:]

        centremost_pt = choose_centremostpoint(all_points)

        centremostpt_match = np.array_equal(centremost_pt,expected_point)

        self.assertTrue(centremostpt_match)

    def test_calccentroid(self):
        pts_x = np.arange(-10,11)
        pts_y = np.arange(-10,11)
        allpts = np.column_stack((pts_x,pts_y))

        self.assertRaises(ValueError,lambda : calc_centroid(pts_x))

        threecolumn_xyz = np.column_stack((allpts,pts_x))
        self.assertRaises(ValueError,lambda : calc_centroid(threecolumn_xyz))

        centroid_pt = calc_centroid(allpts)
        expected_pt = np.apply_along_axis(np.mean,0,allpts)

        centroidpt_match = np.array_equal(expected_pt, centroid_pt)

        self.assertTrue(centroidpt_match)

    def test_findingrowindex(self):
        pts_x = np.arange(0,11)
        pts_y = np.arange(0,11)
        allpts = np.column_stack((pts_x,pts_y))

        target_indx = 3
        target_array = allpts[target_indx,:]

        output_index = find_rowindex(allpts,target_array)

        self.assertEqual(target_indx, output_index)


    def test_generatesurroundpointspoissondisksampling(self):
        '''Test if correct number of points are being generated.
        '''

        nbr_dist = 0.5
        numpoints = np.arange(1,50,5)

        num_genpoints = np.zeros(numpoints.size)
        for index, npoints in enumerate(numpoints):
            gen_pts,centrepts = generate_surroundpoints_w_poissondisksampling(
                                                        npoints, nbr_dist)
            numrows,numcols = gen_pts.shape
            num_genpoints[index] = numrows


        proper_numpointsgenerated = np.array_equal(num_genpoints, numpoints)

        self.assertTrue(proper_numpointsgenerated)

    def test_calculate_angleofarrival(self):
        '''
        '''

        source_xy = np.array([1,1])
        focal_xy = np.array([0,0])

        expected_angle = 45.0

        angle_arrival = calculate_angleofarrival(source_xy, focal_xy)

        self.assertEqual(angle_arrival, expected_angle)


        multiple_xys = np.array(([1,1],[0,0],[0,1],[-1,0],[-1,-1],[0,-1],
                                                                     [-1,0]))
        expected_angles = np.array([45, 0, 0, -90, -135, 180, -90])

        angles_arrival = np.apply_along_axis(calculate_angleofarrival,
                                                     1, multiple_xys, focal_xy)

        angles_match = np.array_equal(angles_arrival, expected_angles)
        self.assertTrue(angles_match)



    def test_calcrundtheta(self):

        multiple_xys = np.array(([1,1],[0,0],[0,1],[-1,0],[-1,-1],[0,-1],
                                                                     [-1,0]))
        focal_point = np.array([0,0])

        expected_angles = np.array([45, 0, 0, -90, -135, 180, -90])
        expected_distances = np.apply_along_axis(spl.distance.euclidean,1,
                                                         multiple_xys,[0,0]  )

        radialdists, arrivalthetas = calculate_r_theta(multiple_xys,
                                                                   focal_point)

        distances_match = np.array_equal(radialdists, expected_distances)
        angles_match = np.array_equal(arrivalthetas, expected_angles)

        self.assertTrue(distances_match)
        self.assertTrue(angles_match)



    def test_implementpoissondiskspatialarrangement(self):
        '''Only do testing of the output pd.DataFrame dimensions
        '''

        nbats = 40
        min_dist = 0.5
        source_level = {'intensity':100,'ref_distance':1.0}

        thetas, intensities = implement_poissondisk_spatial_arrangement(nbats,
                                                                min_dist,
                                                                 source_level)

        self.assertEqual(thetas.size,nbats)
        self.assertEqual(intensities.size,nbats)


# TODO:

class TestingConspecificcallPaths(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.kwargs = {}
        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])
    
    def calc_distmat(self):
        self.distance_matrix = spl.distance_matrix(self.kwargs['bats_xy'], self.kwargs['bats_xy'])

        

    def test_basicpaths(self):
        '''3 bat case that are in a triangular formation
        '''
        consp_paths = calculate_conspecificcall_paths(**self.kwargs)
        output_paths = pd.DataFrame(consp_paths)
        self.calc_distmat()
        
        expected_distances = np.array([self.distance_matrix[0,1], self.distance_matrix[0,2]])
        expected_theta_reception = np.array([90, 45])
        
        self.assertTrue(np.array_equal(expected_distances, output_paths['R_incoming']))
        self.assertTrue(np.array_equal(expected_theta_reception, output_paths['theta_reception']))
        
    def test_inaline(self):
        '''3 bats in a line heading in the same direction
        '''
        self.kwargs['bats_xy'] = np.array(([1,1],[1,2],[1,3]))
        self.kwargs['bats_orientations'] = np.tile(90,3)
        self.calc_distmat()


        consp_paths = calculate_conspecificcall_paths(**self.kwargs)
        output_paths = pd.DataFrame(consp_paths)

        expected_distances = np.array([self.distance_matrix[0,1], self.distance_matrix[0,2]])
        expected_theta_reception = np.array([0, 0])
        
        self.assertTrue(np.array_equal(expected_distances, output_paths['R_incoming']))
        self.assertTrue(np.array_equal(expected_theta_reception, output_paths['theta_reception']))

class Testing2ndaryEchoPaths(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.kwargs = {}
        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])

    def calc_distmat(self):
        self.distance_matrix = spl.distance_matrix(self.kwargs['bats_xy'], self.kwargs['bats_xy'])


    def test_check_correctnumechoes(self):
        '''check if number of secondary echoes is correctly produced;
        '''
        for num_bats in [3, 10, 50]:
        
            self.kwargs['bats_xy'] = np.random.normal(0,1,num_bats*2).reshape(-1,2)
            self.kwargs['bats_orientations'] = np.random.normal(0,90,num_bats)
            self.kwargs['focal_bat'] = self.kwargs['bats_xy'][num_bats-1]
    
            secondechoes_paths = calculate_2ndary_echopaths(**self.kwargs)
            expected_number = (num_bats-1)*(num_bats-2)

            self.assertEqual(expected_number, len(secondechoes_paths['sound_routes']))
        

    def test_basic_2ndaryechoes(self):
        ''' 3 bats in a triangle confuguration
        '''
        secondechoes_paths = calculate_2ndary_echopaths(**self.kwargs)
        output_paths = pd.DataFrame(secondechoes_paths)

        expected_paths = pd.DataFrame(data=[], index=range(2),
                                      columns=output_paths.columns)
        expected_paths['R_incoming'] = [1.0,1.0]
        expected_paths['R_outgoing'] = [np.sqrt(2), 1.0]
        expected_paths['theta_emission'] = [-45.0, -135.0]
        expected_paths['theta_reception'] = [45.0, 90.0]
        expected_paths['sound_routes'] = [(1,2,0),(2,1,0)]
        expected_paths['theta_incoming'] = [-135.0,-45.0]
        expected_paths['theta_outgoing'] = [-90.0,-135.0]
        
        self.assertTrue(expected_paths.equals(output_paths))

    def test_basic_diamond(self):
        '''Test a configuration of 4 bats placed in a diamond configuration 
        around a central bat
        
        This test independently checks if the geometrical parameters are being 
        calculated as expected !
        '''
        
        self.kwargs['bats_xy'] = np.array(([0,0],
                                           [0,1],
                                           [0,-1],
                                           [1,0],
                                           [-1,0]))
        self.kwargs['bats_orientations'] = np.tile(90,5)

        self.kwargs['focal_bat'] = self.kwargs['bats_xy'][0,:]

        self.calc_distmat()

        self.output = calculate_2ndary_echopaths(**self.kwargs)
        exp_R_in = []
        exp_R_out = []
        exp_theta_em = []
        exp_theta_recpn = []
        exp_theta_in = []
        exp_theta_out = []

        for echo_id, each_echo in enumerate(output['sound_routes']):
            emitter, target, focal = each_echo
            
            exp_R_in.append(self.distance_matrix[emitter, target])
            exp_R_out.append(self.distance_matrix[target, focal])

            
            exp_theta_recpn.append(calculate_angleofarrival(self.kwargs['bats_xy'][target],
                                         self.kwargs['bats_xy'][focal],
                                         self.kwargs['bats_orientations'][focal]))

            exp_theta_em.append(calculate_angleofarrival(self.kwargs['bats_xy'][target],
                                                            self.kwargs['bats_xy'][emitter],
                                                            self.kwargs['bats_orientations'][emitter]))

            exp_theta_in.append(calculate_angleofarrival(self.kwargs['bats_xy'][emitter],
                                                            self.kwargs['bats_xy'][target],
                                                            self.kwargs['bats_orientations'][target]))

            exp_theta_out.append(calculate_angleofarrival(self.kwargs['bats_xy'][focal],
                                                            self.kwargs['bats_xy'][target],
                                                            self.kwargs['bats_orientations'][target]))



        # check if the calculated values match up:
        for key, expected in zip(['R_incoming', 'R_outgoing', 'theta_emission', 'theta_reception',
                                  'theta_incoming','theta_outgoing'],
                                [exp_R_in, exp_R_out, exp_theta_em, exp_theta_recpn,
                                 exp_theta_in, exp_theta_out]):
            self.assertEqual(self.output[key], tuple(expected))

class TestingSecondaryEchoReceivedLevels(unittest.TestCase):
    '''Check if the received levels are correct.
    '''
    def setUp(self):
        self.kwargs = {}

        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])
        self.kwargs['reflection_function'] = pd.DataFrame(data=[], index=range(360*360),
                                                          columns=['ref_distance',
                                                                   'incoming_theta',
                                                                   'outgoing_theta',
                                                                   'reflection_strength'])
        self.kwargs['reflection_function']['reflection_strength'] = np.tile(-10, 360*360)
        all_angles = np.array(np.meshgrid(range(-180,180), range(-180,180))).T.reshape(-1,2)
        
        self.kwargs['reflection_function']['incoming_theta'] = all_angles[:,0]
        self.kwargs['reflection_function']['outgoing_theta'] = all_angles[:,1]
        self.kwargs['reflection_function']['ref_distance'] = np.tile(0.1, 360*360)
        
        self.kwargs['call_directionality'] = lambda X : 0 
        self.kwargs['hearing_directionality'] = lambda X : 0
        self.kwargs['source_level'] = {'ref_distance' : 0.1, 'dBSPL':120}
        
    def test_basic(self):
        '''
        '''
        secondary_echoes = calculate_2ndaryecho_levels(**self.kwargs)
        output_received_levels = np.array(secondary_echoes['level']).flatten()

        # calculate the expected received levels of the 2dary echoes
        paths = calculate_2ndary_echopaths(**self.kwargs)
        ref_dist = np.unique(self.kwargs['reflection_function']['ref_distance'])

        reflection_strength = np.unique(self.kwargs['reflection_function']['reflection_strength'])

        expected_received_levels = []
        for echonum, route in enumerate(paths['sound_routes']):
            r_in = paths['R_incoming'][echonum]
            r_out = paths['R_outgoing'][echonum]
            incoming_SPL = calc_RL(r_in-ref_dist,
                                   self.kwargs['source_level']['dBSPL'],
                                   self.kwargs['source_level']['ref_distance'])

            post_reflection_SPL = incoming_SPL + reflection_strength
            rec_levels = calc_RL(r_out, post_reflection_SPL, ref_dist )
            expected_received_levels.append(rec_levels)

        print(expected_received_levels, output_received_levels)
        self.assertTrue(np.array_equal(np.array(expected_received_levels).flatten(),
                                       output_received_levels))
 
class TestCalculateConspecificcall_levels(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.kwargs = {}
        
        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])
        self.kwargs['call_directionality'] = lambda X : 0 
        self.kwargs['hearing_directionality'] = lambda X : 0
        self.kwargs['source_level'] = {'ref_distance' : 0.1, 'dBSPL':120}


    def calc_distmat(self):
        self.distance_matrix = spl.distance_matrix(self.kwargs['bats_xy'], self.kwargs['bats_xy'])

    def test_basic(self):
        '''test 3 bat situation 
        '''
        conspecific_calls = calculate_conspecificcall_levels(**self.kwargs)
        print(conspecific_calls)
        self.calc_distmat()
        
        output_receivedlevels = np.array(conspecific_calls['level']).flatten()

        expected_receivedlevels = np.zeros(2)
        expected_receivedlevels[0] = calc_RL(self.distance_matrix[0,1], self.kwargs['source_level']['dBSPL'],
                                    self.kwargs['source_level']['ref_distance'])
        
        expected_receivedlevels[1] = calc_RL(self.distance_matrix[0,2], self.kwargs['source_level']['dBSPL'],
                                    self.kwargs['source_level']['ref_distance'])
        
        self.assertTrue(np.array_equal(output_receivedlevels, expected_receivedlevels))
        

class TestPropagateSound(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.kwargs = {}
        
        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])
        self.kwargs['call_directionality'] = lambda X : 0 
        self.kwargs['hearing_directionality'] = lambda X : 0
        self.kwargs['source_level'] = {'ref_distance' : 0.1, 'dBSPL':120}

        self.kwargs['reflection_function'] = pd.DataFrame(data=[], index=range(36*36),
                                                          columns=['ref_distance',
                                                                   'incoming_theta',
                                                                   'outgoing_theta',
                                                                   'reflection_strength'])
        self.kwargs['reflection_function']['reflection_strength'] = np.tile(-10, 36*36)
        all_angles = np.array(np.meshgrid(np.linspace(-180,180,36),
                                          np.linspace(-180,180,36))).T.reshape(-1,2)
        
        self.kwargs['reflection_function']['incoming_theta'] = all_angles[:,0]
        self.kwargs['reflection_function']['outgoing_theta'] = all_angles[:,1]
        self.kwargs['reflection_function']['ref_distance'] = np.tile(0.1, 36*36)

    def test_numberofsounds(self):
        
        # calculate the received levels and angles of arrivals of the sounds
        conspecific_calls = propagate_sounds('conspecific_calls', **self.kwargs)
        secondary_echoes = propagate_sounds('2ndary_echoes', **self.kwargs)

        num_conspecificcalls, num_2daryechoes = conspecific_calls.shape[0], secondary_echoes.shape[0]
        
        nbats = self.kwargs['bats_xy'].shape[0]
        exp_numcalls = nbats-1
        exp_num2daryechoes = (nbats-2)*(nbats-1)

        self.assertTrue(np.array_equal([num_conspecificcalls, num_2daryechoes],
                                       [exp_numcalls, exp_num2daryechoes]))


if __name__ == '__main__':

    unittest.main()



