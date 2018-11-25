# -*- coding: utf-8 -*-
"""Tests for calc_num_IPIs
Created on Tue Nov 20 17:25:58 2018

@author: tbeleyur
"""
import collections
import unittest
import numpy as np
from calc_num_IPIs import *

class TestingCountNumrows(unittest.TestCase):
    
    def setUp(self):
        self.echoes_heard = np.array(([1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]))
        
    
    def test_basic_case(self):
        ''' where one unique echo is heard per trial
        '''
        num_trials = count_numrows_to_m_echoes(self.echoes_heard, 4)
        self.assertEqual(num_trials, 4)

    def test_all_zeros(self):
        '''Where there are *no* echoes at all that are heard
        '''
        self.echoes_heard[:,:] = 0
        num_trials = count_numrows_to_m_echoes(self.echoes_heard, 4)
        self.assertTrue(num_trials is None)

    def test_2echo_hearing(self):
        '''where we count the number of trials to hear the first two echoes
        '''
        num_trials = count_numrows_to_m_echoes(self.echoes_heard, 2)
        self.assertEqual(num_trials, 2)
        
    def test_longer_case(self):
        '''where all echoes are present but in the middle  of many empty rows.
        '''
        total_trials = 100
        self.long_case = np.zeros((total_trials,4))
        
        for i, rownum in enumerate([0,24,49,99]):
            self.long_case[rownum,:] = self.echoes_heard[i,:]

        num_trials = count_numrows_to_m_echoes(self.long_case, 4)
        self.assertEqual(num_trials,total_trials)        

    def test_seeing_all_echoes_twice(self):
        '''Test switch to count num trials req to see all 
        echoes *N* times
        '''
        all_echoes_2x = np.row_stack((self.echoes_heard, self.echoes_heard))
        num_times_allechoes = 2 
        numtrials = count_numrows_to_m_echoes(all_echoes_2x, 4,
                                              numtimes_allechoes=2)
        self.assertEqual(numtrials, 8)

    def test_seeing_allechoes_twice_empty(self):
        '''testing if the numtimes switch still works when a zeros array is
        given
        '''
        self.echoes_heard[:,:] = 0
        numtrials = count_numrows_to_m_echoes(self.echoes_heard, 4,
                                              numtimes_allechoes=2)
        self.assertEqual(numtrials, None)
        
        

class TestingCalcNumIpis(unittest.TestCase):

    def setUp(self):
        self.echoesheard_case1 = np.array(([1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]))
        
        self.echoesheard_long = np.zeros((100,4))
        for i, rownum in enumerate([0, 24, 49, 99]):
            self.echoesheard_long[rownum, :] = self.echoesheard_case1[i,:]
    
    def test_basic(self):
        '''check if calc_num_IPIs...gets case1 right
        '''
        trials_to_hear2 = calc_num_IPIs_to_hear_echoes(self.echoesheard_case1, 2, 4)
        expected_results_obtained = np.array_equal(np.array(trials_to_hear2),
                                        np.array([2,2]))
        self.assertTrue(expected_results_obtained)

    def test_noecho(self):
        '''
        '''
        noechoesheard = np.zeros((10,4))
        trials_to_hearnoechoes = calc_num_IPIs_to_hear_echoes(noechoesheard, 2, 4)
        self.assertTrue(trials_to_hearnoechoes[0] is None)

    def test_longcase(self):
        '''
        '''
        num_trials_to_hear = []
        for num_echoes in range(1,5):
            n_trials = calc_num_IPIs_to_hear_echoes(self.echoesheard_long,
                                                    num_echoes, 4)
            num_trials_to_hear.append(n_trials)
        
        expected_results = {1 : [1,24,25,50],
                            2 : [25, 75],
                            3 : [50, None],
                            4 : [100]} 

        # now compare the obtained and expected results 
        for i, obtained_results in enumerate(num_trials_to_hear):
            same_numtrials = self.compare_lists(obtained_results,
                                           expected_results[i+1])
            self.assertTrue(same_numtrials)

    def test_longcase_w_allechoes2x(self):
        ''' check longcase with the numtimes switch set to 2
        '''
        longcase_2x = np.row_stack((self.echoesheard_long, self.echoesheard_long))

        n_trials = calc_num_IPIs_to_hear_echoes(longcase_2x,
                                                    4, 4,
                                                    numtimes_allechoes = 2)
        self.assertEqual(n_trials[0], longcase_2x.shape[0])

    def test_oneecho_allechoes2x(self):
        '''check if one echo case counting is accurate
        '''
        oneecho = np.array([0,0,1,0,0,1]).reshape(6,1)
        n_trials = calc_num_IPIs_to_hear_echoes(oneecho,1,1,
                                                numtimes_allechoes=2)
        self.assertEqual(n_trials[0],6)
        
    def test_onetrialcase(self):
        '''
        '''
        ntrials = calc_num_IPIs_to_hear_echoes(self.echoesheard_case1, 1,4)
        expected = [1]*4
        
        obtained_matches_expected = self.compare_lists(expected, ntrials)
        self.assertTrue(obtained_matches_expected)
                                                               
    
    def compare_lists(self,a,b):
        lists_are_same = collections.Counter(a) == collections.Counter(b)
        return(lists_are_same)
        
        

if __name__ == '__main__':
    unittest.main()
        
        

