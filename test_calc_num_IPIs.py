# -*- coding: utf-8 -*-
"""Tests for calc_num_IPIs
Created on Tue Nov 20 17:25:58 2018

@author: tbeleyur
"""
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
        trials_to_hear_2echoes = calc_num_IPIs_to_hear_echoes(self.echoesheard_long,
                                                              2, 4)
        trials_to_hear_3echoes = calc_num_IPIs_to_hear_echoes(self.echoesheard_long,
                                                              3, 4)
        trials_to_hear_4echoes = calc_num_IPIs_to_hear_echoes(self.echoesheard_long,
                                                              4, 4)
        print(trials_to_hear_2echoes,
              trials_to_hear_3echoes,
              trials_to_hear_4echoes)
        
        

if __name__ == '__main__':
    unittest.main()
        
        

