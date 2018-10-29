# -*- coding: utf-8 -*-
"""
Testing the num echoes heard module :

Created on Thu Sep 28 15:10:06 2017

@author: tbeleyur
"""


import sys
import unittest
import numpy as np
from testfixtures import LogCapture

from num_echoes_heard import *

module_path = 'C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\2016_jamming response modelling\\analytical_modelling\\'
sys.path.append(module_path)
class TestEchoesHeard(unittest.TestCase):

    def test_calc_num_freeechoes(self):
        # when there are only two echoes to be checked :

        echo_ranges = [ [0,3],[3,6] ] # echoes with masking windows included

        test_combination_noechofree = [1]*3+[1]*3+[0]*10

        num_echoesfree_test0 = calc_numfreeechoes(test_combination_noechofree,echo_ranges)

        self.assertEqual(num_echoesfree_test0,0)

        test_combination_1echofree = [0]*3+[1]*3+[0]*10

        self.assertEqual(calc_numfreeechoes(test_combination_1echofree,echo_ranges),1)

        test_combination_2echofree = [0]*3+[0]*3+[0]*10

        self.assertEqual(calc_numfreeechoes(test_combination_2echofree,echo_ranges),2)


        # with large number of calls in each bin
        test_combination_noechofree_v2 = [100000]*3+[1]*3+[0]*10

        num_echoesfree_test0_v2 = calc_numfreeechoes(test_combination_noechofree_v2,echo_ranges)

        self.assertEqual(num_echoesfree_test0_v2,0)

        # no echoes free, but with 1 call in each echo bin:

        test_combination_noechofree_v3 = [0]*2 +[1] + [0]*2 + [1] + [0]*10

        self.assertEqual(calc_numfreeechoes(test_combination_noechofree_v3,echo_ranges),0)


    def test_calculate_fraction_echoes(self):

        combination1 = [0]*3 + [1]*3 +[0]*10 + [1]*3
        combination2 = [1]*3 + [1]*3 +[0]*10 + [1]*3
        combination3 = [0]*3 + [0]*3 +[0]*10 + [1]*3

        three_combinations = [combination1, combination2, combination3]


        self.assertEqual(count_num_echoes(three_combinations,nechoes=2,maskingwindow=3),[1,1,1])

        self.assertEqual(count_num_echoes(three_combinations,nechoes=2,maskingwindow=2),[1,1,1])

        self.assertEqual(count_num_echoes(three_combinations,nechoes=2,maskingwindow=1),[1,0,2])















if __name__ == '__main__':
    unittest.main()