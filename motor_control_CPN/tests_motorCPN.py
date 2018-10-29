# -*- coding: utf-8 -*-
""" Tests for the motor_CPN module
Created on Thu Mar 08 11:22:07 2018

@author: tbeleyur
"""

import unittest
import numpy as np
import scipy.spatial as spatial
from motor_CPN import *




class TestPlanMovementDirection(unittest.TestCase):

    def test_simplecase(self):

        object_locns = np.array(([1.0,-45],[[2.0,-35]]))
        current_heading = 90
        mx_turn = 20.0

        new_heading = plan_movement_direction(object_locns, current_heading,
                                                  mx_turn)

        expected_heading = current_heading + mx_turn

        self.assertEqual(expected_heading, new_heading)



        pass

    def test_ifobjectsareatsamedistance(self):

        objects = np.array([][])



        pass


    def test_ifnoobjectsareseen(self):



        pass





if __name__ == '__main__':

    unittest.main()