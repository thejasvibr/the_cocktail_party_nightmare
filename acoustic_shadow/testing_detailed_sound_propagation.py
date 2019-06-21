# -*- coding: utf-8 -*-
"""Tests for the detailed sound propagation module
Created on Tue Jun 18 11:37:57 2019

@author: tbeleyur
"""
import unittest 
import numpy as np 
np.random.seed(82319)
import scipy.spatial as spatial
from detailed_sound_propagation import *

class TestGetPointsInBetween(unittest.TestCase):
    
    def setUp(self):
        self.start = np.array([1,1])
        self.end = np.array([5,5])
        self.points_between = np.row_stack(([2,2],[3,3],[9,9]))
        self.kwargs = {'rectangle_width':0.1} 
    
    def test_basic(self):
        between = get_points_in_between(self.start, self.end, self.points_between,
                              **self.kwargs)
        self.assertEqual(between.shape[0],2)
    
    def test_basic2(self):
        '''Move start point to -x,-y quadrant
        '''
        self.start *= -1 
        between = get_points_in_between(self.start, self.end, self.points_between,
                              **self.kwargs)
        self.assertEqual(between.shape[0],2)

    def test_basic3(self):
        '''start point to -x,+y quadrant
        '''
        self.start[0] *= -1
        between = get_points_in_between(self.start, self.end, self.points_between,
                              **self.kwargs)
        self.assertEqual(between.shape[0],0)
    def test_basic4(self):
        '''start point to +x,-y 
        '''
        self.start[1] *= -1
        between = get_points_in_between(self.start, self.end, self.points_between,
                              **self.kwargs)
        self.assertEqual(between.shape[0],0)
    def test_basic5(self):
        '''test that points within the rectangle are being picked detected. 
        '''
        self.start = np.array([1,-1])
        self.kwargs['rectangle_width'] = 5.0
        between = get_points_in_between(self.start, self.end, self.points_between,
                              **self.kwargs)
        self.assertEqual(between.shape[0],2)
    
    def test_basic6(self):
        '''
        '''
        self.start = np.array([5,-5])
        between = get_points_in_between(self.start, self.end, self.points_between,
                              **self.kwargs)
        self.assertEqual(between.shape[0],0)
    
    def test_bitmore(self):
        '''Make a bunch of points vertically aligned, and then 
        rotate them - and then check if they're picked up correctly. 
        '''
        theta = np.deg2rad(0)
        width  = 0.6
        rotation_matrix = rot_mat(theta)
        
        self.kwargs['rectangle_width'] = width

        self.start = np.array([0,-1])
        x_coods = np.random.choice(np.arange(-width*0.5, width*0.5, 0.01),10)
        y_coods = np.random.choice(np.arange(0, 2, 0.01),10)
        
        self.between_points = np.column_stack((x_coods, y_coods))
        self.other_points = np.random.choice(np.arange(0.8,0.9,0.1),10).reshape(-1,2)
        
        self.all_other_points = np.row_stack((self.between_points,
                                              self.other_points))
        self.end = np.array([0,5])
        
        rot_end = np.dot(rotation_matrix,self.end)
        rot_start = np.dot(rotation_matrix, self.start)
        rot_allotherpoints = np.apply_along_axis(dot_product_for_rows, 1,
                                                 self.all_other_points,
                                                 rotation_matrix)

        between = get_points_in_between(rot_start, rot_end,
                                        rot_allotherpoints,
                                        **self.kwargs)
        expected_numpoints = self.between_points.shape[0]

        self.assertEqual(between.shape[0], expected_numpoints)



    def test_bitmore_2(self):
        '''Make a bunch of points horizontially aligned, and then 
        rotate them - and then check if they're picked up correctly. 
        '''
        theta = np.radians(-10)
        width  = 0.2
        rotation_matrix = rot_mat(theta)
        
        self.kwargs['rectangle_width'] = width

        self.start = np.array([0,0])
        self.end = np.array([5,0])

        y_coods = np.random.choice(np.arange(-width*0.5, width*0.5, 0.01),10)
        x_coods = np.random.choice(np.arange(0, 2, 0.01),10)
        
        self.between_points = np.column_stack((x_coods, y_coods))
        self.other_points = np.random.choice(np.arange(90,120,1),10).reshape(-1,2)
       
        self.all_other_points = np.row_stack((self.between_points,
                                              self.other_points))

        
        rot_end = np.dot(rotation_matrix,self.end)
        rot_start = np.dot(rotation_matrix, self.start)
        rot_allotherpoints = np.apply_along_axis(dot_product_for_rows, 1,
                                                 self.all_other_points,
                                                 rotation_matrix)

        between = get_points_in_between(rot_start, rot_end,
                                        rot_allotherpoints,
                                        **self.kwargs)

        expected_numpoints = self.between_points.shape[0]

        self.assertEqual(between.shape[0], expected_numpoints)
    
    
    
    def test_pointsonaline(self):
        '''
        '''
        bats_xy = np.array(([1,0],[1,0.05]))
        focal_bat = np.array([0,0])
        receiver = np.array([2,0])
        between = get_points_in_between(focal_bat, receiver,
                                        bats_xy,
                                        **self.kwargs)
        expected = 2
        obtained = between.shape[0]
        self.assertEqual(expected, obtained)
    
    def test_points_onaline2(self):
        bats_xy = np.array([0.5,0.05]).reshape(-1,1)
        start = np.array([2,0])
        end = np.array([0,0])
        between = get_points_in_between(start, end,
                                        bats_xy,
                                        **self.kwargs)
        expected = 1
        obtained = between.shape[0]
        self.assertEqual(expected, obtained)
    
    def test_points_on_yaxis(self):
        #otherpts = np.random.normal(0,5,2000).reshape(-1,2)
        num_points = 100
        y_coods = np.random.choice(np.arange(0.1, 5, 0.01),num_points)
        x_coods = np.tile(0.02,num_points)
                
        between_points = np.column_stack((x_coods, y_coods))
        #otherpts = np.array(([1,0],[1,0.05]))
        #print(get_points_in_between(np.array([2,0]), np.array([0,0]), otherpts, **kwargs ) )
        start = time.time()
        betw_points = get_points_in_between(np.array([0,0]), np.array([0,10]), between_points,
                                       **self.kwargs)
        self.assertEqual(betw_points.shape[0], num_points)
            
        

class TestSoundprop_w_AcousticShadowing(unittest.TestCase):
    
    def setUp(self):
        self.kwargs = {}
        self.kwargs['shadow_strength'] = -3.0
        width = 0.2
        self.kwargs['implement_shadowing'] = True
        self.kwargs['rectangle_width'] = width
        self.kwargs['shadow_TS'] = [-9]
        self.start_point = np.array([0,0])
        self.end_point = np.array([0,10])
        self.kwargs['emitted_source_level'] = {'dBSPL':100, 'ref_distance':1.0}
        x_coods = np.tile(0,2)
        y_coods = np.array([1,2])
        self.kwargs['R'] = 10.0
        self.other_points = np.column_stack((x_coods, y_coods))


    def test_basic(self):
        ''' start and end at 45 degrees and a cloud of points in between. 
        '''
        rl_w_shadowing = soundprop_w_acoustic_shadowing(self.start_point,
                                                    self.end_point,
                                                    self.other_points,
                                                    **self.kwargs)

        expected = 64
        
        self.assertEqual(expected, np.around(rl_w_shadowing))

    def test_nobatsinbetween(self):
        '''
        '''
        self.other_points = np.row_stack(([2,2],[4,4],[9,9],[6,6]))
        shadowing = soundprop_w_acoustic_shadowing(     self.start_point,
                                                        self.end_point,
                                                        self.other_points,
                                                        **self.kwargs)
        
        R_startend = spatial.distance.euclidean(self.start_point,
                                                self.end_point)
        expected =  80.0
        self.assertEqual(expected, shadowing)
    
    def test_3batangle(self):
        self.kwargs['bats_xy'] = np.array(([0,0],[0.5,0.05],[2,0]))
        self.kwargs['implement_shadowing'] = True
        self.kwargs['rectangle_width'] = 0.3
        self.kwargs['shadow_TS'] = [-13]
        self.start_point = self.kwargs['bats_xy'][2,:]
        self.end_point = self.kwargs['bats_xy'][0,:]
        self.other_points = np.array([0.5,0.05]).reshape(1,-1)

        shadowing = soundprop_w_acoustic_shadowing(     self.start_point,
                                                        self.end_point,
                                                        self.other_points,
                                                        **self.kwargs)
        print(shadowing)
        

        

        
        













if __name__ == '__main__':
    unittest.main()