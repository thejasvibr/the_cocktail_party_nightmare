# -*- coding: utf-8 -*-
"""Tests for the_cocktail_party_nightmare_MC
Created on Tue Dec 12 22:07:22 2017

@author: tbeleyur
"""
import sys
sys.path.append('../CPN/')
import unittest
import numpy as np
np.random.seed(82319)
import pandas as pd
#import matplotlib.pyplot as plt

from the_cocktail_party_nightmare import *

    
class TestingCheckIfEchoHeard(unittest.TestCase)    :
    '''
    '''

    def setUp(self):
        # temporal masking function -  make everything linear to ease
        # quick calculation of expected results
        self.kwargs = {}
        self.kwargs['hearing_threshold'] = 0 
        fwd_masking_region = np.linspace(-27, -7, 20000)        
        bkwd_masking_region = np.linspace(-10, -24, 3000)
        simult_masking_regio = np.array([-8])
        self.temporal_masking_fn = (fwd_masking_region,simult_masking_regio,
                                            bkwd_masking_region)
        self.echo = pd.DataFrame(data={'start':[100], 'stop':[3100],
                                       'level':[50], 'theta' : [10]})

        self.kwargs['temporal_masking_thresholds'] = self.temporal_masking_fn
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['interpulse_interval'] = 0.1
        
        spl_masking = pd.DataFrame(data={'deltatheta':np.linspace(0,30,30),
                                         'dB_release':np.linspace(-1,-30,30)})
        self.kwargs['spatial_release_fn'] = np.array(spl_masking)
        self.kwargs['echocall_duration'] = 0.003
        self.cumulative_spl = np.tile(-9, 1000) + np.random.normal(0,1,1000)

        self.other_sounds = pd.DataFrame(data=[], index=range(10),
                                                 columns=['start','stop',
                                                     'level','theta'])
        

    def test_basic_noechoheard(self):
        '''Check that a sub threshold echo is not heard
        '''
        self.echo['level']  = -10
        echo_heard = check_if_echo_heard(self.echo, **self.kwargs)
        self.assertFalse(echo_heard)
    
    def test_basic_echoheard(self):
        '''A very loud echo that should be heard
        '''
        self.echo['level']  = 130
        # make the other sounds :
        for i in range(10):
            self.other_sounds['start'][i] = np.random.choice(np.arange(0,50))
            self.other_sounds['stop'][i] = self.other_sounds['start'][i] + 3
            self.other_sounds['level'][i] = np.random.choice(np.arange(20,30))
            self.other_sounds['theta'][i] = np.random.choice(np.arange(-180,180))

        self.kwargs['other_sounds'] = self.other_sounds
        echo_heard = check_if_echo_heard(self.echo, **self.kwargs)
        self.assertTrue(echo_heard)

    def test_overlapping_calls(self):
        '''What happens when a bunch of calls completely overlap the ipi from end to 
        end
        '''
        self.other_sounds = pd.DataFrame(data=[], index=range(2),
                                                 columns=['start','stop',
                                                     'level','theta'])
        self.other_sounds['start'] = [0,49999]
        self.other_sounds['stop'] = [49999,99999]
        self.other_sounds['level'] = [130, 130]
        self.other_sounds['theta'] = [-35, -35]

        self.echo['level'] = 20
        self.kwargs['other_sounds'] = self.other_sounds
        echo_heard = check_if_echo_heard(self.echo,
                                         **self.kwargs)
        self.assertFalse(echo_heard)
      
        

class TestingNumEchoesHeard(unittest.TestCase):

    def setUp(self):
        # temporal masking function -  make everything linear to ease
        # quick calculation of expected results
        self.kwargs = {}
        self.kwargs['hearing_threshold'] = 0 
        fwd_masking_region = np.linspace(-27, -7, 20000)        
        bkwd_masking_region = np.linspace(-10, -24, 3000)
        simult_masking_regio = np.array([-8])
        self.temporal_masking_fn = (fwd_masking_region,simult_masking_regio,
                                            bkwd_masking_region)
        self.echoes = pd.DataFrame(data={'start':[100,4000], 'stop':[3100, 7000],
                                       'level':[-50, 0], 'theta' : [10, -10]})

        self.kwargs['temporal_masking_thresholds'] = self.temporal_masking_fn
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['interpulse_interval'] = 0.1
        
        spl_masking = pd.DataFrame(data={'deltatheta':np.linspace(0,30,30),
                                         'dB_release':np.linspace(-1,-30,30)})
        self.kwargs['spatial_release_fn'] = np.array(spl_masking)
        self.kwargs['echocall_duration'] = 0.003
        self.cumulative_spl = np.tile(-9, 1000) + np.random.normal(0,1,1000)

        self.other_sounds = pd.DataFrame(data=[], index=range(10),
                                                 columns=['start','stop',
                                                     'level','theta'])


    def test_subthreshold_notheard(self):
        '''all echoes are subthreshold
        '''
        self.other_sounds = pd.DataFrame(data=[], index=range(2),
                                             columns=['start','stop',
                                                 'level','theta'])
        self.other_sounds['start'] = [0,49999]
        self.other_sounds['stop'] = [49999,99999]
        self.other_sounds['level'] = [130, 130]
        self.other_sounds['theta'] = [-35, -35]
        num_heard, _  = calculate_num_heardechoes(self.echoes,
                                              self.other_sounds, 
                                              **self.kwargs)
        self.assertEqual(num_heard, 0)

    def test_loudechoesheard(self):
        '''all echoes are *very* loud
        '''
        self.other_sounds = pd.DataFrame(data=[], index=range(2),
                                             columns=['start','stop',
                                                 'level','theta'])
        self.other_sounds['start'] = [0,49999]
        self.other_sounds['stop'] = [49999,99999]
        self.other_sounds['level'] = [30, 30]
        self.other_sounds['theta'] = [-35, -35]

        self.echoes['level'] = [90,90]
        num_heard, _  = calculate_num_heardechoes(self.echoes,
                                              self.other_sounds, 
                                              **self.kwargs)
        self.assertEqual(num_heard, 2)

    def test_oneecho_heard(self):
        ''' 1/3 echoes should be heard
        '''
        self.echoes = pd.DataFrame(data={'start':[100,4000,10000],
                                         'stop':[3100, 7000, 13000],
                                         'level':[-50, 0, 60], 
                                         'theta' : [10, -10, 0]})

        self.other_sounds = pd.DataFrame(data=[], index=range(2),
                                             columns=['start','stop',
                                                 'level','theta'])
        self.other_sounds['start'] = [0,49999]
        self.other_sounds['stop'] = [49999,99999]
        self.other_sounds['level'] = [30, 30]
        self.other_sounds['theta'] = [30, 30]

        num_heard, _ = calculate_num_heardechoes(self.echoes, self.other_sounds,
                                                 **self.kwargs)
        self.assertEqual(num_heard,1)
#
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
            num_genpoints[index] = numrows + 1 


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
    
            secondechoes_paths = calculate_echopaths('secondary_echoes',**self.kwargs)
            expected_number = (num_bats-1)*(num_bats-2)

            self.assertEqual(expected_number, len(secondechoes_paths['sound_routes']))
        

    def test_basic_2ndaryechoes(self):
        ''' 3 bats in a triangle confuguration
        '''
        secondechoes_paths = calculate_echopaths('secondary_echoes',**self.kwargs)
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

        self.output = calculate_echopaths('secondary_echoes',**self.kwargs)
        exp_R_in = []
        exp_R_out = []
        exp_theta_em = []
        exp_theta_recpn = []
        exp_theta_in = []
        exp_theta_out = []

        for echo_id, each_echo in enumerate(self.output['sound_routes']):
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
        
        self.kwargs['reflection_function']['theta_incoming'] = all_angles[:,0]
        self.kwargs['reflection_function']['theta_outgoing'] = all_angles[:,1]
        self.kwargs['reflection_function']['ref_distance'] = np.tile(0.1, 360*360)
        
        self.kwargs['call_directionality'] = lambda X : 0 
        self.kwargs['hearing_directionality'] = lambda X : 0
        self.kwargs['source_level'] = {'ref_distance' : 1.0, 'dBSPL':100}
        self.kwargs['implement_shadowing'] = False

    def test_basic(self):
        '''
        '''
        secondary_echoes = calculate_secondaryecho_levels(**self.kwargs)
        output_received_levels = np.array(secondary_echoes['level']).flatten()

        # calculate the expected received levels of the 2dary echoes
        paths = calculate_echopaths('secondary_echoes',**self.kwargs)
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

#        self.assertTrue(np.array_equal(np.array(expected_received_levels).flatten(),
#                                       output_received_levels))
# 
class TestCalculateConspecificcall_levels(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.kwargs = {}
        self.kwargs['Nbats'] = 3 
        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])
        self.kwargs['call_directionality'] = lambda X : 0 
        self.kwargs['hearing_directionality'] = lambda X : 0
        self.kwargs['source_level'] = {'ref_distance' : 0.1, 'dBSPL':120}
        self.kwargs['implement_shadowing'] = False


    def calc_distmat(self):
        self.distance_matrix = spl.distance_matrix(self.kwargs['bats_xy'], self.kwargs['bats_xy'])

    def test_basic(self):
        '''test 3 bat situation 
        '''
        conspecific_calls = calculate_conspecificcall_levels(**self.kwargs)
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
        self.kwargs['Nbats'] = 3
        self.kwargs['bats_xy'] = np.array(([1,1],[2,1],[2,2]))
        self.kwargs['focal_bat'] = np.array([1,1])
        self.kwargs['bats_orientations'] = np.array([90,45,135])
        self.kwargs['call_directionality'] = lambda X : 0 
        self.kwargs['hearing_directionality'] = lambda X : 0
        self.kwargs['source_level'] = {'ref_distance' : 1.0, 'dBSPL':100.0}
        self.kwargs['implement_shadowing'] = False


        self.kwargs['reflection_function'] = pd.DataFrame(data=[], index=range(36*36),
                                                          columns=['ref_distance',
                                                                   'incoming_theta',
                                                                   'outgoing_theta',
                                                                   'reflection_strength'])
        self.kwargs['reflection_function']['reflection_strength'] = np.tile(-10, 36*36)
        all_angles = np.array(np.meshgrid(np.linspace(-180,180,36),
                                          np.linspace(-180,180,36))).T.reshape(-1,2)
        
        self.kwargs['reflection_function']['theta_incoming'] = all_angles[:,0]
        self.kwargs['reflection_function']['theta_outgoing'] = all_angles[:,1]
        self.kwargs['reflection_function']['ref_distance'] = np.tile(0.1, 36*36)
        self.kwargs['hearing_threshold'] = 20
        self.kwargs['implement_shadowing'] = False

    def test_numberofsounds(self):
        
        # calculate the received levels and angles of arrivals of the sounds
        conspecific_calls = propagate_sounds('conspecific_calls', **self.kwargs)
        secondary_echoes = propagate_sounds('secondary_echoes', **self.kwargs)
        primary_echoes = propagate_sounds('primary_echoes', **self.kwargs)
        
        num_conspecificcalls, num_2daryechoes = conspecific_calls.shape[0], secondary_echoes.shape[0]
        num_1echoes =  primary_echoes.shape[0]

        nbats = self.kwargs['bats_xy'].shape[0]
        exp_numcalls = nbats-1
        exp_num2daryechoes = (nbats-2)*(nbats-1)
        exp_num1echoes = nbats-1

        self.assertTrue(np.array_equal([num_conspecificcalls, num_2daryechoes, num_1echoes],
                                       [exp_numcalls, exp_num2daryechoes, exp_num1echoes]))



class TestCombineSounds(unittest.TestCase):
    '''
    '''
    def setUp(self):
        # generate two empty DFs with no data yet. 
        self.A = pd.DataFrame(data=[], index=range(10), columns=['start', 'stop',
                         'theta', 'level','id'])
        self.A.loc[:,:] = np.random.normal(0,1,10*5).reshape(-1,5)
        self.B = self.A.copy()
        self.B['identity'] = np.nan

    def test_basic(self):
        '''Check if two dfs with dissimilar columns names will be concateneted correctly:
        '''
        combined_sounds = combine_sounds([self.A, self.B])        
        num_expected_rows = self.A.shape[0] + self.B.shape[0]
        joined_columns = set(self.A.columns).union(set(self.B.columns))
        num_expected_colnums = len(joined_columns)

        self.assertTrue(np.array_equal([num_expected_rows, num_expected_colnums],
                                       list(combined_sounds.shape)))
    def test_2batcase(self):
        '''
        '''
        self.A = pd.DataFrame(data={'start':[0,20,30],
                                    'stop':[20,30,40],
                                    'level':[90,92,10],
                                    'theta':[-10,20,1],
                                    'route':['1-0-1', '2,0,2','3-0-3']}) 
        self.B = self.A.copy()
        self.B['route'] = np.tile(np.nan, self.B.shape[0])
        num_expected_rows = self.A.shape[0] + self.B.shape[0]
        joined_columns = set(self.A.columns).union(set(self.B.columns))
        num_expected_colnums = len(joined_columns)
        combined_sounds = combine_sounds([self.A, self.B])    
        self.assertTrue(np.array_equal([num_expected_rows, num_expected_colnums],
                                       list(combined_sounds.shape)))
        
      

class TestRunCPN(unittest.TestCase):
    '''
    '''
    def setUp(self):
        ''' basic set of kwargs to initiate runCPN
        '''
        self.A = 7
        self.B = 2 

        self.kwargs={}
        self.kwargs['interpulse_interval'] = 0.1
        self.kwargs['v_sound'] = 330
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['echocall_duration'] = 0.003
        self.kwargs['call_directionality'] = lambda X : self.A*(np.cos(np.deg2rad(X))-1)
        self.kwargs['hearing_directionality'] = lambda X : self.B*(np.cos(np.deg2rad(X))-1)
        reflectionfunc = pd.DataFrame(data=[], columns=[], index=range(144))
        thetas = np.linspace(-180,180,12)
        input_output_angles = np.array(np.meshgrid(thetas,thetas)).T.reshape(-1,2)
        reflectionfunc['reflection_strength'] = np.random.normal(-40,5,
                                                  input_output_angles.shape[0])
        reflectionfunc['theta_incoming'] = input_output_angles[:,0]
        reflectionfunc['theta_outgoing'] = input_output_angles[:,1]
        reflectionfunc['ref_distance'] = 0.1
        self.kwargs['reflection_function'] = reflectionfunc
        self.kwargs['heading_variation'] = 10 
        self.kwargs['min_spacing'] = 0.5
        self.kwargs['Nbats'] = 5
        self.kwargs['source_level'] = {'dBSPL' : 120, 'ref_distance':0.1}
        self.kwargs['hearing_threshold'] = 10
        self.kwargs['implement_shadowing'] = False
        
        fwd_masking_region = np.linspace(-27, -7, 20000)        
        bkwd_masking_region = np.linspace(-10, -24, 3000)
        simult_masking_regio = np.array([-8])
        temporal_masking_fn = (fwd_masking_region,simult_masking_regio,
                                                bkwd_masking_region)
        self.kwargs['temporal_masking_thresholds'] = temporal_masking_fn
        
        spl_um = pd.DataFrame(data={'deltatheta':np.linspace(0,30,31),
                                    'dB_release':np.linspace(0, -30,31)})
        self.kwargs['spatial_release_fn'] = np.array(spl_um)

    def test_checkif_type_correct(self):
        '''Check if an integer is the output of a single runCPN
        '''
        self.kwargs['Nbats'] = 6
        num_echoesheard, _ = run_CPN(**self.kwargs)
        self.assertTrue(isinstance(num_echoesheard, int))
    
    def test_check_w_just_2_bats(self):
        '''Make sure that the NaNs are removed in the 2 bat case because
        the 2dary echoes sound dfs are output w Nans.
        '''
        self.kwargs['Nbats'] = 2
        self.kwargs['implement_shadowing'] = False
        
        num_echoesheard, _ = run_CPN(**self.kwargs)
        self.assertTrue(isinstance(num_echoesheard, int))


class TestPlaceSoundsRandomlyinIPI(unittest.TestCase):
    '''
    '''
    def test_basic(self):
        '''simple test to chck if correct number of sounds are being assigned
        '''
        exp_num_sounds = [5,10,100]
        got_num_sounds = []
        for num_sound in exp_num_sounds:
            sound_times = place_sounds_randomly_in_IPI(range(10**5),30,num_sound)
            got_num_sounds.append(sound_times.shape[0])
        
        self.assertTrue(np.array_equal(exp_num_sounds, got_num_sounds))

    def test_weird(self):
        '''make sure an error is thrown when the call duration is longer than the ipi
        '''
        with self.assertRaises(AssertionError) as context:
            place_sounds_randomly_in_IPI(range(10**5),10**6,2)

        self.assertTrue('Call duration cannot be greater than ipi!!' in context.exception)

class TestAssignRandomArrivalTimes(unittest.TestCase):
    
    def setUp(self):
        '''
        '''
        self.calls = pd.DataFrame(data=[], columns = ['start','stop','theta'],index=range(10))
        self.kwargs ={}
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['interpulse_interval'] = 0.1
        self.kwargs['echocall_duration'] = 0.003
        
    def check_num_nulls(self,X):
        num_nulls = sum(pd.isnull(X))
        return(num_nulls)

    def test_basis(self):
        '''Check if the number of start-stop times correspond to the number of sounds in sound_df
        '''
        self.calls = assign_random_arrival_times(self.calls, **self.kwargs)
        startstop_null = [self.check_num_nulls(self.calls['start']),
                          self.check_num_nulls(self.calls['stop'])]

        # check that there are no missing values
        self.assertTrue(np.array_equal([0,0], startstop_null))

class TestAssignRealArrivalTimes(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.echoes = pd.DataFrame(data=[], columns = ['start','stop','theta'],index=range(1))
        self.kwargs ={}
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['interpulse_interval'] = 0.1
        self.kwargs['echocall_duration'] = 0.003
        self.kwargs['v_sound'] = 330.0
        self.kwargs['echocall_duration'] = 0.003
    
    def test_oneecho(self):
        '''check that the time of arrival is proper for just one target
        '''
        self.kwargs['echocall_duration'] = 0.002
        self.kwargs['bats_xy'] = np.array(([0,0],[1,0]))
        exp_start = int(np.around((- self.kwargs['echocall_duration'] +2/self.kwargs['v_sound'])/self.kwargs['simtime_resolution'])) 
        exp_echocall_timesteps = int(np.around(self.kwargs['echocall_duration']/self.kwargs['simtime_resolution']))
        exp_stop = exp_start + exp_echocall_timesteps -1 
        assign_real_arrival_times(self.echoes, **self.kwargs)
        self.assertTrue(np.array_equal([exp_start, exp_stop],
                                        np.array(self.echoes[['start','stop']]).flatten()))
        
    
    def test_basiccross(self):
        '''check if the echo TOA are proper for a basic cross configuration of 4 bats 
        with the focal in the centre
        '''
        ipi_timesteps = calc_num_timesteps_in_IPI(**self.kwargs)
        self.echoes = pd.DataFrame(data=[], columns = ['start','stop','theta'],index=range(4))
        self.kwargs['bats_xy'] = np.array(([0,0],[2,0],[-3,0],[0,-4],[0,5]))
        assign_real_arrival_times(self.echoes, **self.kwargs)
        
        echo_dists = 2*spl.distance_matrix(self.kwargs['bats_xy'], self.kwargs['bats_xy'])
        distances = echo_dists[1:,0]
        
        starts = distances/self.kwargs['v_sound'] - self.kwargs['echocall_duration']
        relative_starts = np.float64(starts/self.kwargs['interpulse_interval'])
        exp_start = np.int64(np.around(relative_starts*ipi_timesteps)) 
        exp_echocall_timesteps = np.around(self.kwargs['echocall_duration']/self.kwargs['simtime_resolution'])
        exp_stop = np.int64(exp_start + exp_echocall_timesteps -1 )
        exp_start_stop = np.column_stack((exp_start, exp_stop)).reshape(-1,2)
        got_start_stop = np.array(self.echoes[['start','stop']]).reshape(-1,2)

        self.assertTrue(np.array_equal(exp_start_stop, got_start_stop))
    
    def test_callecho_overlap(self):
        '''Simulate the case where the echo returns before call emission is complete
        '''
        self.echoes = pd.DataFrame(data=[], columns = ['start','stop','theta'],index=range(2))
        self.kwargs['echocall_duration'] = 0.010
        self.kwargs['bats_xy'] = np.array(([0,0],[0.5,0]))
        exp_start = int(np.around((2/self.kwargs['v_sound'])/self.kwargs['simtime_resolution'])) - self.kwargs['echocall_duration']
        exp_echocall_timesteps = int(np.around(self.kwargs['echocall_duration']/self.kwargs['simtime_resolution']))
        exp_stop = exp_start + exp_echocall_timesteps -1 
        exp_msg = 'Some echoes are arriving before the call is over! Please check call duration or neighbour distance'
        with self.assertRaises(ValueError) as context:
            assign_real_arrival_times(self.echoes, **self.kwargs)
            
        self.assertTrue(exp_msg in context.exception)


class TestRelativeAngleCalculations(unittest.TestCase):
    
    def test_basic(self):
        incoming = -45
        outgoing = 45
        expected = 90.0
        theta_sep  = get_relative_echo_angular_separation(incoming, outgoing)
        self.assertEqual(expected, theta_sep)
    
    def test_w180(self):
        incoming = -45
        outgoing = 180
        expected = 135.0
        theta_sep  = get_relative_echo_angular_separation(incoming, outgoing)
        self.assertEqual(expected, theta_sep)
    
    def test_wsameside(self):
        incoming = 45
        outgoing = 180
        expected = 135.0
        theta_sep  = get_relative_echo_angular_separation(incoming, outgoing)
        self.assertEqual(expected, theta_sep)

class TestCalcEchoThetas(unittest.TestCase):
    ''' 
    '''
    def setUp(self):
        # calc_echo_thetas(bats_xy, bat_orientations, sound_routes, which_angles)
        self.xy = np.row_stack(([0,0],[1,0],[1,1],[0,1],[-1,1]))
        self.bat_orientations = np.tile(90, self.xy.shape[0])
        self.sound_routes = [(0,1,2),(0,3,2),(0,3,4),(0,3,1)]
    def test_basic(self):
        '''
        '''
        whichangles= 'emission_reception'
        towards_tgt, from_tgt = calc_echo_thetas(self.xy, self.bat_orientations,
                                                 self.sound_routes, whichangles)
        concat_output = np.concatenate((towards_tgt,from_tgt)).flatten()
        expected_towards = [90.0, 0.0, 0.0, 0.0]
        expected_from = [180.0, -90.0 , 90.0, -45.0]
        concat_expected = np.concatenate((expected_towards, expected_from)).flatten()

        self.assertTrue( np.array_equal(concat_expected,
                         concat_output) )
    
    def test_basic2(self):
        whichangles= 'incoming_outgoing'
        
        towards_tgt, from_tgt = calc_echo_thetas(self.xy, self.bat_orientations,
                                                 self.sound_routes, whichangles)

        concat_output = np.concatenate((towards_tgt,from_tgt)).flatten()
        expected_towards = [-90.0, 180.0, 180.0, 180]
        expected_from = [0.0, 90.0 , -90.0, 135.0]
        concat_expected = np.concatenate((expected_towards, expected_from)).flatten()

        self.assertTrue( np.array_equal(concat_expected,
                         concat_output) )
        
        
        
        
class TestCheckCumSPL(unittest.TestCase):

    def setUp(self):
        self.echo = pd.DataFrame(data={'start':[50], 'stop':[53], 'level':[50]})
        fwd_masking_region = np.linspace(-27, -7, 21)        
        bkwd_masking_region = np.linspace(-10, -24, 3)
        simult_masking_regio = np.array([-8])
        self.temporal_masking_fn = (fwd_masking_region,simult_masking_regio,
                                            bkwd_masking_region)

        self.kwargs = {}
        self.kwargs['temporal_masking_thresholds'] = self.temporal_masking_fn
        self.cum_SPL = np.ones(100) * -8
        self.kwargs['simtime_resolution'] = 10**-3
        self.kwargs['echocall_duration'] = 0.003
        self.kwargs['hearing_threshold'] = 0.0

    def test_basic_echo_heard(self):
        '''Echo should be heard when delta echo-masker is above the threshold all throughout
        '''
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertTrue(echo_heard)
    
    def test_echo_at_edges1(self):
        '''Test what happens when the echo is at the edges of the ipi:
        '''
        self.echo = pd.DataFrame(data={'start':[0], 'stop':[3], 'level':[50]})
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertTrue(echo_heard)
        
    def test_echo_at_edges2(self):
        '''Test what happens when the echo is at the edges of the ipi Part2
        '''
        self.echo = pd.DataFrame(data={'start':[97], 'stop':[99], 'level':[50]})
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertTrue(echo_heard)
    
    def test_echo_beyond_ipi_on_right(self):
        '''Test what happens when the echo is at the edges of the ipi Part2
        '''
        self.echo = pd.DataFrame(data={'start':[100], 'stop':[102], 'level':[50]})
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertFalse(echo_heard)
    
    def test_echo_beyond_ipi_on_left(self):
        '''Test what happens when the echo is at the edges of the ipi Part2
        '''
        self.echo = pd.DataFrame(data={'start':[-10], 'stop':[0], 'level':[50]})
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertFalse(echo_heard)
    

    def test_echo_at_edges3(self):
        '''Test what happens when the echo ends a bit beyond the ipi 
        '''
        self.echo = pd.DataFrame(data={'start':[97], 'stop':[99], 'level':[50]})
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertTrue(echo_heard)
    
    def test_heard_w_loudburst(self):
        '''What happens when the echmasker profile falls below the 
        threshold for < echocall duraiton but is above it in general.
        '''
        self.cum_SPL = np.ones(100) *5
        self.cum_SPL[int(self.echo['start']):int(self.echo['stop'])] = 90 # one timestep has a strong drop in echomasker ratio 
        echo_heard = check_if_cum_SPL_above_masking_threshold(self.echo, self.cum_SPL, 
                                                              **self.kwargs)
        self.assertFalse(echo_heard)
    

class CheckRealArrivalTimes_AreIn_IPI(unittest.TestCase)  :
    
    def setUp(self):
        self.sound_df = pd.DataFrame(data=[], columns=['start','stop'])
        self.kwargs = {}
        self.kwargs['v_sound'] = 330.0
        self.kwargs['bats_xy'] = np.array(([0,0],[10,0],[0,2]))
        self.kwargs['simtime_resolution'] = 10**-6
        self.kwargs['interpulse_interval'] = 0.025
        self.kwargs['echocall_duration'] = 0.001
        self.kwargs['echoes_beyond_ipi'] = True
    
    def test_echoes_falling_out_of_IPI(self):
        '''Make sure that there's 
        '''
        self.kwargs['echoes_beyond_ipi'] = False
        
        with self.assertRaises(EchoesOutOfIPI) as context:
            assign_real_arrival_times(self.sound_df, **self.kwargs)
        self.assertTrue('Some echoes fall out of the interpulse interval - change the IPI or reduce distance of reflecting objects' in context.exception)

    def test_echoes_in_IPI_allow_echoes_outof_IPI(self):
        '''A lack of error is good here 
        '''
        self.kwargs['echoes_beyond_ipi'] = True
        print(self.kwargs)
        assign_real_arrival_times(self.sound_df, **self.kwargs)

class TestingIfEchoIsMostlyInIPI(unittest.TestCase):
    
    def setUp(self):
        self.kwargs = {}
        self.kwargs['simtime_resolution'] = 10**-3
        self.kwargs['echocall_duration'] = 0.005
        self.kwargs['masking_tolerance'] = 0.25
        self.num_ipi_timesteps = 100

    def test_basic(self):
        start = 100
        stop = 102
        echo_in_ipi = is_the_echo_mostly_in_the_ipi(start, stop,
                           self.num_ipi_timesteps,
                           **self.kwargs)
        self.assertFalse(echo_in_ipi)

    def test_basic_positive(self):
        start = 52
        stop = 56
        echo_in_ipi = is_the_echo_mostly_in_the_ipi(start, stop,
                           self.num_ipi_timesteps,
                           **self.kwargs)
        self.assertTrue(echo_in_ipi)
            
class TestNonCentralBat(unittest.TestCase):
    
    def setUp(self):
        '''
        '''
        self.noncentral_bat = (1.0, np.pi/2.0) # at 12 o'clock and furthest bat
        self.nearby = np.array(([1,1],[0,3],[-1,-1],[1,-1],[-1,1]))
        self.centremost_pt = np.array([0,0])
    
    def test_basic_outermost(self):
        '''The target bat is at 12 o'clock and the furthest from the centre
        '''
        nearby, focal = choose_a_noncentral_bat(self.noncentral_bat,
                                                self.nearby,
                                                self.centremost_pt)
        focal_is_as_expected = np.array_equal(focal, np.array([0,3]))
        
        expected_nearby = np.row_stack((self.centremost_pt,self.nearby))
        expected_nearby = np.delete(expected_nearby, 2,0)
        nearby_as_expected = np.array_equal(nearby, expected_nearby)
        
        output_as_expected = np.all([nearby_as_expected, focal_is_as_expected])
        self.assertTrue(output_as_expected)
    
    def test_more_complicated(self):
        '''Set up more bats all around and check if the correct noncentral 
        bat is chosen
        '''
        central_bat = np.array([0,0])
        radii = [2.5, 5.0]
        thetas = np.radians(np.arange(0,360,90))
        other_positions = []
        for r in radii:
            for theta in thetas:
                x,y = r*np.cos(theta), np.sin(theta)
                other_positions.append(np.array([x,y]))
        
        noncentral = (0.4, np.pi)
        other_positions = np.row_stack(other_positions)

        nearby, focal = choose_a_noncentral_bat(noncentral, 
                                                other_positions, 
                                                central_bat)
        expected_focal = np.array([-2.5,0])
        print('Obtained focal:', focal)
        obtained_as_expected = np.array_equal(np.around(expected_focal),
                                              np.around(focal) )
        self.assertTrue(obtained_as_expected)
 
    
class TestPlaceBatsInSpace(unittest.TestCase):
    '''Check that bats are placed as expected and the focal individual
    is assigned properly each time
    '''
    def setUp(self):
        self.kwargs = {}
        self.kwargs['Nbats'] = 10
        self.kwargs['min_spacing'] = 0.5
        self.kwargs['heading_variation'] = 40
        
    
    def test_basic_central_bat(self):
        '''
        '''
        bats_xy, headings = place_bats_inspace(**self.kwargs)
        dimensions_correct = self.check_if_output_dimensions_match(bats_xy,
                                                                   headings)
        self.assertTrue(dimensions_correct)
    
        
    def test_w_noncentral_bat(self):
        '''
        '''
        self.kwargs['noncentral_bat'] = (0.5, np.pi/2.0)
        bats_xy, headings = place_bats_inspace(**self.kwargs)
        dimensions_correct = self.check_if_output_dimensions_match(bats_xy,
                                                                   headings)
        self.assertTrue(dimensions_correct)

    def test_noncentral_bat_is_really_noncentral(self):
        self.kwargs['noncentral_bat'] = (0.5, np.pi/2.0)
        bats_xy, headings = place_bats_inspace(**self.kwargs)
        
        nearby, focal = bats_xy
        centremost = choose_centremostpoint(np.row_stack((focal, nearby)))
        focal_is_not_central = np.invert(np.array_equal(centremost, focal))
        self.assertTrue(focal_is_not_central)
        

    def check_if_output_dimensions_match(self, bats_xy, headings):
        # check if the headings and bats xy sizes match
        num_bats_obtained = np.sum([bats_xy[0].size, bats_xy[1].size])*0.5
        num_headings_obtained = headings.size
        obtained_dimensions = np.array([num_headings_obtained, num_bats_obtained])
        expected_dimensions = np.array([self.kwargs['Nbats']]*2)
        dimensions_correct = np.array_equal(obtained_dimensions, 
                                            expected_dimensions)
        return(dimensions_correct)
        
        
        

        
        

if __name__ == '__main__':

    unittest.main()



