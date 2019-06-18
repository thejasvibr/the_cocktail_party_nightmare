# -*- coding: utf-8 -*-
""" Module to implement acoustic shadowing along with spherical spreading
of sound

Created on Mon Jun 17 16:08:50 2019

@author: tbeleyur
"""
import sys 
sys.path.append('..//bridson//')
sys.path.append('..//')
import numpy as np 
import scipy.spatial as spatial
from the_cocktail_party_nightmare import calc_RL


def soundprop_w_acoustic_shadowing(SL,
                                   start_point, end_point, all_other_points,
                                   **kwargs):
    '''Calculate the received level of a sound emitter at the start point 
    and reaching the end point after potentially passing through other points on 
    the way. 
    
    Each point sound passes through creates a drop in the intensity because of 
    acoustic shadowing. 

    Parameters
    ----------
    SL : tuple with 2 entries. 
         entry 0 : sound pressure level in dB SPL re 20 microPascals
         entry 1 : reference distance in metres at which the measurement was made. 

    start_point : 1 x 2 array like

    end_point :  1 x 2 array like

    all_other_points : Nbats-2 x 2 array like
                       xy coordinates of all points between start and end point
                       in a rectangular area of given width 
    TS_shadow : float < 0. 
                Acoustic shadow effect in dB. This refers to 
    
    Keyword Arguments
    ----------------
    rectangle_width : float >0. width of the rectangle 

    shadow_strength : the 'blocking' effect of an object being in the path.

    Returns:
        received_level: float. 
                        received level in dB SPL re 20 muPa.
                        
    '''    
    all_points_between = get_points_in_between(start_point, end_point, all_other_points, **kwargs)
    
    # if there are not points in between then calculate normal radial spreading. 
    if all_points_between.shape[0] == 0 :
        R = spatial.distance.euclidean(start_point, end_point)
        received_level = calc_RL(R,  SL[0], SL[1])
    else:
        received_level = calc_RL(R,  SL[0], SL[1]) + all_points_between.shape[0]*kwargs['shadow_strength']

    return(received_level)    

def get_points_in_between(start_point, end_point, all_other_points,
                          **kwargs):
    '''

    rectangle_width : float >0.
                      The width of the rectangle between the start and end point.

    
    '''
    rectangle_limits, rotation_matrix = make_rectangle_between_2_points(start_point, 
                                                       end_point, 
                                                       **kwargs)

    points_between = get_points_in_rectangle(rectangle_limits, start_point,
                                                   all_other_points, rotation_matrix)

    return(points_between)


def make_rectangle_between_2_points(A, B, **kwargs):
    '''
    '''
    # treat A as origin, calculate slope between B and A
    B_rel = B-A 
    slope = B_rel[1]/B_rel[0]
    # 'un-rotate' B and thus form a vertical rectangle easily
    theta = np.arctan(slope)
    rotation_matrix = rot_mat(theta)
    B_rotated = np.dot(rotation_matrix, B_rel)
    x0, x1 = -kwargs['rectangle_width']*0.5, kwargs['rectangle_width']*0.5
    y0, y1 = 0, B_rotated[1]
    
    return([x0,x1,y0,y1], rotation_matrix)




def get_points_in_rectangle(corner_limits, startpt,
                            many_points, rotn_matrix):
    '''
    x0,x1 the x coordinates defining the width of the rectangle
    y0,y1 the y coordinates defininf the height of the rectanlge

    '''
    x0,x1,y0,y1 = corner_limits
    relative_posns = many_points - startpt
    rotated_pts = np.apply_along_axis(dot_product_for_rows, 1, relative_posns,
                                      rotn_matrix)
    within_x = np.logical_and(rotated_pts[:,0] >= x0, rotated_pts[:,0] <= x1)
    within_y = np.logical_and(rotated_pts[:,1] >= y0,  rotated_pts[:,1] <= y1 )
    
    within_pts = np.logical_and(within_x, within_y)
    return(many_points[within_pts])
    
    

def dot_product_for_rows(xy_row, rotation_matrix):
    return(np.dot(rotation_matrix, xy_row))

def rot_mat(theta):
    return(np.row_stack(([np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)])))




    
  
if __name__ == '__main__':
    kwargs = {'rectangle_width':0.2}
    otherpts = np.random.normal(0,5,2000).reshape(-1,2)
    print(get_points_in_between(np.array([0,0]), np.array([5,5]), otherpts, **kwargs ) )
