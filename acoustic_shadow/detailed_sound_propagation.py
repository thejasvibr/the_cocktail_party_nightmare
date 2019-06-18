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


def soundprop_w_acoustic_shadowing(start_point, end_point, all_other_points,
                                   **kwargs):
    '''Calculate the received level of a sound emitter at the start point 
    and reaching the end point after potentially passing through other points on 
    the way. 
    
    Each point sound passes through creates a drop in the intensity because of 
    acoustic shadowing. 

    TODO
    ----
    1) implement a randomly varying acoustic shadowing for each bat - to mimic
    the flapping of wings 
        

    Parameters
    ----------

    start_point : 1 x 2 array like

    end_point :  1 x 2 array like

    all_other_points : Nbats-2 x 2 array like
                       xy coordinates of all points between start and end point
                       in a rectangular area of given width 
    shadow_strength : float < 0. 
                      Acoustic shadow effect in dB. This refers to 


    
    Keyword Arguments
    ----------------
    rectangle_width : float >0. width of the rectangle 

    shadow_strength : the 'blocking' effect of an object being in the path in dB



    Returns
    -------
    acoustic_shadowing: float<0 
                        reduction in received level of a sound due to acoustic shadowing in dB
    '''    
    all_points_between = get_points_in_between(start_point, end_point, 
                                                       all_other_points, **kwargs)
    
    acoustic_shadowing = all_points_between.shape[0]*kwargs['shadow_strength']

    return(acoustic_shadowing)    


def calc_RL(distance, SL, ref_dist):
    '''calculates received level only because of spherical spreading.

    Parameters

    distance : float>0. receiver distance from source in metres.

    SL : float. source level in dB SPL re 20 muPa.

    ref_dist : float >0. distance at which source level was measured in metres.


      Returns
    -------

    RL : received level in dB SPL re 20muPa.

    '''

    RL = SL - 20*np.log10(distance/ref_dist)

    return(RL)



def get_points_in_between(start_point, end_point, all_other_points,
                          **kwargs):
    '''
    
    Parameters
    ----------
    start_point : 1x2 array like
                  xy coordinates of starting point  

    end_point : 1x2 array like
                xy coordinates of end points

    all_other_points : N x 2 array like
                       xy coordinates of all other points  
    
    Keyword Arguments
    -----------------
    rectangle_width : float >0.
                      The width of the rectangle between the start and end point.

    Returns
    -------
    points_between : Mpoints_between x 2 np.array where Mpoints can be >= 0. 

    '''
    rectangle_limits, rotation_matrix = make_rectangle_between_2_points(start_point, 
                                                       end_point, 
                                                       **kwargs)

    points_between = get_points_in_rectangle(rectangle_limits, start_point,
                                                   all_other_points, rotation_matrix)

    return(points_between)


def make_rectangle_between_2_points(A, B, **kwargs):
    '''First calculate the relative coordinates of B wref to A. 
    Then draw a straight line between 0,0 and B_rel and 'undo' the
    slope. To this vertical line not apply rectangular bands on left and right.
    
    Output bottom left and top right vertices along with the rotation matrix
    along with the rotation matrix used to undo the slope of the 0,0-B_rel line
    for application to other poitns. 

    Parameters
    ----------
    A, B : 1x2 array like. xy coordinates of start(A) and end(B) points

    Keyword Arguments
    -----------------
    rectangle_width : float>0. The width of the rectangle between A and B. 

    Returns
    -------
    corner_limits : tuple.
                    Consisting of x0,x1,y0,y1

    '''
    # treat A as origin, calculate slope between B and A
    B_rel = B-A 
    # 'un-rotate' B and thus form a vertical rectangle easily
    try:
        theta = np.arctan(B_rel[1]/B_rel[0])
    except:
        theta = np.arctan2(B_rel[1], B_rel[0])

    theta_tobe_rotated = np.remainder(theta, np.pi/2)
    rotation_matrix = rot_mat(theta_tobe_rotated)
    B_rotated = np.dot(rotation_matrix, B_rel)
    x0, x1 = -kwargs['rectangle_width']*0.5, kwargs['rectangle_width']*0.5
    y0, y1 = 0, B_rotated[1]
    
    return([x0,x1,y0,y1], rotation_matrix)


def get_points_in_rectangle(corner_limits, startpt,
                            many_points, rotn_matrix):
    ''' Many points are checked if they are within 
    the rectangle defined by the bottom left point (x0,y0) and 
    the top right corner (x1,y1).

    Corner limits is a tuple with four entries (x0,x1,y0,y1)
    x0,x1 the x coordinates defining the width of the rectangle
    y0,y1 the y coordinates defininf the height of the rectanlge
   
    rotn_matrix is the rotation matrix to rotate the many points into 
    the same frame of reference as the rectangle. it is in the 
    form of a 2 x 2 array with the form described 
    [here](https://en.wikipedia.org/wiki/Rotation_matrix)



    '''
    x0,x1,y0,y1 = corner_limits
    relative_posns = many_points - startpt
    rotated_pts = np.apply_along_axis(dot_product_for_rows, 1, relative_posns,
                                      rotn_matrix)
    within_x = np.logical_and(rotated_pts[:,0] >= np.min([x0,x1]),
                              rotated_pts[:,0] <= np.max([x1,x0]))

    within_y = np.logical_and(rotated_pts[:,1] >= np.min([y0,y1]),
                              rotated_pts[:,1] <= np.max([y0,y1]))
    
    within_pts = np.logical_and(within_x, within_y)
    return(many_points[within_pts])
    
    

def dot_product_for_rows(xy_row, rotation_matrix):
    return(np.dot(rotation_matrix, xy_row))

def rot_mat(theta):
    rotation_matrix = np.row_stack(([np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]))
    return(rotation_matrix)




    
  
if __name__ == '__main__':
    kwargs = {'rectangle_width':0.2}
    otherpts = np.random.normal(0,5,2000).reshape(-1,2)
    #print(get_points_in_between(np.array([-1,1]), np.array([5,5]), otherpts, **kwargs ) )
    
    theta = np.deg2rad(45)
    width  = 0.6
    rotation_matrix = rot_mat(theta)
    
    