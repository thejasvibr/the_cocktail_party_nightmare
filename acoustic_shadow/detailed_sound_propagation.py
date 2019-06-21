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

        

    Parameters
    ----------

    start_point : 1 x 2 array like

    end_point :  1 x 2 array like

    all_other_points : Nbats-2 x 2 array like
                       xy coordinates of all points between start and end point
                       in a rectangular area of given width 
      
    Keyword Arguments
    ----------------
    implement_shadowing : Boolean. If True then shadowing calculations are done, else only simple spherical spreading. 

    R : float. straight line distance between soure and receiver

    rectangle_width : float >0. width of the rectangle 

    shadow_TS : list/array with >=1 values
                The bistatic target strengtha at emitter-receiver separation of 180 degrres. 

    emitted_source_level : dictionary with key 'dBSPL' and 'ref_distance' ,
                    indicating source level in dB SPL re 20muPa and reference distance in metres. 
    
    R : float >0. straight line distance between source and receiver. This is used
        in case there are no obstacles between them. 
   


    Returns
    -------
    received_level : float. 
                    received level of the sound 
    '''    
    
    #print(start_point,end_point)
    if kwargs['implement_shadowing']:
        all_points_between = get_points_in_between(start_point, end_point, 
                                                       all_other_points, **kwargs)
        if all_points_between.shape[0] >= 1 :
            #print('PING!!!', all_points_between)
            point2point_dists = get_distances_between_points(all_points_between, start_point,
                                                                 end_point)
            received_level = calculate_RL_w_shadowing(point2point_dists, 
                                                      kwargs['emitted_source_level']['dBSPL'],
                                                         kwargs['shadow_TS'])
            #print(received_level)
        else:
            #print('MIAAAAAOW')
            received_level = calc_RL(kwargs['R'],kwargs['emitted_source_level']['dBSPL'],
                                             kwargs['emitted_source_level']['ref_distance'])
           
    else:
        #print('BOWWWW')
        received_level = calc_RL(kwargs['R'],kwargs['emitted_source_level']['dBSPL'],
                                             kwargs['emitted_source_level']['ref_distance'])
        
    return(received_level) 



def get_distances_between_points(xy_between, start, end):
    '''
    
    '''
    all_xy =  np.row_stack((start, xy_between, end))
    distance_matrix = spatial.distance_matrix(all_xy, all_xy)
    distance_to_source = np.argsort(distance_matrix[:,0])
    
    points_sorted = all_xy[distance_to_source,:]
    distances_sorted = spatial.distance_matrix(points_sorted, points_sorted)

    num_distances = all_xy.shape[0] - 1
    point_2_point_distances = np.zeros(num_distances)
    for i, (point0, point1) in enumerate(zip(range(all_xy.shape[0]-1),
                                         range(1,all_xy.shape[0]))):
        point_2_point_distances[i] = distances_sorted[point0, point1]
    return(point_2_point_distances)
        
    

    

def calculate_RL_w_shadowing(distances, SL=100, shadow_TS=[-9]):
    '''Calculates received level of a call with acoustic shadowing included. 
    The received level of the call with shadowing is calculated with an iterative 
    application of the bistatic sonar equation. The TS used here is the bistatic 
    target strength at emitter-receiver angular separations of 180 degrees. 
    
    Parameters
    ----------
    distances : array-like, >= 3 entries. 
               The first entry is distance between emitter and closest target, followed by distance between 1st target
               and second target etc. The sum of all distances is the receiver-emitter distance

    SL : float.
         source level at 1metre distance in dB SPl re 20 muPa

    TS : array-like. 
         Bistatic target strengths of object at 180 degrees separation between emitter and receiver. 
         If multiple target strengths then a random choice is made within these values for each obstacle.
         This mimics each object having a different TS because of its orientation or position. 

    Returns
    -------
    RL_w_shadowing : float. 
                    Received level of sound after accounting for multiple object blocking the direct path. 
  
    '''
    all_distance_pairs = []
    for i, dist in enumerate(distances[:-1]):
        if i == 0:
            distance_pair = (dist,distances[i+1])
        else:
            distance_pair = ( np.sum(all_distance_pairs[-1]),distances[i+1])
        all_distance_pairs.append(distance_pair)
 
    SL_1m = SL
    for (r0,r1) in  all_distance_pairs:
        RL_apparent = SL_1m - 20*np.log10(r0) - 20*np.log10(r1) + np.random.choice(shadow_TS,1) # received level at the target
        SL_1m = RL_apparent - 20*np.log10(1.0/(r0+r1)) # equivalent source level at 1metre
    return(RL_apparent)


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
    theta = np.arctan2(B_rel[1], B_rel[0])

    #theta_tobe_rotated = np.remainder(theta, np.pi/2)
    theta_tobe_rotated = np.pi/2.0 - theta
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
    kwargs = {'rectangle_width':0.2, 'implement_shadowing':True,
              'emitted_source_level': {'dBSPL':90, 'ref_distance':1.0}}
    kwargs['shadow_TS'] = [-15]
    #otherpts = np.random.normal(0,5,2000).reshape(-1,2)
    x_coods = np.array([0.5])#np.random.choice(np.arange(-width*0.25, width*0.25, 0.01),10)
    y_coods = np.array([0.05])
    otherpts = np.column_stack((x_coods, y_coods))
    #otherpts = np.array(([1,0],[1,0.05]))
    #print(get_points_in_between(np.array([2,0]), np.array([0,0]), otherpts, **kwargs ) )

    soundprop_w_acoustic_shadowing(np.array([2,0]), np.array([0,0]), otherpts,
                                   **kwargs)

#    kwargs['bats_xy'] = np.array(([0,0],[1,0],[2,0]))
#    kwargs['focal_bat'] = np.array([0,0])
#    kwargs['implement_shadowing'] = True
#    kwargs['rectangle_width'] = 0.3
#    kwargs['shadow_TS'] = [-9]

    