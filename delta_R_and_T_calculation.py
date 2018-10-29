# -*- coding: utf-8 -*-
"""
Trying to see how to calculate the
Delta R for bat ears !

Created on Wed Oct 25 12:07:24 2017

@author: tbeleyur
"""
import numpy as np
import matplotlib.pyplot as  plt



def calc_polar_distance (p1,p2):
    '''
    Calculates distance between 2 points in
    a polar coordinate space

    Inputs:
        p1 & p2 : each is a tuple with 2 entries: radius and theta.
            radius: float. radial distance of the points
            theta : 0<= float < 2pi. Angle in radians.
    Outputs:
        distance: float. distance between the two points

    '''
    r1, theta1 = p1
    r2, theta2 = p2

    sumsquare_radii  = r1**2 + r2**2
    joint_product = 2*r1*r2*np.cos(theta1-theta2)

    distance = np.sqrt( sumsquare_radii - joint_product   )

    return (distance)

def calc_ITD(sound_angle,R1 = 1, ear_dist = 0.05,vsound=330.0):
    '''
    calculates inter-aural time difference given
    the spherical coordinates of the sound source
    from the right ear.

    The coordinate origin is loacted on the right ear.
    The location of coordinate system origin is irrelevant as such because
    we're only interested in calculating the difference of radial distances
    the sound source travels to reach the two ears.

    Keeping the origin on the right ear is convenient because then
    the left ear automatically has a known position in spherical coordinates:
            ( inter_ear_distance, pi )




    Inputs:

        R1 : float. radial distance from right ear
        angle: 0<=float<2pi. angle in radians
        ear_dist: float. inter-ear distance in meters.

    Outputs:
        delta_time : float. ITD in seconds

    '''

    leftear_coods = (ear_dist,np.pi)
    sound_source = (R1,sound_angle)

    R2 = calc_polar_distance(leftear_coods,sound_source) # radial distance
    # from sound source to right ear

    deltaR = R2 - R1

    ITD = deltaR / vsound

    return(ITD)






if __name__ == '__main__':

    angles_deg = np.linspace(0,120,50)
    sound_angles = np.deg2rad(  angles_deg  )
    r_dist = 1.0;
    inter_mic_dist = 0.05

    itds =  [ calc_ITD(angles,r_dist,inter_mic_dist) for angles in sound_angles]
    itds_np = np.array(itds)

    plt.figure(555)
    plt.plot(angles_deg,itds_np*10**6,'*-')
    plt.ylabel('ITD, microseconds')
    plt.xlabel('Angle, degrees \n' + '0 degrees corresponds to sound on right ear side')








