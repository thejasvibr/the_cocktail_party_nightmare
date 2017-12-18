# -*- coding: utf-8 -*-
""" A clean approach to making Monte-Carlo simulations of
the cocktail party nightmare

Created on Tue Dec 12 21:55:48 2017

@author: tbeleyur
"""
import random
import numpy as np
import pandas as pd
from MC_echo_call_overlap import generate_calls_randomly





def pechoesheard_multipledensities(call_densities,**kwargs):
    ''' For a given set of echoes to be heard, vary the call density
    and provide the cumulative probabilities of 0,1,2..N echoes being
    heard per pulse interval.


    '''
    pechoes = []
    for call_density in call_densities:
        pechoes.append(calc_pechoesheard(call_density,**kwargs))

    pechoes_heard = pd.DataFrame(pechoes)

    return(pechoes_heard)




def calc_pechoesheard(call_density,numreplicates=10**5,**kwargs):
    '''Calculates the cumulative probabilities of 1,2,3,...N echoes
    being heard in the midst of a given number of calls in the pulse
    interval acting as maskers


    Parameters:



    Returns:

    pheard_cumulative : 1 x (Nechoes + 1) np.array
    with P_heard(0echoes), P_heard(1>= echoes),P_heard(2>= echoes)...
                                    ...P_heard(N>=echoes)

    '''
    # repeat one_density many times

    pheard_cumulative = None
    return(pheard_cumulative)


def run_one_pulse_interval():
    calls = populate_sounds()
    echoes = populate_sounds

    calculate_num_heardechoes # run this function along each row



def populate_sounds(sound_range, sound_duration,
                    sound_intensityrange, sound_arrivalangles,
                    num_sounds = 1):
    '''Creates properties for a set of sounds (echo/call) given the
    limits in kwargs. The output is considered a 'sound' DataFrame

    Parameters:

    sound_range : np.array with integer numbers of the discretised timesteps
                of the pulse interval.
    sound_duration: integer. number of time steps that the call occupies.
    sound_intensityrange : tuple with lower value in first index. minimum and maximum intensitiy that the sounds
                            arrive with.

    sound_arrivalangles : tuple with lower value in first index. minimum and maximum range of angles that sounds
                        arrive with in degrees.

    num_sounds: integer. number of sounds to generate

    Returns:

    all_sounds : pd.DataFrame with the following column names :
                start | stop | theta | intensity |

                Note : All thetas and intensity values are integer values.
    '''

    column_names = ['start','stop','theta','level']
    all_sounds = pd.DataFrame(index=range(num_sounds),columns = column_names)

    # generate the start and stop times of the sounds :
    # [0] index required to avoid sublisting behaviour of generate_calls_randomly
    start_stop = generate_calls_randomly(sound_range,sound_duration,num_sounds,
                                                                            1)[0]
    startstop_array = np.asarray(start_stop)

    all_sounds['start'] = startstop_array[:,0]
    all_sounds['stop'] = startstop_array[:,1]

    angle1,angle2 = sound_arrivalangles
    all_sounds['theta'] = np.random.random_integers(angle1,angle2,num_sounds)

    level1, level2 = sound_intensityrange
    all_sounds['level'] = np.random.random_integers(level1,level2,num_sounds)


    return(all_sounds)


def calculate_num_heardechoes(echoes,calls,temporalmasking_fn,spatialrelease_fn):
    '''Given an echo and call sound dataframes, outputs the total number of
    echoes that a bat might have heard.

    Parameters:

    echoes : pandas.DataFrame. It is a 'sound' DataFrame with >=1 echoes and 4
             columns (see 'populate_sounds' for more documentation)

    calls : pandas.DataFrame. It is a 'sound' DataFrame with >=1 calls and 4
            columns (see 'populate_sounds' for more documentation)

    temporalmasking_fn : pandas.DataFrame. The DataFrame has 2 columns:
                        |time_delay_ms|delta_dB|.

                    time_delay_ms: positive values are a forward masking regime,
                                and stand for the time gap between the end
                                of the masker and beginning of the echo.

                                Negative values are backward masking regimes
                                and stand for the time gap between the start
                                of the echo and start of the masker.

                                0 time delay is in the case of simultaneous mas-
                                king.

                    delta_dB : the difference between the intensity of the echo
                                and masker: echo_dBSPL - masker_dBSPL

    spatial_release_fn : pandas.DataFrame

    '''

    echoes_heard = []
    for echoindex,each_echo in echoes.iterrows():

        this_echoheard = False

        for callindex,each_call in calls.iterrows():


            heard = check_if_echo_heard(each_echo,each_call,temporalmasking_fn,spatialrelease_fn)
            if heard:
                this_echoheard =True

        echoes_heard.append(this_echoheard)
    num_echoes = sum(echoes_heard)
    return(num_echoes)







def check_if_echo_heard(echo,call,temporalmasking_fn,spatialrelease_fn,
                         **kwargs):
    '''
    Given a single echo and single call, check if the echo could be heard
    given their time gap and the angular separation.

    Parameters:
    echo : 1 x 4 pd.DataFrame

    call : 1 x 4 pd.DataFrame.

    temporalmasking_fn : Ntimepoints x 2 pd.DataFrame with following column
                        names :
                        |timegap_ms|delta_dB|

    spatialrelease_fn : Nthetas x 2 pd.DataFrame

    Returns :

    echo_heard : Boolean. True if echo is heard, False if it could be masked
    '''
    if not 'simtime_resoln' in kwargs.keys():
        simtime_resoln = 10**-4
    else :
        simtime_resoln = kwargs['simtime_resoln']


    time_gap = quantify_temporalmasking(echo,call)

    # if there's a lot of time gap between the echo and call then there's no problem hearing
    call_muchb4_echo = time_gap*simtime_resoln > np.max(temporalmasking_fn['timegap_ms'])
    call_muchafter_echo = time_gap*simtime_resoln < np.min(temporalmasking_fn['timegap_ms'])

    #print call_muchafter_echo.bool() , call_muchb4_echo.bool()

    try:
        if call_muchafter_echo.bool() or call_muchb4_echo.bool():
            return(True)
    except:
        if call_muchafter_echo or call_muchb4_echo:
            return(True)


    timegap_inms = time_gap*simtime_resoln
    colloc_deltadB = get_collocalised_deltadB(timegap_inms,temporalmasking_fn)

    echocall_deltadB = float(echo['level'] - call['level'])

    if echocall_deltadB > colloc_deltadB:
        return(True)

    # if the time gap between the call and echo is within the
    # temporal masking function

    angular_separation = calc_angular_separation(echo['theta'],call['theta'])
    spatial_release = calc_spatial_release(angular_separation, spatialrelease_fn)

    if echocall_deltadB > float(colloc_deltadB + spatial_release):
            return(True)
    else:
            return(False)


def quantify_temporalmasking(echo,call):
    '''
    Gives the timedelay according to the temporal masking function.

    If the values are positive, it is forward masking.
    If negative, it is backward masking.

    Note : Any forward/backward masking overlap of less than 10% of the echoes
    length is considered equivalent to simultaneous masking.

    Parameters:

    echo: a 1X4 pd.DataFrame row with the indices of the echo's location accessed with
                    'start' and 'stop' keys.

    call : Ncalls x 4 pd.DataFrame, with same columns names as echo.


    Returns:
            timegap : integer delay in integers.
    '''

    echo_range = set(range(echo['start'],echo['stop']+1))
    call_range = set(range(call['start'],call['stop']+1))

    # Check for overlap :
    overlap = echo_range & call_range

    # if the length of overlap is 90% of the whole echoes duration,
    # then treat it as a simultaneous overlap ..as '0' time gap

    if len(overlap) >= int(0.9*len(echo_range)):
        return(0)

    time_gap = which_masking(call,echo)

    return(time_gap)


def which_masking(call,echo):
    '''Decides which kind of masking is occuring and how much timegap there is
    between the call and echo.

    If the timegap is >0 then it is forward masking, if timegap<0, then it is
    backward masking, if it is 0 -then it is simultaneous masking.

    Parameters:

    call: dictionary with (at least the) following keys:
        start: simulation index of call start
        stop:  simulation index of call stop

    echo: same as above.


    Returns:

    timegap : integer. number of iterations length of forward/backward masking




    '''

    fwd_nonoverlap = (call['stop'] < echo['start']) & (call['start'] < echo['start'])

    fwd_overlap = (call['stop'] >= echo['start']) & (call['start'] < echo['start'])

    bkwd_nonoverlap = (call['start'] < echo['stop']) & (call['stop'] > echo['stop'])

    bkwd_overlap = (call['start'] > echo['start']) & (call['stop'] > echo['stop'])


    try:
        if fwd_nonoverlap.bool():
            timegap = echo['start'] - call['stop']
            return(timegap)
    except:
        if fwd_nonoverlap:
            timegap = echo['start'] - call['stop']
            return(timegap)


    try:
        if fwd_overlap.bool():
            # In the absence of data -  treat a slight forward fringe masking
            # conservatively as if it were simultaneous masking.
            timegap = 0
            return(timegap)
    except:
        if fwd_overlap:
            # In the absence of data -  treat a slight forward fringe masking
            # conservatively as if it were simultaneous masking.
            timegap = 0
            return(timegap)
    try:

        if bkwd_nonoverlap.bool() or bkwd_overlap.bool():
            timegap = echo['start'] - call['start']
            return(timegap)
    except:
        if bkwd_nonoverlap or bkwd_overlap:
            timegap = echo['start'] - call['start']
            return(timegap)




def get_collocalised_deltadB(timegap_ms, temp_mask_fn):
    '''Return the closest corresponding deltadB in the temporal masking function
    '''
    closest_indx = np.argmin(np.abs(timegap_ms-temp_mask_fn.iloc[:,0]))

    return(temp_mask_fn.iloc[closest_indx,1])



def calc_angular_separation(angle1,angle2):
    '''Calculates the minimum separation between two angles, the 'inner' angle.

    Parameters:

        angle1: float. degrees

        angle2: float. degrees

    Returns:

        diff_angle. float.

    Example:

    If there are two angles given, eg. 90deg and 350deg. The diff angle will be
    100deg and not 260 deg.

    calc_angular_separation(90,350) --> 100

    '''
    diff_angle = abs(float(angle1-angle2))
    if diff_angle > 180:
        diff_angle = 360 - diff_angle

    return(diff_angle)

def calc_spatial_release(angular_separation,spatial_release):
    '''Gives the spatial release value closest to the input

    Parameters:

    angular_separation: integer. angular separation in degrees.

    spatial_release : 2 x Nangularseparations pd.DataFrame.
                    The 0'th column has the angular separations
                    and the 1st column has the spatial release value:
                    |deltatheta|dB_release|

    Returns:

    dB_release: float. Amount of spatial release due to spatial unmasking in dB

    Example:

    let's say we have a spatial release functio with this data :
    theta  spatial release
    0      -10
    5      -20
    10     -30

    and the angular separation we give is 2 degrees.

    calc_spatial_release(2,spatial_release) --> -10

    if angular separation is 7.5 , then the function returns the first index which
    is close to 7.5.

    calc_spatial_release(3,spatial_release) --> -20


    '''
    if angular_separation > np.max(spatial_release['deltatheta']):
        return(np.min(spatial_release['dB_release']))

    else:
        closest_index = np.argmin( np.abs( spatial_release.iloc[:,0] - angular_separation )  )
        dB_release = spatial_release.iloc[closest_index,1]
        return(dB_release)



if __name__ == '__main__':

    calls = populate_sounds(np.arange(0,1000,1),3,(100,106),(0,360),5)
    echoes = populate_sounds(np.arange(0,500,1),3,(60,82),(0,360),3)
    print(calls)
    col_names = ['start','stop','theta','level']
    call = pd.DataFrame(index=[0],columns=col_names)
    echo = pd.DataFrame(index=[0],columns=col_names)

    call['start'] = 0; call['stop'] = 10
    echo['start'] = 12 ; echo['stop'] = 14
    echo['level'] = 65; call['level'] = 80

    fwdmask_nonovlp = quantify_temporalmasking(echo,call)
    print fwdmask_nonovlp
