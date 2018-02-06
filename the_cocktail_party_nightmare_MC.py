# -*- coding: utf-8 -*-
""" A clean approach to making Monte-Carlo simulations of
the cocktail party nightmare

TODO :
1) implement the effect of spatial arrangement on the cocktail party nightmare

Created on Tue Dec 12 21:55:48 2017

@author: tbeleyur
"""
import sys
folder = 'C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\2016_jamming response modelling\\analytical_modelling\\poisson-disc-master\\poisson-disc-master'
sys.path.append(folder)
import numpy as np
import pandas as pd
import scipy.misc as misc
import scipy.spatial as spl
from poisson_disc import Grid

def generate_calls_randomly(timeline,calldurn_steps,Ncalls = 1,replicates=10**5):
    '''
    Function which does a Monte Carlo simulation of call arrival in the pulse
    interval

    Inputs:

    timeline: list. range object with iteration numbers ranging from 0 to the
             the number of iterations the inter pulse interval consists of

    calldurn_steps : integer. the length of the calls which are arriving in the
              pulse interval

    Ncalls : integer. number of calls to generate per pulse interval.
    replicates: integer. number of times to generate Ncalls in the pulse interval

    Outputs:

    calls : list with sublists. The 1st sublist layer contains calls from multiple
            replicates. The 2nd layer contains the multiple calls within each replicate


    '''

    # Achtung: I actually assume a right truncated timeline here
    # because I want to ensure the full length of the call is always
    # assigned within the inter-pulse interval

    actual_timeline = timeline[:-calldurn_steps]

    multi_replicate_calls =[]

    for each_replicate in range(replicates):

        this_replicate = []

        for every_call in range(Ncalls):

            call_start = np.random.choice(actual_timeline)
            call_end = call_start + calldurn_steps -1

            if call_end > len(timeline):

                raise Exception('call_end is beyond current timeline')

            else:

               this_replicate.append([call_start,call_end])

        multi_replicate_calls.append(this_replicate)

    return(multi_replicate_calls)


def calc_pechoesheard(num_echoes_heard, total_echoes):
    '''Calculates the cumulative probabilities of 1,2,3,...N echoes
    being heard in the midst of a given number of calls in the pulse
    interval acting as maskers


    Parameters:

    num_echoes_heard : array like. Entries are the number of echoes heard.
    total_echoes : integer. Total number of echoes placed in the interpulse
                                                                    interval

    Returns:

    heard_probs : 1 x (Nechoes+1). Probability of hearing 0,1,2,..N echoes

    pheard_cumulative : 1 x (Nechoes + 1) np.array with probabilities
                        of hearing 0, <=1, <=2 ..<=N echoes.


    Example:

    numechoesheard = [0,2,3,4,2,2]

    calc_pechoesheard(numechoesheard,3) --> heardprobs, p_heardcumul

    where:
        heardprobs : array([0.16666667, 0., 0.5, 0.16666667, 0.16666667,  0.])
        p_heardcumul: array([ 0.16666667, 0.16666667, 0.66666667, 0.83333333, 1.
                                                     ,  1. ])
    '''

    try:
        occurence = np.bincount(num_echoes_heard, minlength = total_echoes+1)
    except TypeError:
        occurence = np.bincount(num_echoes_heard.astype('int64'),
                                            minlength = total_echoes+1)

    heard_probs = occurence/float(sum(occurence))

    cumulative_prob = np.cumsum(heard_probs)


    return(heard_probs, cumulative_prob)

def calc_echoes2Pgeqechoes(many_heardechoes, num_echoes, geqN):
    '''Calculates P of hearing >= Nechoes for multiple heard_echo objects
    laid row on row

    Parameters:

    many_heardechoes : ncases x Ntrials np.array. Each row contains an integer
                    number of values with the echoes heard for each trial.

    num_echoes : integer. Number of target echoes placed in the inter-pulse
                interval.

    geqN : integer. The probability of hearing at least or more than geqN echoes
            will be calculated for each row.


    Returns :

    PgeqNechoes : ncases x 1 np.array with probability of hearing at least geqN
                  for each simulation case.


    Examples :
    input_heardechoes = [ [0,1,1,2,3],
                         [1,1,1,1,1] ]
    geq = 1

    calc_multiple_PgeqNechoes(input_heardechoes,geq ) --> [0.8, 1.0]

    '''
    cols = None
    try:
        rows, cols = many_heardechoes.shape
    except ValueError:
        rows = many_heardechoes.shape[0] # for 1 row case

    PgeqNechoes = np.zeros((rows,1))

    if cols is None :
        probs, cum_prob = calc_pechoesheard(many_heardechoes, num_echoes)
        PgeqNechoes = calc_PgeqNechoes(probs,geqN)

    else :

        for rownum, each_case in enumerate(many_heardechoes):
            probs, cum_prob = calc_pechoesheard(each_case, num_echoes)
            PgeqNechoes[rownum] = calc_PgeqNechoes(probs,geqN)

    return(PgeqNechoes)









def calc_PgeqNechoes(heard_probs,geqN):
    '''Calculates probability of at least >=N echoes being heard.

    Parameters:

    heard_probs : 1 x (Nechoes+1 ) array-like. Contains probabilities of hearing
                 0,1,....Nechoes in that order.

    geqN : integer. The number of echoes or greater for which the probability
            is required.

    Returns :

    p_geqN : 0<=float<=1. Probability of hearing at least N echoes in the inter
            pulse interval.

    Example:

    calc_PgeqNechoes( [0.1, 0.05, 0.55, 0.25, 0.05] ,  2 ) --> 0.85

    '''

    if not np.isclose(sum(heard_probs),1):
        raise ValueError('The input probabilities do not sum to 1')

    p_geqN = sum(heard_probs[geqN:])

    return(p_geqN)



def populate_sounds(time_range, sound_duration,
                    sound_intensityrange, sound_arrivalangles,
                    num_sounds = 1,**kwargs):
    '''Creates properties for a set of sounds (echo/call) given the
    limits in kwargs. The output is considered a 'sound' DataFrame

    Parameters:

    time_range : np.array with integer numbers of the discretised timesteps
                of the pulse interval.

    sound_duration: integer. number of time steps that the call occupies.

    sound_intensityrange : tuple with lower value in first index. minimum and
                           maximum intensitiy that the sounds   arrive with.

                           Note : This argument can be set to None if poisson disk
                           sampling is being used to simulate spatial arrangem-
                           ent


    sound_arrivalangles : tuple with lower value in first index. minimum and maximum range of angles that sounds
                        arrive with in degrees.

                        Note : This argument can be set to None if poisson disk
                           sampling is being used to simulate spatial arrangem-
                           ent

    num_sounds: integer. number of sounds to generate

    There are optional keyword arguments to include call directionality
    with particular assumptions (please refer to the other notebooks or
    the published paper for the assumptions with which call directionality is
    implemented)

    kwargs:
        with_dirnlcall: with directional call. dictionary with the following keys.
                A : float>0. Asymmetry parameter of Giuggioli et al. 2015

        poisson_disk : implements bats as if they were placed nearly
                                uniformly in space with a Poisson disk arrange-
                                ment. The centremost point is set as the focal
                                bat's location, and the source level and recei-
                                ved angles are calculated according to the nei-
                                ghbouring points around it.

                Dictionary with 2 keys :

            'source_level' : dictionary with two keys describing
                                    emitted call  (ref function calculate_recei
                                    -vedlevels)

            'min_nbrdist' : float >0. Minimum distance maintained be-
                                        tween neighbouring bats.



    Returns:

    all_sounds : pd.DataFrame with the following column names :
                start | stop | theta | level |

                Note : All thetas and intensity values are integer values.
    '''

    column_names = ['start','stop','theta','level']
    all_sounds = pd.DataFrame(index=range(num_sounds),columns = column_names)

    # generate the start and stop times of the sounds :
    # [0] index required to avoid sublisting behaviour of generate_calls_randomly
    start_stop = generate_calls_randomly(time_range,sound_duration,num_sounds,
                                                                            1)[0]
    startstop_array = np.asarray(start_stop)

    all_sounds['start'] = startstop_array[:,0]
    all_sounds['stop'] = startstop_array[:,1]
    # LOCATION OF SCENARIO SPECIFIC SWITCHES - FOR GEOMETRY OF THE SWARM.

    if not 'poisson_disk' in kwargs.keys():
        angle1,angle2 = sound_arrivalangles
        all_sounds['theta'] = np.random.random_integers(angle1,angle2,
                                                                    num_sounds)

        level1, level2 = sound_intensityrange

        all_sounds['level'] = np.random.random_integers(level1,level2,
                                                                    num_sounds)
    else:
        min_nbrdist = kwargs['poisson_disk']['min_nbrdist']
        source_level = kwargs['poisson_disk']['source_level']

        pois_theta, pois_intensity = implement_poissondisk_spatial_arrangement(
                                num_sounds,min_nbrdist, source_level)
        all_sounds['theta'] = pois_theta
        all_sounds['level'] = np.around(pois_intensity)



    if 'with_dirnlcall' in kwargs.keys():
        implement_call_directionality(all_sounds,
                                                 kwargs['with_dirnlcall']['A'])

    return(all_sounds)

def implement_call_directionality(sound_df,A):
    '''Calculates the received level of a conspecific call
    given a set of received levels, angles

    This function assumes all bats are flying in the same direction.

    Parameters:
        sound_df: pandas dataframe with the following column names :
                |start|stop|theta|level|
                (see populate_sounds documentation for more details)

        A : float>0. Asymmetry parameter from Giuggioli et al. 2015 The higher
            this value, the more drop there is off-axis.
    Returns:
        sound_df : returns the input dataframe with altered level values post
                incorporation of call directionality factor.

    '''

    emission_angles = np.pi - np.deg2rad(sound_df['theta'])
    cd_factor = np.array([call_directionality_factor(A,em_angle)
                            for em_angle  in emission_angles])

    sound_df['level'] += cd_factor

    return(sound_df)


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


    Returns:

    num_echoes: integer. number of echoes that are heard.

    '''

    echoes_heard = []

    for echoindex,each_echo in echoes.iterrows():

        this_echoheard = False

        call_doesnotmask = []
        num_calls,ncol = calls.shape

        for callindex,each_call in calls.iterrows():

            heard = check_if_echo_heard(each_echo,each_call,temporalmasking_fn,
                                                          spatialrelease_fn)
            call_doesnotmask.append(heard)

        #The echo is heard only if all calls do not mask it
        this_echoheard = num_calls ==  sum(call_doesnotmask)

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


    #print('enter check if echo heard')
    try:
        if call_muchafter_echo.bool() or call_muchb4_echo.bool():
            return(True)
    except:
        if call_muchafter_echo or call_muchb4_echo:
            return(True)


    timegap_inms = time_gap*simtime_resoln

#    print('timegap_inms: ',timegap_inms)

    colloc_deltadB = get_collocalised_deltadB(float(timegap_inms),temporalmasking_fn)
#    print('colloc_deltadB: ',colloc_deltadB)

    echocall_deltadB = float(echo['level'] - call['level'])
#    print('echocall deltadB: ',echocall_deltadB)

    if echocall_deltadB >= colloc_deltadB:
#        print('echo heard! - > colloc')
        return(True)

    # if the time gap between the call and echo is within the
    # temporal masking function

    angular_separation = calc_angular_separation(echo['theta'],call['theta'])
    spatial_release = calc_spatial_release(angular_separation, spatialrelease_fn)
#    print('total req deltadB: ',float(colloc_deltadB + spatial_release))
    if echocall_deltadB >= float(colloc_deltadB + spatial_release):
#        print('echo heard!')
        return(True)
    else:
#        print('echo not heard!')
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

    try:
        echo_range = set(range(echo['start'],echo['stop']+1))
        call_range = set(range(call['start'],call['stop']+1))
    except :
        # because pd.iterrows() converts the whole row into a float if there
        # is even a single float value in the midst of int entries
        echo_range = set(range(int(echo['start']),int(echo['stop']+1)))
        call_range = set(range(int(call['start']),int(call['stop']+1)))

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


    Example :

    eg_call, eg_echo, where eg_call edge is 10ms from the eg_echo in a case
    of potential forward masking

    which_masking(eg_call, eg_echo) --> 100 # iterations, with time resolution
                                        of 10**-4 seconds per iteration.

    '''

    fwd_nonoverlap = (call['stop'] < echo['start']) & (call['start'] < echo['start'])
    # to do:  refactor to avoid the multiple try except clauses
    try:
        if fwd_nonoverlap.bool():
            timegap = echo['start'] - call['stop']
            return(timegap)
    except:
        if fwd_nonoverlap:
            timegap = echo['start'] - call['stop']
            return(timegap)


    fwd_overlap = (call['stop'] >= echo['start']) & (call['start'] < echo['start'])

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


    bkwd_nonoverlap = (call['start'] < echo['stop']) & (call['stop'] > echo['stop'])

    bkwd_overlap = (call['start'] > echo['start']) & (call['stop'] > echo['stop'])

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
    if angular_separation >= np.max(spatial_release['deltatheta']):
        return(np.min(spatial_release['dB_release']))

    else:
        closest_index = np.argmin( np.abs( spatial_release.iloc[:,0] - angular_separation )  )
        dB_release = spatial_release.iloc[closest_index,1]
        return(dB_release)


def run_multiple_trials(num_trials, call_densities, temporal_masking_fn,
                        spatial_release_fn, spatial_unmasking=True,**kwargs):
    '''Wrapper function which runs each call density num_trials number of times.
    See run_one_trial for more information on all parameters except num_trials


    Parameters :


    num_trials: integer. Number of replicates of run_one_trial to run

    call_densities : list with integers. Contains the call densities which need
            to be simulated in the inter-pulse interval.

    temporal_masking_fn : see run_one_trial

    spatial_release_fn : see run_one_trial

    spatial_unmasking : see run_one_trial

    **kwargs : see run_one_trial

    Returns :

    all_echoes_heard : len(call_densities)xnum_trials np.array with number of
                    echoes heard for each trial in every call density.

    '''

    rows_tmfn, cols_tmfn = temporal_masking_fn.shape
    rows_spfn, cols_spfn = spatial_release_fn.shape

    if not np.all([cols_tmfn, cols_spfn] == [2,2]) :
        raise IndexError('Number of columns in input masking or spatial \
        release functions !=2')


    all_echoes_heard = np.zeros((len(call_densities),num_trials))

    for row_num, call_density in enumerate(call_densities):

        echoes_heard = [ run_one_trial(call_density, temporal_masking_fn,
                                    spatial_release_fn, spatial_unmasking,
                                    **kwargs) for a_trial in range(num_trials)]
        all_echoes_heard[row_num,:] = echoes_heard

    return(all_echoes_heard)


def run_one_trial(call_density, temporal_masking_fn,spatial_release_fn,
                  spatial_unmasking=True,**kwargs):
    '''
    Runs one trial with/out spatial unmasking.

    Parameters:

    call_density : array-like. Entries indicate the number of calls per
                    interpulse interval

    temporal_masking_fn : Npointsx2 pd.DataFrame. The first column has the
                          the time delay in MILLISECONDS. The second column has
                          the required deltadB levels between echo and masker.

    spatial_release_fn : Npointsx2 pd.DataFrame. The first column has the
                         angular separation in degrees. The second column has
                         the amount of spatial release obtained by this angular
                         separation in dB.

    spatial_unmasking : Boolean. If False, then the second
                        spatial column of the release function is replaced by
                        zeros, to indicate no spatial unmasking.


    **kwargs:
    The list of parameters below will re-write the default values of the
    following echolocation variables.

        num_echoes : integer. number of echoes to be placed in the interpulse interval
                     Defaults to 5 echoes.

        call_level_range : 1x2 array like. the minimum and maximum values of
                           the arriving masking calls.
                           Defaults to 100-106 dB SPl re 20muPa

        echo_level_range : 1x2 array like. The minimum and maximum values of the
                            the arriving target echoes.
                            Defaults to 60-82 dB SPL re 20 muPa

        echo_arrival_angles : 1x2 array like. Minimum and maximum angles of arrival
                             of echoes.
                             Defaults to a 30 degree cone in the front between
                             75-105 degrees (0 degrees is set to 3 o'clock)
        call_arrival_angles : 1x2 array like. Minimum and maximum angles of arrival
                             of masking calls.
                             Defaults to 0,360 -  which implies calls can arrive
                             from any direction

        interpulse_interval : float. duration of the interpulse interval in
                             seconds.
                             Defaults to 0.1 secondss

        call_durn : float. duration of the call in seconds. The duration of the
                    echoes is the same as the calls.
                    Defaults to 3 milliseconds.

    On inclusion the parameters below activate additional switches such as call
    directionality and spatial arrangement of the conspecifics emitting the calls.

        with_dirnlcall : dictionary with one key:
            A : float >0. Asymmetry parameter of Giuggioli et al. 2015


        poisson_disk : dictionary with two keys (ref implement_poissondisk_spa-
                        tial_arrangement). Implements a random placement of
                        individuals in the xy plane without causing problems
                        like clustering or grid formation. Poisson disk sampling
                        is the name of the algorithm used to generate such a
                        distribution of points.



    Returns:

    num_echoesheard : 0<=integer<=num_echoes. Number of echoes that were heard
                      in this one simulation run.


    Example:

    run_one_trial(.....with all the arguments) --> 4

    '''


    num_echoes = 5
    call_level_range = (100,106)
    echo_level_range = (60,82)
    echo_arrival_angles = (75,105) # treating 0 degrees as 3'oclock
    call_arrival_angles = (0,360)

    timeresolution = 10**-4
    interpulse_interval = 0.1
    call_durn = 3*10**-3

    # if any other non-default values are assigned for the temporal and acoustic
    # parameters - re-assign them below.
    if len(kwargs.keys())>0:
        for key in kwargs.keys():
            exec('%s=%s'%(key,kwargs[key]))


    pi_timesteps = np.arange(0,int(interpulse_interval/timeresolution))

    call_steps = int(call_durn/timeresolution)


    calls = populate_sounds(pi_timesteps,call_steps,call_level_range,
                                              call_arrival_angles,call_density,
                                              **kwargs)

    # re-assign the echo start and stop points to avoid echo-echo overlap and
    #  keep them at the beggining 1/2 of the pulse interval

    echoes = populate_sounds(pi_timesteps,call_steps,echo_level_range,
                                               echo_arrival_angles,num_echoes)
    echo_starts = np.int16(np.linspace(0,interpulse_interval/2.0,5)/timeresolution)
    echo_ends = echo_starts + call_steps -1
    echoes['start'] = echo_starts
    echoes['stop'] = echo_ends

    if spatial_unmasking:
        num_echoesheard = calculate_num_heardechoes(echoes,calls,
                                                    temporal_masking_fn,
                                                    spatial_release_fn)
    # No spatial unmasking :
    else :

        spl_rel_fn = spatial_release_fn.copy()

        spl_rel_fn.iloc[:,1] = 0

        num_echoesheard = calculate_num_heardechoes(echoes,calls,
                                                    temporal_masking_fn,
                                                    spl_rel_fn)

    return(num_echoesheard)


def calculate_directionalcall_level(call_params, receiver_distance):
    '''Calculates the received level of a call according to a given emission
    angle,  on-axis source level and distane to the receiver

    Note. This calculation DOES NOT include atmospheric attenuation !
    - and is therefore an overly pessimistic calculation

    Parameters:

    call_params: dictioanry with following keys:
            A : float>0. Asymmetry parameter
            source_level: float>0. On-axis source level of a call at 1 metre in
                dB SPL re 20 microPa
            emission_angle : value between -pi and pi radians.

    receiver_distance: float>0. Distance of the receiving bat from the emitting
                    bat   in metres.

    Returns :
    received_level : float. Intensity of sound heard by the receiver in dB SPL
                    re 20 microPa
    '''
    off_axis_factor = call_directionality_factor(call_params['A'],
                                                 call_params['emission_angle'])
    directional_call_level = call_params['source_level'] + off_axis_factor
    received_level = directional_call_level -20*np.log10(receiver_distance/1.0)

    return(received_level)


def call_directionality_factor(A,theta):
    '''Calculates the drop in source level as the angle
    increases from on-axis.

    The function calculates the drop using the third term
    in equation 11 of Giuggioli et al. 2015

    Parameters:
    A : float >0. Asymmetry parameter
    theta : float. Angle at which the call directionality factor is
            to be calculated in radians. 0 radians is on-axis.
    Returns:

    call_dirn : float <=0. The amount of drop in dB which occurs when the call
                is measured off-axis.
    '''
    if A <=0 :
        raise ValueError('A should be >0 ! ')

    call_dirn = A*(np.cos(theta)-1)

    return(call_dirn)

def calc_num_times(Ntrials,p):
    '''Calculates the probability of an event (eg. at least 3 echoes heard
    in an inter-pulse interval) happening when the trial is repeated Ntrials
    times.

    Parameters:

        Ntrials : integer >0. Number of trials that are played.

        p : 0<=float<=1. probability of an event occuring.

    Returns:

    prob_occurence : pd.DataFrame with the following columns.

        num_times : Number of times that the event could occur.

        probability : Probability that an even will occur num_times
                        in Ntrials.

    '''
    if p>1 or p<0 :
        raise ValueError('Probability must be >=0 and <=1 ')

    if not Ntrials >0:
        raise ValueError('Ntrials must be >0')

    if not isinstance(Ntrials,int):
         try:
            isitnpinteger = Ntrials.dtype.kind == 'i'
            if not isitnpinteger:
                raise TypeError('Ntrials must be an integer value')
         except:
            raise TypeError('Ntrials must be an integer value')



    probability = np.zeros(Ntrials+1)
    num_times = np.arange(Ntrials+1)

    for k in num_times:
        probability[k] = misc.comb(Ntrials,k)*(p**(k))*(1-p)**(Ntrials-k)

    prob_occurence = pd.DataFrame(index = range(Ntrials+1),
                                  columns=['num_times','probability'])
    prob_occurence['num_times'] = num_times
    prob_occurence['probability'] = probability

    return(prob_occurence)





def implement_hexagonal_spatial_arrangement(Nbats,nbr_distance,source_level):
    '''Implements the fact that neigbouring bats in a group will occur at
    different distances.

    The focal bat is placed in the centre, and all neighbours are placed from
    nearest to furthest ring according to the geometry of the group. By default
    a hexagonal array arrangement is assumed, which means the focal bat has six
    neighbours in the its first ring, followed by 12 neighbours in the second
    ring, and so on.

    Parameters :

    Nbats : integer >0. number of bats to simulate as neighbours

    nbr_distance : float>0. Distance to the nearest neighbour in the first ring
                    in metres.

    source_level : dictionary with two keys :
            'intensity' : float. Sound pressure level at which a bat calls ref
                         ref 20 muPa.
            'ref_distamce' : float>0. distance in meters at which the call
                            intensity has been measured at.



    Returns :

    RL : 1x Nrings np.array. received level in dB SPL. The calculations are
        done *without*  any assumptions of atmospheric absorption - and only
        considering spherical spreading. Where Nrings is the maximum number of
        rings required to fill up Nbats in the hexagonal array.

    num_calls : 1x Nrings np.array .number of calls at each received level that
                arrive at the focal bat (without call directionality
                implemented).    See above for description of Nrings.


    Example :

    implement
    '''

    ring_nums, bat_nums = fillup_hexagonalrings(Nbats)

    ring_distances = ring_nums * nbr_distance

    RL_fromrings = calculate_receivedlevels(ring_distances, source_level)

    return(RL_fromrings, bat_nums)





def fillup_hexagonalrings(numbats):
    ''' Fills up a given number bats in a group around a focal bat.

    All the bats are placed within the centres of regular hexagonal
    cells. The first ring refers to the immediate neighbours around the focal
    bat. The second ring refers to the second concentric ring of cells centred
    around the focal bat and so on.

    All rings close to the bat need to be filled up before neighbouring bats
    are added to the outer rings.


    Parameters:

    numbats : integer >0. The total number of neighbouring bats being simulated


    Returns

    occupied_rings : np.array with ring numbers starting from 1.

    bats_in_eachring : number of bats in each ring.



    Example :

    fillup_hexagonalrings(10) --> np.array([1,2]) , np.array([6,4])

    fillup_hexagonalrings(23) --> np.array([1,2,3]), np.array(6,12,5)

    '''
    if numbats <= 0 :
        raise ValueError('Number of neighbouring bats must be >0')

    n = np.arange(1,11)
    bats_per_ring = 6*n
    cumsum_bats = np.cumsum(bats_per_ring)

    if numbats <= 6:

        return(np.array(1),np.array(numbats))

    outer_ring = np.argwhere(numbats<=cumsum_bats)[0] + 1

    inner_rings = outer_ring -1

    if inner_rings >1:
        inner_numbats = 6*np.arange(1,inner_rings+1)
    else:
        inner_numbats = 6

    bats_outermostring = numbats - np.sum(inner_numbats)

    bats_in_eachring = np.append(inner_numbats,bats_outermostring)
    occupied_rings = np.arange(1,outer_ring+1)

    return(occupied_rings,bats_in_eachring)



def calculate_receivedlevels(nbr_distances,source_level):
    '''Calculates received levels at the focal bat given a source level value
    at which neighbours are all calling at. All received levels are calculated
    assuming only spherical spreading and no atmospheric absorption.

    Parameters:

    nbr_distance : np.array. nearest neighbour distances in meters.

    source_level : dictionary with two keys :
            'intensity' : float. Sound pressure level at which a bat calls ref
                         ref 20 muPa.
            'ref_distance' : float>0. distance in meters at which the call
                            intensity has been measured at.
    Returns :

    received_levels : 1x nbr_distance.size np.array. Intensity at which
                focal bat hears the calls of the neighbouring bats.

    '''
    if not type(nbr_distances) is np.ndarray:
        nbr_distances = np.asanyarray(nbr_distances).reshape(1,-1)

    try:
        received_levels = np.apply_along_axis(calc_RL,0,nbr_distances,
                                          source_level['intensity'],
                                            source_level['ref_distance'])
    except:
        received_levels = np.apply_along_axis(calc_RL,0,
                                        nbr_distances.reshape(1,-1),
                                          source_level['intensity'],
                                            source_level['ref_distance'])



    return(received_levels)

def calc_RL(distance, SL, ref_dist):
    '''calculates received level only because of spherical spreading.

    Parameters

    distance : float>0. receiver distance from source in metres.

    SL : float. source level in dB SPL re 20 muPa.

    ref_dist : float >0. distance at which source level was measured in metres.


    Returns:

    RL : received level in dB SPL re 20muPa.

    '''

    if sum( np.array([distance,ref_dist]) <= 0.0):
        raise ValueError('distances cannot be <= 0 !')

    RL = SL - 20*np.log10(distance/ref_dist)

    return(RL)



def implement_poissondisk_spatial_arrangement(numbats,nbr_distance,
                                                                 source_level):
    '''Distributes points/bats over space randomly with a minimum distance
    between points through the Poisson disk sampling algorithm of Robert Brids
    -on. Based on the distances to the focal bat the received level is calcula-
    ted. The angle of arrival is also calculated.

    The Poisson disk generation is implemented through code from IHautI
    https://github.com/IHautaI/poisson-disc

    Parameters:

    numbats : integer>0. Number of bats in the group, excluding the focal bat

    nbr_distance : float>0. Minimum distance between points in the group in met
                    -res

    source_level : dictionary with two keys :
            'intensity' : float. Sound pressure level at which a bat calls ref
                         ref 20 muPa.
            'ref_distance' : float>0. distance in meters at which the call
                            intensity has been measured at.


    Returns:

    sounds_thetaintensity : pd.DataFrame with the following column names :
                | theta | intensity |
                These columns can then be replaced in the original
                pd.DataFrame which additionally has the start and stop times
                of the different calls.
    '''

    nbr_points, centre_pt = generate_surroundpoints_w_poissondisksampling(
                                                        numbats, nbr_distance)

    radial_dist, thetas  = calculate_r_theta(nbr_points,centre_pt)

    sounds_intensity = np.apply_along_axis(
                            calculate_receivedlevels,1, radial_dist.reshape(-1,1),
                                                                source_level)
    return(thetas, sounds_intensity)



def generate_surroundpoints_w_poissondisksampling(npoints, nbr_distance):
    '''Generates a set of npoints+1 roughly equally placed points using the
    Poisson disk sampling algorithm. The point closest to the centroid is
    considered the centremost point. The closest points to the centremost point
    are then chosen.

    Parameters:

    npoints: integer. Number of neighbouring points around the focal point

    nbr_distance : float>0. Minimum distance between adjacent points.

    Returns:

    nearby_points : npoints x 2 np.array. XY coordinates of neighbouring points
                    around the centremost point.

    centremost_point : 1 x 2 np.array. XY coordinates of centremost point.


    '''

    if nbr_distance <= 0.0 :
        raise ValueError('nbr_distance cannot be < 0')

    if npoints < 1:
         raise ValueError('Number of neighbouring points must  be >=1 ')

    insufficient_points = True
    sidelength = 1.5
    while insufficient_points :


        length, width = sidelength, sidelength
        grid = Grid(nbr_distance, length, width)

        data = grid.poisson((length,width))
        data_np = np.asanyarray(data)

        rows, columns = data_np.shape
        if rows <= npoints:
            sidelength += 0.5
        else :
            insufficient_points = False


    centremost_pt = choose_centremostpoint(data_np)
    centremost_index = find_rowindex(data_np, centremost_pt)

    nearby_points = find_nearbypoints(data_np, centremost_index, npoints)

    return(nearby_points, centremost_pt)


def calculate_r_theta(target_points, focal_point):
    '''Calculates radial distance and angle from a set of target points to the
    focal point.

    The angle calculated is the arrival angle for the focal point.

    The focal_point is considered to be moving with a 90 degree direction by
    default.


    Parameters:

    target_points: Npoints x 2 np.array. XY coordinates of the target points.

    focal_point : 1 x 2 np.array. XY coordinates of the focal point.


    Returns:

    radial_distances : Npoints x 1 np.array. The radial distance between each
                        of the target points and the focal point.

    arrival_angles : Npoints x 1 np.array. The angle of arrival at which a call
                    emitted from target points will arrive at the focal bat.

                    The angles are in degrees. Angles that are negative, imply
                    that the target point is on the left side of the focal poin
                    -t.


    Example:

    target_points = np.array([ [1,1],
                               [-1,-1]
                                    ])
    focal_point = np.array([0,0])


    rad_dists, angles_arrival = calculate_r_theta(target_points, focal_point)

    rad_dists --> np.array([1.4142, 1.4142])
    angles_arrival --> np.array([45,-45])

    '''

    radial_distances = np.apply_along_axis(calculate_radialdistance,1,
                                           target_points,
                                           focal_point)

    arrival_angles = np.apply_along_axis(calculate_angleofarrival,1,
                                                             target_points,
                                                             focal_point)

    return(radial_distances, arrival_angles)



def calculate_radialdistance(sourcepoint, focalpoint):
    '''

    Parameters:

    sourcepoint, focalpoint : 1 x Ndimensions np.arraya with coordinates

    Returns:

    radialdist : float > 0. radial distance between sourcepoint and focalpoint

    '''
    radialdist = spl.distance.euclidean(sourcepoint, focalpoint)

    return(radialdist)




def calculate_angleofarrival(sourcepoint_xy,focalpoint_xy,focalpoint_orientation=90.0):
    '''Calculates the relative angle of arrival if a neighbouring bat were to
    call.

    Function based on https://tinyurl.com/y9qpq84l - thanks MK83


    Parameters :

    sourcepoint_xy : 1x2 np.array. xy coordinates of call emitting bat

    focalpoint_xy : 1x2 np.array. xy coordinates of focal bat

    focalpoint_orientation : float. heading of focal bat. 0 degrees means the
                            focal bat is facing 3 o'clock. Angles increase in
                            anti-clockwise fashion
                            Default is 90 degrees.

    Returns :

    angle_ofarrival : float. Angle of arrivals are between 0 and +/- 180 degrees
                    Positive angles of arrival imply the emitted sound will be
                    received on the right side, while negative angles of arrival
                    mean the sound is received on the left side.

    Example :

    source_xy = np.array([1,1])
    focal_xy = np.array([0,0])

    calculate_angleofarrival(source_xy, focal_xy) --> +45.0
    '''

    radagntorntn=np.radians(float(focalpoint_orientation))


    p0= np.array([np.cos( radagntorntn ), np.sin( radagntorntn) ])+ focalpoint_xy
    p1= focalpoint_xy
    v0 =  p0-p1

    agentvects=(sourcepoint_xy-p1).flatten()


    relangle = np.rad2deg( np.math.atan2( np.linalg.det([agentvects,v0]),
                                                    np.dot(agentvects,v0) ) )
    relangle=np.around(relangle,4)

    return( relangle )



def find_rowindex(multirow_array, target_array):
    '''Given a multi-row array and a target 2D array which is one of the rows
    from the multi-row array - gets the row index of the target array.

    Parameters:

    multirow_array : N x 2 np.array.

    target_array : 1 x 2 np.array.

    Returns:

    row_index : integer. row index of the target_array within the multirow_array
    '''
    values_sqerror = multirow_array**2 - target_array**2
    row_error = np.sum(values_sqerror,axis=1)
    try:
        row_index = int(np.where(row_error==0)[0])
        return(row_index)
    except:
        raise ValueError('Multiple matching points - please check inputs')




def choose_centremostpoint(points):
    '''Select the point at the centre of a set of points. This is done by calc-
    ulating the centroid of a set of points and checking which of the given
    points are closest to the centroid.

    If there are multiple points at equal distance to the centroid, then
    one of them is assigned arbitrarily.


    Parameters:

    points : Npoints x 2 np.array. With X and Y coordinates of points

    Returns:

    centremost_point : 1 x 2 np.array.

    '''
    centroid_point = calc_centroid(points)
    centremost_point = find_closestpoint(points, centroid_point)

    return(centremost_point)



def calc_centroid(data):
    '''
    based on code thanks to Retozi, https://tinyurl.com/ybumhquf
    '''
    try:
        if data.shape[1] != 2:
            raise ValueError('Input data must be a 2 column numpy array')
    except:
        raise ValueError('Input data must be a 2 column numpy array')


    centroids = np.apply_along_axis(np.mean,0,data)
    return(centroids)

def find_closestpoint(all_points, target_point):
    '''Given a target point and a set of generated points,
    outputs the point closest to the target point.


    Example:
    target_point = np.array([0,0])
    generated_points = np.array([0,1],[2,3],[3,10])

    find_closestpointindex(generated_points, target_point) --> np.array([0,1])

    '''
    distance2tgt = np.apply_along_axis(calc_distance,1,all_points,target_point)
    closestpointindx = np.argmin(distance2tgt)
    closestpoint = all_points[closestpointindx,:]

    return(closestpoint)


calc_distance = lambda point1, point2 : spl.distance.euclidean(point1,point2)

def find_nearbypoints(all_points, focalpoint_index, numnearpoints):
    '''Given a set of points choose a fixed number of them that are closest
    to a focal point among them.

    Parameters:

    all_points : Npoints x 2 np.array. All generated points.

    focalpoint_index : integer. Row index from all_points of the focal point.

    numnearpoints : integer. Number of neighbouring points that must be chosen.

    Returns:

    nearbypoints : numnearpoints x 2 np.array.



    Example :
    all_points = np.array([ [0,0],[0,1],[1,0],[2,2],[3,5] ])
    focalpoint_index = 0
    numnearpoints = 3




    find_nearestpoints(all_points, focalpoint_index, numnearpoints) -->
        np.array([ [0,1],[1,0],[2,2] ])

    '''

    numrows, numcols = all_points.shape
    if numnearpoints >  numrows-1 :
        raise ValueError('The number of neighbours requested is more than \
            the number of points given!! ')

    if not focalpoint_index in range(numrows):
        raise IndexError('The given focalpoint index is not within the range\
            of the array!')

    validpoints = np.delete(all_points,focalpoint_index,0)
    focal_point = all_points[focalpoint_index,:]
    nearbypoints_dist = np.apply_along_axis(calc_distance,1,validpoints,
                                                                focal_point)

    nearbypoints_indices = nearbypoints_dist.argsort()[:numnearpoints]


    nearbypoints = validpoints[nearbypoints_indices,:]

    return(nearbypoints)



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

    temp_masking = pd.DataFrame(index=range(50),columns=['timegap_ms','dB'])
    temp_masking['timegap_ms'] = np.linspace(10,-1,50)
    temp_masking['dB'] = np.linspace(-30,10,50)
    spl_release = temp_masking.copy()
    spl_release.columns = ['deltatheta','dB_release']

    num = run_one_trial(5,temp_masking,spl_release,False)

    call_ds = np.array([10,20,40])
    num_replicates = 50
    p_leq3= {'wsum':[],'wosum':[]}

    for calldens in call_ds:
        w_sum = [ run_one_trial(calldens,temp_masking,spl_release,True,echo_level_range=(80,92)) for k in range(num_replicates)]
        wo_sum = [ run_one_trial(calldens,temp_masking,spl_release,False,echo_level_range=(80,92)) for k in range(num_replicates)]
        probs,cum_probs = calc_pechoesheard(w_sum,5)
        probs_wo, cum_probs_wo = calc_pechoesheard(wo_sum,5)

        p_leq3['wsum'].append(sum(probs[1:]))
        p_leq3['wosum'].append(sum(probs_wo[1:]))

    p_leq3