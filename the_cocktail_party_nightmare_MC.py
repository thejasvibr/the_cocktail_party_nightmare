# -*- coding: utf-8 -*-
""" A clean approach to making Monte-Carlo simulations of
the cocktail party nightmare

TODO :
1) implement the effect of spatial arrangement on the cocktail party nightmare

Created on Tue Dec 12 21:55:48 2017

@author: tbeleyur
"""
import numpy as np
import pandas as pd

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
    sound_intensityrange : tuple with lower value in first index. minimum and maximum intensitiy that the sounds
                            arrive with.

    sound_arrivalangles : tuple with lower value in first index. minimum and maximum range of angles that sounds
                        arrive with in degrees.

    num_sounds: integer. number of sounds to generate

    There are optional keyword arguments to include call directionality
    with particular assumptions (please refer to the other notebooks or
    the published paper for the assumptions with which call directionality is
    implemented)

    kwargs:
        with_dirnlcall: with directional call. dictionary with the following keys.
                A : float>0. Asymmetry parameter of Giuggioli et al. 2015



    Returns:

    all_sounds : pd.DataFrame with the following column names :
                start | stop | theta | intensity |

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

    angle1,angle2 = sound_arrivalangles
    all_sounds['theta'] = np.random.random_integers(angle1,angle2,num_sounds)

    level1, level2 = sound_intensityrange

    all_sounds['level'] = np.random.random_integers(level1,level2,num_sounds)

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


    try:
        if call_muchafter_echo.bool() or call_muchb4_echo.bool():
            return(True)
    except:
        if call_muchafter_echo or call_muchb4_echo:
            return(True)


    timegap_inms = time_gap*simtime_resoln
    colloc_deltadB = get_collocalised_deltadB(timegap_inms,temporalmasking_fn)

    echocall_deltadB = float(echo['level'] - call['level'])

    if echocall_deltadB >= colloc_deltadB:
        return(True)

    # if the time gap between the call and echo is within the
    # temporal masking function

    angular_separation = calc_angular_separation(echo['theta'],call['theta'])
    spatial_release = calc_spatial_release(angular_separation, spatialrelease_fn)

    if echocall_deltadB >= float(colloc_deltadB + spatial_release):
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



    Returns:

    num_echoesheard : 0<=integer<=num_echoes. Number of echoes that were heard in this one
                      simulation run.


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

    # assign the non-default values to the variable names in the kwargs
    if len(kwargs.keys())>0:
        for key in kwargs.keys():
            exec('%s=%s'%(key,kwargs[key]))


    pi_timesteps = np.arange(0,int(interpulse_interval/timeresolution))

    call_steps = int(call_durn/timeresolution)


    calls = populate_sounds(pi_timesteps,call_steps,call_level_range,
                                              call_arrival_angles,call_density,
                                              **kwargs)

    # re-assign the echo start and stop points to avoid echo-echo overlap and keep them at the beggining 1.2
    # of the pulse interval
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
    else :

        # with NO spatial release
        spl_rel_fn = spatial_release_fn.copy()

        spl_rel_fn.iloc[:,1] = 0

        num_echoesheard = calculate_num_heardechoes(echoes,calls,
                                                    temporal_masking_fn,
                                                    spl_rel_fn)
#    print('echoes',echoes)
#    print('calls',calls)
    return(num_echoesheard)


def calculate_directionalcall_level(call_params, receiver_distance):
    '''Calculates the received level of a call according to a given emission angle,
    on-axis source level and distane to the receiver

    Note. This calculation DOES NOT include atmospheric attenuation !
    - and is therefore an overly pessimistic calculation

    Parameters:

    call_params: dictioanry with following keys:
            A : float>0. Asymmetry parameter
            source_level: float>0. On-axis source level of a call at 1 metre in dB SPL re 20 microPa
            emission_angle : value between -pi and pi radians.

    receiver_distance: float>0. Distance of the receiving bat from the emitting bat
                in metres.

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