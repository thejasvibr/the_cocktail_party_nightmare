# -*- coding: utf-8 -*-
""" A clean approach to making Monte-Carlo simulations of
the cocktail party nightmare

TODO :
1) implement the effect of spatial arrangement on the cocktail party nightmare
2) hearing directionality
3) include switch to make each bat call in a different direction
4) calculate 2dary echoes 
5) many calculations assume all bats fly in the same direction - fix this assumption
   by including a switch to allow different flight directions

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


class MissingKeyword(ValueError):
    '''raised when there's a missing keywork argument
    '''

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


def calculate_num_heardechoes(echoes,calls,temporalmasking_fn,spatialrelease_fn,**kwargs):
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

    Keyword arguments:
    
        one_hot : boolean. If True then the echoes that are heard are 
                  output as an 1 x Nechoes one-hot binary array. 


    Returns:

    num_echoes: integer. number of echoes that are heard.

    heardechoes_id : (if one_hot is True) 1x Nechoes one-hot binary np.array.
                  eg. for a 5 echo situation in which echoes 2,3 are heard, 
                  the output will be --> np.array([0,1,1,0,0])

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

    if 'one_hot' in kwargs.keys():
        heardechoes_id = np.array(echoes_heard).astype('int')
        return(num_echoes, heardechoes_id)
    else:
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

    onehot_echoesheard : (based on keyword argument one_hot) num_calldensities x Nechoes x num_trials np.array. 
                        with one-hot encoding of which echoes were heard.

    '''

    rows_tmfn, cols_tmfn = temporal_masking_fn.shape
    rows_spfn, cols_spfn = spatial_release_fn.shape

    if not np.all([cols_tmfn, cols_spfn] == [2,2]) :
        raise IndexError('Number of columns in input masking or spatial release functions !=2')


    all_echoes_heard = np.zeros((len(call_densities),num_trials))
    output_number_and_id = 'one_hot' in kwargs.keys() and kwargs['one_hot']
    if output_number_and_id:
        all_echo_ids = []

    for row_num, call_density in enumerate(call_densities):

        echoes_heard = [ run_one_trial(call_density, temporal_masking_fn,
                                    spatial_release_fn, spatial_unmasking,
                                    **kwargs) for a_trial in xrange(num_trials)]
        if not output_number_and_id:
            all_echoes_heard[row_num,:] = echoes_heard
        else:
            # get the total number of echoes heard
            all_echoes_heard[row_num,:] = map(extract_numechoesheard, echoes_heard)

            num_echoes = echoes_heard[0][1].size
            onehot_echoesheard = np.zeros((num_trials,num_echoes))            
            # get the identity of the echoes heard
            for i, echoespertrial in enumerate(echoes_heard):
                onehot_echoesheard[i,:] = extract_echoids(echoespertrial)
            all_echo_ids.append(onehot_echoesheard)
        
            # assemble the identities of echoes heard across various call densities
            multidensity_echoids = assemble_echoids(all_echo_ids, call_densities,
                                                          num_echoes, num_trials)        
    if output_number_and_id:
        return(all_echoes_heard, multidensity_echoids)
    else:
        return(all_echoes_heard)

# helper functions to separate out tuples 
extract_numechoesheard = lambda X : X[0]

extract_echoids = lambda X : X[1]

def assemble_echoids(echoids_per_calldensity, call_densities, num_echoes,
                           num_trials):
    '''reshapes multiple 2D onehot echoids in a list into a 3D array
    '''
    multidensity_echoids = np.zeros((len(call_densities),num_trials,num_echoes))
    for i, echoids_at_calldensity in enumerate(echoids_per_calldensity):
        multidensity_echoids[i,:,:] = echoids_at_calldensity
    return(multidensity_echoids)
        


def run_one_trial(call_density, temporal_masking_fn,spatial_release_fn,
                  spatial_unmasking=True,**kwargs):
    '''
    Places conspecific maskers into an interpulse interval with echos and calc-
    ulates the number of echoes heard. The complexity of the model can be twea-
    ked by changing the various parameters and biological phenomena included.

    Parameters:

    call_density : integer. Entries indicate the number of calls per
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

        one_hot : boolean. See calculate_num_heardechoes.

        echoes_within : 1x2 array-like. Interval of the interpulse interval within 
                        which the echoes are uniformly placed. 
                        The values should lie between 0 and interpulse interval
                        length. Defaults to [0, ipi-0.004]. 


    Returns:

    num_echoesheard : 0<=integer<=num_echoes. Number of echoes that were heard
                      in this one simulation run.
    
    id_echoesheard : 1 x num_echoes binary np.array. Output is conditional on
                     keyword argument one_hot==True. See documentation in 
                     calculate_num_heardechoes  for more. 


    Example:

    run_one_trial(.....with all the arguments) --> 4

    '''


    num_echoes = 5
    call_level_range = (100,106)
    echo_level_range = (60,82)
    echo_arrival_angles = (60,120) # treating 0 degrees as 3'oclock
    call_arrival_angles = (0,360)

    timeresolution = 10**-4
    interpulse_interval = 0.1
    call_durn = 3*10**-3
    echoes_within = [0, interpulse_interval-0.004]


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

    echo_timewindow = np.linspace(echoes_within[0],echoes_within[1],num_echoes)
    echo_starts = np.int16(echo_timewindow/timeresolution)
    echo_ends = echo_starts + call_steps -1
    echoes['start'] = echo_starts
    echoes['stop'] = echo_ends

    if spatial_unmasking:
        echoesheard = calculate_num_heardechoes(echoes,calls,
                                                    temporal_masking_fn,
                                                    spatial_release_fn,**kwargs)
    # No spatial unmasking :
    else :

        spl_rel_fn = spatial_release_fn.copy()

        spl_rel_fn.iloc[:,1] = 0

        echoesheard = calculate_num_heardechoes(echoes,calls,
                                                    temporal_masking_fn,
                                                    spl_rel_fn, **kwargs)
    return(echoesheard)


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
    grid_size = 2*(nbr_distance/np.sqrt(2))

    sidelength = (np.ceil(np.sqrt(npoints))+0.1)*grid_size

    while insufficient_points :
        length, width = sidelength, sidelength
        grid = Grid(nbr_distance, length, width)

        data = grid.poisson((0,0))
        data_np = np.asanyarray(data)

        rows, columns = data_np.shape
        if rows <= npoints:
            sidelength += 0.5
        else :
            insufficient_points = False

    centremost_pt = choose_centremostpoint(data_np)
    centremost_index = find_rowindex(data_np, centremost_pt)

    nearby_points = find_nearbypoints(data_np, centremost_index, npoints-1)

    return(nearby_points, centremost_pt)


def calculate_r_theta(target_points, focal_point):
    '''Calculates radial distance and angle from a set of target points to the
    focal point.

    TODO : accept heading orientation other than 90 degrees
    
        

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
        raise ValueError('The number of neighbours requested is more than the number of points given!! ')

    if not focalpoint_index in xrange(numrows):
        raise IndexError('The given focalpoint index is not within the range of the array!')

    validpoints = np.delete(all_points,focalpoint_index,0)
    focal_point = all_points[focalpoint_index,:]
    nearbypoints_dist = np.apply_along_axis(calc_distance,1,validpoints,
                                                                focal_point)

    nearbypoints_indices = nearbypoints_dist.argsort()[:numnearpoints]


    nearbypoints = validpoints[nearbypoints_indices,:]

    return(nearbypoints)


def make_focal_first(xy_posns, focal_xy):
    '''shifts the focal_xy to the 1st row
    '''
    row_index = find_rowindex(xy_posns, focal_xy)
    focal_first = np.roll(xy_posns, np.tile(-row_index, 2))
    return(focal_first)

def propagate_sounds(sound_type, **kwargs):
    '''Propagates a sound and calculates the received levels and angles 
    
    Parameters:
        sound_type : str. defines which kind of sound is being propagated. 
                     The valid entries are either '2ndary_echoes' or 'conspecific_calls'

    Keyword Arguments:
        focal_bat
        bats_xy
        bats_orientations
        source_level
        call_directionality,
        hearing_directionality,
        reflection_strength 

    Returns:
        received_sounds : pd.DataFrame . A 'sound' DataFrame - see populate_sounds

    '''
  
    sound_type_propagation = {'2ndary_echoes': calculate_2ndaryecho_levels,
                               'conspecific_calls' : calculate_conspecificcall_levels}

    try:
        received_sounds = sound_type_propagation[sound_type](**kwargs)
        return(received_sounds)
    except:
        print(sound_type)
        

def calculate_conspecificcall_levels(**kwargs):
    '''Calculates the received levels and angles of reception of conspecific calls

     Keyword Arguments:
        focal_bat
        bats_xy
        bats_orientations
        reflection_function
        hearing_directionality
        call_directionality
        source_level
    
    Returns:

        conspecific_calls : pd.DataFrame. A 'sound' df with received level, angle of arrival
                                       and other related attributes of the sound : 
                                       start | stop | theta | level |
    '''
    if kwargs['bats_xy'].shape[0] < 2:
        raise ValueError('Çonspecific calls cannot propagated for Nbats < 2')
    else:
        try:
            hearing_directionality = kwargs['hearing_directionality']
            call_directionality = kwargs['call_directionality']
            source_level = kwargs['source_level']
            focal_bat = kwargs['focal_bat']
            bats_xy = kwargs['bats_xy']
        except:
            pass
            
        conspecific_call_paths = calculate_conspecificcall_paths(**kwargs)
        conspecific_calls = calculate_conspecificcallreceived_levels(conspecific_call_paths,
                                                                     call_directionality,
                                                                     hearing_directionality,
                                                                     source_level)
        
        return(conspecific_calls)

def calculate_2ndaryecho_levels(**kwargs):
    ''' Calculates the received levels and angle of reception of 2ndary echoes from a conspecific call
    bouncing off conspecifics once and reaching the focal bat


    Keyword Arguments:
        focal_bat
        bats_xy
        bats_orientations
        reflection_function
        hearing_directionality
        call_directionality
        source_level
    
    Returns:

        secondary_echoes : pd.DataFrame. A 'sound' df with received level, angle of arrival
                                       and other related attributes of the sound : 
                                       start | stop | theta | level |
       

    '''
    # There will be secondary echoes only when there are >= 3 bats
    if kwargs['bats_xy'].shape[0] >= 3:
        try:
            reflection_function = kwargs['reflection_function']
            hearing_directionality = kwargs['hearing_directionality']
            call_directionality = kwargs['call_directionality']
            source_level = kwargs['source_level']
            # calculate the distances and angles involved in the secondary echo paths
            secondary_echopaths = calculate_2ndary_echopaths(**kwargs)
            
            # calculate the sound pressure levels based on the geometry + emission-reception directionalities        
            secondary_echoes = calculate_2ndaryechoreceived_levels(secondary_echopaths,
                                                                       reflection_function,
                                                                       call_directionality,
                                                                       hearing_directionality,
                                                                       source_level)
        except:
            pass

    else:
        # other wise return an empty sound df. 
        secondary_echoes = pd.DataFrame(data=[], index=[0],
                                   columns=['start','stop','theta','level'])
    return(secondary_echoes)

def calculate_conspecificcallreceived_levels(conspecificcall_paths,
                                             call_directionality,
                                             hearing_directionality,
                                             source_level):
    '''Calculates the final sound pressure levels at the focal receiver bat
    of a conspecific call emitted by a group member. 

    Parameters:

        bats_xy :
        conspecificall_paths :  dictionary with the following keys:
                call_routes
                R_incoming
                theta_emission
                theta_reception

        call_directionality : function with one input. The call_direcitonality
                              function accepts one input theta_emission, which 
                              is the relative angle of emission with reference to the
                              on-axis heading angle of the bat.

        hearing_directionality : function with one input. The hearing_directionality
                                 function acccepts one input theta_reception, 
                                 which is the relative angle of sound reception 
                                 with reference to the on-axis heading angle of 
                                 the bat. 

        source_level : dictionary with two keys:
                        dBSPL : float. the sound pressure level in dB SPL relative to 20 microPascals
                        ref_distance : float >0. the reference distance at which the bat's source
                                level is specified at. 
    
    Returns :

        conspecific_calls : pd.DataFrame.  A 'sound' df with received level, angle of arrival
                                       and other related attributes of the sound : 
                                       start | stop | theta | level |
        
    '''
    num_calls = len(conspecificcall_paths['call_routes'])
    conspecific_calls = pd.DataFrame(data=[], index=xrange(num_calls),
                                     columns=['start','stop','theta', 'level'])
    for call_id, each_callpath in enumerate(conspecificcall_paths['call_routes']):
        # get the reception angle and the received SPL
        conspecific_calls['theta'][call_id] = conspecificcall_paths['theta_reception'][call_id]
        # outgoing SPL at emitter after call directionality 
        outgoing_SPL = source_level['dBSPL'] + call_directionality(conspecificcall_paths['theta_emission'][call_id])
        # SPL at focal bat after hearing directionality
        incoming_SPL = calc_RL(conspecificcall_paths['R_incoming'][call_id],
                               outgoing_SPL, source_level['ref_distance']) +hearing_directionality(conspecificcall_paths['theta_reception'][call_id])
        conspecific_calls['level'][call_id] = incoming_SPL

    return(conspecific_calls)
    
    

def calculate_2ndaryechoreceived_levels(secondary_echopaths,
                                        reflection_function,
                                        call_directionality,
                                        hearing_directionality,
                                        source_level):
    '''Calculates the final sound pressure levels at the focal receiver bat given
    the geometry of the problem. 

    Parameters:

        secondary_echopaths : dictionary with geometry related entries. 
                              See calculate_2ndary_echopaths for details.

        reflection_function : pd.DataFrame with the following columns:
            
                    ref_distance : float>0. Distance at which the reflection strength is calculated in metres.

                    incoming_theta : float. Angle at which the incoming sound arrives at. This 
                                     angle is with relation to the heading direciton of the target bat. 

                    outgoing_theta : float. Angle at which the outgoing sound reflects at. This 
                                     angle is with relation to the heading direciton of the target bat. 

                    reflection_strength : float. The ratio of incoming and outgoing sound pressure levels in dB (20log10)

                       see get_reflection_strength for Details.

        call_directionality : function with one input. The call_direcitonality
                              function accepts one input theta_emission, which 
                              is the relative angle of emission with reference to the
                              on-axis heading angle of the bat.

        hearing_directionality : function with one input. The hearing_directionality
                                 function acccepts one input theta_reception, 
                                 which is the relative angle of sound reception 
                                 with reference to the on-axis heading angle of 
                                 the bat. 

        source_level : dictionary with two keys:
                        dBSPL : float. the sound pressure level in dB SPL relative to 20 microPascals
                        ref_distance : float >0. the reference distance at which the bat's source
                                level is specified at. 

    Returns:

        secondaryechoes : pd.DataFrame. A 'sound' df with received level, angle of arrival
                                       and other related attributes of the sound : 
                                       start | stop | theta | level |

    '''
    reflection_ref_distance = np.unique(reflection_function['ref_distance'])

    num_echoes = len(secondary_echopaths['sound_routes'])
    secondaryechoes = pd.DataFrame(data=[], index=range(num_echoes),
                                   columns=['start','stop','theta','level'])
    # for each secondary echo
    for echo_id, echo_route in enumerate(secondary_echopaths['sound_routes']):
        
        R_incoming = secondary_echopaths['R_incoming'][echo_id]
        R_outgoing = secondary_echopaths['R_outgoing'][echo_id]
        theta_emission = secondary_echopaths['theta_emission'][echo_id]
        theta_reception = secondary_echopaths['theta_reception'][echo_id]
        theta_incoming = secondary_echopaths['theta_incoming'][echo_id]
        theta_outgoing = secondary_echopaths['theta_outgoing'][echo_id]
        
        # calculate soundpressure level at the reference radius around the target.
        emitted_SPL = source_level['dBSPL'] + call_directionality(theta_emission)

        
        # calculate the incoming SPL at the reference distance of the 
        if R_incoming < reflection_ref_distance:
            raise ValueError('The distance is less than the reference distance \
                             unable to calculate incoming SPL')
        else:
            effective_incoming_R = R_incoming - reflection_ref_distance
            incoming_SPL = calc_RL(effective_incoming_R, emitted_SPL,
                                   source_level['ref_distance'])

        # calculate the reflection strength given the incoming and outgoing angle 
        reflection_strength = get_reflection_strength(reflection_function,
                                                      theta_incoming,
                                                      theta_outgoing)

        # calculate the received level:
        if R_outgoing <= reflection_ref_distance:
            raise ValueError('The outgoing distance is less than the reference distance \
                             unable to calculate outgoing SPL')
        else:
            outgoing_SPL = incoming_SPL + reflection_strength
            received_level = calc_RL(R_outgoing, outgoing_SPL, reflection_ref_distance) + hearing_directionality(theta_reception)

        secondaryechoes['theta'][echo_id] = theta_reception
        secondaryechoes['level'][echo_id] = received_level

    return(secondaryechoes)


def get_reflection_strength(reflection_function, 
                            theta_in,
                            theta_out,
                            max_theta_error = 30):
    '''The reflection_function is a mapping between the reflection strength
    and the input + output angles of the sound. 
    
    In simulatoins, the incoming and outgoing angles will likely vary continuously.
    However, experimental measurements of course cannot handle all possible values. 
    This function tries to choose the angle pair which matches the
    unmeasured theta incoming and outgoing to best possible extent. 

    Parameters:

        reflection_function : pd.DataFrame with the following columns:
            
                    ref_distance : float>0. Distance at which the reflection strength is calculated in metres.

                    incoming_theta : float. Angle at which the incoming sound arrives at. This 
                                     angle is with relation to the heading direciton of the target bat. 

                    outgoing_theta : float. Angle at which the outgoing sound reflects at. This 
                                     angle is with relation to the heading direciton of the target bat. 

                    reflection_strength : float. The ratio of incoming and outgoing sound pressure levels in dB (20log10)

                    For example, one row entry could be :
                                ref_distance | incoming_theta | outgoing_theta | reflection_strength 
                                      0.1    |       30       |       90       |       -60  
            
                              The above example refers to a situation where sound
                              arrives at the object at 30 degrees and its reflection
                              is received at 90 degrees. The outgoing sound id 60 dB fainter than the 
                              incoming sound when measured at a 10 cm radius around the centre
                              of the target object. 

    Returns:

        reflection_strength : float <=0. The ratio of incoming and outgoing sound pressure levels in dB

    

    Example Usage : 
        get_reflection_strength(reflection_funciton, 90, 0) --> -50

    '''
    
    # get best fit angle pair:
    theta_diffs = np.apply_along_axis(angle_difference, 1,
                                      np.array(reflection_function[['incoming_theta','outgoing_theta']]),
                                      theta_in, theta_out)

    if np.min(abs(theta_diffs)) > max_theta_error:
        print(theta_in, theta_out, np.min(abs(theta_diffs)))
        raise ValueError('Reflection function is coarser than ' + str(max_theta_error)+'..aborting calculation' )
    else:
        best_index = np.argmin(abs(theta_diffs))
        return(reflection_function['reflection_strength'][best_index])

def angle_difference(df_row, theta_in, theta_out):
        angle_diff = spl.distance.euclidean([df_row[0],df_row[1]],
                     [theta_in, theta_out])
        return(angle_diff)
    


def calculate_conspecificcall_paths(**kwargs):
    '''Given the positions and orientations of all bat, the output is 
    the distances and angles relevant to the geometry of conspecific call
    propagation. 

    The straight line distances are calculated without assuming any occlusion
    by conspecifics. 

    Keyword Arguments:

        focal_bat :
        bats_xy
        bats_orientations

    Returns:
        conspecificall_paths : dictionary with the following keys:
            call_routes
            R_incoming
            theta_emission
            theta_reception
    '''
    bats_xy = kwargs['bats_xy']
    bats_orientations = kwargs['bats_orientations']
    distance_matrix = spl.distance_matrix(bats_xy,bats_xy)
    focal_bat = find_rowindex(bats_xy, kwargs['focal_bat'])

    # make all emitter-target and target-receiver paths using the row indices as 
    # an identifier
    emitters = set(range(bats_xy.shape[0])) - set([focal_bat])

    conspecificcall_routes = []
    
    # generate all possible conspecific call routes 
    for each_emitter in emitters:
        emitter_focal = (each_emitter, focal_bat)
        conspecificcall_routes.append(emitter_focal)

    conspecificall_paths = {}
    conspecificall_paths['call_routes'] = tuple(conspecificcall_routes)
    conspecificall_paths['R_incoming'] = get_conspecificcall_Rin(distance_matrix, conspecificcall_routes)
    conspecificall_paths['theta_emission'], conspecificall_paths['theta_reception'] = calc_conspecificcall_thetas(bats_xy,
                                                                                                                 bats_orientations,
                                                                                                                 conspecificcall_routes)
    return(conspecificall_paths)

def calculate_2ndary_echopaths(**kwargs):
    '''Given the positions and orientations of all bats, the output is the
    distances, angles and secondary echo routes required to calculate the
    received levels at the focal bat.

    For Nbats the total number of secondary echoes a focal bat will hear is:
        N_emitters x N_targets -->  (Nbats-1) x (Nbats-2)
    
    Keyword Arguments:
        focal_bat : 1x2 np.array. xy position of focal bat. This is the end point of all the 2ndary
                    paths that are calculated

        bats_xy : Nbats x 2 np.array. xy positions of bats 

        bats_orientations : np.array. heading directions of all bats. This direction is with reference to 
                            a global zero which is set at 3 o'clock. 

    Returns:

        secondary_echopaths : dictionary with following 7 keys - each with 
                    sound_routes|theta_emission|R_incoming|theta_incoming|theta_outgoing|R_outgoing|theta_reception

                    All theta values are in degrees. 
                    theta_emission, theta_incoming, theta_outgoing and theta_reception are calculated
                    with the heading direction of the bat as the zero. 

                    R_incoming and R_outgoing are in metres. 
    '''
    bats_xy = kwargs['bats_xy']
    bats_orientations = kwargs['bats_orientations']

    focal_bat = find_rowindex(bats_xy, kwargs['focal_bat'])

    # make all emitter-target and target-receiver paths using the row indices as 
    # an identifier
    emitters = set(range(bats_xy.shape[0])) - set([focal_bat])
    targets = set(range(bats_xy.shape[0])) - set([focal_bat])

    secondary_echo_routes = []
    for an_emitter in emitters:
        for a_target in targets:
            if not a_target is an_emitter:
                emitter_target_focal = (an_emitter, a_target, focal_bat)
                secondary_echo_routes.append(emitter_target_focal)

    distance_matrix = spl.distance_matrix(bats_xy, bats_xy)

    secondary_echo_paths = {}
    secondary_echo_paths['R_incoming'], secondary_echo_paths['R_outgoing'] = calc_R_in_out(distance_matrix,
                                                                                           secondary_echo_routes)
    secondary_echo_paths['theta_incoming'], secondary_echo_paths['theta_outgoing'] = calc_2daryecho_thetas(bats_xy,
                                                                                                           bats_orientations,
                                                                                                           secondary_echo_routes,
                                                                                                           'incoming_outgoing')
    secondary_echo_paths['theta_emission'], secondary_echo_paths['theta_reception'] = calc_2daryecho_thetas(bats_xy,
                                                                                                            bats_orientations,
                                                                                                            secondary_echo_routes,
                                                                                                            'emission_reception')
    secondary_echo_paths['sound_routes'] = tuple(secondary_echo_routes)
    return(secondary_echo_paths)

def calc_R_in_out(distance_matrix, sound_routes):
    '''Calculates the direct path length that a sound needs to travel between
    an emitter bat to target bat (R_in) and target bat to focal receiver bat (R_out)
    
    Parameters:

        distance_matrix : Nbats x Nbats np.array. Distance between each of the bats in metres. 

        sound_routes : list with (Nbats-1)x(Nbats-2) tuples of 3 entries. Each tuple refers to the path 
                        of a call emission. 
                        Eg. (20, 10, 5) refers to a sound emitted by bat 20 onto bat 10 and then 
                        reflecting and reaching bat 5. 

    Returns:

        R_in : tuple with (Nbats-1)x(Nbats-2) floats. Distances of sound on its inbound journey from emitter to target
        R_in : tuple with (Nbats-1)x(Nbats-2) floats. Distances of sound on its outbound journey from target to receiver
    
    '''
    R_in = [distance_matrix[each_route[0],each_route[1]] for each_route in sound_routes]
    R_out = [distance_matrix[each_route[1],each_route[2]] for each_route in sound_routes]
    return(tuple(R_in), tuple(R_out))
    
def  get_conspecificcall_Rin(distance_matrix, conspecificcall_routes):
    '''calculate one-way radial distane between an emitter and focal bat.
    '''
    R_emitter_focal = []
    for route in conspecificcall_routes:
        emitter, focal = route
        R_emitter_focal.append(distance_matrix[emitter,focal])

    return(tuple(R_emitter_focal))

def calc_conspecificcall_thetas(bats_xy, bat_orientations, conspecificcall_routes):
    '''Calculates the relative angle of conspecific call emission and reception. 

    Parameters:

        bats_xy

        bat_orientations

        conspecificcall_routes

    Returns:
        theta_emission : tuple with Nbats-1 abs(floats) <= 180 . Relative angle of call emission with reference to the
                         heading direction of the calling bat. -ve angles are to the left and +ve are to the right

        theta_reception : tuple with Nbats-1 abs(floats) <= 180 . RElative angle of call reception with reference to the
                         heading direction of the focal bat. -ve angles are to the left and +ve are to the right
    '''
    theta_emission = []
    theta_reception = []

    for each_callroute in conspecificcall_routes:
        emitter, focal = each_callroute
        theta_emitter = calculate_angleofarrival(bats_xy[focal], bats_xy[emitter], bat_orientations[emitter])
        theta_emission.append(theta_emitter)

        theta_receiver = calculate_angleofarrival(bats_xy[emitter], bats_xy[focal], bat_orientations[focal])
        theta_reception.append(theta_receiver)

    return(tuple(theta_emission), tuple(theta_reception))


def calc_2daryecho_thetas(bats_xy, bat_orientations, sound_routes, which_angles):
    '''Calculates relative  angles of sound with reference to the heading direction
    of the bats concerned. 
    
    The switch 'which_angles' decides if the theta_ingoing and theta_outgoing 
    or theta_emission and theta_reception pairs are calculated.

    Parameters:

        bats_xy : Nbats x 2 np.array. XY positions of bats. 

        bat_orientations : Nbats x 1 np.array. Heading directions of all bats in degrees. 
                           Zero degrees is 3 o'clock and increases in anti clockwise manner
        
        sound_routes : list w (Nbats-1)x(Nbats-2) tuples.  Each tuple has the index number of 
                       the emitter, target and focal receiver bat. 
                       
        which_angles : str. Switch which decides which pair of angles are calculated. 
                       
            If which_angles is 'emission_reception'  then :
                       
                       theta_towardstarget is the relative angle of emission angle with reference to 
                       the heading direction of the emitter bat
                       AND 
                       theta_fromtarget is the relative angle of reception with reference to the heading 
                       direction of the focal bat 
            If which_angles is 'incoming_outgoing' then the angles are calculated with reference ot 
                        the incoming call and outgoing echo on the surface of the target bat. 

                     theta_towardstarget is the relative angle of emitted call arrival wrt to the 
                     heading direction of the target bat
                     AND
                     theta_fromtarget is the relative angle of the reflected sound wrt to the 
                     heading direction of the target bat
                       

    Returns:

        theta_towardstarget : tuple with Nbats-1 x Nbats-2 floats. Angle of incoming sound with reference to heading direction
                         of target bat or emitting bat (see which_angles). Heading direction is zero and it increases off-axis till +/- 180 degrees.

        theta_fromtarget : tuple with  Nbats-1 x Nbats-2 floats. Angle of outgoing sound with reference to heading direction
                         of target bat or focal bat (see which_angles). Heading direction is zero and it increases off-axis till +/- 180 degrees.
    '''
    # towardstarget refers to the emission angle OR to the incoming angle
    theta_towardstarget = []
    theta_fromtarget = []

    for route in sound_routes:
        emitter, target, focal = route
        if which_angles is 'incoming_outgoing':
            theta_in = calculate_angleofarrival(bats_xy[emitter], bats_xy[target], bat_orientations[target])
            theta_towardstarget.append(theta_in)
            theta_out = calculate_angleofarrival(bats_xy[focal], bats_xy[target], bat_orientations[target])
            theta_fromtarget.append(theta_out)
        elif which_angles is 'emission_reception':
            theta_in = calculate_angleofarrival(bats_xy[target], bats_xy[emitter], bat_orientations[emitter])
            theta_towardstarget.append(theta_in)
            theta_out = calculate_angleofarrival(bats_xy[target], bats_xy[focal], bat_orientations[focal])
            theta_fromtarget.append(theta_out)

    return(tuple(theta_towardstarget), tuple(theta_fromtarget))

def implement_hearing_directionality(arrival_angle, hearing_dirnlty):
    '''
    '''
    
    pass

def combine_sounds(sounddf_list):
    '''Combine 2>= sound dfs into one df. 
    EAch sound df is expected to at least have the following columns:
        start : start time of 
        stop : 
        level : 
        theta : 

    '''
    combined_sounds = pd.concat(sounddf_list, ignore_index=True)
    return(combined_sounds)

def place_bats_inspace(**kwargs):
    ''' Assign bats their positions and heading directions in 2D
    The positions of the bats are placed randomly so that 
    they are at least a fixed distance away from their nearest neighbours.
    This is achieved through Poisson disk sampling.
    

    Keyword Arguments:
        Nbats : integer >0. Number of bats in the group

        min_spacing : float> 0. Minimum distance between neighbouring bats in metres.

        heading_variation : float >0. Range of heading directions in degrees.
                            eg. if heading_variation = 10, 
                            then all bats will have a uniform prob. of 
                            having headings between [90-10, 90+10] degrees.

    Returns : 
        bat_xy : list with [Nbats x 2 np.array, focal bat XY]. XY positions of nearby and focal bats

        headings : 1x Nbats np.array. Heading direction of bats in degrees.
       
    '''
    min_heading, max_heading = 90 - kwargs['heading_variation'], 90 + kwargs['heading_variation']
    headings = np.random.choice(np.arange(min_heading, max_heading+1),
                                kwargs['Nbats'])
    
    nearby, focal = generate_surroundpoints_w_poissondisksampling(kwargs['Nbats'],
                                                                   kwargs['min_spacing'])

    return([nearby, focal], headings)


def run_CPN(**kwargs):
    ''' Run one iteration of the spatially explicit CPN, and calculate the
    number of echoes heard.

    This version of the simulation places bats in space, calculates the 
    received conspecific call levels and the 2dary echoes that may arrive at a
    focal bat. 

    
    Keyword Arguments:

            Nbats : integer. Number of bats in the group.

            min_spacing : float>0. Minimum distance between one bat and its closest neighbour

            heading_variation : float>0. Variation in heading direction of each bat in degrees. 
                            The reference direction is 90 degrees (3 o'clock is 0 degrees), and 
                            all bat headings are within +/- heading_variation of 90 degrees. 
                            This is also assumed to be the direction in which the bat is calling and hearing at.
                            Example  : a heading variation of 10 degrees will mean that all bats in the group
                            are facing between 100-80 degrees with uniform probability.

           source_level : dictionary with the following keys and entries:
                          'dB_SPL' : float. Sound pressure level in dB with 20microPascals as reference
                          'ref_distance' : float. Reference distance in metres. 

           call_directionality : function. Relates the decrease in source level in dB with emission angle
                          The on-axis angle is set to 0 here, and the angle right behind the bat is 180 degrees. 

           hearing_directionality : function. Relates the change in received level in dB with sound arrival angle
                          The heading angle is set to 0 here, and the angle right behind the bat is 180 degrees. 

           reflection_function : function. Describes the reflection characteristics of echoes bouncing off
                               bats. This is different from the target_strength because the position of
                               reception and position of emission are different. 

           N_echoes : integer >0. The number of target echoes to be placed in the interpulse interval

           echo_properties : pd.DatFrame. A 'sound'df -- see previous documentation

           temporal_masking_fn : ?? . temporal masking funciton

           spatial_release_fn : ?? . spatial release function. 

    Returns:

        num_echoes_heard : Number of echoes heard        

    '''
    # place Nbats out 
    bat_positions, bats_orientations = place_bats_inspace(**kwargs)
    nearby_bats, focal_bat = bat_positions

    bats_xy = np.row_stack((focal_bat, nearby_bats))

    kwargs['bats_xy'] = bats_xy
    kwargs['bats_orientations'] = bats_orientations
    kwargs['focal_bat'] = focal_bat

    print(kwargs.keys())
    # calculate the received levels and angles of arrivals of the sounds
    conspecific_calls = propagate_sounds('conspecific_calls', **kwargs)
    secondary_echoes = propagate_sounds('2ndary_echoes', **kwargs)

    # place the conspecific calls and 2dary echoes in the IPI
    calls_and_2daryechoes = combine_sounds([conspecific_calls, secondary_echoes])
#
#    # place target echoes in the IPI and check how many of them are heard
#    target_echoes = generate_echoes(Nechoes, echo_properties)
#
#    num_echoes_heard = calculate_num_heardechoes(target_echoes, calls_and_2daryechoes,
#                              temporal_masking_fn,
#                              spatial_release_fn)
#    return(num_echoes_heard)
    return(calls_and_2daryechoes)
