# -*- coding: utf-8 -*-
"""Parallelising calculations of
the cocktail party nightmare
Created on Wed Jan 10 17:18:01 2018

@author: tbeleyur
"""

import multiprocessing as mp
from multiprocessing import Pool
import time
import copy
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
import pandas as pd
from the_cocktail_party_nightmare_MC import run_multiple_trials, calc_echoes2Pgeqechoes

st = time.time()
print('1',time.time()-st)
# load the temporal-masking function
tempmasking_fn = pd.read_csv('data//temporal_masking_fn.csv').iloc[:,1:]
print('loaded tempmasking')
# load the spatial-release function
spatialrelease_fn = pd.read_csv('data//spatial_release_fn.csv').iloc[:,1:]

# set up simulation parameters
numtrials = 1000
#call_densities = 5*np.arange(1,8)
call_densities = np.insert( np.arange(1,7)*5,0,1)
echorange = (60,82)
callrange = (100,106)

asym_param = 7
call_directionality = {'A':asym_param}


print('2',time.time()-st)
scenario_dict = {
    'only_TM': lambda : run_multiple_trials(numtrials, call_densities, tempmasking_fn,
                                   spatialrelease_fn, spatial_unmasking=False,
                                   echo_level_range = echorange,
                                   call_level_range = callrange),

    'with_SUm': lambda : run_multiple_trials(numtrials, call_densities, tempmasking_fn,
                                    spatialrelease_fn, spatial_unmasking=True,
                                    echo_level_range = echorange,
                                    call_level_range = callrange),

    'with_CallDirn': lambda : run_multiple_trials(numtrials, call_densities,
                                         tempmasking_fn, spatialrelease_fn,
                                         spatial_unmasking=False,
                                         echo_level_range = echorange,
                                         with_dirnlcall = call_directionality,
                                         call_level_range = callrange),

    'with_SUm_CallDirn': lambda : run_multiple_trials(numtrials, call_densities,
                                             tempmasking_fn, spatialrelease_fn,
                                             spatial_unmasking=True,
                                             echo_level_range = echorange,
                                        with_dirnlcall = call_directionality,
                                                  call_level_range = callrange)
                }
print('function dictionary set ')

def run_each_scenario(scenario_name):
    print(scenario_name)
    heardechoes = scenario_dict[scenario_name]()
    return(heardechoes,scenario_name)
print('3',time.time()-st)


if __name__== '__main__':
    print('started....')
    scenarios_2bcalculated = ['only_TM','with_SUm','with_CallDirn',
                                                      'with_SUm_CallDirn']
    print('Calculations started')

#    start = time.time()
#    serial_echoesheard  = map(run_each_scenario, scenarios_2bcalculated)
#    print('serial :', time.time()-start)

    print('parallel starting:')
    start_pll = time.time()
    pool = Pool(processes=4)
    all_heardechoes = pool.map(run_each_scenario,scenarios_2bcalculated)
    print('parallel time: ',time.time()-start_pll)

    pgeq3 = []
    for each_scenario,scenario_name in all_heardechoes:
        pgeq3.append((calc_echoes2Pgeqechoes(each_scenario,5,3),
                                   scenario_name))


    column_names = copy.deepcopy(scenarios_2bcalculated)
    column_names.append('call_density')

    pgeq3_data = pd.DataFrame(index= range(call_densities.size),
                                         columns = column_names)
    pgeq3_data['call_density'] = call_densities

    for each_scenariodata in pgeq3:
        each_scenario, scenario_name = each_scenariodata
        pgeq3_data[scenario_name] = each_scenario

        plt.plot(call_densities,each_scenario,'*-',label=scenario_name)

    plt.legend(); plt.show()

    timestamp = dt.datetime.now()
    fmtd_timestamp = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    file_name = 'results/pgeq3_'+fmtd_timestamp+'_'+str(numtrials)+'replicates.csv'
    pgeq3_data.to_csv(file_name,index=False)





