# -*- coding: utf-8 -*-
""" Effect of group size 
Created on Mon Jun 10 13:55:43 2019

@author: tbeleyur
"""


import glob
import multiprocessing as mp

from run_simulations import run_multiple_simulations
    

# run simulations across all parameter sets present in the folder :
parameter_sets = glob.glob('*.paramset')

all_cpus = mp.cpu_count()

print('Starting processing...')
success = run_multiple_simulations('Group Size', 'Group Size simulations with shadowing included', 
                             '*.paramset',
                             all_cpus)
