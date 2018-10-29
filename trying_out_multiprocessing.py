# -*- coding: utf-8 -*-
"""
First time trying out multiprocessing in Python

Created on Sun Oct 01 17:46:11 2017

@author: tbeleyur
"""

def combinations_with_replacement_counts(n, r):
    # code credit: https://stackoverflow.com/questions/37711817/generate-all-possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego
    size = n + r - 1
    for indices in itertools.combinations(range(size), n-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield tuple(map(operator.sub, stops, starts))
