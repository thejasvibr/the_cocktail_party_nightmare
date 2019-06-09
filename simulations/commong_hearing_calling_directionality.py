# -*- coding: utf-8 -*-
"""Create common hearing and calling direcitonality functions
Created on Sat Jun 08 19:52:45 2019

@author: tbeleyur
"""
import numpy as np 

def call_directionality_fn(X,A=7):
    cd_factor = A*(np.cos(np.deg2rad(X))-1)
    return(cd_factor)

def hearing_directionality_fn(X, B=2):
    hearing_factor = B*(np.cos(np.deg2rad(X))-1)
    return(hearing_factor)

