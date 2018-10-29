# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 13:23:53 2018

@author: tbeleyur
"""

import numpy as np
import matplotlib.pyplot as plt

theta=np.linspace(0,2*np.pi,100)
A = 7.3
direction = (A+2)*(np.cos(theta)-1)


plt.figure(1,figsize=(10,8))
ax = plt.subplot(111,projection='polar')
ax.plot(theta,direction,'*-')
ax.set_rticks(np.arange(-24,6,6))
ax.set_theta_zero_location('N')
ax.set_rlabel_position(180)