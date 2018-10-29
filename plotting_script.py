# -*- coding: utf-8 -*-
"""Common plotting script
Created on Thu Jan 11 13:26:08 2018

@author: tbeleyur
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv('results/pgeq3_2018-01-10-20-37-19_10000replicates.csv')

treatment_names = {'only_TM':'Only Temporal Masking',
            'with_CallDirn':'Call Directionality',
            'with_SUm':'Spatial Unmasking',
            'with_SUm_CallDirn':'Call Directionality and Spatial Unmasking'}

plt.figure(1,figsize=(12,8))

for column in data.iloc[:,:-1]:
    plt.plot(data['call_density'],data[column],'*-',
                                             label=treatment_names[column])

plt.legend(title='Cocktail Party Scenarios', prop={'size':13})
plt.xlabel('Number of calls in Inter Pulse Interval', size=25)
plt.ylabel('$P(3 \geq echoes\ heard)$', size=25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.grid()
plt.show()