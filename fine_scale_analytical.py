# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:30:24 2017

@author: tbeleyur
"""
import numpy as np
import matplotlib.pyplot as plt

ipi = 0.1
echo_durn = 0.003

timeres = np.array([echo_durn,10**-3,10**-4,10**-5,10**-6])

E = np.int64(echo_durn/timeres)

N = np.int64(ipi/timeres)

p_echoheard = (N - (2*E+1.0) ) / N

print(p_echoheard)


plt.plot(timeres,p_echoheard,'*-');
plt.ylim(0,1);plt.xlim(np.max(timeres),np.min(timeres))
plt.ylabel('P $(1\ echo\ heard |\ 1\ call)$');
plt.xlabel('Time resolution of analysis (seconds)'+'\n interpulse interval '+str(ipi)+' seconds',fontsize=10);

fwdmask_period = 0.003
fwdmask = np.int64(fwdmask_period/timeres)
p_echoheard_fwdmask = (N - (2*E+1.0 + fwdmask) ) / N

bkwdmask_period = 0.0024
bkwdmask = np.int64(bkwdmask_period/timeres)

p_echoheard_fwdmask = (N - (2*E+1.0 + fwdmask) ) / N
p_echoheard_bkwdfwdmask = (N - (2*E+1.0 + fwdmask+bkwdmask) ) / N

plt.plot(timeres,p_echoheard_fwdmask,'*-')
plt.plot(timeres,p_echoheard_bkwdfwdmask,'*-')
plt.xscale('log')
print('nomask',p_echoheard)
print('fwdmask',p_echoheard_fwdmask)
print('fwdbkwdmask',p_echoheard_bkwdfwdmask)