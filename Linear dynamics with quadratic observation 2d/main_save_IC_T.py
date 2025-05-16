#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 18:18:00 2025

@author: jarrah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:12:58 2025

@author: jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from SIR import SIR
from EnKF import EnKF
from OTF_save_IC import OTF_param

plt.close('all')

# Choose h(x) here, the observation rule
def h(x):
    # return x[0].reshape(1,-1)*x[0].reshape(1,-1)
    return x*x

def A(x,t=0):
    return F @ x




def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    x0 = x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

    
    for i in range(N-1):
        x[i+1,:] = A(x[i,:])  + sai[i,:] 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
        
        # x[i+1,] = np.ones_like(x[i+1,]) + sai[i,:] 
        # y[i+1,] = h(x[i+1,]) + eta[i+1,]
        
    return x,y

#%%   
n = 1
L = n*2 # number of states
tau = 1e-1 # timpe step 
T = tau*10 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = L # number of states observed

# dynmaical system
# H = np.array([[1,0]]) 
# H = np.eye(1,dy)
alpha = 0.9
a = alpha
b = np.sqrt(1-alpha**2)
c = alpha

F = np.array([[a, b],[-b,c]]) 
# F = np.array([[a, b, 0, 0],[-b,c,0,0],[0,0,a,b],[0,0,-b,c]]) 

F = np.kron(np.eye(int(n)), F)
# F = np.eye(2*n)*0.96


noise = np.sqrt(1e-2) # noise level std
sigmma = noise # Noise in the hidden state
sigmma0 = 1#5*noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1#/noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]

J = int(1e4/1) # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*2) #128
parameters['BATCH_SIZE'] = int(64/1) #64
parameters['LearningRate'] = 1e-3*1 # 1e-3
parameters['ITERATION'] = int(1024*8/1) # 1e4
parameters['Final_Number_ITERATION'] = int(1024*4) #int(64) #ITERATION 
parameters['Time_step'] = N


t = np.arange(0, N)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = 1+x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


X_EnKF = EnKF(Y_True,X0,A,h,t,tau,Noise)
X_SIR = SIR(Y_True,X0,A,h,t,tau,Noise)
X_OT , OT_param_dict, f_param_dict = OTF_param(Y_True,X0,parameters,A,h,t,tau,Noise)

#%%

p = 1000 # number of particles to plot
num_plot_state = 1 # number of state to plot
l=0

plt.figure(figsize=(15,10))
plt.subplot(3,1,1)
plt.plot(t,X_EnKF[l,:,num_plot_state,:p],'g',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'kx',label='True state')
plt.ylabel('EnKF')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t,X_SIR[l,:,num_plot_state,:p],'b',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'kx')
plt.ylabel('SIR')


plt.subplot(3,1,3)
plt.plot(t,X_OT[l,:,num_plot_state,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'kx')
plt.plot(t,-X_True[l,:,num_plot_state],'kx')
plt.ylabel(r'$OT$')
plt.xlabel('time')

#%%
# j = N-2
j=1
plt.figure(figsize=(20,6))
for i in range(L):
    plt.subplot(1,L,1+i)
    plt.hist(X_OT[l,j,i,:p],density=True)
    plt.axvline(x=X_True[l,j,i], color='black',linestyle='--')
    plt.axvline(x=-X_True[l,j,i], color='black',linestyle='--')
    plt.xlim([-3,3])

sys.exit()
#%%
np.savez('./DATA_XX/DATA_file_param_IC.npz',OT_param_dict = OT_param_dict,f_param_dict=f_param_dict)