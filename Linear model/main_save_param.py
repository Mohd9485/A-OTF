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
from OT_save_param import OT_param

plt.close('all')


# Choose h(x) here, the observation rule
def h(x):
    return x[0].reshape(1,-1)
    # return x[0].reshape(1,-1)*x[0].reshape(1,-1)

def A(x,t=0):
    return F @ (x)




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
        
    return x,y

#%%    
L = 2 # number of states
tau = 1e-1 # timpe step 
T = 10 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = 1 # number of states observed

# dynmaical system
H = np.array([[1,0]]) 
# =============================================================================
# H = np.eye(dy)
# =============================================================================
alpha = 0.9
a = alpha
b= np.sqrt(1-alpha**2)
c = alpha
F = np.array([[a, -b],[b,c]]) 


noise = np.sqrt(1e-1) # noise level std
sigmma = noise # Noise in the hidden state
sigmma0 = 1#5*noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1#/noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]

J = int(1e3/1) # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32) #64
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64*1) #128
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024/1) 
parameters['Final_Number_ITERATION'] = int(64/1) #int(64) #ITERATION 
parameters['Time_step'] = N


t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


X_EnKF = EnKF(Y_True,X0,A,h,t,tau,Noise)
X_SIR = SIR(Y_True,X0,A,h,t,tau,Noise)
X_OT , X_prior,Y_prior, OT_param_dict = OT_param(Y_True,X0,parameters,A,h,t,tau,Noise)

#%%

p = 100 # number of particles to plot
num_plot_state = 1 # number of state to plot
l=0

plt.figure(figsize=(15,10))
plt.subplot(3,1,1)
plt.plot(t,X_EnKF[l,:,num_plot_state,:p],'g',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--',label='True state')
plt.ylabel('EnKF')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t,X_SIR[l,:,num_plot_state,:p],'b',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--')
plt.ylabel('SIR')


plt.subplot(3,1,3)
plt.plot(t,X_OT[l,:,num_plot_state,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--')
plt.ylabel(r'$OT$')
plt.xlabel('time')



sys.exit()
#%%
np.savez('./DATA_X/DATA_file_param_1.npz',\
    X_prior = X_prior,Y_prior = Y_prior,OT_param_dict = OT_param_dict)