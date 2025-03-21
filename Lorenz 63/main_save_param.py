#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:54:44 2025

@author: jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import sys
from OTF_save_param import OTF_param

plt.close('all')

# Choose h(x) here, the observation rule
def h(x):
    # return x[0,].reshape(dy,-1)
    return x[2,].reshape(dy,-1)
# =============================================================================
#     return x[::2,]
# =============================================================================

def L63(x, t):
    """Lorenz 96 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3

    d[0] = sigma*(x[1]-x[0])
    d[1] = x[0]*(r-x[2])-x[1]
    d[2] = x[0]*x[1]-b*x[2]
    return d


def Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    # x0 = (2*torch.randint(0,2,(1,)).item()-1)*10 + np.random.multivariate_normal(np.zeros(L), np.eye(L),1)
    x0 = 0 + np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0*np.eye(L),1)
    x[0,] = x0

    
    for i in range(N-1):
        x[i+1,:] = x[i,:] + L63(x[i,:],t[i])*tau 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
    
    return x,y

def mse(x,x_true):
    x_mean = (x-x_true.reshape(AVG_SIM,N,L,1)).mean(axis=3)
    return ((x_mean*x_mean).sum(axis=2)).mean(axis=0)

#%%   
L = 3 # number of states
tau = 1e-2 # timpe step 
T = 2 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s

dy = 1 # number of states observed
H = np.zeros((dy,L))

noise = np.sqrt(10) # noise level std
sigmma = noise/10 # Noise in the hidden state
sigmma0 = 10 # Noise in the initial state distribution
gamma = noise*1 # Noise in the observation
x0_amp = 1 # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]
Odeint = False

J = int(1000) # Number of ensembles EnKF

AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*2/1)
parameters['BATCH_SIZE'] = int(64/1)
parameters['LearningRate'] = 1e-4
parameters['ITERATION'] = int(1024/1) #1024*2 
parameters['Final_Number_ITERATION'] = int(64*2/1) #int(64*2) #ITERATION 

t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


X_OT , X_prior,Y_prior, OT_param_dict = OTF_param(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint) 


p = 100 # number of particles to plot
# num_plot_state = 0 # number of state to plot
# l=0
for l in range(AVG_SIM):
    plt.figure(figsize=(15,10))
    for num_plot_state in range(L):
        plt.subplot(3,1,num_plot_state+1)
        plt.plot(t,X_OT[l,:,num_plot_state,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
        plt.plot(t,X_True[l,:,num_plot_state],'k--')
        plt.ylabel(r'$OT$')
        plt.xlabel('time')

#%%
for l in range(AVG_SIM):
    plt.figure(figsize=(15,10))
    for num_plot_state in range(L):
        plt.subplot(3,1,num_plot_state+1)
        plt.hist(X_OT[l,30,num_plot_state,:],density=True,bins=20)
sys.exit()
#%%
np.savez('./DATA_L63/DATA_file_param_5.npz',\
    X_prior = X_prior,Y_prior = Y_prior,OT_param_dict = OT_param_dict)
  
    
    
    
    
    
    
    
    
    
    