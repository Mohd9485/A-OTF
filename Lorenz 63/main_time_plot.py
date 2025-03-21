#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:43:47 2025

@author: jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import sys
import matplotlib
from EnKF import EnKF
from SIR import SIR
from OTF import OTF
from A_OTF_MMD import A_OTF_MMD
from A_OTF_W2 import A_OTF_W2

from select_maps_fun import select_maps_fun


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=13)          # controls default text sizes


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
T = int(10) # final time in seconds
N = int(T/tau) # number of time steps T = 20 s

dy = 1 # number of states observed
H = np.zeros((dy,L))
# =============================================================================
# H[0,0] = 1
# =============================================================================


noise = np.sqrt(10) # noise level std
sigmma = noise/10 # Noise in the hidden state
sigmma0 = 10 # Noise in the initial state distribution
gamma = noise*1 # Noise in the observation
x0_amp = 1 # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]
Odeint = False

J = int(1000/2) # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(64/1)
parameters['BATCH_SIZE'] = int(64/1)
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024/1) #1024*2 
parameters['Final_Number_ITERATION'] = int(64) #int(64*2) #ITERATION 



Num_selected_maps = [1,2,5,10,20,50]
# Num_selected_maps = [1,2]

t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


time_nearest_w2 = []
time_nearest_mmd = []
for k in Num_selected_maps:
    print(k)
    S = select_maps_fun(k,method='d_w2') # method='d_mmd',d_w2,d_T default : d_T
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=False) 
    time_nearest_w2.append(t_nearest)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=False) 
    time_nearest_mmd.append(t_nearest)


# data is AVG_SIM x N x L x J
X_OT, time_OT = OTF(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint) 
# X_OT =  np.load('./particles_run/X_OT_particles_3554.npz')['X_OT']
# time_OT =  np.load('./particles_run/X_OT_particles_3554.npz')['time_OT']

  
start_time = time.time()
X_EnKF = EnKF(Y_True,X0,L63,h,t,tau,Noise,Odeint)
time_EnKF = time.time() - start_time

start_time = time.time()
X_SIR = SIR(Y_True,X0,L63,h,t,tau,Noise,Odeint)
time_SIR = time.time() - start_time 
  
    

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


plt.figure(figsize=(9,8))    

plt.axhline(y=time_EnKF/N,color='g',label='EnKF',lw=2,linestyle='--')
plt.axhline(y=time_SIR/N,color='b',label='SIR',lw=2,linestyle='--')
plt.axhline(y=time_OT/N,color='m',label='OTF',lw=2,linestyle='--')

plt.plot(Num_selected_maps,np.array(time_nearest_w2)/N,'s-',lw=2,label=r'A-OTF,$\rho_{W_2}$',markersize=10)
plt.plot(Num_selected_maps,np.array(time_nearest_mmd)/N,'D-',lw=2,label=r'A-OTF,$\rho_{MMD}$',markersize=10)

plt.xlabel("# of selected maps (K)",fontsize=20)
plt.ylabel('time',fontsize=20)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()