#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:39:06 2025

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

delta = [0.01,0.01] # lambda_T lambda_f 
J = int(1000/4) # Number of ensembles EnKF
AVG_SIM = 5 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(64/1)
parameters['BATCH_SIZE'] = int(64/1)
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024/1) #1024*2 
parameters['Final_Number_ITERATION'] = int(64/2) #int(64*2) #ITERATION 


# Num_selected_maps = [1,5]
Num_selected_maps = [1,2,5,10,20]

method='d_mmd' # method='d_mmd',d_w2,d_T default : d_T

t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))




Map_index_offline = []

X_NEAREST = []
Distance = []
time_nearest = []
Map_index_online = []

X_NEAREST_MMD = []
X_WEIGHTED_W2 = []
X_WEIGHTED_MMD = []

for k in Num_selected_maps:
    print(k)
    S = select_maps_fun(k,method=method) # method='d_mmd',d_w2,d_T default : d_T
    Map_index_offline.append(S)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=True) 
    
    X_NEAREST.append(X_OT_nearest)
    Distance.append(distance)
    time_nearest.append(t_nearest)
    Map_index_online.append(nearest_index)
    
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=True) 
    X_NEAREST_MMD.append(X_OT_nearest)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=False) 
    X_WEIGHTED_W2.append(X_OT_nearest)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=False) 
    X_WEIGHTED_MMD.append(X_OT_nearest)


# data is AVG_SIM x N x L x J
X_OT, time_OT = OTF(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint) 

  

X_EnKF = EnKF(Y_True,X0,L63,h,t,tau,Noise,Odeint)
X_SIR = SIR(Y_True,X0,L63,h,t,tau,Noise,Odeint)
      
#%%
n_true = int(1e5)
X0_true = np.zeros((AVG_SIM,L,n_true))
for k in range(AVG_SIM):    
    X0_true[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),n_true))
X_true_particles = SIR(Y_True,X0_true,L63,h,t,tau,Noise,Odeint)

for l in range(AVG_SIM):
    plt.figure(figsize=(8,8))
    for num_plot_state in range(L):
        plt.subplot(3,1,num_plot_state+1)
        plt.plot(t,X_true_particles[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
        plt.plot(t,X_True[l,:,num_plot_state],'k--')
        plt.ylabel(r'$X_{True}$')
        plt.xlabel('time')
#%%
import ot
W2 = {}
for i in range(len(Num_selected_maps)):
    W2[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_NEAREST[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M)) 
        W2[str(int(i))].append(w2/AVG_SIM)

w2_nearest = []
for i in range(len(Num_selected_maps)):
    w2_nearest.append(sum(W2[str(int(i))])/len(W2[str(int(i))]))

W2_ot = []
for j in range(len(t)):
    if j%100==0:
        print("OT , j: ",j)
    
    w2 = 0
    for k in range(AVG_SIM):
        M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_OT[k,j,:].T) 
            
        # Uniform weights if distributions are unweighted
        a = np.ones(1000) / 1000 # Uniform weights for X
        b = np.ones(J) / J # Uniform weights for Y
            
        # Compute the Wasserstein distance (emd2 returns the squared distance)
        w2 += np.sqrt(ot.emd2(a, b, M))
    W2_ot.append(w2/AVG_SIM)
 

w2_enkf = []
w2_sir = []
for j in range(len(t)):
    if j%100==0:
        print("EnKF, SIR , j: ",j)
        
    w2_en = 0
    w2_s = 0
    for k in range(AVG_SIM):
        M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_EnKF[k,j,:].T) 
            
        # Uniform weights if distributions are unweighted
        a = np.ones(1000) / 1000 # Uniform weights for X
        b = np.ones(J) / J # Uniform weights for Y
            
        # Compute the Wasserstein distance (emd2 returns the squared distance)
        w2_en += np.sqrt(ot.emd2(a, b, M))
        
        M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_SIR[k,j,:].T) 
        w2_s += np.sqrt(ot.emd2(a, b, M))
        
    w2_enkf.append(w2_en/AVG_SIM)
    w2_sir.append(w2_s/AVG_SIM)
    


W2 = {}
for i in range(len(Num_selected_maps)):
    W2[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_WEIGHTED_W2[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M)) 
        W2[str(int(i))].append(w2/AVG_SIM)

w2_weighted = []
for i in range(len(Num_selected_maps)):
    w2_weighted.append(sum(W2[str(int(i))])/len(W2[str(int(i))]))

    
mmd = {}
for i in range(len(Num_selected_maps)):
    mmd[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        mmd_sum = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_NEAREST_MMD[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            mmd_sum += np.sqrt(ot.emd2(a, b, M)) 
        mmd[str(int(i))].append(mmd_sum/AVG_SIM)

mmd_nearest = []
for i in range(len(Num_selected_maps)):
    mmd_nearest.append(sum(mmd[str(int(i))])/len(mmd[str(int(i))]))
    
    
mmd = {}
for i in range(len(Num_selected_maps)):
    mmd[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        mmd_sum = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_WEIGHTED_MMD[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            mmd_sum += np.sqrt(ot.emd2(a, b, M)) 
        mmd[str(int(i))].append(mmd_sum/AVG_SIM)

mmd_weighted = []
for i in range(len(Num_selected_maps)):
    mmd_weighted.append(sum(mmd[str(int(i))])/len(mmd[str(int(i))]))
#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)          # controls default text sizes

 
plt.figure(figsize=(9,8))    
plt.axhline(y=sum(w2_enkf)/len(w2_enkf),color='g',label='EnKF',lw=2,linestyle='--')
plt.axhline(y=sum(w2_sir)/len(w2_sir),color='b',label='SIR',lw=2,linestyle='--')
plt.axhline(y=sum(W2_ot)/len(W2_ot),color='m',label='OT',lw=2,linestyle='--')



plt.plot(Num_selected_maps,w2_weighted,'v-',lw=2,label=r'$\rho_{W_2}$, Weighted',markersize=10)
plt.plot(Num_selected_maps,w2_nearest,'s-',lw=2,label=r'$\rho_{W_2}$, Nearest',markersize=10)

plt.plot(Num_selected_maps,mmd_weighted,'o-',lw=2,label=r'$\rho_{MMD}$, Weighted',markersize=10)
plt.plot(Num_selected_maps,mmd_nearest,'D-',lw=2,label=r'$\rho_{MMD}$, Nearest',markersize=10)


plt.xlabel("# of selected maps (K)",fontsize=20)
plt.ylabel(r"$W_2$",fontsize=20)
# plt.title('')
plt.legend(loc=2,fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.show()