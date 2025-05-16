#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:48:18 2025

@author: jarrah
"""


import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from EnKF import EnKF
from SIR import SIR

from OTF import OTF
from scipy.integrate import odeint

from A_OTF_MMD import A_OTF_MMD
from A_OTF_W2 import A_OTF_W2
from select_maps_fun import select_maps_fun
import ot
#%matplotlib auto

import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=15)          # controls default text sizes


plt.close('all')

# Choose h(x) here, the observation rule
def h(x):
    return x*x

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
n = 4
L = n*2 # number of states
tau = 1e-1 # timpe step 
T = 5 # final time in seconds
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


noise = np.sqrt(1e-2) # noise level std
sigmma = noise # Noise in the hidden state
sigmma0 = 1#5*noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1#/noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]


Num_selected_maps = 10
# Num_selected_maps = [1,2,5,10,20]
# Num_selected_maps = [1,2,5,10,20,50,100]
S = select_maps_fun(Num_selected_maps,method='d_w2') # method='d_mmd',d_w2,d_T default : d_T
# S = select_maps_fun(Num_selected_maps,method='d_mmd') # method='d_mmd',d_w2,d_T default : d_T

J = int(1e4/1) # Number of ensembles EnKF
true_particle = int(1e5)
J_dist = min(1000,J)

AVG_SIM = 5 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*2) #128
parameters['BATCH_SIZE'] = int(64) #64
parameters['LearningRate'] = 1e-5 # 1e-3
parameters['ITERATION'] = int(1024/1) # 1e4
parameters['Final_Number_ITERATION'] = int(1024/4) #int(64) #ITERATION 
parameters['Time_step'] = N


t = np.arange(0.0, tau*N, tau)

# MU_0 = [1,5]
# MU_0 = [10,8,6,4,2,0]
SIGMA0 = [1,2,3,4,5] 

X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))

for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y

F = np.array([[a, b],[-b,c]]) 
X_true_particles = np.zeros((AVG_SIM,N,L,true_particle))
for i in range(n):
    x = X_True[:,:,2*i:(2*i+2)].reshape(AVG_SIM,N,2)
    y = Y_True[:,:,2*i:(2*i+2)].reshape(AVG_SIM,N,2)
    x0 = np.zeros((AVG_SIM,2,true_particle))
    for j in range(AVG_SIM):
        x0[j] = np.transpose(np.random.multivariate_normal(np.zeros(2),sigmma0*sigmma0 * np.eye(2),true_particle))
    x_true = SIR(y,x0,A,h,t,tau,Noise)
    X_true_particles[:,:,2*i:(2*i+2),:] = x_true.reshape(AVG_SIM,N,2,true_particle)

# for l in range(AVG_SIM):
#     plt.figure(figsize=(8,8))
#     for num_plot_state in range(L):
#         plt.subplot(L,1,num_plot_state+1)
#         plt.plot(t,X_true_particles[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,num_plot_state],'k--')
#         plt.ylabel(r'$X_{True}$')
#         plt.xlabel('time')

F = np.kron(np.eye(int(n)), F)
     
W2_enkf = []
W2_sir = []  
W2_ot = []  
W2_amortized = []
for sigmma0 in SIGMA0:
    
    X0 = np.zeros((AVG_SIM,L,J))
    for k in range(AVG_SIM):    
        X0[k,] = 0 + x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))
    
   
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False)  
    # X_OT_nearest,_,_,_ = A_OTF_MMD(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False) 
        

    # X_KF  = KF(Y_True,X0,F,H,t,tau,Noise)
    X_SIR = SIR(Y_True,X0,A,h,t,tau,Noise)
    
    X_EnKF = EnKF(Y_True,X0,A,h,t,tau,Noise)
    
    X_OT, time_OT = OTF(Y_True,X0,parameters,A,h,t,tau,Noise)
    
    
    # num_plot_state = 1
    # p=100
    # for l in range(AVG_SIM):
    #     plt.figure(figsize=(15,10))
    #     plt.subplot(4,1,1)
    #     plt.plot(t,X_EnKF[l,:,num_plot_state,:p],'g',ls='none',marker='o',ms=4,alpha = 0.1)
    #     plt.plot(t,X_True[l,:,num_plot_state],'k--',label='True state')
    #     plt.ylabel('EnKF')
    #     plt.title('sigma 0 =  %d'%(sigmma0))
    #     plt.legend()
        
    #     plt.subplot(4,1,2)
    #     plt.plot(t,X_SIR[l,:,num_plot_state,:p],'b',ls='none',marker='o',ms=4,alpha = 0.1)
    #     plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #     plt.ylabel('SIR')
        
    #     plt.subplot(4,1,3)
    #     plt.plot(t,X_OT_nearest[l,:,num_plot_state,:p],'C4',ls='none',marker='o',ms=4,alpha = 0.1)
    #     plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #     plt.ylabel(r'$OT_{nearest}$')
        
    #     plt.subplot(4,1,4)
    #     plt.plot(t,X_OT[l,:,num_plot_state,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
    #     plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #     plt.ylabel(r'$OT$')
    #     plt.xlabel('time')
         
#%%       
    w2_enkf = []
    w2_sir = []  
    w2_ot = []  
    w2_amortized = []
    
    # Uniform weights if distributions are unweighted
    a = np.ones(1000) / 1000 # Uniform weights for X
    
    b = np.ones(J_dist) / J_dist # Uniform weights for Y
    w2_start_time = 1
    for j in range(w2_start_time,len(t)):
        if j%100==0:
            print("Amortized, K: ",Num_selected_maps,", j: ",j)
            
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_OT_nearest[k,j,:,:J_dist].T) 
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M)) 
        w2_amortized.append(w2/AVG_SIM)
    
    

    
    for j in range(w2_start_time,len(t)):
        if j%100==0:
            print("OT , j: ",j)
        
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_OT[k,j,:,:J_dist].T) 
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M))
        w2_ot.append(w2/AVG_SIM)
     
    for j in range(w2_start_time,len(t)):
        if j%100==0:
            print("EnKF, SIR , j: ",j)
            
        w2_en = 0
        w2_s = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_EnKF[k,j,:,:J_dist].T) 
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2_en += np.sqrt(ot.emd2(a, b, M))
            
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_SIR[k,j,:,:J_dist].T) 
            w2_s += np.sqrt(ot.emd2(a, b, M))
            
        w2_enkf.append(w2_en/AVG_SIM)
        w2_sir.append(w2_s/AVG_SIM)
        
        
    W2_enkf.append(sum(w2_enkf)/len(w2_enkf))
    W2_sir.append(sum(w2_sir)/len(w2_sir)) 
    W2_ot.append(sum(w2_ot)/len(w2_ot))
    W2_amortized.append(sum(w2_amortized)/len(w2_amortized))
    
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)          # controls default text sizes




plt.figure(figsize=(9,8))    

plt.plot(SIGMA0,W2_enkf,'v--',color='g',label='EnKF',lw=2,markersize=10)
plt.plot(SIGMA0,W2_sir,'s--',color='b',label='SIR',lw=2,markersize=10)
plt.plot(SIGMA0,W2_ot,'o--',color='m',label='OTF',lw=2,markersize=10)

plt.plot(SIGMA0,W2_amortized,'D-',lw=2,label='A-OTF',markersize=10)

plt.xlabel(r"$\sigma_{0}$",fontsize=20)
# plt.ylabel(r"$W_2$",fontsize=20)
# plt.title('')
# plt.legend()
plt.show()
