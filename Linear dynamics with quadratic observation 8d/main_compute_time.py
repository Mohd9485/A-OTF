#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 11:35:21 2025

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
#%matplotlib auto
import ot
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
S = select_maps_fun(Num_selected_maps,method='d_w2') # method='d_mmd',d_w2,d_T default : d_T


AVG_SIM = 5 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*2) #128
parameters['BATCH_SIZE'] = int(64) #64
parameters['LearningRate'] = 1e-5 # 1e-3
parameters['ITERATION'] = int(1024/1) # 64
parameters['Final_Number_ITERATION'] = int(1024/4) #int(64) #ITERATION 
parameters['Time_step'] = N


t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))

for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y


# sim=0
true_particle = int(1e5)
F_2d = np.array([[a, b],[-b,c]])
def A_2d(x,t=0):
    return F_2d @ (x) 
X_true_dist = np.zeros((AVG_SIM,N,L,true_particle))
for i in range(n):
    x = X_True[:,:,2*i:(2*i+2)]#.reshape(1,N,2)
    y = Y_True[:,:,2*i:(2*i+2)]#.reshape(1,N,2)
    x0 = np.zeros((AVG_SIM,2,true_particle))
    for sim in range(AVG_SIM):
        x0[sim] = np.transpose(np.random.multivariate_normal(np.zeros(2),sigmma0*sigmma0 * np.eye(2),true_particle))
    x_true = SIR(y,x0,A_2d,h,t,tau,Noise)
    X_true_dist[:,:,2*i:(2*i+2),:] = x_true#.reshape(A,N,2,true_particle)

X_true_dist = X_true_dist[:,:,:,:1000] 

# plt.figure()
# plt.figure(figsize=(8,16))
# for num_plot_state in range(L):
#     plt.subplot(L,1,num_plot_state+1)
#     plt.plot(t,X_true_dist[0,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
#     plt.plot(t,X_True[0,:,num_plot_state],'k--')
#     plt.ylabel(r'$X_{True}$')
#     plt.xlabel('time')



W2_EnKF = []
W2_SIR = []
W2_OTF = []
W2_AOTF = []

time_EnKF = []
time_SIR = []
time_OTF = []
time_AOTF = []

J_OT = 1e6
# J_list = [int(5*10**i) for i in range(4,1,-1)] # Number of ensembles 
J_list = [int(10**i) for i in range(6,0,-1)] # Number of ensembles 
# J_list.append(int(5*1e6))
# J_list = [int(1e7)]
for J in J_list:
    print("J: ", J)
    J_ot = min(1000,J)
    X0 = np.zeros((AVG_SIM,L,J))
    for k in range(AVG_SIM):
        X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))
    
    
    start_time = time.time()
    X_SIR = SIR(Y_True,X0,A,h,t,tau,Noise)
    time_SIR.append((time.time()-start_time)/(AVG_SIM*N))
    
    start_time = time.time()
    X_EnKF = EnKF(Y_True,X0,A,h,t,tau,Noise)
    time_EnKF.append((time.time()-start_time)/(AVG_SIM*N))
    
    
    
    if J <= J_OT:    
        start_time = time.time()
        X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False) 
        # X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False) 
        time_AOTF.append((time.time()-start_time)/(AVG_SIM*N))
        
        start_time = time.time()
        X_OT, _ = OTF(Y_True,X0,parameters,A,h,t,tau,Noise)
        time_OTF.append((time.time()-start_time)/(AVG_SIM*N))
    
    W2_enkf = []
    W2_sir = []
    W2_otf = []
    W2_aotf = []
    for j in range(len(t)):
        w2_enkf = 0
        w2_sir = 0 
        w2_otf = 0
        w2_aotf = 0
        for k in range(AVG_SIM):
            M_enkf =  ot.dist(X_true_dist[k,j,:,:1000].T, X_EnKF[k,j,:,:J_ot].T) 
            M_sir =  ot.dist(X_true_dist[k,j,:,:1000].T, X_SIR[k,j,:,:J_ot].T) 
            
            if J <= J_OT:  
                M_otf =  ot.dist(X_true_dist[k,j,:,:1000].T, X_OT[k,j,:,:J_ot].T) 
                M_aotf =  ot.dist(X_true_dist[k,j,:,:1000].T, X_OT_nearest[k,j,:,:J_ot].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J_ot) / J_ot # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2_enkf += np.sqrt(ot.emd2(a, b, M_enkf))
            w2_sir += np.sqrt(ot.emd2(a, b, M_sir))
            
            if J <= J_OT:  
                w2_otf += np.sqrt(ot.emd2(a, b, M_otf))
                w2_aotf += np.sqrt(ot.emd2(a, b, M_aotf))

        W2_enkf.append(w2_enkf/AVG_SIM)
        W2_sir.append(w2_sir/AVG_SIM)
        
        if J <= J_OT:  
            W2_otf.append(w2_otf/AVG_SIM)
            W2_aotf.append(w2_aotf/AVG_SIM)

    W2_EnKF.append(sum(W2_enkf)/len(W2_enkf))
    W2_SIR.append(sum(W2_sir)/len(W2_sir))
    if J <= J_OT:  
        W2_OTF.append(sum(W2_otf)/len(W2_otf))
        W2_AOTF.append(sum(W2_aotf)/len(W2_aotf))

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)          # controls default text sizes

plt.figure(figsize=(9,8))    


# plt.plot(MU_0,W2_enkf,'v--',color='g',label='EnKF',lw=2,markersize=10)
# plt.plot(MU_0,W2_sir,'s--',color='b',label='SIR',lw=2,markersize=10)
# plt.plot(MU_0,W2_ot,'o--',color='m',label='OTF',lw=2,markersize=10)

# plt.plot(MU_0,W2_amortized,'D-',lw=2,label='A-OTF',markersize=10)



plt.plot(time_EnKF[:-1],W2_EnKF[:-1],'v--',color='g',lw=2,label=r'EnKF',markersize=10)

j = 0
J_list_print = [6,5,4,3,2]
for xi, yi in zip(time_EnKF[:-1], W2_EnKF[:-1]):
    plt.text(
        xi,                 # x-coordinate
        yi,                 # y-coordinate
        rf'$10^{{{J_list_print[j]}}}$',        # text: y-value formatted to 2 decimals
        ha='center',        # horizontal alignment
        va='top',        # vertical alignment: just above the point
        fontsize=20,         # you can tweak font size
        position = (xi,yi-0.1)
    )
    j+=1
    
plt.plot(time_SIR[:-1],W2_SIR[:-1],'s--',color='b',lw=2,label=r'SIR',markersize=10)

plt.plot(time_OTF[:-1],W2_OTF[:-1],'o--',color='m',lw=2,label=r'OTF',markersize=10)
plt.plot(time_AOTF[:-1],W2_AOTF[:-1],'D-',lw=2,label=r'A-OTF',markersize=10)


plt.xlabel("computational time",fontsize=20)
plt.ylabel(r"$W_2$",fontsize=20)
plt.legend(loc=1,fontsize=20, bbox_to_anchor=(0.5, 0.6))
# plt.legend(loc=6,fontsize=20)
plt.xscale('log')
# plt.yscale('log')

plt.show()


#%%

plt.figure(figsize=(9,8))    

plt.plot(J_list[:-1],W2_EnKF[:-1],'v--',color='g',lw=2,label=r'EnKF',markersize=10)
plt.plot(J_list[:-1],W2_SIR[:-1],'s--',color='b',lw=2,label=r'SIR',markersize=10)

plt.plot(J_list[:-1],W2_OTF[:-1],'o--',color='m',lw=2,label=r'OTF',markersize=10)
plt.plot(J_list[:-1],W2_AOTF[:-1],'D-',lw=2,label=r'A-OTF',markersize=10)


plt.xlabel("N",fontsize=20)
plt.ylabel(r"$W_2$",fontsize=20)
# plt.legend(loc=1,fontsize=20, bbox_to_anchor=(1, 0.7))
plt.legend(loc=0,fontsize=20)
plt.xscale('log')
# plt.yscale('log')

plt.show()




