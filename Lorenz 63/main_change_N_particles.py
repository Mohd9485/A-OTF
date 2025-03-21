#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:34:14 2025

@author: jarrah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:59:55 2025

@author: jarrah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:57:14 2025

@author: jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import sys
import matplotlib
from EnKF import EnKF
from SIR import SIR
# from OT_norm import OT
from OTF_change_N import OTF

from A_OTF_MMD import A_OTF_MMD
from A_OTF_W2 import A_OTF_W2

from select_maps_fun import select_maps_fun


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)          # controls default text sizes

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
    x0 = mu_0 + np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0*np.eye(L),1)
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
T = int(5) # final time in seconds
N = int(T/tau) # number of time steps T = 20 s

dy = 1 # number of states observed
H = np.zeros((dy,L))
# =============================================================================
# H[0,0] = 1
# =============================================================================

# J = int(1000/4) # Number of ensembles EnKF
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


Num_selected_maps = 20 #,5,10,20]
S = select_maps_fun(Num_selected_maps,method='d_w2') # method='d_mmd',d_w2,d_T default : d_T

t = np.arange(0.0, tau*N, tau)


noise = np.sqrt(10) # noise level std
sigmma = noise/10 # Noise in the hidden state
gamma = noise*1 # Noise in the observation
sigmma0 = 5 # Noise in the initial state distribution
mu_0 = 4
x0_amp = 1 # Amplifiying the initial state 
Odeint = False

Noise = [noise,sigmma,sigmma0,gamma,x0_amp]




Particles_num = [10000,5000,1000,500,200,100]
# Particles_num = [100,500]


X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
for k in range(AVG_SIM):    
    x,y = Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y

    
n_true = int(1e5)
X0_true = np.zeros((AVG_SIM,L,n_true))
for k in range(AVG_SIM):    
    X0_true[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),n_true))
X_true_particles = SIR(Y_True,X0_true,L63,h,t,tau,Noise,Odeint)

# for l in range(AVG_SIM):
#     plt.figure(figsize=(8,8))
#     for num_plot_state in range(L):
#         plt.subplot(3,1,num_plot_state+1)
#         plt.plot(t,X_true_particles[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,num_plot_state],'k--')
#         plt.ylabel(r'$X_{True}$')
#         plt.xlabel('time')


W2_enkf = []
W2_sir = []  
W2_ot = []  
W2_amortized = []
for J in Particles_num:
    
    n_particles_w2 = min(1000,J)  # number of particles selected to compute W2

    X0 = np.zeros((AVG_SIM,L,J))
    for k in range(AVG_SIM):    
        X0[k,] = mu_0 + np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


    parameters['NUM_NEURON'] =  int(64/1)
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=False) 



    # data is AVG_SIM x N x L x J
    parameters['NUM_NEURON'] = int(32)# int(64/4) #64/8
    # parameters['NUM_NEURON'] = 48# int(64/4) #64/8
    X_OT, time_OT = OTF(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint) 
    
    
  

    X_EnKF = EnKF(Y_True,X0,L63,h,t,tau,Noise,Odeint)
    X_SIR = SIR(Y_True,X0,L63,h,t,tau,Noise,Odeint)
    
    # for l in range(AVG_SIM):
    #     plt.figure(figsize=(8,8))
    #     for num_plot_state in range(L):
    #         plt.subplot(3,1,num_plot_state+1)
    #         plt.plot(t,X_OT_nearest[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
    #         plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #         plt.ylabel(r'$X_{nearest}$')
    #         plt.xlabel('time')
            
    # for l in range(AVG_SIM):
    #     plt.figure(figsize=(8,8))
    #     for num_plot_state in range(L):
    #         plt.subplot(3,1,num_plot_state+1)
    #         plt.plot(t,X_OT[l,:,num_plot_state,:250],'b',ls='none',marker='o',ms=4,alpha = 0.1)
    #         plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #         plt.ylabel(r'$X_{OT}$')
    #         plt.xlabel('time')
            
    # for l in range(AVG_SIM):
    #     plt.figure(figsize=(8,8))
    #     for num_plot_state in range(L):
    #         plt.subplot(3,1,num_plot_state+1)
    #         plt.plot(t,X_SIR[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
    #         plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #         plt.ylabel(r'$X_{SIR}$')
    #         plt.xlabel('time')
    # for l in range(AVG_SIM):
    #     plt.figure(figsize=(8,8))
    #     for num_plot_state in range(L):
    #         plt.subplot(3,1,num_plot_state+1)
    #         plt.plot(t,X_EnKF[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
    #         plt.plot(t,X_True[l,:,num_plot_state],'k--')
    #         plt.ylabel(r'$X_{EnKF}$')
    #         plt.xlabel('time')
              
    #%%
    import ot
    w2_enkf = []
    w2_sir = []  
    w2_ot = []  
    w2_amortized = []
    
    
    for j in range(len(t)):
        if j%100==0:
            print("Amortized, N: ",J ,", j: ",j)
            
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_OT_nearest[k,j,:,:n_particles_w2].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(n_particles_w2) / n_particles_w2 # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M)) 
        w2_amortized.append(w2/AVG_SIM)
    
    

    
    for j in range(len(t)):
        if j%100==0:
            print("OT , j: ",j)
        
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_OT[k,j,:,:n_particles_w2].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(n_particles_w2) / n_particles_w2 # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M))
        w2_ot.append(w2/AVG_SIM)
     
    #%% 
    

    for j in range(len(t)):
        if j%100==0:
            print("EnKF, SIR , j: ",j)
            
        w2_en = 0
        w2_s = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_EnKF[k,j,:,:n_particles_w2].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(n_particles_w2) / n_particles_w2 # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2_en += np.sqrt(ot.emd2(a, b, M))
            
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_SIR[k,j,:,:n_particles_w2].T) 
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
plt.plot(Particles_num,W2_enkf,'v--',color='g',label='EnKF',lw=2,markersize=10)
plt.plot(Particles_num,W2_sir,'s--',color='b',label='SIR',lw=2,markersize=10)
plt.plot(Particles_num,W2_ot,'o--',color='m',label='OTF',lw=2,markersize=10)

plt.xscale('log')

plt.plot(Particles_num,W2_amortized,'D-',lw=2,label='A-OTF',markersize=10)

plt.xlabel(r"$N$",fontsize=20)
# plt.ylabel(r"$W_2$",fontsize=20)
# plt.title('')
# plt.legend(fontsize=20)
plt.show()