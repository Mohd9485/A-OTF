#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:21:47 2024

@author: jarrah
"""

import numpy as np
import time

def EnKF(Y,X0,A,h,t,tau,Noise,Odeint):
    np.random.seed(0)
    # Y is AVG_SIM x N x dy
    # X0 is AVG_SIM x L x J
    AVG_SIM = X0.shape[0]
    L = X0.shape[1]
    J = X0.shape[2]
    
    N = Y.shape[1]
    dy = Y.shape[2]

    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4] # Amplifiying the initial state 

    start_time = time.time()
    SAVE_X_EnKF =  np.zeros((AVG_SIM,N,J,L))
    
    for k in range(AVG_SIM):

        y = Y[k,]

        x_EnKF  = np.zeros((N,J,L))
        x_EnKF[0,] = X0[k,].T 
        
        SAVE_X_EnKF[k,0,:,:] = x_EnKF[0,]
        # EnKF & 3DVAR
        for i in range(N-1):

            sai_EnKF = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J) #  J x L
            x_hatEnKF = x_EnKF[i,] + tau*A(x_EnKF[i,].T,t[i]).T + sai_EnKF # J x L
            
            eta_EnKF = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),J)  # J x dy 
            y_hatEnKF = h(x_hatEnKF.T).T + eta_EnKF # J x dy
            
            X_hat = x_hatEnKF.mean(axis=0,keepdims=True)
            Y_hat = y_hatEnKF.mean(axis=0,keepdims=True)

            a = (x_hatEnKF - X_hat)
            b = (y_hatEnKF - Y_hat)

            C_xy = 1/J * a.T@b #np.matmul(a.transpose(),b)/J
            C_yy = 1/J * b.T@b #np.matmul(b.transpose(),b)/J
            
            K = C_xy @ np.linalg.inv(C_yy + np.eye(dy)*1e-6)#gamma*gamma)
            x_EnKF[i+1,:,:] = x_hatEnKF + (K@ (y[i+1,:] - y_hatEnKF).T).T 
        
            SAVE_X_EnKF[k,i+1,:,:] = x_EnKF[i+1,]

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_EnKF.transpose(0,1,3,2)