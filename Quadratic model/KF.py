#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:23:44 2024

@author: jarrah
"""

import numpy as np
import time
def KF(Y,X0,A,H,t,tau,Noise):
    AVG_SIM = X0.shape[0]
    N = Y.shape[1]
    L = X0.shape[1]
    dy = Y.shape[2]
    J = X0.shape[2]
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = 1/noise # Amplifiying the initial state 
    start_time = time.time()
    SAVE_X_KF =  np.zeros((AVG_SIM,N,L,J))
    SAVE_cov = np.zeros((AVG_SIM,N,L,L))
    
    P0 = sigmma0*sigmma0*np.eye(L)
    R = gamma*gamma*np.eye(dy)
    Q = sigmma*sigmma*np.eye(L)
    for k in range(AVG_SIM):
        
        y = Y[k,]

        x_hatKF = X0[k,] 
        

        
        SAVE_X_KF[k,0,:,:] = X0[k,] 
        # EnKF & 3DVAR
        for i in range(N):
            # Gain
            K = P0 @ H.T @ np.linalg.inv(H @ P0 @ H.T + R) 
            
            # Update
            x_hatKF = x_hatKF + K @ (y[i,:].reshape(dy,1) - H@x_hatKF)
            P0 = (np.eye(L) - K@H)@P0
            
            
            SAVE_X_KF[k,i,:,:] = x_hatKF
            SAVE_cov[k,i,:,:] = P0
            
            # Propagation
            x_hatKF = A@x_hatKF 
            P0 = A@P0@A.T + Q
            

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_KF#, SAVE_cov