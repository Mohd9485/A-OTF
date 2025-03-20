#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:00:44 2025

@author: jarrah
"""


import numpy as np
import torch
import torch.nn as nn
import ot
from scipy.spatial.distance import cdist
import sys
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import copy 

def select_maps_fun(k,method='d_T'):
            
    def kernel(X,Y,sigma=1):
        return torch.exp(-sigma*torch.cdist(X.T,Y.T)*torch.cdist(X.T,Y.T))
        
    
    def MMD(XY, XY_target, kernel,sigma ):
    # =============================================================================
    #     N = 1000
    #     quantile = torch.quantile(torch.cdist(XY[:,:N].T,XY_target.T).reshape(1,-1),q=0.25).item()
    #     print(quantile)
    # =============================================================================
    # =============================================================================
    #     sigma = 0.003#1/(2*quantile**2)
    # =============================================================================
    # =============================================================================
    #     sigma = 1/(2*10**2)
    # =============================================================================
    # =============================================================================
    #     print(sigma)
    # =============================================================================
        device = torch.device('mps')
        XY = XY.T.to(device)
        XY_target = XY_target.T.to(device)
        
        return torch.sqrt(kernel(XY,XY,sigma=sigma).mean() 
                          + kernel(XY_target,XY_target,sigma=sigma).mean()
                          - 2*kernel(XY,XY_target,sigma=sigma).mean())

    def h(x):
        # return x[0,].reshape(dy,-1)
        return x[2,].reshape(1,-1)

    i = 1
    # file_name = '/Users/jarrah/Documents/CDC2025_DATA/L63/DATA_file_param_'+str(i)+'.npz'
    file_name = './DATA_X/DATA_file_param_'+str(i)+'.npz'
    X_prior = np.load(file_name,allow_pickle=True)['X_prior']
    OT_param_dict = np.load(file_name,allow_pickle=True)['OT_param_dict'].tolist()
    for i in range(2,6):
        
        # file_name = '/Users/jarrah/Documents/CDC2025_DATA/L63/DATA_file_param_'+str(i)+'.npz'
        file_name =  './DATA_X/DATA_file_param_'+str(i)+'.npz'
        X_prior = np.concatenate((X_prior,np.load(file_name,allow_pickle=True)['X_prior']),axis=0) 
        OT_param_dict += np.load(file_name,allow_pickle=True)['OT_param_dict'].tolist()



    # MAP_T = T_NeuralNet([3,1], 32)
    # MAP_T.load_state_dict(OT_param_dict[0])

    J = X_prior.shape[1]
    N = len(OT_param_dict)



    if method.lower() == 'd_t':
        D = np.load('distance_between_maps.npz')['D']
        # D = np.load('distance_between_maps.npz')['D']
        print('Distance between maps')
    elif method.lower() == 'd_mmd':
        # D = np.load('distance_MMD_ls_0.1.npz')['D']
        D = np.load('distance_MMD.npz')['D']
        # D = np.load('distance_MMD_ls_10.npz')['D']
        print('MMD distance')
    elif method.lower() == 'd_w2':
        D = np.load('distance_W2.npz')['D']
        print('W2 distance')
    else:
        raise ValueError("Select distance matrix D")

    if k==1:
        U, s, VT = np.linalg.svd(D, full_matrices=False)
        S = np.argmax(U[:,:1],axis=0)
        select_map = []
        for i in range(k):
            select_map.append(copy.deepcopy(OT_param_dict[S[i]]))
        np.savez('selected_maps.npz',\
                 prior = X_prior[S], select_map = select_map,S=S )
        return S
        
    # 3. Fit k-medoids with 'precomputed' distance metric:
    kmedoids = KMedoids(
        n_clusters=k,
        metric='precomputed',
        method='pam',         # the PAM (Partitioning Around Medoids) algorithm
        init='k-medoids++'    # or 'random'; 'k-medoids++' can help initialization
    ).fit(D)
    
    # 4. Extract results:
    labels = kmedoids.labels_           # Cluster assignments for each distribution
    S = kmedoids.medoid_indices_  # Indices of the chosen medoids
    
    # print("Cluster labels:", labels)
    print("Indices of chosen medoids:", S)
    # print(X[S])
    
    
    select_map = []
    for i in range(k):
        select_map.append(copy.deepcopy(OT_param_dict[S[i]]))
    
        
    np.savez('selected_maps.npz',\
             prior = X_prior[S], select_map = select_map,S=S )
    
    # plt.figure(figsize=(20,8))
    # for i in range(k):
    #     plt.subplot(1,k,i+1)
    #     plt.scatter(X_prior[S][i,:,0], X_prior[S][i,:,1])
    #     plt.xlabel(r'$X_1$')
    #     plt.ylabel(r'$X_2$')
    return S