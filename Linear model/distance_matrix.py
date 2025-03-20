#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:05:55 2025

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

from sklearn.metrics import silhouette_score


plt.close('all')

class T_NeuralNet(nn.Module):
        
        def __init__(self, input_dim, hidden_dim):
            super(T_NeuralNet, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activation = nn.ReLU()
            # self.activation = nn.ELU()
            # self.activation = nn.Sigmoid()
            self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
            self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out = nn.Linear(self.hidden_dim, input_dim[0], bias=False)
            
            self.layer31 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer32 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            
        # Input is of size
        def forward(self, x, y):
            X = self.layer_input(torch.concat((x,y),dim=1))
            
            # xy = self.layer11(self.activation(X))
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12 (xy)
            
            xy = self.activation(xy)+X
            
            xy = self.layer21(xy)
            xy = self.activation(xy)
            xy = self.layer22 (xy)
            
            # xy = self.activation(xy)+X
            
            # xy = self.layer31(xy)
            # xy = self.activation(xy)
            # xy = self.layer32 (xy)
            
            xy = self.layer_out(self.activation(xy)+X)
            return xy
            
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
#%%
i = 1
# file_name = '/Users/jarrah/Documents/CDC2025_DATA/L63/DATA_file_param_'+str(i)+'.npz'
file_name = './DATA_X/DATA_file_param_'+str(i)+'.npz'
X_prior = np.load(file_name,allow_pickle=True)['X_prior']
Y_prior = np.load(file_name,allow_pickle=True)['Y_prior']
OT_param_dict = np.load(file_name,allow_pickle=True)['OT_param_dict'].tolist()
for i in range(2,6):
    
    # file_name = '/Users/jarrah/Documents/CDC2025_DATA/L63/DATA_file_param_'+str(i)+'.npz'
    file_name =  './DATA_X/DATA_file_param_'+str(i)+'.npz'
    X_prior = np.concatenate((X_prior,np.load(file_name,allow_pickle=True)['X_prior']),axis=0) 
    Y_prior = np.concatenate((Y_prior,np.load(file_name,allow_pickle=True)['Y_prior']),axis=0) 
    OT_param_dict += np.load(file_name,allow_pickle=True)['OT_param_dict'].tolist()



# MAP_T = T_NeuralNet([3,1], 32)
# MAP_T.load_state_dict(OT_param_dict[0])

J = X_prior.shape[1]
N = len(OT_param_dict)

method = 'd_t' # 'd_t' for distance between maps, 'd_w2' for Wasserstein-2, and 'd_mmd' for Maximum mean discrepancy
# method = 'd_w2'
# method = 'd_mmd'
#%%

if method.lower()=='d_t':
    # Distance between maps
    print('Distance between maps')
    MAP_T1 = T_NeuralNet([2,1], 32)
    MAP_T2 = T_NeuralNet([2,1], 32)
    D = np.zeros((N,N))
    device = torch.device('mps')
    for i in range(N):
        print("i: ",i)
        MAP_T1.load_state_dict(OT_param_dict[i])
        
        x1 = torch.from_numpy(X_prior[i,:,:]).to(torch.float32)
        
        perm_1 = torch.randperm(len(Y_prior[i,:,:]))
        y1 = torch.from_numpy(Y_prior[i,perm_1,:]).to(torch.float32)
        
        x11 = MAP_T1(torch.from_numpy(X_prior[i,:,:]).to(torch.float32),y1)
        for j in range(i+1,N):
            MAP_T2.load_state_dict(OT_param_dict[j])
            
            x2 = torch.from_numpy(X_prior[j,:,:]).to(torch.float32)
            
            perm_2 = torch.randperm(len(Y_prior[j,:,:]))
            y2 = torch.from_numpy(Y_prior[j,perm_2,:]).to(torch.float32)
            
            x12 = MAP_T1(x2,y2)
            x21 = MAP_T2(x2,y1)
            x22 = MAP_T2(x2,y2)

            D[i,j] = 0.5*(torch.norm(x11-x21,dim=1).mean() + torch.norm(x12-x22,dim=1).mean()) 
            
    D = D + D.T

    np.savez('distance_between_maps.npz', D=D )
    
elif method.lower()=='d_w2':
    # W2 on X
    print('Distance W2')
    D = np.zeros((N,N))
    for i in range(N):
        print("i: ",i)
        for j in range(i+1,N):
            # print("i: ",i," j: ",j)
            M =  ot.dist(X_prior[i,:,:], X_prior[j,:,:]) 
            
            # Uniform weights if distributions are unweighted
            a = np.ones(J) / J # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
            
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            D[i,j] = np.sqrt(ot.emd2(a, b, M))
            
    D = D + D.T
    np.savez('distance_W2.npz', D=D )
    
elif  method.lower()=='d_mmd':
    # MMD
    print('Distance MMD')
    D = np.zeros((N,N))
    sigma = 0.1
    for i in range(N):
        print("i: ",i)
        
        mean =  X_prior[i,:,:].mean(axis=0,keepdims=True)
        std = X_prior[i,:,:].std(axis=0,keepdims=True)
        x1 = torch.from_numpy((X_prior[i,:,:] - mean)/std).to(torch.float32)
        
        for j in range(i+1,N):
            mean =  X_prior[j,:,:].mean(axis=0,keepdims=True)
            std = X_prior[j,:,:].std(axis=0,keepdims=True)
            x2 = torch.from_numpy((X_prior[j,:,:] - mean)/std).to(torch.float32)
            D[i,j] = MMD(x1, x2, kernel,sigma)    
            
    D = D + D.T
    np.savez('distance_MMD.npz', D=D )
    
    
        

       



