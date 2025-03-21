#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:46:05 2025

@author: jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from torch.distributions.multivariate_normal import MultivariateNormal


def A_OTF_MMD(Y,X0_C,parameters,A,h,t,tau,Noise,Odeint,nearest=True):
    # Y is AVG_SIM x N x dy
    # X0_C is AVG_SIM x L x J
    AVG_SIM = X0_C.shape[0]
    L = X0_C.shape[1]
    J = X0_C.shape[2]
    
    N = Y.shape[1]
    dy = Y.shape[2]

    
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4] # Amplifiying the initial state 
    T = N*tau
    
    # OT networks parameters
    normalization = parameters['normalization']
    NUM_NEURON = parameters['NUM_NEURON']
    INPUT_DIM = parameters['INPUT_DIM']
    BATCH_SIZE =  parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']
    
    #device = torch.device('mps' if torch.has_mps else 'cpu') # M1 Chip
    device = torch.device('cpu')
    # NN , initialization and training    
# =============================================================================
#     class NeuralNet(nn.Module):
#             
#             def __init__(self, input_dim, hidden_dim):
#                 super(NeuralNet, self).__init__()
#                 self.input_dim = input_dim
#                 self.hidden_dim = hidden_dim
#                 self.activationSigmoid = nn.Sigmoid()
#                 self.activationReLu = nn.ReLU()
#                 self.activationNonLinear = nn.Sigmoid()
#                 self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
#                 self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#                 self.layer_out = nn.Linear(self.hidden_dim, 1, bias=False)
#                 
#             # Input is of size
#             def forward(self, x, y):
#                 X = self.layer_input(torch.concat((x,y),dim=1))
#                 
#                 xy = self.layer11(self.activationReLu(X))
#    
#                 xy = self.layer_out(self.activationReLu(xy)+X)
#                 return xy
# =============================================================================
        
        
    class NeuralNet(nn.Module):
            
            def __init__(self, input_dim, hidden_dim):
                super(NeuralNet, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.activation = nn.ReLU()
                # self.activation = nn.Sigmoid()
                self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
                self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                # self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                # self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer_out = nn.Linear(self.hidden_dim, 1, bias=False)
                
            # Input is of size
            def forward(self, x, y):
                X = self.layer_input(torch.concat((x,y),dim=1))

                xy = self.layer11(X)
                xy = self.activation(xy)
                xy = self.layer12 (xy)
                
                # xy = self.activation(xy)+X
                
                # xy = self.layer21(xy)
                # xy = self.activation(xy)
                # xy = self.layer22 (xy)
                
                xy = self.layer_out(self.activation(xy)+X)
                return xy


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
                
                # self.layer31 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                # self.layer32 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                
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
    
        
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.xavier_normal_(m.weight)
            #torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            #torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def train(f,T,X_Train,Y_Train, iterations,learning_rate,ts,Ts,batch_size,k,K):
        f.train()
        T.train()

# =============================================================================
#         optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate*1000) 
#         optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate*1000)
# =============================================================================
        
        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/1) # *10
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/5)
# =============================================================================
#         optimizer_T = torch.optim.SGD(T.parameters(), lr=learning_rate,momentum=0.9) 
#         optimizer_f = torch.optim.SGD(f.parameters(), lr=learning_rate,momentum=0.9)
# =============================================================================
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
       
        inner_iterations = 10
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        for i in range(iterations):
            idx = torch.randperm(X1.shape[0])[:batch_size]
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            for j in range(inner_iterations):
                map_T = T.forward(X_train,Y_shuffled)
                f_of_map_T= f.forward(map_T,Y_shuffled) 
                loss_T = - f_of_map_T.mean() + 0.5*((X_train-map_T)*(X_train-map_T)).sum(axis=1).mean()

                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()
                
            f_of_xy = f.forward(X_train,Y_train) 
            map_T = T.forward(X_train,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            loss_f = -f_of_xy.mean() + f_of_map_T.mean()

            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()
            
            scheduler_f.step()
            scheduler_T.step()
                
            if  (i+1)==iterations:# or i%500==0:
                with torch.no_grad():
                    f_of_xy = f.forward(X_Train,Y_Train) 
                    map_T = T.forward(X_Train,Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T,Y_Train_shuffled) 
                    loss_f = f_of_xy.mean() - f_of_map_T.mean()
                    loss = f_of_xy.mean() - f_of_map_T.mean() + 0.5*((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %(k+1,K,ts,Ts-1,i+1,iterations,loss.item()))
                    
            

            

    def Normalization(X,Type = 'None'):
        ''' Normalize Date with type 'MinMax' out data between [0,1] or 'Mean' for mean 0 and std 1 '''
        if Type == 'None':
            return 0,0,X
        elif Type == 'Mean':
            Mean_X_training_data = torch.mean(X)
            Std_X_training_data = torch.std(X)
            return Mean_X_training_data , Std_X_training_data , (X - Mean_X_training_data)/Std_X_training_data
        elif Type == 'MinMax':
            Min = torch.min(X) 
            Max = torch.max(X)
            return Min , Max , (X-Min)/(Max-Min)

            
    def Transfer(M,S,X,Type='None'):
        '''Trasfer test Data to normalized data using knowledge of training data
        M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization '''
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return (X - M)/S
        elif Type == 'MinMax':
            return (X - M)/(S - M)
        
    def deTransfer(M,S,X , Type = 'None'):
        ''' Detransfer the normalized data to the origin set
         M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization'''  
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return X*S + M
        elif Type == 'MinMax':
            return X*(S - M) + M
        
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

    #
    
    SAVE_all_X_OT = np.zeros((AVG_SIM,N,J,L))

    MAP_T_param = np.load('selected_maps.npz',allow_pickle=True)['select_map'].tolist()
    X_prior = np.load('selected_maps.npz',allow_pickle=True)['prior']
    # i = 1
    # file_name = './DATA/DATA_file_param_'+str(i)+'.npz'
    # MAP_T_param = np.load(file_name,allow_pickle=True)['OT_param_dict'].tolist()
    # X_prior = np.load(file_name,allow_pickle=True)['X_prior']
    
    avg_dist = np.mean(np.load('distance_MMD.npz')['D'])
    
    start_time = time.time()
    
    Distance = np.zeros((AVG_SIM,N-1))
    save_index = np.zeros((AVG_SIM,N-1))
    
    N_maps = len(MAP_T_param)
    for k in range(AVG_SIM):
        
        sigma = 1
        
        y = Y[k,]

        ITERS = ITERATION
        LR = LearningRate
        
        # convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)
        MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON)
        
        # convex_f.apply(init_weights)
        # MAP_T.apply(init_weights)     
        
        
       
        X0 = X0_C[k,].T
        X1 = np.zeros((J,L))
        Y1 = np.zeros((J,dy))
        SAVE_all_X_OT[k,0,:,:] = X0
        #plt.figure()
        for i in range(N-1):
           
            sai_train = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J)
            if Odeint:
                sai_train = sai_train.T
                X1 = ((odeint(A, (X0.T).reshape(-1), t[i:i+2])[1,]).reshape(L,J) + sai_train).T
            else: 
                X1 = X0 + (((A(X0.T,t[i]).T)*tau)  + sai_train)
            
            # eta_train = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),J)
            # Y1 = np.array(h(X1.T).T + eta_train)
            # X1_train = torch.from_numpy(X1)
            # X1_train = X1_train.to(torch.float32)
            # Y1_train = torch.from_numpy(Y1)
            # Y1_train = Y1_train.to(torch.float32)
            # X1_train = X1_train.to(device)
            # Y1_train = Y1_train.to(device)
            
            
            Y1_true = y[i+1,:]
            Y1_true = torch.from_numpy(Y1_true)
            Y1_true = Y1_true.to(torch.float32)
            
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_true = Y1_true.to(device)
            
                
            distance = np.zeros(N_maps)
            
            mean =  X1_test.mean(axis=0,keepdims=True)
            std = X1_test.std(axis=0,keepdims=True)
            x1 = (X1_test - mean)/std
            
            N_particles_check = J#int(J/5) #J
            for j in range(N_maps):
                x_ot = (torch.from_numpy(X_prior[j,:,:]).to(torch.float32) - mean)/std 
                # print(X_prior[j,:J,:].shape)
                # s
                distance[j] = MMD(x1[:N_particles_check,], x_ot[:N_particles_check,], kernel,sigma)
                
            
            if nearest:    
                # nearest map
                j = np.argmin(distance)
                MAP_T.load_state_dict(MAP_T_param[j])
                map_T = MAP_T.forward(X1_test, Y1_true*torch.ones((X1_test.shape[0],dy)))
            else:
                # weighted map
                distance = np.exp(-(distance - distance.min()))
                weight = distance/distance.sum()
                map_T = 0
                for j in range(N_maps):
                    MAP_T.load_state_dict(MAP_T_param[j])
                    map_T += weight[j]*MAP_T.forward(X1_test, Y1_true*torch.ones((X1_test.shape[0],dy)))

            print('K:%d'%(N_maps),'Simu#%d/%d  '%(k+1,AVG_SIM), 'time:%d/%d'%(i+1,N), 'Select ind:%d'%(j), "distance: ",distance[j]/avg_dist)

            Distance[k,i] = distance[j]/avg_dist
            save_index[k,i] = j
            # print(i)
            
            
            if device.type == 'mps':
                X0 = map_T.cpu().detach().numpy()
            else:
                X0 = map_T.detach().numpy()
            
            SAVE_all_X_OT[k,i+1,:,:] = map_T.detach().numpy()
            
            

    print("--- OT_nearst time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT.transpose(0,1,3,2),Distance,save_index, time.time() - start_time