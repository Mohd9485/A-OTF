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
from KF import KF

from A_OTF_MMD import A_OTF_MMD
from A_OTF_W2 import A_OTF_W2
from select_maps_fun import select_maps_fun
#%matplotlib auto

import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=15)          # controls default text sizes


plt.close('all')

# Choose h(x) here, the observation rule
def h(x):
    return x[0].reshape(1,-1)*x[0].reshape(1,-1)

def A(x,t=0):
    return F @ (x)




def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    Odeint = True*0
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
L = 2 # number of states
tau = 1e-1 # timpe step 
T = 10 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = 1 # number of states observed

# dynmaical system
H = np.array([[1,0]]) 
alpha = 0.9
a = alpha
b= np.sqrt(1-alpha**2)
c = alpha
F = np.array([[a, -b],[b,c]]) 


noise = np.sqrt(1e-1) # noise level std
sigmma = noise # Noise in the hidden state
sigmma0 = 1#5*noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1#/noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]

J = int(1e3/4) # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over


Num_selected_maps = [20]
# Num_selected_maps = [1,2,5,10,20]
# Num_selected_maps = [1,2,5,10,20,50,100]


# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32) #64
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64*1/1) #128
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024/1) 
parameters['Final_Number_ITERATION'] = int(64/1) #int(64) #ITERATION 
parameters['Time_step'] = N


t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))



X_NEAREST = []
Distance = []
time_nearest = []
Map_index_offline = []
Map_index_online = []
for k in Num_selected_maps:
    print(k)
    S = select_maps_fun(k,method='d_w2') # method='d_mmd',d_w2,d_T default : d_T
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False) 
    # X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False) 
    
    X_NEAREST.append(X_OT_nearest)
    Distance.append(distance)
    time_nearest.append(t_nearest)
    Map_index_offline.append(S)
    Map_index_online.append(nearest_index)

X_KF  = KF(Y_True,X0,F,H,t,tau,Noise)
X_SIR = SIR(Y_True,X0,A,h,t,tau,Noise)
X_EnKF = EnKF(Y_True,X0,A,h,t,tau,Noise)


X_OT, time_OT = OTF(Y_True,X0,parameters,A,h,t,tau,Noise)
#%%
num_plot_state = 1
p=100
for l in range(AVG_SIM):
    plt.figure(figsize=(15,10))
    plt.subplot(4,1,1)
    plt.plot(t,X_EnKF[l,:,num_plot_state,:p],'g',ls='none',marker='o',ms=4,alpha = 0.1)
    plt.plot(t,X_True[l,:,num_plot_state],'k--',label='True state')
    plt.ylabel('EnKF')
    plt.title('k =  %d'%(k))
    plt.legend()
    
    plt.subplot(4,1,2)
    plt.plot(t,X_SIR[l,:,num_plot_state,:p],'b',ls='none',marker='o',ms=4,alpha = 0.1)
    plt.plot(t,X_True[l,:,num_plot_state],'k--')
    plt.ylabel('SIR')
    
    plt.subplot(4,1,3)
    plt.plot(t,X_OT_nearest[l,:,num_plot_state,:p],'C4',ls='none',marker='o',ms=4,alpha = 0.1)
    plt.plot(t,X_True[l,:,num_plot_state],'k--')
    plt.ylabel(r'$OT_{nearest}$')
    
    plt.subplot(4,1,4)
    plt.plot(t,X_OT[l,:,num_plot_state,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
    plt.plot(t,X_True[l,:,num_plot_state],'k--')
    plt.ylabel(r'$OT$')
    plt.xlabel('time')


#%%

import seaborn as sns
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=15)          # controls default text sizes
 
sim=0
true_particle = int(1e5)
X0_true = np.zeros((1,L,true_particle))
X0_true[0,] = 1*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),true_particle))
X_true_dist = SIR(Y_True[sim,].reshape(1,N,dy),X0_true,A,h,t,tau,Noise)


for num_plot_state in range(1,2):
    # num_plot_state = 2
    n_bins = 50
    fontsize = 20
    if num_plot_state == 0:
        y_lim = [-10,10]
        n_ylabel = 3
    elif num_plot_state == 1:
        y_lim = [-10,10]
        n_ylabel = 3
    
    
    position_bins = np.linspace(y_lim[0], y_lim[1], n_bins)  # Define position bins
    
    plt.figure(figsize=(8,8))
    # plt.figure(figsize=(16,8),facecolor='#F9F9EE')
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_true_dist[0,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
    
    plt.subplot(5,1,1)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel('True',fontsize=fontsize)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_EnKF[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,2)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel('EnKF',fontsize=fontsize)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_SIR[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,3)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel('SIR',fontsize=fontsize)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_OT[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,4)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel('OTF',fontsize=fontsize)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_OT_nearest[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,5)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    # ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.xticks(ticks=np.linspace(0, N, 11),labels= np.round(np.linspace(0, N*tau, 11),1))
    plt.ylabel('A-OTF',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    plt.show()