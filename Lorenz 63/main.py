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
AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(64/1)
parameters['BATCH_SIZE'] = int(64/1)
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024/1) #1024*2 
parameters['Final_Number_ITERATION'] = int(64/2) #int(64*2) #ITERATION 



t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))




Num_selected_maps = [20]#,5,10,20]

X_NEAREST = []
Distance = []
time_nearest = []
Map_index_offline = []
Map_index_online = []
for k in Num_selected_maps:
    print(k)
    S = select_maps_fun(k,method='d_w2') # method='d_mmd',d_w2,d_T default : d_T
    Map_index_offline.append(S)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,nearest=False) 
    
    X_NEAREST.append(X_OT_nearest)
    Distance.append(distance)
    time_nearest.append(t_nearest)
    
    Map_index_online.append(nearest_index)


# data is AVG_SIM x N x L x J
X_OT, time_OT = OTF(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint) 

  

X_EnKF = EnKF(Y_True,X0,L63,h,t,tau,Noise,Odeint)
X_SIR = SIR(Y_True,X0,L63,h,t,tau,Noise,Odeint)
#%%
import seaborn as sns 
sim=0
true_particle = int(1e5)
X0_true = np.zeros((1,L,true_particle))
X0_true[0,] = 1*np.transpose(np.random.multivariate_normal(np.zeros(L),Noise[2]*Noise[2] * np.eye(L),true_particle))
X_true_dist = SIR(Y_True[sim,].reshape(1,N,dy),X0_true,L63,h,t,tau,Noise,Odeint)

for num_plot_state in range(1):
    # num_plot_state = 2
    n_bins = 50
    fontsize = 20
    if num_plot_state == 0:
        y_lim = [-30,30]
        n_ylabel = 3
    elif num_plot_state == 1:
        y_lim = [-35,35]
        n_ylabel = 3
    elif num_plot_state == 2:
         y_lim = [-20,60]
         n_ylabel = 3
    
    
    position_bins = np.linspace(y_lim[0], y_lim[1], n_bins)  # Define position bins
    
    plt.figure(figsize=(9,8))
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
        density, _ = np.histogram(X_NEAREST[0][sim,:,num_plot_state,][n], bins=position_bins, density=True)
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