import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from EnKF import EnKF
from SIR import SIR
# =============================================================================
# from OT_new import OT
# =============================================================================
# from OT_skip_window import OT_skip
from OTF import OTF
from scipy.integrate import odeint
from KF import KF
# from OT_nearest import OT_nearest

from A_OTF_MMD import A_OTF_MMD
from A_OTF_W2 import A_OTF_W2
from select_maps_fun import select_maps_fun
#%matplotlib auto
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=13)          # controls default text sizes


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
T =  10 # final time in seconds
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
AVG_SIM = 5 # Number of Simulations to average over


# Num_selected_maps = [20]
Num_selected_maps = [1,2,5,10,20,50]
# Num_selected_maps = [1,20]

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



X_KF  = KF(Y_True,X0,F,H,t,tau,Noise)
X_SIR = SIR(Y_True,X0,A,h,t,tau,Noise)
X_EnKF = EnKF(Y_True,X0,A,h,t,tau,Noise)


X_OT, time_OT = OTF(Y_True,X0,parameters,A,h,t,tau,Noise)



Map_index_offline = []

X_NEAREST = []
Distance = []
time_nearest = []
Map_index_online = []

X_NEAREST_MMD = []
X_WEIGHTED_W2 = []
X_WEIGHTED_MMD = []

for k in Num_selected_maps:
    print(k)
    S = select_maps_fun(k,method='d_mmd') # method='d_mmd',d_w2,d_T default : d_T
    Map_index_offline.append(S)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=True) 
    
    X_NEAREST.append(X_OT_nearest)
    Distance.append(distance)
    time_nearest.append(t_nearest)
    Map_index_online.append(nearest_index)
    
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=True) 
    X_NEAREST_MMD.append(X_OT_nearest)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_W2(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False) 
    X_WEIGHTED_W2.append(X_OT_nearest)
    
    X_OT_nearest,distance,nearest_index,t_nearest = A_OTF_MMD(Y_True,X0,parameters,A,h,t,tau,Noise,nearest=False)  
    X_WEIGHTED_MMD.append(X_OT_nearest)


           


#%%
n_true = int(1e5)
X0_true = np.zeros((AVG_SIM,L,n_true))
for k in range(AVG_SIM):    
    X0_true[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),n_true))
X_true_particles = SIR(Y_True,X0_true,A,h,t,tau,Noise)

# for l in range(AVG_SIM):
#     plt.figure(figsize=(8,8))
#     for num_plot_state in range(L):
#         plt.subplot(2,1,num_plot_state+1)
#         plt.plot(t,X_true_particles[l,:,num_plot_state,:1000],'b',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,num_plot_state],'k--')
#         plt.ylabel(r'$X_{True}$')
#         plt.xlabel('time')

#%%
import ot
W2 = {}
for i in range(len(Num_selected_maps)):
    W2[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_NEAREST[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M)) 
        W2[str(int(i))].append(w2/AVG_SIM)

w2_nearest = []
for i in range(len(Num_selected_maps)):
    w2_nearest.append(sum(W2[str(int(i))])/len(W2[str(int(i))]))

W2_ot = []
for j in range(len(t)):
    if j%100==0:
        print("OT , j: ",j)
    
    w2 = 0
    for k in range(AVG_SIM):
        M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_OT[k,j,:].T) 
            
        # Uniform weights if distributions are unweighted
        a = np.ones(1000) / 1000 # Uniform weights for X
        b = np.ones(J) / J # Uniform weights for Y
            
        # Compute the Wasserstein distance (emd2 returns the squared distance)
        w2 += np.sqrt(ot.emd2(a, b, M))
    W2_ot.append(w2/AVG_SIM)
  
w2_enkf = []
w2_sir = []
for j in range(len(t)):
    if j%100==0:
        print("EnKF, SIR , j: ",j)
        
    w2_en = 0
    w2_s = 0
    for k in range(AVG_SIM):
        M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_EnKF[k,j,:].T) 
            
        # Uniform weights if distributions are unweighted
        a = np.ones(1000) / 1000 # Uniform weights for X
        b = np.ones(J) / J # Uniform weights for Y
            
        # Compute the Wasserstein distance (emd2 returns the squared distance)
        w2_en += np.sqrt(ot.emd2(a, b, M))
        
        M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_SIR[k,j,:].T) 
        w2_s += np.sqrt(ot.emd2(a, b, M))
        
    w2_enkf.append(w2_en/AVG_SIM)
    w2_sir.append(w2_s/AVG_SIM)

 
W2 = {}
for i in range(len(Num_selected_maps)):
    W2[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        w2 = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_WEIGHTED_W2[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M)) 
        W2[str(int(i))].append(w2/AVG_SIM)

w2_weighted = []
for i in range(len(Num_selected_maps)):
    w2_weighted.append(sum(W2[str(int(i))])/len(W2[str(int(i))]))

    
mmd = {}
for i in range(len(Num_selected_maps)):
    mmd[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        mmd_sum = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_NEAREST_MMD[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            mmd_sum += np.sqrt(ot.emd2(a, b, M)) 
        mmd[str(int(i))].append(mmd_sum/AVG_SIM)

mmd_nearest = []
for i in range(len(Num_selected_maps)):
    mmd_nearest.append(sum(mmd[str(int(i))])/len(mmd[str(int(i))]))

mmd = {}
for i in range(len(Num_selected_maps)):
    mmd[str(int(i))] = []
for i in range(len(Num_selected_maps)):
    for j in range(len(t)):
        if j%100==0:
            print("k: ",Num_selected_maps[i],", j: ",j)
            
        mmd_sum = 0
        for k in range(AVG_SIM):
            M =  ot.dist(X_true_particles[k,j,:,:1000].T, X_WEIGHTED_MMD[i][k,j,:].T) 
                
            # Uniform weights if distributions are unweighted
            a = np.ones(1000) / 1000 # Uniform weights for X
            b = np.ones(J) / J # Uniform weights for Y
                
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            mmd_sum += np.sqrt(ot.emd2(a, b, M)) 
        mmd[str(int(i))].append(mmd_sum/AVG_SIM)

mmd_weighted = []
for i in range(len(Num_selected_maps)):
    mmd_weighted.append(sum(mmd[str(int(i))])/len(mmd[str(int(i))]))

#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)          # controls default text sizes
 
plt.figure(figsize=(9,8))    
plt.axhline(y=sum(w2_enkf)/len(w2_enkf),color='g',label='EnKF',lw=2,linestyle='--')
plt.axhline(y=sum(w2_sir)/len(w2_sir),color='b',label='SIR',lw=2,linestyle='--')
plt.axhline(y=sum(W2_ot)/len(W2_ot),color='m',label='OT',lw=2,linestyle='--')



plt.plot(Num_selected_maps,w2_weighted,'v-',lw=2,label=r'$\rho_{W_2}$, Weighted',markersize=10)
plt.plot(Num_selected_maps,w2_nearest,'s-',lw=2,label=r'$\rho_{W_2}$, Nearest',markersize=10)

plt.plot(Num_selected_maps,mmd_weighted,'o-',lw=2,label=r'$\rho_{MMD}$, Weighted',markersize=10)
plt.plot(Num_selected_maps,mmd_nearest,'D-',lw=2,label=r'$\rho_{MMD}$, Nearest',markersize=10)


plt.xlabel("# of selected maps (K)",fontsize=20)
plt.ylabel(r"$W_2$",fontsize=20)
# plt.title('')
# plt.legend(loc=1,fontsize=20, bbox_to_anchor=(1, 0.95))
# plt.legend(loc=2,fontsize=20)
plt.xscale('log')
# plt.yscale('log')

# Define the text and properties of the text box.
textstr = r'$d_{T}$'
# textstr = r'$d_{W_2}$'
# textstr = r'$d_{MMD}$'
bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# Add the text box in the top left corner of the axes.
plt.text(5, 3,textstr ,fontsize=30, bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)) 
plt.show()