import numpy as np
import time

def SIR(Y,X0,A,h,t,tau,Noise,Odeint):
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
    x_SIR =  np.zeros((AVG_SIM,N,L,J))

    rng = np.random.default_rng()
    for k in range(AVG_SIM):
        x_SIR[k,0,] = X0[k,]
        y = Y[k,]
        
        for i in range(N-1):
            sai_SIR = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
            x_SIR[k,i+1,] = x_SIR[k,i,]+ A(x_SIR[k,i,],t[i])*tau + sai_SIR
            
            
            W = np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)
            W = W - np.min(W)
            W = np.exp(-W).T
            W = W/np.sum(W)
            
            #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
            index = rng.choice(np.arange(J), J, p = W)
            x_SIR[k,i+1,] = x_SIR[k,i+1,:,index].T
    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    return x_SIR