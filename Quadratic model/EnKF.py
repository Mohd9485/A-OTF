import numpy as np
import time
def EnKF(Y,X0,A,h,t,tau,Noise):
    AVG_SIM = X0.shape[0]
    L = X0.shape[1]
    N = Y.shape[1]
    dy = Y.shape[2]
    J = X0.shape[2]
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = 1/noise # Amplifiying the initial state 
    start_time = time.time()
    SAVE_X_EnKF =  np.zeros((AVG_SIM,N,L,J))
# =============================================================================
#     SAVE_True_X = np.zeros((AVG_SIM,N,L))
#     SAVE_True_Y = np.zeros((AVG_SIM,N,dy))
# =============================================================================
    mse_EnKF =  np.zeros((N,AVG_SIM))
    for k in range(AVG_SIM):
        
# =============================================================================
#         x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma)
#         SAVE_True_X[k,] = x
#         SAVE_True_Y[k,] = y
# =============================================================================
        y = Y[k,]
        
        x_EnKF  = np.zeros((N,L,J))
        #x_EnKF[0,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))
        x_EnKF[0,] = X0[k,] 
        x_hatEnKF = np.transpose(np.random.multivariate_normal(np.zeros(L),noise*noise * np.eye(L),J))
        y_hatEnKF = np.zeros((dy,J)) 
        
        m_hatEnKF = np.zeros(L)
        o_hatEnKF = np.zeros((N,dy))
        C_hat_vh = np.zeros((L,dy))
        C_hat_hh = np.zeros((dy,dy))
        
        K_EnKF = np.zeros((L,dy))
        
        SAVE_X_EnKF[k,0,:,:] = x_EnKF[0,]
        # EnKF & 3DVAR
        for i in range(N-1):
    # =============================================================================
    #         if (i+1)%50 == 0:
    #             print("EnKF : Simu#%d/%d ,Time Step:%d/%d" %(k,AVG_SIM,i+1,N-1))
    # =============================================================================
            sai_EnKF = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
                
            x_hatEnKF = A(x_EnKF[i,],t[i]) + sai_EnKF
                
            eta_EnKF = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),J)    
            #y_hatEnKF = (x_hatEnKF**3).transpose() + eta_EnKF
            y_hatEnKF = (h(x_hatEnKF)).transpose() + eta_EnKF
            
            m_hatEnKF = x_hatEnKF.mean(axis=1)
            #o_hatEnKF = (x_hatEnKF**3).mean(axis=1)
            o_hatEnKF = (y_hatEnKF.T).mean(axis=1)

            a = (x_hatEnKF.transpose() - m_hatEnKF)
            #b = ((x_hatEnKF**3).transpose() - o_hatEnKF)
            b = (h(x_hatEnKF).transpose() - o_hatEnKF)
            
 
            C_hat_vh = np.matmul(a.transpose(),b)/J
            C_hat_hh = np.matmul(b.transpose(),b)/J
            K_EnKF = np.matmul(C_hat_vh,np.linalg.inv(C_hat_hh + np.eye(dy)*1e-6))
                
            x_EnKF[i+1,:,:] = x_hatEnKF + np.matmul( K_EnKF, (y[i+1,:] - y_hatEnKF).transpose())
            
            SAVE_X_EnKF[k,i+1,:,:] = x_EnKF[i+1,]
    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_EnKF