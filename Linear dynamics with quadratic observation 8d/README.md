This folder corresponds to the Linear dynamics with quadratic observation example in the paper.
Consider

$$
\begin{aligned}
        X_{t} &= \begin{bmatrix}
        \alpha & \sqrt{1-\alpha^2}
        \\
        -\sqrt{1-\alpha^2} & \alpha
    \end{bmatrix}
    X_{t-1} + \sigma V_t\\
    Y_t &= X_t^2 + \sigma W_t
\end{aligned}
$$

for $t=1,2,\dots$ where $X_t\in \mathbb{R}^2,~ Y_t \in \mathbb{R}^2,~ V_t$ and $W_t$ are i.i.d sequences of $8$-dimensional standard Gaussian random variables, $\alpha=0.9$ and $\sigma^2=0.01$. 

The files and their content:
- 'main_save_param.py': Train the OTF maps and save their parameters using the 'OTF_save_param.py' function.
- 'distance_matrix.py': Generate the distance matrix D according to the desired distance function $d_{T},d_{W_2}$, and $d_{MMD}$ 
- 'main.py': Run the code using the distance matrix D and generate the first panel of Figure 3.
- 'main_compute_time.py': Generate the $W_2$ distance as a function of $N$ and computational time in Figure 4 third and fourth panels.
- 'main_different_methods.py': Generate the $W_2$ distance as a function of $K$ for different distance functions $d$ and $\rho$ in Figure 3 the right three panels.
- 'main_change_mu0.py': Generate the $W_2$ distance as a function of $\mu_0$ in Figure 4 first panel.
- 'main_change_sigma0.py': Generate the $W_2$ distance as a function of $\sigma_0$ in Figure 4 second panel.
- 'EnKF.py', 'SIR.py', and 'OTF.py' are the ensemble Kalman filter, sequential import resampling particle filter, and optimal transport filtering, respectively.
- 'A_OTF_MMD.py', and 'A_OTF_W2' are the A_OTF algorithms using $\rho_{MMD}$ and $\rho_{W_2}$ distance function, respectively.
- 'select_maps_fun.py' is the function used to apply the K-Medoids algorithm to select K maps.



