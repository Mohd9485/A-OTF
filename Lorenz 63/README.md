This folder corresponds to the Lorenz 63 example in the paper.

$$
\begin{aligned}
\begin{bmatrix}
    \dot{X}(1) \\ 
    \dot{X}(2) \\
    \dot{X}(3)
\end{bmatrix}
&= 
\begin{bmatrix}
    \sigma (X(2) - X(1)) \\
    X(1) (\gamma - X(3)) - X(2) \\
    X(1)X(2) - \beta X(3)   
\end{bmatrix},\quad X_0 \sim \mathcal{N}(\mu_0,\sigma_0^2I_3),
\\
Y_t &= X_t(1) + \sigma_{obs}W_t,
\end{aligned}
$$

where $[X(1),X(2),X(3)]^\top$ are the variables representing the hidden states of the system, and $\sigma$, $\gamma$, and $\beta$ are the model parameters. We choose $\sigma=10$, $\gamma=28$, $\beta=8/3$, $\mu_0 = [0,0,0]^\top$, and $\sigma_{0}^2=10$. The observed noise $W$ is a $1$-dimensional standard Gaussian random variable with $\sigma_{obs}^2=10.$

The files and their content:
- 'main_save_param.py': Train the OTF maps and save their parameters using the 'OTF_save_param.py' function.
- 'distance_matrix.py': Generate the distance matrix D according to the desired distance function $d_{T},d_{W_2}$, and $d_{MMD}$ 
- 'main.py': Run the code using the distance matrix D and generate the final figures.
- 'main_different_methods.py': Using different desired distance function $d_{T},d_{W_2}$, and $d_{MMD}$ method in line $135$ to produce Figure 3. 
- 'KF.py', 'EnKF.py', 'SIR.py', and 'OTF.py' are the Kalman filter, ensemble Kalman filter, sequential import resampling particle filter, and optimal transport filtering, respectively.
- 'A_OTF_MMD.py', and 'A_OTF_W2' are the A_OTF algorithms using $\rho_{MMD}$ and $\rho_{W_2}$ distance function, respectively.
- 'select_maps_fun.py' is the function used to apply the K-Medoids algorithm to select K maps.

If you wish to run the code directly, you can run 'main.py' with the saved '.npz' data files. 
