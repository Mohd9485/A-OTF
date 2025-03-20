This folder corresponds to the Linear dynamics with linear observation example in the paper.
Consider

$$
\begin{aligned}
        X_{t} &= \begin{bmatrix}
        \alpha & \sqrt{1-\alpha^2}
        \\
        -\sqrt{1-\alpha^2} & \alpha
    \end{bmatrix}
    X_{t-1} + \sigma V_t\\
    Y_t &= h(X_t) + \sigma W_t
\end{aligned}
$$

for $t=1,2,\dots$ where $X_t\in \mathbb{R}^2,~ Y_t \in \mathbb{R},~ V_t$ and $W_t$ are i.i.d sequences of $2$-dimensional and one-dimensional standard Gaussian random variables, $\alpha=0.9$ and $\sigma^2=0.1$. Two observation functions are of interest:

$$
\begin{aligned}
    h(X_t)=X_t(1),
\end{aligned}
$$

where $X_t(1)$ is the first component of the vector $X_t$. We refer to these observation models as linear and quadratic, respectively.

The files and their content:
- 'main_save_param.py': Train the OTF maps and save their parameters using the 'OTF_save_param.py' function.
- 'distance_matrix.py': Generate the distance matrix D according to the desired distance function $d_{T},d_{W_2}$, and $d_{MMD}$ 
- 'main.py': Run the code using the distance matrix D and generate the final figures.
- 'KF.py', 'EnKF.py', 'SIR.py', and 'OTF.py' are the Kalman filter, ensemble Kalman filter, sequential import resampling particle filter, and optimal transport filtering, respectively.
- 'A_OTF_MMD.py', and 'A_OTF_W2' are the A_OTF algorithms using $\rho_{MMD}$ and $\rho_{W_2}$ distance function, respectively.
- 'select_maps_fun.py' is the function used to apply the K-Medoids algorithm to select K maps.

If you wish to run the code directly, you can run 'main.py' with the saved data as '.npz' files. 
