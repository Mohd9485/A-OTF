The files and their content:
- 'main_save_param.py': Train the OTF maps and save their parameters using the 'OTF_save_param.py' function.
- 'distance_matrix.py': Generate the distance matrix D according to the desired distance function $d_{T},d_{W_2}$, and $d_{MMD}$ 
- 'main.py': Run the code using the distance matrix D and generate the final figures.
- 'KF.py', 'EnKF.py', 'SIR.py', and 'OTF.py' are the Kalman filter, ensemble Kalman filter, sequential import resampling particle filter, and optimal transport filtering, respectively.
- 'A_OTF_MMD.py', and 'A_OTF_W2' are the A_OTF algorithms using $\rho_{MMD}$ and $\rho_{W_2}$ distance function, respectively.
- 'select_maps_fun.py' is the function used to apply the K-Medoids algorithm to select K maps.

If you wish to run the code directly, you can run 'main.py' with the saved data as '.npz' files. 
