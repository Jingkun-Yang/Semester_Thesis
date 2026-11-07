"""Parameters for the NAFEMS T4 steady-state mesh-convergence study."""
Lx, Ly = 0.6, 1.0
k = 52.0
h = 750.0
T_bottom = 100.0
T_amb = 0.0
Nx_Ny = [[6, 10], [12, 20], [24, 40], [48, 80], [96, 160], [192, 320]]
N_modes = 1000
ele_type = "QUAD4"
