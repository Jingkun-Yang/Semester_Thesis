import numpy as np

k1 = 0.142
k2 = 3.4
k3 = 0.088
c1 = 2.03e6
c2 = 1.42e6
c3 = 0.419e6
L1 = 0.43e-3
L2 = 9.14e-3
L3 = 25.4e-3
Lx = L1 + L2 + L3

sensor1_point = np.array([L1, 0.])
sensor2_point = np.array([L1 + L2, 0.])

Width = 0.01e-3

Nx1 = 2
Nx2 = 20
Nx3 = 40

T_t0 = 31. # initial temperature

q = 7800. # heat flux starts at t = 1.9, ends at t = 21.9

q_guess = 4000.

dt = 0.1
t_end = 40.
t_heat = 1.9
t_cool_down = 21.9

ele_type = 'QUAD4'