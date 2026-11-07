import numpy as np

k = 35.
rho = 7200.
cp = 440.5
A = 100.
omega = np.pi / 40.
alpha = k / rho / cp

Lx = 0.1
Ly = 0.01
Lz = 0.01

Nx = 100
Ny = 1
Nz = 1

def analytical_solution(x, t, N=500):
    x = np.asarray(x, dtype=float)
    T = x / Lx * A * np.sin( omega * t )
    for n in range(1, N + 1):
        beta = alpha * ( ( n * np.pi / Lx ) ** 2 )
        u = - (( 2 * A * omega * ((-1) ** (n + 1)) / (n * np.pi) )
               * (beta * np.cos(omega * t) + omega * np.sin(omega * t) - beta * np.exp(- beta * t)) / (beta ** 2 + omega ** 2)
               * np.sin(n * np.pi * x / Lx))
        T += u
    return T

t = 32.
x = np.linspace(0, Lx, 11)
print(x)
Temperature1 = analytical_solution(x, t)
print(Temperature1)