import jax
import jax.numpy as jnp
from jax_fem.problem import Problem

import numpy as np
from scipy.optimize import brentq
import T4_Configure as cfg

# numerical solution
class nafems_t4_steady(Problem):
    def get_tensor_map(self):
        return lambda T_grad: cfg.k * T_grad
    def get_surface_maps(self):
        def robin(T, x):
            return cfg.h * (T - cfg.T_amb)
        return [robin, robin]

def left(point):
    return jnp.isclose(point[0], 0., atol=1e-5)

def right(point):
    return jnp.isclose(point[0], cfg.Lx, atol=1e-5)

def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-5)

def top(point):
    return jnp.isclose(point[1], cfg.Ly, atol=1e-5)

def dirichlet_val_bottom(point):
    return cfg.T_bottom

# analytical solution
def calculate_eigenvalues(N_modes):
    eigenvalues = []
    def eigenvalue_equation(mu_n):
        return mu_n * np.tan(mu_n * cfg.Lx) - cfg.h / cfg.k
    for n in range(N_modes):
        lower = (n + 1e-9) * np.pi / cfg.Lx
        upper = (n - 1e-9 + 0.5) * np.pi / cfg.Lx
        mu_n = brentq(
            eigenvalue_equation,
            lower,
            upper
        )
        eigenvalues.append(mu_n)
    return np.array(eigenvalues)

def calculate_an(mu):
    numerator = np.sin(mu * cfg.Lx) / mu
    denominator = (cfg.Lx / 2
                   + np.sin(2 * mu * cfg.Lx) / (4 * mu))
    an = ((cfg.T_bottom - cfg.T_amb)
          * numerator / denominator)
    return an

def Fy_function(mu, y): # only one mu value
    numerator = (
        (1 + cfg.h / cfg.k / mu)
        + (1 - cfg.h / cfg.k / mu) * np.exp(- 2 * mu * (cfg.Ly - y))
    )
    denominator = (
        (1 + cfg.h / cfg.k / mu)
        + (1 - cfg.h / cfg.k / mu) * np.exp(- 2 * mu * cfg.Ly)
    )
    return np.exp(- mu * y) * numerator / denominator

def analytical_temperature(x, y, N_modes, mu_values, an_values):
    T = cfg.T_amb
    for n in range(N_modes):
        mu = mu_values[n]
        an = an_values[n]
        X = np.cos(mu * x)
        Y = Fy_function(mu, y)
        T += an * X * Y
    return T