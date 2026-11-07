"""JAX-FEM problem, boundary functions, and analytical reference solution."""
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from jax_fem.problem import Problem
import config as cfg


class nafems_t4_steady(Problem):
    def get_tensor_map(self):
        return lambda T_grad: cfg.k * T_grad

    def get_surface_maps(self):
        def robin(T, x):
            return cfg.h * (T - cfg.T_amb)
        return [robin, robin]


def left(point):
    return jnp.isclose(point[0], 0.0, atol=1e-5)


def right(point):
    return jnp.isclose(point[0], cfg.Lx, atol=1e-5)


def bottom(point):
    return jnp.isclose(point[1], 0.0, atol=1e-5)


def top(point):
    return jnp.isclose(point[1], cfg.Ly, atol=1e-5)


def dirichlet_val_bottom(point):
    return cfg.T_bottom


def calculate_eigenvalues(N_modes):
    eigenvalues = []

    def eigenvalue_equation(mu_n):
        return mu_n * np.tan(mu_n * cfg.Lx) - cfg.h / cfg.k

    for n in range(N_modes):
        lower = (n + 1e-9) * np.pi / cfg.Lx
        upper = (n - 1e-9 + 0.5) * np.pi / cfg.Lx
        mu_n = brentq(eigenvalue_equation, lower, upper)
        eigenvalues.append(mu_n)
    return np.array(eigenvalues)


def calculate_an(mu):
    numerator = np.sin(mu * cfg.Lx) / mu
    denominator = cfg.Lx / 2 + np.sin(2 * mu * cfg.Lx) / (4 * mu)
    return (cfg.T_bottom - cfg.T_amb) * numerator / denominator


def Fy_function(mu, y):
    numerator = ((1 + cfg.h / cfg.k / mu)
                 + (1 - cfg.h / cfg.k / mu) * np.exp(-2 * mu * (cfg.Ly - y)))
    denominator = ((1 + cfg.h / cfg.k / mu)
                   + (1 - cfg.h / cfg.k / mu) * np.exp(-2 * mu * cfg.Ly))
    return np.exp(-mu * y) * numerator / denominator


def analytical_temperature(x, y, mu_values, an_values):
    """Temperature at a scalar point or arrays of points, using precomputed modes."""
    T = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), cfg.T_amb, dtype=float)
    for mu, an in zip(mu_values, an_values):
        T += an * np.cos(mu * x) * Fy_function(mu, y)
    return T


def calculate_global_L2_error(problem, T, mu_values, an_values):
    """Quadrature approximation of sqrt(integral_domain (T_h - T_exact)^2 dOmega)."""
    fe = problem.fes[0]
    T_numerical_quad = np.asarray(fe.convert_from_dof_to_quad(T)[..., 0])
    quad_points = np.asarray(problem.physical_quad_points)
    coordinates = quad_points.reshape(-1, 2)
    T_analytical = analytical_temperature(
        coordinates[:, 0], coordinates[:, 1], mu_values, an_values
    )
    T_analytical_quad = T_analytical.reshape(quad_points.shape[:2])
    JxW = np.asarray(fe.JxW)
    error_squared = (T_numerical_quad - T_analytical_quad) ** 2
    elements_integral = np.sum(error_squared * JxW, axis=1)
    domain_integral = np.sum(elements_integral, axis=0)
    return float(np.sqrt(domain_integral))
