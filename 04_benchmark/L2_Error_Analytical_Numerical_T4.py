import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import numpy as np
from scipy.optimize import brentq

Lx, Ly = 0.6, 1.


k = 52.
h = 750.

T_bottom = 100.
T_amb = 0.

Nx = 6 # 6, 12, 24, 48, 96, may be 192
Ny = 10 # 10, 20, 40, 80, 160, may be 320

N_modes = 1000

class nafems_t4_steady(Problem):
    def get_tensor_map(self):
        return lambda T_grad: k * T_grad
    def get_surface_maps(self):
        def robin(T, x):
            return h * (T - T_amb)
        return [robin, robin]


def left(point):
    return jnp.isclose(point[0], 0., atol=1e-5)

def right(point):
    return jnp.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-5)

def top(point):
    return jnp.isclose(point[1], Ly, atol=1e-5)

def dirichlet_val_bottom(point):
    return T_bottom



# finish to calculate the numerical temperature

def calculate_eigenvalues(N_modes):
    eigenvalues = []
    def eigenvalue_equation(mu_n):
        return mu_n * np.tan(mu_n * Lx) - h / k
    for n in range(N_modes):
        lower = (n + 1e-9) * np.pi / Lx
        upper = (n - 1e-9 + 0.5) * np.pi / Lx
        mu_n = brentq(
            eigenvalue_equation,
            lower,
            upper
        )
        eigenvalues.append(mu_n)
    return np.array(eigenvalues)
mu_values = calculate_eigenvalues(N_modes)

# print(mu[:5])

def calculate_an(mu):
    numerator = np.sin(mu * Lx) / mu
    denominator = (Lx / 2
                   + np.sin(2 * mu * Lx) / (4 * mu))
    an = ((T_bottom - T_amb)
          * numerator / denominator)
    return an
an_values = calculate_an(mu_values)

def Fy_function(mu, y): # only one mu value
    numerator = (
        (1 + h / k / mu)
        + (1 - h / k / mu) * np.exp(- 2 * mu * (Ly - y))
    )
    denominator = (
        (1 + h / k / mu)
        + (1 - h / k / mu) * np.exp(- 2 * mu * Ly)
    )
    return np.exp(- mu * y) * numerator / denominator

def analytical_temperature(x, y, N_modes):
    T = T_amb
    for n in range(N_modes):
        mu = mu_values[n]
        an = an_values[n]
        X = np.cos(mu * x)
        Y = Fy_function(mu, y)
        T += an * X * Y
    return T

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

location_fns_dirichlet = [bottom]
value_fns = [dirichlet_val_bottom]
vecs = [0]

dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

location_fns = [top, right]

problem = nafems_t4_steady(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

sol_list = solver(problem, solver_options={'spsolve_solver':{}})
T = sol_list[0]

fe = problem.fes[0]

T_gaussian_point = fe.convert_from_dof_to_quad(T)[..., 0]
# [..., 0] means [:, :, 0], I need an array like:
# [number of elements, number of points, number of values]
# we have the temperature at gaussian points
# (60, 4)

quad_points = problem.physical_quad_points
# (60, 4, 2)

# calculate the analytical temperature at gaussian points

points_coordinats = np.asarray(quad_points).reshape(-1, 2)
# -1 means let python automatically calculate this number
# if I input -1 in the second position, like (240, -1) if I have 60 elements
# python will calculate the number of column automatically
# (240, 2)

T_analytical = np.array([
    analytical_temperature(x, y, N_modes)
    for x, y in points_coordinats
    ])
# here x and y mean points_coordinats[i, 0] and points_coordinats[i, 1]
# distinction: points_coordinats[0] = [x0, y0], points_coordinats[1] = [x1, y1]
# points_coordinates[:, 0] = all 240 x, points_coordinates[:, 1] = all 240 y
# (240, 1)

T_analytical_quad = T_analytical.reshape(quad_points.shape[:2])
# (60, 4)

# N = fe.shape_vals
# only get the shape functions' values at each Gaussian points
# shape_functions.shape = (4, 4), which means 4 values
# N1 to N4 for each points, there are four Gaussian points 
# each element

# print(T_gaussian_point.shape)
# print(T_analytical.shape)
# print(quad_points.shape)
# print(T_analytical_quad.shape)

T_numerical_quad = np.asarray(T_gaussian_point)
# from jnp array to np array
JxW = np.asarray(fe.JxW)

# print(JxW.shape)

# now I have three (60, 4) arrays
error_squared = (T_numerical_quad - T_analytical_quad) ** 2
# print(error_squared.shape)
# (60, 4)
elements_integral = np.sum(error_squared * JxW, axis=1)
# print(elements_integral.shape)
# (60, )
domain_integral = np.sum(elements_integral, axis=0)
# print(domain_integral.shape)
# ()
E_L2 = domain_integral ** 0.5