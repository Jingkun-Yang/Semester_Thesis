import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

Lx, Ly = 0.6, 1.


k = 52.
h = 750.

T_bottom = 100.
T_amb = 0.

Nx_Ny = [[6, 10],
         [12, 20],
         [24, 40],
         [48, 80],
         [96, 160],
         [192, 320]]

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

mesh_size = []
E_L2 = []
for Nx, Ny in Nx_Ny:
    dx = Lx / Nx
    dy = Ly / Ny
    mesh_size.append(max(dx, dy))

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
    quad_points = problem.physical_quad_points
    points_coordinates = np.asarray(quad_points).reshape(-1, 2)
    T_analytical = np.array([
        analytical_temperature(x, y, N_modes)
        for x, y in points_coordinates
        ])
    T_analytical_quad = T_analytical.reshape(quad_points.shape[:2])
    T_numerical_quad = np.asarray(T_gaussian_point)
    JxW = np.asarray(fe.JxW)
    error_squared = (T_numerical_quad - T_analytical_quad) ** 2
    elements_integral = np.sum(error_squared * JxW, axis=1)
    domain_integral = np.sum(elements_integral, axis=0)
    E_L2.append(np.sqrt(domain_integral))
    print(f"current Nx and Ny: {Nx, Ny}")

print(mesh_size)
print(E_L2)

fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.loglog(mesh_size, E_L2, 'o-', label='FEM L2 Error')
ax1.set_xlabel('Mesh size h [m]')
ax1.set_ylabel('L2 error')
ax1.set_title('L2 error convergence')
ax1.grid(True, which='both')
plt.show()