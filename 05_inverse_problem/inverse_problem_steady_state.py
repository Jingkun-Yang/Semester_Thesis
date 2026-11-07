import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import os
import numpy as np

from jax_fem.problem import Problem # inherit the class Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh
from scipy.optimize import minimize

k = 35. # for a steady-state only k is required.

class nafems_t3_inverse(Problem):
    def get_tensor_map(self):
        return lambda T_grad: k * T_grad

    def get_surface_maps(self):
        def surface_map(T, x, q):
            return -jnp.array([q]) # here I pass q as a parameter, but not a global variable.
        return [surface_map]

    def set_params(self, q):
        num_faces = len(self.boundary_inds_list[0])
        num_face_quads = self.fes[0].num_face_quads
        surface_params = q * jnp.ones(
            (num_faces, num_face_quads)
        )
        self.internal_vars_surfaces = [[surface_params]]

ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 0.1, 0.01, 0.01

meshio_mesh = box_mesh(Nx=5, Ny=1, Nz=1,
                        domain_x=Lx, domain_y=Ly, domain_z=Lz)

mesh = Mesh(meshio_mesh.points, 
            meshio_mesh.cells_dict[cell_type])

# print(meshio_mesh.points)
# print(meshio_mesh.cells)

def left(point):
    return jnp.isclose(point[0], 0, atol=1e-5)

def right(point):
    return jnp.isclose(point[0], Lx, atol=1e-5)

def dirichlet_val_right(point):
    return 0

location_fns_dirichlet = [right]
vecs = [0]
value_fns = [dirichlet_val_right]
dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

location_fns = [left]

problem = nafems_t3_inverse(mesh=mesh,
                            vec=1,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns)

fwd_pred = ad_wrapper(problem)

q_true = 1000.
q_array = jnp.array([q_true])
sol_list = fwd_pred(q_array)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/temperature_t3_sensor.vtu')
os.makedirs(os.path.dirname(vtk_path), exist_ok=True)
save_sol(problem.fes[0], sol_list[0], vtk_path)

T = sol_list[0]
sensor_point = jnp.array([0.02, 0., 0.])

def get_sensor_temperature(T, sensor_point):

    points = jnp.asarray(mesh.points)

    point_index = jnp.where(
        jnp.all(
            jnp.isclose(points, sensor_point, atol=1e-8, rtol=0),
            axis=1 # compare alonge each row, x, y, and z are all the same
        )
    )[0]

    index = int(point_index[0])
    T_sensor = T[index, 0]
    return T_sensor, index

T_sensor_true, sensor_index = get_sensor_temperature(T, sensor_point)
print(f"The temperature of the sensor is {T_sensor_true}")

q_guess = 500.

def objective(q_guess):
    T_guess = fwd_pred(q_guess)[0][sensor_index, 0]
    residual = T_sensor_true - T_guess
    return 0.5 * residual ** 2

objective_grad = jax.grad(objective)

def scipy_objective(q_array):
    q_jax = jnp.asarray(q_array)
    value, gradient = jax.value_and_grad(objective)(q_jax)
    return float(value), np.array(gradient, dtype=float)

result = minimize(
    scipy_objective,
    x0=np.array([q_guess]),
    method="L-BFGS-B",
    jac=True,
    bounds=[(0.0, None)],
    options={
        "gtol": 1e-12,
        "ftol": 1e-15,
        "maxiter": 100
    }
)

q_inverse = float(result.x[0])
relative_error = abs(q_inverse - q_true) / abs(q_true) * 100

print(f"q_true = {q_true:.6f} W/m²")
print(f"q_guess = {q_guess:.6f} W/m²")
print(f"q_inverse = {q_inverse:.6f} W/m²")
print(f"Relative error = {relative_error:.6f} %")
print(f"Final objective = {result.fun:.12e}")
print(f"Optimization success = {result.success}")

sol_inverse = fwd_pred(jnp.array(q_inverse))
T = sol_inverse[0]
T_sensor_inverse, index_inverse = get_sensor_temperature(T, sensor_point)

print(f"T_observed = {float(T_sensor_true):.8f}")
print(f"T_inverse = {float(T_sensor_inverse):.8f}")