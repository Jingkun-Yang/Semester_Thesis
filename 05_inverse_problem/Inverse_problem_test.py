
import os
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from scipy.optimize import minimize

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import (
    get_meshio_cell_type,
    Mesh,
    box_mesh,
)


# ============================================================
# 1. Physical parameters
# ============================================================

k = 35.0             # W/(m K)
Lx = 0.1             # m
Ly = 0.01            # m
Lz = 0.01            # m

q_true = 1000.0      # W/m^2
q_initial = 500.0    # W/m^2

x_sensor = 0.08      # m


# ============================================================
# 2. FEM problem
# ============================================================

class NAFEMST3Inverse(Problem):

    def get_tensor_map(self):
        return lambda T_grad: k * T_grad

    def get_surface_maps(self):

        def surface_map(T, x, q):
            # Negative sign: inward heat flux
            return -jnp.array([q])

        return [surface_map]

    def set_params(self, q):
        # One scalar heat flux applied uniformly
        # at all left-boundary surface quadrature points.

        surface_points = self.physical_surface_quad_points[0]

        q_surface = q * jnp.ones(
            surface_points.shape[:2]
        )

        self.internal_vars_surfaces = [[q_surface]]


# ============================================================
# 3. Mesh
# ============================================================

ele_type = "HEX8"
cell_type = get_meshio_cell_type(ele_type)

meshio_mesh = box_mesh(
    Nx=5,
    Ny=1,
    Nz=1,
    domain_x=Lx,
    domain_y=Ly,
    domain_z=Lz,
)

mesh = Mesh(
    meshio_mesh.points,
    meshio_mesh.cells_dict[cell_type],
)


# ============================================================
# 4. Boundary conditions
# ============================================================

def left(point):
    return jnp.isclose(point[0], 0.0, atol=1e-5)


def right(point):
    return jnp.isclose(point[0], Lx, atol=1e-5)


def dirichlet_val_right(point):
    return 0.0


dirichlet_bc_info = [
    [right],
    [0],
    [dirichlet_val_right],
]

location_fns = [left]


# ============================================================
# 5. Initialize problem
# ============================================================

problem = NAFEMST3Inverse(
    mesh=mesh,
    vec=1,
    dim=3,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
)

# Differentiable FEM solver
fwd_pred = ad_wrapper(problem)


# ============================================================
# 6. Find sensor node
# ============================================================

points = np.asarray(mesh.points)

# Select a node on the centerline if possible.
distances = (
    (points[:, 0] - x_sensor)**2
    + (points[:, 1] - Ly / 2)**2
    + (points[:, 2] - Lz / 2)**2
)

sensor_node_id = int(np.argmin(distances))

sensor_position = points[sensor_node_id]

print("Sensor node ID:", sensor_node_id)
print("Sensor position:", sensor_position)

# The chosen node must have the requested x-coordinate.
assert np.isclose(
    sensor_position[0], x_sensor, atol=1e-8
), "Sensor x-position does not coincide with a mesh node."


# ============================================================
# 7. Generate synthetic observations
# ============================================================

sol_list_true = fwd_pred(jnp.array(q_true))

T_obs = sol_list_true[0][sensor_node_id, 0]

T_analytical = (
    q_true * (Lx - x_sensor) / k
)

print("\nTrue heat flux:", q_true)
print("Observed temperature:", float(T_obs))
print("Analytical temperature:", T_analytical)


# ============================================================
# 8. Define objective function
# ============================================================

def sensor_temperature(q):
    sol_list = fwd_pred(q)
    return sol_list[0][sensor_node_id, 0]


def loss(q):
    T_pred = sensor_temperature(q)
    return 0.5 * (T_pred - T_obs)**2


# ============================================================
# 9. Automatic differentiation
# ============================================================

grad_loss = jax.grad(loss)

q0 = jnp.array(q_initial)

print("\nInitial heat flux:", float(q0))
print("Initial loss:", float(loss(q0)))
print("Initial loss gradient:", float(grad_loss(q0)))


# ============================================================
# 10. Verify gradient using finite differences
# ============================================================

h = 1.0  # W/m^2

grad_fd = (
    loss(q0 + h) - loss(q0 - h)
) / (2.0 * h)

grad_ad = grad_loss(q0)

print("\nAD gradient:", float(grad_ad))
print("FD gradient:", float(grad_fd))


# ============================================================
# 11. Inverse optimization
# ============================================================

# Normalize the parameter for better optimizer scaling:
# q = 1000 * p

q_scale = 1000.0

def objective_np(p):
    q = jnp.asarray(p[0] * q_scale)
    return float(loss(q))


def gradient_np(p):
    q = jnp.asarray(p[0] * q_scale)
    dL_dq = grad_loss(q)
    return np.array([float(dL_dq) * q_scale])


result = minimize(
    fun=objective_np,
    x0=np.array([q_initial / q_scale]),
    jac=gradient_np,
    method="BFGS",
    options={
        "gtol": 1e-12,
        "maxiter": 100,
    },
)

q_recovered = float(result.x[0] * q_scale)

print("\n========== INVERSE RESULTS ==========")
print("True heat flux:", q_true)
print("Recovered heat flux:", q_recovered)
print("Final loss:", result.fun)
print("Optimizer success:", result.success)
print("Optimizer message:", result.message)


# ============================================================
# 12. Verify analytical sensitivity
# ============================================================

S_analytical = (Lx - x_sensor) / k

# For the scalar sensor output, differentiate
# the sensor temperature directly.
S_ad = jax.grad(sensor_temperature)(
    jnp.array(q_recovered)
)

print("\nAnalytical sensitivity:", S_analytical)
print("AD sensitivity:", float(S_ad))


# ============================================================
# 13. Save forward and recovered temperature fields
# ============================================================

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "vtk",
)

os.makedirs(output_dir, exist_ok=True)

true_path = os.path.join(
    output_dir, "temperature_true.vtu"
)

recovered_path = os.path.join(
    output_dir, "temperature_recovered.vtu"
)

sol_list_recovered = fwd_pred(
    jnp.array(q_recovered)
)

save_sol(
    problem.fes[0],
    sol_list_true[0],
    true_path,
)

save_sol(
    problem.fes[0],
    sol_list_recovered[0],
    recovered_path,
)

print("\nResults saved to:", output_dir)
