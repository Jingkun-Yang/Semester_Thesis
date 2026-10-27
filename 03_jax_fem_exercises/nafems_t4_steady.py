import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import os

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

k = 52.
h = 750.

T_amb = 0

class nafems_t4_steady(Problem):
    def get_tensor_map(self):
        return lambda T_grad: k * T_grad
    def get_surface_maps(self):
        def robin(T, x):
            return h * (T - T_amb)
        return [robin, robin]

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 0.6, 1.
meshio_mesh = rectangle_mesh(Nx=60, Ny=100, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return jnp.isclose(point[0], 0., atol=1e-5)

def right(point):
    return jnp.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-5)

def top(point):
    return jnp.isclose(point[1], Ly, atol=1e-5)

def dirichlet_val_bottom(point):
    return 100.

location_fns_dirichlet = [bottom]
value_fns = [dirichlet_val_bottom]
vecs = [0]

dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

location_fns = [top, right]

problem = nafems_t4_steady(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

sol_list = solver(problem, solver_options={'spsolve_solver':{}})
T = sol_list[0]

points = jnp.asarray(mesh.points[:, :2])

target = jnp.array([0.6, 0.2])

mask = (
    jnp.isclose(points[:, 0], target[0]) &
    jnp.isclose(points[:, 1], target[1])
)

node_id = jnp.where(mask)[0][0]

T_target = T[node_id, 0]

print(f"Temperature at (0.6, 0.2) = {T_target:.6f} °C")

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/temperature_steady.vtu')

save_sol(problem.fes[0], sol_list[0], vtk_path)