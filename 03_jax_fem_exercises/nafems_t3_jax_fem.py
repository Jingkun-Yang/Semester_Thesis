import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh

class nafems_t3_jax_fem(Problem):
    def get_tensor_map(self):
        return lambda T_grad, T_old: k * T_grad

    # backward Euler
    def get_mass_map(self):
        def mass_map(T_new, x, T_old):
            return rho * cp * (T_new - T_old) / dt
        return mass_map

    def set_old_temperature(self, T_old):
        T_old_quad = self.fes[0].convert_from_dof_to_quad(T_old)
        self.internal_vars = [T_old_quad]

k = 35.
cp = 440.5
rho = 7200

dt = 0.5

ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 0.1, 0.01, 0.01
meshio_mesh = box_mesh(Nx=20, Ny=1, Nz=1, domain_x=Lx, domain_y=Ly, domain_z=Lz)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return jnp.isclose(point[0], 0, atol=1e-5)

def right(point):
    return jnp.isclose(point[0], Lx, atol=1e-5)

def walls(point):
    bottom = jnp.isclose(point[2], 0, atol=1e-5)
    top = jnp.isclose(point[2], Lz, atol=1e-5)
    front = jnp.isclose(point[1], 0, atol=1e-5)
    back = jnp.isclose(point[1], Ly, atol=1e-5)
    return bottom | top | front | back

def dirichlet_val_left(point):
    return 0

def make_dirichlet_val_right(t):
    def dirichlet_value_right(point):
        return 100 * jnp.sin( jnp.pi * t / 40)
    return dirichlet_value_right

t_end = 32
num_steps = int(jnp.round(t_end / dt))

# initial values
location_fns_dirichlet = [left, right]
vecs = [0, 0]
right_bc = make_dirichlet_val_right(0.)
value_fns = [dirichlet_val_left, right_bc]
dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

# location_fns = [walls] # commented out because here Neumann boundary conditions are naturally satisfied
problem = nafems_t3_jax_fem(mesh=mesh, vec=1, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

T_old = jnp.zeros((problem.fes[0].num_total_nodes, 1))

for step in range(1, num_steps + 1):
    t = step * dt
    right_bc = make_dirichlet_val_right(t)
    value_fns = [dirichlet_val_left, right_bc]
    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    problem.set_old_temperature(T_old)
    sol_list = solver(problem)
    T_new = sol_list[0]
    T_old = T_new

print('t = ', t)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/t3_temperature.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)