import jax
import jax.numpy as jnp
from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import numpy as np
import Beck_Config as cfg

def heat_flux(t):
    q = jnp.where(
        t < cfg.t_heat,
        0.,
        jnp.where(t < cfg.t_cool_down, cfg.q, 0.)
    )
    return q


class Beck_1996(Problem):
    def get_tensor_map(self):
        def tensor_map(T_grad, x):
            k = jnp.where(
                x[0] < cfg.L1,
                cfg.k1,
                jnp.where(x[0] < cfg.L2, cfg.k2, cfg.k3)
            )
            return k * T_grad
        return tensor_map

    def get_mass_map(self):
        def mass_map(T_new, x, T_old):
            c = jnp.where(
                x[0] < cfg.L1,
                cfg.c1,
                jnp.where(x[0] < cfg.L2, cfg.c2, cfg.c3)
            )
            return c * (T_new - T_old) / cfg.dt
        return mass_map

    def get_surface_map(self):
        def surface_map(T, x, q_current):
            return q_current * jnp.ones_like(T)
        return surface_map

    def set_old_temperature(self, T_old):
        T_old_quad = self.fes[0].convert_from_dof_to_quad(T_old)
        self.internal_vars = [T_old_quad]

    def set_params(self, q_current):
        self.internal_vars_surfaces = [[q_current]]

def left(point):
    return jnp.isclose(point[0], 0., atol=1e-5)

def right(point):
    return jnp.isclose(point[0], cfg.Lx, atol=1e-5)

def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-5)

def top(point):
    return jnp.isclose(point[1], cfg.Width, atol=1e-5)

def get_sensor_temperature_index(sensor_point, mesh):
    points = np.asarray(mesh.points)
    point_index = np.where(
        np.all(
            np.isclose(points, sensor_point, atol=1e-5, rtol=0),
            axis=1
        )
    )[0]

    if len(point_index) == 0:
        raise ValueError("No such point")

    index = int(point_index[0])
    return index

def get_new_x_coords():
    Nx = cfg.Nx1 + cfg.Nx2 + cfg.Nx3
    Lx = cfg.L1 + cfg.L2 + cfg.L3
    cell_type = get_meshio_cell_type(cfg.ele_type)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=1,
                                 domain_x=Lx, domain_y=cfg.Width)

    x1 = np.linspace(
        0.,
        cfg.L1,
        cfg.Nx + 1 # number of elements pluses one
    )

    x2 = np.linspace(
        cfg.L1,
        cfg.L1 + cfg.L2,
        cfg.Nx2 + 1
    )

    x3 = np.linspace(
        cfg.L2,
        cfg.L3,
        cfg.Nx3 + 1
    )

    x_nonuni = np.concatenate([
        x1,
        x2[1:],
        x3[1:]
    ]) # [n, ] n is the number of x coordinates

    points = meshio_mesh.points.copy()
    x_uniform = np.linspace(
        0.,
        Lx,
        Nx + 1
    )

    # map uniform x-coordinates to nonuniform coordinates
    points[:, 0] = np.interp( # use interpolation to map the coordinates
        points[:, 0],
        x_uniform,
        x_nonuni
    )

    mesh = Mesh(
        points,
        meshio_mesh.cells_dict[cell_type]
    )

    return mesh