import jax.numpy as jnp
from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import numpy as np
import Beck_Config as cfg

def heat_flux(step):
    heat_step = round(cfg.t_heat / cfg.dt)
    cool_step = round(cfg.t_cool_down / cfg.dt)
    if heat_step <= step < cool_step:
        return cfg.q
    return 0.


class Beck_1996(Problem):
    def get_tensor_map(self):
        def tensor_map(T_grad, k_quad, T_old):
            return k_quad * T_grad
        return tensor_map

    def get_mass_map(self):
        def mass_map(T_new, x, k_quad, T_old):
            c = jnp.where(
                x[0] < cfg.L1,
                cfg.c1,
                jnp.where(x[0] < (cfg.L1 + cfg.L2),
                          cfg.c2, cfg.c3)
            )
            return c * (T_new - T_old) / cfg.dt
        return mass_map

    def get_surface_maps(self):
        def surface_map(T, x, q_current):
            return - q_current * jnp.ones_like(T)
        return [surface_map]

    def set_old_temperature(self, T_old):
        T_old_quad = self.fes[0].convert_from_dof_to_quad(T_old)
        
        x_quad = self.physical_quad_points[..., 0]
        
        k_quad = jnp.where(
            x_quad < cfg.L1,
            cfg.k1,
            jnp.where(
                x_quad < (cfg.L1 + cfg.L2),
                cfg.k2,
                cfg.k3
            )
        )        
        self.internal_vars = [k_quad, T_old_quad]

    def set_params(self, params):
        T_old, q_current = params

        # Update the previous temperature
        self.set_old_temperature(T_old)

        # Update the heat-flux boundary condition
        num_faces = len(self.boundary_inds_list[0])
        num_quads = self.selected_face_shape_vals[0].shape[1]

        q_surface = q_current * jnp.ones(
            (num_faces, num_quads)
        )

        self.internal_vars_surfaces = [[q_surface]]

def left(point):
    return jnp.isclose(point[0], 0., atol=1e-9)

def right(point):
    return jnp.isclose(point[0], cfg.Lx, atol=1e-9)

def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-9)

def top(point):
    return jnp.isclose(point[1], cfg.Width, atol=1e-9)

def get_sensor_temperature_index(sensor_point, mesh):
    points = np.asarray(mesh.points)
    point_index = np.where(
        np.all(
            np.isclose(points, sensor_point, atol=1e-9, rtol=0),
            axis=1
        )
    )[0]

    if len(point_index) == 0:
        raise ValueError("No such point")

    index = int(point_index[0])
    return index

def get_new_x_coords():
    Nx = cfg.Nx1 + cfg.Nx2 + cfg.Nx3
    cell_type = get_meshio_cell_type(cfg.ele_type)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=1,
                                 domain_x=cfg.Lx, domain_y=cfg.Width)

    x1 = np.linspace(
        0.,
        cfg.L1,
        cfg.Nx1 + 1 # number of elements pluses one
    )

    x2 = np.linspace(
        cfg.L1,
        cfg.L1 + cfg.L2,
        cfg.Nx2 + 1
    )

    x3 = np.linspace(
        cfg.L1 + cfg.L2,
        cfg.Lx,
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
        cfg.Lx,
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