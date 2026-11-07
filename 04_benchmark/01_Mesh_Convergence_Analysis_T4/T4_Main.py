import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import numpy as np

import T4_Configure as cfg

from T4_Functions import (
    nafems_t4_steady,
    left,
    right,
    bottom,
    top,
    dirichlet_val_bottom,
    calculate_eigenvalues,
    calculate_an,
    analytical_temperature
)

import os

def main():
    cell_type = get_meshio_cell_type(cfg.ele_type)
    mu_values = calculate_eigenvalues(cfg.N_modes)
    an_values = calculate_an(mu_values)
    mesh_size = []
    E_L2 = []
    for Nx, Ny in cfg.Nx_Ny:
        dx = cfg.Lx / Nx
        dy = cfg.Ly / Ny
        mesh_size.append(max(dx, dy))

        meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=cfg.Lx, domain_y=cfg.Ly)
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
        location_fns_dirichlet = [bottom]
        value_fns = [dirichlet_val_bottom]
        vecs = [0]

        dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]
        location_fns = [top, right]
        problem = nafems_t4_steady(mesh=mesh, vec=1, dim=2, ele_type=cfg.ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
        sol_list = solver(problem, solver_options={'spsolve_solver':{}})
        T = sol_list[0]

        fe = problem.fes[0]
        
        T_gaussian_point = fe.convert_from_dof_to_quad(T)[..., 0]
        quad_points = problem.physical_quad_points
        points_coordinates = np.asarray(quad_points).reshape(-1, 2)
        T_analytical = np.array([
            analytical_temperature(x, y, cfg.N_modes, mu_values, an_values)
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

    mesh_size = np.array(mesh_size)
    E_L2 = np.array(E_L2)
    os.makedirs("results", exist_ok=True)
    results = np.column_stack((mesh_size, E_L2))
    np.save("results/T4_Convergence_Analysis_Results.npy", results)
if __name__ == '__main__':
    main()