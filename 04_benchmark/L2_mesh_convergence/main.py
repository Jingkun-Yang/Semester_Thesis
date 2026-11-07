"""Run NAFEMS T4 steady-state FEM mesh convergence and plot L2 error."""
# Configure JAX before importing JAX-FEM or modules that import JAX.
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import config as cfg
from functions import (
    nafems_t4_steady, bottom, top, right, dirichlet_val_bottom,
    calculate_eigenvalues, calculate_an, calculate_global_L2_error,
)


def main():
    mu_values = calculate_eigenvalues(cfg.N_modes)
    an_values = calculate_an(mu_values)
    cell_type = get_meshio_cell_type(cfg.ele_type)

    mesh_size = []
    E_L2 = []
    for Nx, Ny in cfg.Nx_Ny:
        dx = cfg.Lx / Nx
        dy = cfg.Ly / Ny
        mesh_size.append(max(dx, dy))

        meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=cfg.Lx, domain_y=cfg.Ly)
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
        dirichlet_bc_info = [[bottom], [0], [dirichlet_val_bottom]]
        location_fns = [top, right]
        problem = nafems_t4_steady(
            mesh=mesh, vec=1, dim=2, ele_type=cfg.ele_type,
            dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns,
        )
        sol_list = solver(problem, solver_options={"spsolve_solver": {}})
        T = sol_list[0]
        E_L2.append(calculate_global_L2_error(problem, T, mu_values, an_values))
        print(f"current Nx and Ny: {Nx, Ny}")

    print(mesh_size)
    print(E_L2)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.loglog(mesh_size, E_L2, "o-", label="FEM L2 Error")
    ax1.set_xlabel("Mesh size h [m]")
    ax1.set_ylabel("L2 error")
    ax1.set_title("L2 error convergence")
    ax1.grid(True, which="both")
    plt.show()


if __name__ == "__main__":
    main()
