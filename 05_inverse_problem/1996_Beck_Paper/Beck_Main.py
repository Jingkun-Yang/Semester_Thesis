import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

import jax.numpy as jnp
import os
import numpy as np

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from scipy.optimize import minimize

import Beck_Config as cfg
import Beck_Functions as fnc

def main():
    mesh = fnc.get_new_x_coords()
    location_fns = [fnc.left]
    problem = fnc.Beck_1996(mesh, vec=1, dim=2, ele_type=cfg.ele_type,
                            location_fns=location_fns)

    T_old = cfg.T_t0 * jnp.ones((problem.fes[0].num_total_nodes, 1))

    sensor1_index = fnc.get_sensor_temperature_index(cfg.sensor1_point, mesh)
    sensor2_index = fnc.get_sensor_temperature_index(cfg.sensor2_point, mesh)

    T_sensor1 = []
    T_sensor2 = []

    num_steps = int(round(cfg.t_end / cfg.dt))

    for step in range(1, num_steps + 1):
        
        problem.set_old_temperature(T_old)
        q = fnc.heat_flux(step)
        problem.set_params(q)

        print(f"\nStep = {step}")
        print(f"Time = {step * cfg.dt:.2f} s")
        print(f"Heat flux = {q}")

        print(f"T_old min = {jnp.min(T_old)}")
        print(f"T_old max = {jnp.max(T_old)}")
        print(f"T_old finite = {jnp.all(jnp.isfinite(T_old))}")

        sol_list = solver(problem,
                        solver_options={
                            "spsolve_solver":{}
                        })
        T_new = sol_list[0]
        T_sensor1.append(T_new[sensor1_index, 0])
        T_sensor2.append(T_new[sensor2_index, 0])
        T_old = T_new
        print('==============================================================')
        print(f'iteration = {step}')

    T_sensor1 = np.array(T_sensor1)
    T_sensor2 = np.array(T_sensor2)
    time = np.arange(1, len(T_sensor1) + 1) * cfg.dt

    data_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(data_dir, exist_ok=True)
    results = np.column_stack((time, T_sensor1, T_sensor2))
    np.save(os.path.join(data_dir, 'results.npy'), results)

if __name__ == '__main__':
    main()