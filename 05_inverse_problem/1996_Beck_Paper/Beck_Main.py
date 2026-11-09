import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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

mesh = fnc.get_new_x_coords()
location_fns = [fnc.left]
problem = fnc.Beck_1996(mesh, vec=1, dim=2, ele_type=cfg.ele_type,
                        location_fns=location_fns)

t_steps = np.arange(cfg.dt, cfg.t_end + cfg.dt, cfg.dt)

T_old = 31 * jnp.ones((problem.fes[0].num_total_nodes, 1))

sensor1_point = np.array([0.43e-3, 0])
sensor2_point = np.array([9.14e-3, 0])
sensor1_index = fnc.get_sensor_temperature_index(sensor1_point, mesh)
sensor2_index = fnc.get_sensor_temperature_index(sensor2_point, mesh)


T_sensor1 = np.array([])
T_sensor2 = np.array([])

for t_now in t_steps:
    problem.set_old_temperature(T_old)
    q = fnc.heat_flux(t_now)
    problem.set_params(q)
    sol_list = solver(problem)
    T_new = sol_list[0]
    T_old = T_new
    T_sensor1 = np.append(T_sensor1,
                          T_old[sensor1_index])
    T_sensor2 = np.append(T_sensor2,
                          T_old[sensor2_index])
    print('==============================================================')
    print(f'iteration = {t_now / cfg.dt}')