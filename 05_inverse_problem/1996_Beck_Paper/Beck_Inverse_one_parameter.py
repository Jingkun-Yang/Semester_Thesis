import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

from scipy.optimize import minimize

import Beck_Config as cfg
import Beck_Functions as fnc

# I know when the mica is heated up.
# from definding the mesh to definding the class problem
# then use wrapper to make the problem differentiable
# difine the objective equation
# optimize the objective equation to get the result

def main():
    mesh = fnc.get_new_x_coords()
    location_fns = [fnc.left]
    problem = fnc.Beck_1996(mesh, vec=1, dim=2, ele_type=cfg.ele_type,
                            location_fns=location_fns)
    T_old = cfg.T_t0 * jnp.ones((problem.fes[0].num_total_nodes, 1))

    T_sensor1 = []
    T_sensor2 = []

    sensor1_index = fnc.get_sensor_temperature_index(cfg.sensor1_point, mesh)
    sensor2_index = fnc.get_sensor_temperature_index(cfg.sensor2_point, mesh)
    fwd_pred = ad_wrapper(problem)

    q_array_guess = jnp.array([cfg.q_guess])

    num_steps = int(round(cfg.t_end / cfg.dt))
    for step in range(1, num_steps + 1):
        problem.set_old_temperature(T_old)
        # q = fnc.heat_flux(step) # I don't use this function to find q

        if step * cfg.dt < cfg.t_heat or step * cfg.dt > cfg.t_cool_down:
            q = 0.
        else:
            q = cfg.q

        problem.set_params(q)
        q_array = jnp.array([q])

        sol_list = fwd_pred(q_array)
        T_new = sol_list[0]
        T_sensor1.append(T_new[sensor1_index, 0])
        T_sensor2.append(T_new[sensor2_index, 0])
        T_old = T_new

    T_sensor1 = np.array(T_sensor1)
    T_sensor2 = np.array(T_sensor2)

    time = np.arange(1, len(T_sensor1) + 1) * cfg.dt

    def objective(q_guess_array):
        T_guess = []

        for step in range(1, num_steps + 1):
            pass