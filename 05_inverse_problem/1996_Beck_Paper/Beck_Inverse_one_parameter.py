import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jax_fem.solver import ad_wrapper

from scipy.optimize import minimize

import Beck_Config as cfg
import Beck_Functions as fnc

# I know when the mica is heated up.
# from definding the mesh to definding the class problem
# then use wrapper to make the problem differentiable
# difine the objective equation
# optimize the objective equation to get the result
# only use the temperature at sensor1

def main():
    mesh = fnc.get_new_x_coords()
    location_fns = [fnc.left]
    problem = fnc.Beck_1996(mesh, vec=1, dim=2, ele_type=cfg.ele_type,
                            location_fns=location_fns)
    sensor1_index = fnc.get_sensor_temperature_index(cfg.sensor1_point, mesh)
    fwd_pred = ad_wrapper(
        problem,
        solver_options={
            "spsolve_solver":{}
        },
        adjoint_solver_options={
            "spsolve_solver": {}
        }
    )
    # fwd_pred = ad_wrapper(problem)

    def solve_fwd_temperature(q):
        T_old = cfg.T_t0 * jnp.ones(
            (problem.fes[0].num_total_nodes, 1)
        )

        T_sensor1 = []
        num_steps = int(round(cfg.t_end / cfg.dt))
        heat_step = int(round(cfg.t_heat / cfg.dt))
        cool_step = int(round(cfg.t_cool_down / cfg.dt))

        for step in range(1, num_steps + 1):
            if heat_step <= step < cool_step:
                q_now = q
            else:
                q_now = jnp.zeros_like(q)
            q_array = jnp.array([q_now])
            params = (T_old, q_array)

            T_new = fwd_pred(params)[0]

            T_sensor1.append(T_new[sensor1_index, 0])

            T_old = T_new
    
            print(f'Iteration: {step}')

        return jnp.stack(T_sensor1)
    
    T_sensor1_true = solve_fwd_temperature(q=cfg.q)
    T_sensor1_true = jax.lax.stop_gradient(T_sensor1_true)
    
    def objective(q_array): # q_array is an array with (1, )
        q_guess = q_array[0] # make q_guess a scalar
        T_guess = solve_fwd_temperature(q_guess) # in the solver q_guess will be changed to an array
        residual = T_sensor1_true - T_guess
        return 0.5 * jnp.sum(residual ** 2)
    
    q_test = jnp.array([cfg.q_guess])

    grad_ad = jax.grad(objective)(q_test)[0]

    h = 10.0
    J_plus = objective(q_test + h)
    J_minus = objective(q_test - h)

    grad_fd = (J_plus - J_minus) / (2.0 * h)

    print("AD gradient:", float(grad_ad))
    print("FD gradient:", float(grad_fd))
    print(
        "Gradients close:",
        np.isclose(
            float(grad_ad),
            float(grad_fd),
            rtol=1e-2,
            atol=1e-8
        )
    )
    value_and_grad = jax.value_and_grad(objective)
    def scipy_objective(q_array):
        q_jax = jnp.asarray(q_array)
        value, gradient = value_and_grad(q_jax)
        return float(value), np.array(gradient, dtype=float)

    result = minimize(
        scipy_objective,
        x0=np.array([cfg.q_guess]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0., None)],
        options={
            "gtol": 1e-12,
            "ftol": 1e-15,
            "maxiter": 100
        }
    )

    q_inverse = float(result.x[0])
    print("Optimization success:", result.success)
    print("Message:", result.message)
    print("Number of iterations:", result.nit)
    print("True heat flux:", cfg.q)
    print("Recovered heat flux:", result.x[0])
    print("Final objective:", result.fun)

   

if __name__ == "__main__":
    main()