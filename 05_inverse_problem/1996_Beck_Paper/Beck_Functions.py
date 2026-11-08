import jax
import jax.numpy as jnp
from jax_fem.problem import Problem

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