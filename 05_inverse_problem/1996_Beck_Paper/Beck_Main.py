import jax
jax.config.updata("jax_enable_x64", True)
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