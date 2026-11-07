import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

import T4_Configure
from T4_Functions import (
    nafems_t4_steady,
    left,
    right,
    bottom,
    top,
    dirichlet_val_bottom,
    calculate_eigenvalues,
    calculate_an,
    Fy_function,
    analytical_temperature
)