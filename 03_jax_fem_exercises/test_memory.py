import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

A = jnp.array([
    [4., 1.],
    [1., 3.]
])

b = jnp.array([1., 2.])

x, info = bicgstab(A, b)

x.block_until_ready()

print("Solution:", x)
print("Info:", info)