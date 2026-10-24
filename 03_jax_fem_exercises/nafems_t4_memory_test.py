
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_log_compiles", True)

import scipy.sparse as sp
from jax.experimental.sparse import BCOO

import jax.numpy as jnp
import numpy as np
import os
import csv
import time
import traceback
import psutil

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import (
    get_meshio_cell_type,
    Mesh,
    rectangle_mesh
)

# ==========================================
# 1. Physical parameters
# ==========================================

k = 52.
h = 750.
dt = 10.
rho = 7850.
cp = 460.
T_amb = 0.


# ==========================================
# 2. JAX-FEM problem
# ==========================================

class nafems_t4(Problem):

    def get_tensor_map(self):
        return lambda T_grad, T_old: k * T_grad

    def get_mass_map(self):
        def mass_map(T_new, x, T_old):
            return rho * cp * (T_new - T_old) / dt
        return mass_map

    def get_surface_maps(self):
        def surface_map(T, x):
            return h * (T - T_amb)
        return [surface_map, surface_map]

    def set_old_temperature(self, T_old):
        T_old_quad = self.fes[0].convert_from_dof_to_quad(T_old)
        self.internal_vars = [T_old_quad]

@jax.jit
def compiled_bicgstab(A, b, x0, inv_diag):

    def preconditioner(x):
        return x * inv_diag

    x, info = jax.scipy.sparse.linalg.bicgstab(
        A,
        b,
        x0=x0,
        M=preconditioner,
        tol=1e-10,
        atol=1e-10,
        maxiter=10000
    )

    return x


def reusable_jax_solver(A, b, x0, linear_options):

    # PETSc sparse matrix -> SciPy CSR
    indptr, indices, data = A.getValuesCSR()

    A_scipy = sp.csr_matrix(
        (data, indices, indptr),
        shape=A.getSize()
    )

    A_scipy.sum_duplicates()
    A_scipy.sort_indices()

    # Jacobi preconditioner
    diagonal = A_scipy.diagonal()

    if np.any(diagonal == 0):
        raise RuntimeError(
            "Zero diagonal detected in Jacobi preconditioner"
        )

    inv_diag = jnp.asarray(1.0 / diagonal)

    # SciPy CSR -> JAX BCOO
    A_jax = BCOO.from_scipy_sparse(A_scipy)

    b = jnp.asarray(b)

    if x0 is None:
        x0 = jnp.zeros_like(b)
    else:
        x0 = jnp.asarray(x0)

    # Reuse the same JIT-compiled solver
    x = compiled_bicgstab(
        A_jax,
        b,
        x0,
        inv_diag
    )

    # Verify the linear residual
    residual = float(
        jnp.linalg.norm(A_jax @ x - b)
    )

    if not np.isfinite(residual) or residual >= 0.1:
        raise RuntimeError(
            f"BiCGSTAB failed: residual = {residual}"
        )

    return x

# ==========================================
# 3. Mesh
# ==========================================

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)

Lx, Ly = 0.6, 1.

meshio_mesh = rectangle_mesh(
    Nx=48,
    Ny=80,
    domain_x=Lx,
    domain_y=Ly
)

mesh = Mesh(
    meshio_mesh.points,
    meshio_mesh.cells_dict[cell_type]
)


# ==========================================
# 4. Boundary conditions
# ==========================================

def left(point):
    return jnp.isclose(point[0], 0., atol=1e-5)

def right(point):
    return jnp.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-5)

def top(point):
    return jnp.isclose(point[1], Ly, atol=1e-5)

def dirichlet_val_bottom(point):
    return 100.


location_fns_dirichlet = [bottom]
value_fns = [dirichlet_val_bottom]
vecs = [0]

dirichlet_bc_info = [
    location_fns_dirichlet,
    vecs,
    value_fns
]

location_fns = [top, right]


# ==========================================
# 5. Create problem
# ==========================================

problem = nafems_t4(
    mesh=mesh,
    vec=1,
    dim=2,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns
)


# ==========================================
# 6. Memory monitoring setup
# ==========================================

process = psutil.Process(os.getpid())

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data"
)

os.makedirs(data_dir, exist_ok=True)

memory_csv = os.path.join(
    data_dir,
    "memory_monitor.csv"
)

def get_memory_usage():

    memory = process.memory_info()

    rss_mb = memory.rss / (1024 ** 2)
    vms_mb = memory.vms / (1024 ** 2)

    # Number of Linux memory mappings
    try:
        with open("/proc/self/maps", "r") as f:
            map_count = sum(1 for _ in f)
    except OSError:
        map_count = -1

    return rss_mb, vms_mb, map_count


print(f"Python PID: {os.getpid()}")
print(f"Memory CSV: {memory_csv}")


# ==========================================
# 7. Transient simulation
# ==========================================

t_end = 20000
num_steps = int(round(t_end / dt))

T_old = jnp.zeros(
    (problem.fes[0].num_total_nodes, 1)
)

monitor_interval = 50

start_time = time.perf_counter()

with open(memory_csv, "w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "Step",
        "Physical_Time_s",
        "RSS_MB",
        "VMS_MB",
        "Memory_Map_Count",
        "Elapsed_Time_s"
    ])

    # Record initial memory
    rss, vms, maps = get_memory_usage()

    writer.writerow([
        0, 0, rss, vms, maps, 0
    ])

    csv_file.flush()

    try:

        for step in range(num_steps):

            problem.set_old_temperature(T_old)

            # Keep the original JAX BiCGSTAB solver
            # sol_list = solver(problem)
            sol_list = solver(problem,
                              solver_options={
                                  "custom_solver": reusable_jax_solver
                                          }
                                          )

            T_new = sol_list[0]
            T_old = T_new

            current_step = step + 1

            if (
                current_step % monitor_interval == 0
                or current_step == 1
            ):

                # Wait for JAX asynchronous work
                T_new.block_until_ready()

                rss, vms, maps = get_memory_usage()

                elapsed = (
                    time.perf_counter() - start_time
                )

                physical_time = current_step * dt

                print(
                    f"[MEMORY] "
                    f"Step={current_step}, "
                    f"Time={physical_time:.1f}s, "
                    f"RSS={rss:.2f}MB, "
                    f"VMS={vms:.2f}MB, "
                    f"Maps={maps}, "
                    f"Elapsed={elapsed:.2f}s",
                    flush=True
                )

                writer.writerow([
                    current_step,
                    physical_time,
                    rss,
                    vms,
                    maps,
                    elapsed
                ])

                csv_file.flush()

    except Exception:

        print(
            "\nERROR OCCURRED DURING SIMULATION",
            flush=True
        )

        traceback.print_exc()

        rss, vms, maps = get_memory_usage()

        print(
            f"Memory at failure: "
            f"RSS={rss:.2f}MB, "
            f"VMS={vms:.2f}MB, "
            f"Maps={maps}",
            flush=True
        )

        csv_file.flush()

        raise


# ==========================================
# 8. Extract target temperature
# ==========================================

points = jnp.asarray(mesh.points[:, :2])

target = jnp.array([0.6, 0.2])

mask = (
    jnp.isclose(points[:, 0], target[0]) &
    jnp.isclose(points[:, 1], target[1])
)

node_ids = np.flatnonzero(np.asarray(mask))

if len(node_ids) == 0:
    print(
        "Target point is not an exact mesh node. "
        "Interpolation is required."
    )
else:
    node_id = int(node_ids[0])

    T_target = T_new[node_id, 0]

    print(
        f"Temperature at (0.6, 0.2) = "
        f"{float(T_target):.6f} °C"
    )


# ==========================================
# 9. Save VTK result
# ==========================================

vtk_dir = os.path.join(data_dir, "vtk")
os.makedirs(vtk_dir, exist_ok=True)

vtk_path = os.path.join(
    vtk_dir,
    "temperature.vtu"
)

save_sol(
    problem.fes[0],
    sol_list[0],
    vtk_path
)

print("Simulation finished successfully.")
print(f"Memory data saved to: {memory_csv}")
