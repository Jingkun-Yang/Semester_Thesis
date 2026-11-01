import numpy as np
from scipy.optimize import brentq
import meshio
import os
import pandas as pd

Lx = 0.6
Ly = 1.

k = 52.
h = 750.
T_bottom = 100.
T_amb = 0.

N_modes = 500

def calculate_eigenvalues(N_modes):
    eigenvalues = []
    def eigenvalue_equation(mu_n):
        return mu_n * np.tan(mu_n * Lx) - h / k
    for n in range(N_modes):
        lower = (n + 1e-9) * np.pi / Lx
        upper = (n - 1e-9 + 0.5) * np.pi / Lx
        mu_n = brentq(
            eigenvalue_equation,
            lower,
            upper
        )
        eigenvalues.append(mu_n)
    return np.array(eigenvalues)
mu_values = calculate_eigenvalues(N_modes)

# print(mu[:5])

def calculate_an(mu):
    numerator = np.sin(mu * Lx) / mu
    denominator = (Lx / 2
                   + np.sin(2 * mu * Lx) / (4 * mu))
    an = ((T_bottom - T_amb)
          * numerator / denominator)
    return an
an_values = calculate_an(mu_values)

def Fy_function(mu, y): # only one mu value
    numerator = (
        (1 + h / k / mu)
        + (1 - h / k / mu) * np.exp(- 2 * mu * (Ly - y))
    )
    denominator = (
        (1 + h / k / mu)
        + (1 - h / k / mu) * np.exp(- 2 * mu * Ly)
    )
    return np.exp(- mu * y) * numerator / denominator

def analytical_temperature(x, y, N_modes):
    T = T_amb
    for n in range(N_modes):
        mu = mu_values[n]
        an = an_values[n]
        X = np.cos(mu * x)
        Y = Fy_function(mu, y)
        T += an * X * Y
    return T

x_E = 0.6
y_E = 0.2
T_E = analytical_temperature(x_E, y_E, N_modes)
print(T_E)

# build up the mesh and calculate the temperature at each point

Nx = 60
Ny = 100

x_coords = np.linspace(0, Lx, Nx + 1)
y_coords = np.linspace(0, Ly, Ny + 1)
x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

points = np.column_stack((
    x_mesh.ravel(),
    y_mesh.ravel(),
    np.zeros(x_mesh.size)
)) # for each column there should be like x0, x1, x2... and y0, y1, y2...

quads = []
for j in range(Ny): # the indices for all element nodes
    for i in range(Nx):
        n0  = j * (Nx + 1) + i
        n1 = n0 + 1
        n2 = n1 + (Nx + 1)
        n3 = n0 + (Nx + 1) # or n2 - 1
        quads.append([n0, n1, n2, n3]) # counter clockwise
quads = np.array(quads)

temperatures = np.array([
    analytical_temperature(p[0], p[1], N_modes)
    for p in points
])
bottom_nodes = np.isclose(points[:, 1], 0.0, atol=1e-10)
temperatures[bottom_nodes] = T_bottom

vtk_mesh = meshio.Mesh(
    points=points,
    cells=[("quad", quads)],
    point_data={
        "T_analytical": temperatures
    }
)
data_dir = os.path.join(os.path.dirname(__file__), 'data/vtk')
os.makedirs(data_dir, exist_ok=True)
vtk_path = os.path.join(data_dir, f'nafems_t4_analytical.vtu')
vtk_mesh.write(
    os.path.join(data_dir, vtk_path)
)
print("output complete")

# comparison of the analytical and numerical results


# fem_mesh = meshio.read("data/vtk/nafems_t4_numerical.vtu")
# points = fem_mesh.points
# T_fem = np.asarray(fem_mesh.point_data["sol"]).reshape(-1)

# T_analytical = np.array([
#     analytical_temperature(p[0], p[1], N_modes)
#     for p in points
# ])

# error = T_fem - T_analytical
# abs_error = np.abs(error)

# comparison = pd.DataFrame({
#     "Node": np.arange(len(points)),
#     "x": points[:, 0],
#     "y": points[:, 1],
#     "T_FEM": T_fem,
#     "T_analytical": T_analytical,
#     "Error": error,
#     "Absolute_error": abs_error
# })

# os.makedirs("data", exist_ok=True)

# comparison.to_csv(
#     "data/nafems_t4_node_comparison.csv",
#     index=False
# )