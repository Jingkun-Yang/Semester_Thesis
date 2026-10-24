import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print(jax.devices())

L = 0.1
width = 0.01
thickness = 0.01

area = width * thickness

t_final = 32
dt = 0.01

k = 35.0 # conductivity [W/(m degree C)]
cp = 440.5 # specific heat [J/(kg degree C)]
rho = 7200 # density [kg/(m3)]

num_element = 120
num_nodes = num_element + 1
nodes = jnp.linspace(0, L, num_nodes)
print("nodes: ", nodes)

h = L/num_element
print("h", h)

# connectivity = jnp.array([
#     [0, 1],
#     [1, 2],
#     [2, 3],
#     [3, 4],
#     [4, 5]
# ])
# print("connectivity:", connectivity)

connectivity = jnp.zeros((num_element, 2), dtype=int)

for e in range(num_element):
    connectivity = connectivity.at[e].set(
        jnp.array([e, e + 1])
    )
    # connectivity = connectivity.at[e, 0].set(e)
    # connectivity = connectivity.at[e, 1].set(e + 1)

print(connectivity)

# shape function mapping into an interval from -1 to 1
# reference element
def shape_function(xi):
    N = jnp.array([
        (1 - xi) / 2,
        (1 + xi) / 2
    ])
    dN_dxi = jnp.array([
        - 1 / 2,
          1 / 2
    ])

    return N, dN_dxi

def assemble(K, M, global_nodes, Ke, Me):
    K = K.at[global_nodes[0], global_nodes[0]].add(Ke[0, 0])
    K = K.at[global_nodes[0], global_nodes[1]].add(Ke[0, 1])
    K = K.at[global_nodes[1], global_nodes[0]].add(Ke[1, 0])
    K = K.at[global_nodes[1], global_nodes[1]].add(Ke[1, 1])

    M = M.at[global_nodes[0], global_nodes[0]].add(Me[0, 0])
    M = M.at[global_nodes[0], global_nodes[1]].add(Me[0, 1])
    M = M.at[global_nodes[1], global_nodes[0]].add(Me[1, 0])
    M = M.at[global_nodes[1], global_nodes[1]].add(Me[1, 1])

    return K, M

#===================================
#-----------------------------------

# xi = 1
# N, dN_dxi = shape_function(xi)
# print(N, dN_dxi)

# # here only test element 2,  0, 1, 2 actually the third element
# xe = nodes[connectivity[2]]
# print(xe)

# x = N @ xe
# print(x)
# J = dN_dxi @ xe
# print(J)
# dN_dx = dN_dxi / J
# print(dN_dx)

#-----------------------------------

# gauss_points = jnp.array([
#     - 1 / jnp.sqrt(3),
#       1 / jnp.sqrt(3)
#     ])
# gauss_weights = jnp.array([
#     1,
#     1
# ])

# Ke = jnp.zeros((2, 2))
# for gp in range(2):
#     xi = gauss_points[gp]
#     w = gauss_weights[gp]

#     N, dN_dxi = shape_function(xi)

#     x = N @ xe
#     J = dN_dxi @ xe
#     dN_dx = dN_dxi / J

#     Ke += k * (jnp.outer(dN_dx, dN_dx)) * J * w

# print(Ke)

# Me = jnp.zeros((2, 2))
# for gp in range(2):
#     xi = gauss_points[gp]
#     w = gauss_weights[gp]

#     N, dN_dxi = shape_function(xi)

#     J = dN_dxi @ xe

#     Me += rho * cp * jnp.outer(N, N) * J * w

# print(Me)

#======================================

gauss_points = jnp.array([
    - 1 / jnp.sqrt(3),
      1 / jnp.sqrt(3)
    ])
gauss_weights = jnp.array([
    1,
    1
])
K = jnp.zeros((num_nodes, num_nodes))
M = jnp.zeros((num_nodes, num_nodes))
for e in range(num_element):
    xe = nodes[connectivity[e]]
    Ke = jnp.zeros((2, 2))
    Me = jnp.zeros((2, 2))

    for gp in range(2):
        xi = gauss_points[gp]
        w = gauss_weights[gp]

        N, dN_dxi = shape_function(xi)

        # x = N @ xe
        J = dN_dxi @ xe
        dN_dx = dN_dxi / J

        Ke += k * area * (jnp.outer(dN_dx, dN_dx)) * J * w
        Me += rho * cp * area * jnp.outer(N, N) * J * w

    global_nodes = connectivity[e]

    K, M = assemble(K, M, global_nodes, Ke, Me)

    # K = K.at[global_nodes[0], global_nodes[0]].add(Ke[0, 0])
    # K = K.at[global_nodes[0], global_nodes[1]].add(Ke[0, 1])
    # K = K.at[global_nodes[1], global_nodes[0]].add(Ke[1, 0])
    # K = K.at[global_nodes[1], global_nodes[1]].add(Ke[1, 1])

    # M = M.at[global_nodes[0], global_nodes[0]].add(Me[0, 0])
    # M = M.at[global_nodes[0], global_nodes[1]].add(Me[0, 1])
    # M = M.at[global_nodes[1], global_nodes[0]].add(Me[1, 0])
    # M = M.at[global_nodes[1], global_nodes[1]].add(Me[1, 1])

print("K = ")
print(K)
print("M = ")
print(M)


A = M / dt + K
A = A.at[0, :].set(0.0)
A = A.at[0, 0].set(1.0)
A = A.at[- 1, :].set(0.0)
A = A.at[- 1, - 1].set(1.0)
print("A = ")
print(A)

Tn = jnp.zeros(num_nodes)

n_t = jnp.arange(0.0, t_final + dt, dt)

for n in range(len(n_t) - 1):
    bn = (M / dt) @ Tn
    bn = bn.at[0].set(0)
    bn = bn.at[- 1].set(100 * jnp.sin(jnp.pi * n_t[n + 1] / 40))
    T_next = jnp.linalg.solve(A, bn)
    Tn = T_next

print("Tn : ")
print(Tn)
print("The final time is ", (n + 1) * dt)