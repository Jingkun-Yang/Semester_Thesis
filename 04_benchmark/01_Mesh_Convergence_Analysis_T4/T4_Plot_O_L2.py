import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/T4_Convergence_Analysis_Results.npy")

mesh = data[:, 0]
E_L2 = data[:, 1]

O_h = E_L2[0] * (mesh) * 100
O_h2 = E_L2[0] * (mesh) ** 2 * 100

fig, ax = plt.subplots(figsize=(10, 10))
ax.loglog(mesh, E_L2, 'o-', label=r'$E_{L^2}$', color='blue')
ax.loglog(mesh, O_h, label=r'$O(h)$', color='orange')
ax.loglog(mesh, O_h2, label=r'$O(h^2)$', color='green')

ax.legend(loc='lower right', prop={'size': 20})
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('h', fontsize=18)
ax.set_ylabel('L2 error', fontsize=18)
ax.set_title('L2 error convergence', fontsize=24)
ax.grid(True, which='both')
plt.savefig("results/T4_Convergence_Analysis_Results.png", dpi=300, bbox_inches='tight')
plt.show()