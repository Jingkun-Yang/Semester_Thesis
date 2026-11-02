import numpy as np
import matplotlib.pyplot as plt

mesh_size = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
E_L2 = np.array([np.float64(0.47823708584115815), 
        np.float64(0.14891312124683642), 
        np.float64(0.04401159924580776), 
        np.float64(0.01255169745461659), 
        np.float64(0.003492394900094547), 
        np.float64(0.0009542348833226444)])

O_h = E_L2[0] * (mesh_size) * 100
O_h2 = E_L2[0] * (mesh_size) ** 2 * 100

fig, ax = plt.subplots(figsize=(10, 10))
ax.loglog(mesh_size, E_L2, 'o-', label=r'$E_{L^2}$', color='blue')
ax.loglog(mesh_size, O_h, '--', label=r'$O(h)$', color='orange')
ax.loglog(mesh_size, O_h2, ':', label=r'$O(h^2)$', color='green')

ax.legend(loc='lower right', prop={'size': 20})
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('h', fontsize=18)
ax.set_ylabel('L2 Error', fontsize=18)
ax.set_title('L2 error convergence', fontsize=24)
ax.grid(True, which='both')
plt.show()