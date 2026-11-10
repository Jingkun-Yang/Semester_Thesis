import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = os.path.join(os.path.dirname(__file__),
                        'results/results.npy')

data = np.load(data_dir)

time = data[:, 0]
T_sensor1 = data[:, 1]
T_sensor2 = data[:, 2]

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(time, T_sensor1, '-',
        label='Temperature between mica and specimen',
        color='blue')
ax.plot(time, T_sensor2, '-',
        label='Temperature between specimen and insulation',
        color='red')
ax.legend(loc='lower right', prop={'size': 12})
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('time', fontsize=18)
ax.set_ylabel('Temperature [°C]', fontsize=18)
ax.set_title('The temperature at two interfaces', fontsize=20)
ax.set_ylim(30, 48)
ax.set_yticks(np.arange(30, 49, 2))
ax.grid(True, which='both')
plt.savefig(os.path.join(os.path.dirname(__file__), 'results/sensor_temperature.png'),
            dpi=300, bbox_inches='tight')
plt.show()