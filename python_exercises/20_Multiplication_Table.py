import numpy as np

multi = np.zeros((10, 10))
for i in range(1, 11, 1):
    for j in range(1, 11, 1):
        multi[i - 1, j - 1] = i * j

print(multi)