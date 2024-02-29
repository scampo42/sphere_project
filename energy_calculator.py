import numpy as np
from gen_functions import *
arrangement = np.loadtxt('/150gen2.txt')

arrangement = relax_arrangement(arrangement,10000)

n = len(arrangement)

energy = 0.0
for i in range(0, n):
    for j in range(i + 1, n):
        distance = np.sqrt(sum((arrangement[i] - arrangement[j]) ** 2))
        energy = energy + 1.0 / distance
print(energy)