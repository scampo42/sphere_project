#This does multistart to find the best arrangement. This is then also saved to a text file

import copy
import random
import matplotlib
import math
import numpy
import numpy as np
from central_functions import *

n = 100
runs = 10
loops = 10000


lowest_energy = 999999
#
# for i in range(0,runs):
#     start_arrange = random_central(n)
#     arrangement, arrangement_energy = relax_arrangement(start_arrange,loops)
#     if arrangement_energy < lowest_energy:
#         lowest_energy = arrangement_energy
#         best_arrangement = copy.deepcopy(arrangement)
#         print("New Low:",lowest_energy,"Shells:", shells(best_arrangement)[0:2])
#         np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central2.txt', best_arrangement)
#     else:
#         print("Energy Too High:",arrangement_energy,"Shells:",shells(arrangement)[0:2])

best_arrangement = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central2.txt')

print("Energy:",energy_arrangement(best_arrangement))
print("Exact Shells:",exact_norm_list(best_arrangement))
print("Rounded Shells:",round_norm_list(best_arrangement))
print("Shell details:",shells(best_arrangement)[0:2])

draw_shells(shells(best_arrangement)[2])