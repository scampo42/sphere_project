#This just does a single run of central configurations using MoSD

import copy
import random
import matplotlib
import math
import numpy
import numpy as np
from central_functions import *

n = 32
loops = 10000

start_arrange = random_central(n)
end_arrangement = relax_arrangement(start_arrange,loops)[0]
np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central1.txt', end_arrangement)
# end_arrangement = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central1.txt')

print("Energy:",energy_arrangement(end_arrangement))
# print("Exact Shells:",exact_norm_list(end_arrangement))
# print("Rounded Shells:",round_norm_list(end_arrangement))
print("Shell details:",shells(end_arrangement)[0:2])

draw_shells(shells(end_arrangement)[2])
# draw(shell_arr[0],"black")
# draw(shell_arr[1],"black")