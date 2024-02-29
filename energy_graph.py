import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from my_functions import *


#Calculating the energies for the corresponding n (6dp))
n_lst = []
energy_lst = []

for i in range(2,31):
    n_lst = n_lst + [i]
    # energy_lst = energy_lst +[round(min_energy_n(i),6)]


#Setting the energy list manually to save time (not-recalculate every time) - 2 to 30
energy_lst = [0.5, 1.732051, 3.674235, 6.474691, 9.985281, 14.452977, 19.675288, 25.759987, 32.716949, 40.596451, 49.165253, 58.853231, 69.306363, 80.670244, 92.911655, 106.050405, 120.084467, 135.089468, 150.881568, 167.641622, 185.287536, 203.930191, 223.347074, 243.81276, 265.133326, 287.302615, 310.491542, 334.634440, 359.603946]

print(len(energy_lst),len(n_lst))
#Setting the form of our function to plot against
def f(x,alpha,beta):
    y = 1/2*(x**2) + alpha*(x**(3/2)) + beta*(x)
    return y

#Calculating the least squares fit of the data for our given form
parameters, covariance = curve_fit(f, n_lst, energy_lst)
alpha, beta = parameters[0], parameters[1]
print("alpha:",alpha,"beta:", beta)

#Calculating the differences between f(x) and energy_n(x)
dif_list = []
for  i in range (0, len(n_lst)):
    dif_list = dif_list + [energy_lst[i] - f(n_lst[i], alpha, beta)]


# Plotting Stuff
# Printing the energy vs the particle size
# plt.subplot(2,1,1)
# plt.plot(n_lst, energy_lst, marker = 'x')
# plt.plot(n_lst, f(numpy.array(n_lst),alpha,beta), color='red')
# plt.xlabel("Number of Particles", fontsize = 30)
# plt.ylabel("Corresponding Coulomb Energy", fontsize = 30)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)

# #Printing the energy vs the particle size
# plt.subplot(2,1,2)
plt.plot(n_lst, dif_list, marker = 'x')
plt.plot(n_lst, [0]*len(n_lst), linestyle = 'dotted')
plt.xlabel("Number of Particles", fontsize = 30)
plt.ylabel("Residual Energy (True - Polynomial)", fontsize = 30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

#Show the Graphs
plt.show()