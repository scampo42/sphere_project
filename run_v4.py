#Run V4:
#This moves particles in the same format as run_v2.py (Method of steepest descent)
#It then rotates the particles so that we are looking principle axis (so we can see certain symmetry types)

import numpy as np
import numpy
import matplotlib.pyplot as plt
import random
from my_functions import *
from gen_functions import *



# project one or all points to the sphere
def proj(x, j):
    if (j == n):
        for i in range(0, n):
            norm = np.sqrt(sum(x[i] ** 2))
            x[i] = x[i] / norm
    else:
        norm = np.sqrt(sum(x[j] ** 2))
        x[j] = x[j] / norm
    return x

# set the number of points
n = 28

# set the number of loops
loops = 6000

# set how often to output the energy
often = 1000

# set the maximum amplitude of the change
amplitude = 1

# open file to output energy during minimization
filename = "energy" + str(n).zfill(3)
out = open(filename, 'w')
out.close()

looplist = []
energylist = []

# assign random start points on the sphere
random.seed()
x = proj((2.0 * np.random.random((n, 3)) - 1.0), n)

# calculate the initial energy
energy = 0.0
for i in range(0, n):
    for j in range(i + 1, n):
        distance = np.sqrt(sum((x[i] - x[j]) ** 2))
        energy = energy + 1.0 / distance

# the main loop to reduce the energy

for loop in range(0, loops + 1):

    ###########################################################
    ###########################################################
    # temp variable old_energy to see if new arrangement has more or less
    old_energy = energy

    # move all the particles in force*gamma
    x = x_new(x, amplitude)

    # project all new points onto a unit sphere
    x = proj(x, n)

    ###########################################################
    # calculate new energy
    energy = 0.0
    for i in range(0, n):
        for j in range(i + 1, n):
            distance = np.sqrt(sum((x[i] - x[j]) ** 2))
            energy = energy + 1.0 / distance

    # calculate difference in energy
    difference = 0.0
    difference = energy - old_energy

    # accept move either way but scale gamma down if negative
    if (difference > 0.0):
        amplitude = amplitude * 0.95
    ###########################################################
    ###########################################################


    ###########################################################
    #For some reason the "rotation" does not seem to always keep point on the sphere
    #Works well for different ones- i.e. 11,15,5 work well, 3,7 work badly (but sometimes right)
    ###########################################################

#For testing
# for point in x:
#     print("Pre Rotate:",dist([0,0,0],point))

#Actual doing the rotation function
x = rotate_to_principle(x)

#For testing
# for point in x:
#     print("Post T_eigenvector:",dist([0,0,0],point))

print("Symmetry in reflection in plane containing z axis:",zaxis_reflection_sym(x))
print("Symmetry in reflection in xy plane:",xy_reflection_sym(x))
print("Foppl:",foppl(x,101))
print(all_c_symmetry(x))

# output final energy to the screen and points to a file
print("Final energy = {0:.6f} \n".format(energy))
filename2 = "points" + str(n).zfill(3)
points = open(filename2, 'w')
for i in range(0, n):
    for j in range(0, 3):
        points.write("{0:.6f} ".format(x[i, j]))
    points.write('\n')
points.close()

# Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
xs = np.sin(theta) * np.cos(phi)
ys = np.sin(theta) * np.sin(phi)
zs = np.cos(theta)

# convert data
x1 = []
x2 = []
x3 = []
for i in range(0, n):
    x1.append(x[i, 0])
    x2.append(x[i, 1])
    x3.append(x[i, 2])

# Render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs, rstride=4, cstride=4, color='cyan', alpha=0.4, linewidth=0) #cyan0.4, yellow0.5
ax.scatter(x1, x2, x3, color="black", s=100)
plt.axis('off')
ax.view_init(azim=90.0, elev=90.0)

ax.set_xlim([-1.0, 1.0])
ax.set_ylim([-1.0, 1.0])
ax.set_zlim([-1.0, 1.0])
ax.set_aspect("auto")
ax.set_box_aspect(aspect=(1, 1, 1))
ax.set_title("{0} ".format(n) + "points on a sphere")

# #To plot Principal axis:
# ax.plot3D(np.linspace(0, 0, 100), np.linspace(0, 0, 100), np.linspace(-1.5, 1.5, 100))
# ax.plot3D(np.linspace(0, 0, 100), np.linspace(-1.5, 1.5, 100), np.linspace(0,0, 100))
# ax.plot3D(np.linspace(-1.5, 1.5, 100), np.linspace(0, 0, 100), np.linspace(0,0, 100))

# print(x)

plt.subplots_adjust(hspace=0) #Attempting to make the images larger

plt.show()