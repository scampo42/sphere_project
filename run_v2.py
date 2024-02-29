#Run V2:
#This moves all particle at once by the the force vector (not a unit vector) scaled by gamma
#steepest descent/gradient flow/gradient descent

import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *

# project one or all points to the sphere
def proj(x,j):
    if(j==n):
        for i in range(0,n):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm
    return x

#set the number of points
n=7

# set the number of loops
loops=round(7000/n)

# set how often to output the energy
often=1

# set the maximum amplitude of the change
amplitude=0.9 #RETURN TO 0.9

# open file to output energy during minimization
filename="energy"+str(n).zfill(3)
out = open(filename,'w')
out.close()

#assign random start points on the sphere
random.seed()
x=proj((2.0*np.random.random((n,3))-1.0),n)

# calculate the initial energy
energy=0.0
for i in range(0,n):
    for j in range(i+1,n):
        distance=np.sqrt(sum((x[i]-x[j])**2))
        energy=energy+1.0/distance

looplist=[0]
energylist=[energy]

# the main loop to reduce the energy

for loop in range(0,loops+1):

###########################################################
###########################################################
    #temp variable old_energy to see if new arrangement has more or less
    old_energy = energy

    #move all the particles in force*gamma
    x = x_new(x,amplitude)

    #project all new points onto a unit sphere
    x = proj(x,n)

###########################################################
    # calculate new energy
    energy=0.0
    for i in range(0,n):
        for j in range(i+1,n):
            distance=np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance

    # calculate difference in energy
    difference=0.0
    difference = energy - old_energy

    # accept move either way but scale gamma down if negative
    if(difference>0.0):
        amplitude = amplitude*0.5 #For higher values of n we use more loops, use a higher scale factor (0.9)
###########################################################
###########################################################
    # output energy to screen and a file

    if(loop%often==0):
        print("{0} {1:.6f}".format(loop*n,energy))
        out = open(filename,'a')
        out.write("{0} {1:.6f} \n".format(loop*n,energy))
        out.close()
        looplist.append(loop*n)
        energylist.append(energy)

print("Final gamma:", amplitude)

# output final energy to the screen and points to a file
print("Final energy = {0:.6f} \n".format(energy))
filename2= "points"+str(n).zfill(3)
points=open(filename2,'w')
for i in range(0,n):
    for j in range(0,3):
        points.write("{0:.6f} ".format(x[i,j]))
    points.write('\n')
points.close()

# Writing to energies to a file
np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/v2_energy.txt', energylist)
np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/v2_list.txt', looplist)

#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)

#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,n):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])

#Render
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.5, linewidth=0)
ax.scatter(x1,x2,x3,color="black",s=80)
plt.axis('off')
ax.view_init(azim=90.0, elev=90.0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("auto")
ax.set_box_aspect(aspect=(1,1,1))
ax.set_title("{0} ".format(n)+"points on a sphere")

ax2=fig.add_subplot(212)
ax2.plot(looplist,energylist)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("energy")

plt.show()