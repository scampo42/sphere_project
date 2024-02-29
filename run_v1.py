#Run V1:
#This selects 1 point to move at a time and moves it in the direction of the force
#The force is scaled to a unit vector and then applied * gamma
# so that the effective force on the particle is gamma(*1 = gamma)
#If the energy of the system increases then the particle is returned and gamma scaled by a 1/2

# python code to compute points on a sphere
# uses a simple Monte Carlo method 
# draws the points on the sphere and plots the energy against iteration number


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
n=10

# set the number of loops
loops=10000

# set how often to output the energy 
often=100

# set the maximum amplitude of the change
amplitude=1

# open file to output energy during minimization 
filename="energy"+str(n).zfill(3)
out = open(filename,'w')
out.close()

looplist=[]
energylist=[]

#assign random start points on the sphere
random.seed()
x=proj((2.0*np.random.random((n,3))-1.0),n)

# calculate the initial energy
energy=0.0
for i in range(0,n):
    for j in range(i+1,n):
        distance=np.sqrt(sum((x[i]-x[j])**2))
        energy=energy+1.0/distance

# the main loop to reduce the energy        

for loop in range(0,loops+1):

###########################################################
    # randomly choose a point to move
    i=random.randint(0,n-1)

    # store the old coordinates of this point
    old=np.array(x[i])

    # move in direction of force
    x[i]=x[i]+unit_force_vector(i,x,amplitude)
    x=proj(x,i)

###########################################################
    # calculate the difference in energy
    difference=0.0
    for j in range(0,n):
        if(j!=i):
          distance=np.sqrt(sum((x[i]-x[j])**2))
          distanceold=np.sqrt(sum((old-x[j])**2))
          difference=difference+1.0/distance-1.0/distanceold;

    # accept or reject the move 
    if(difference<0.0):
        energy=energy+difference
    else:
        x[i]=old
        #reduce amplitude (gamma) size
        amplitude = amplitude*0.95
    # output energy to screen and a file
    
    if(loop%often==0):
        print("{0} {1:.6f}".format(loop,energy))
        out = open(filename,'a')
        out.write("{0} {1:.6f} \n".format(loop,energy))
        out.close()
        looplist.append(loop)
        energylist.append(energy)

# output final energy to the screen and points to a file
print("Final energy = {0:.6f} \n".format(energy))
filename2= "points"+str(n).zfill(3)       
points=open(filename2,'w')
for i in range(0,n):
    for j in range(0,3):
        points.write("{0:.6f} ".format(x[i,j]))
    points.write('\n')              
points.close()

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