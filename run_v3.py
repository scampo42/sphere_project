#Run V3:
#This moves particles in the same format as run_v2.py but this time does
#multi start to calculate the actual global minimum (hopefully)


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

###########################################################
###########################################################

# set how many starts (multistart) that you want
runs = 100

# initialise matrix for energy values
final_energies = []

#initialise matrix for storing the point arrangements for each final
points_arrangement_list = []

###########################################################
###########################################################

#set the number of points
n=282


# set the number of loops
loops=2000

# set how often to output the energy
# adpted so only shows the final energy for each
often=loops
# often = 1000

# open file to output energy during minimization 
filename="energy"+str(n).zfill(3)
out = open(filename,'w')
out.close()

looplist=[]
energylist=[]

for count in range (0,runs):

    # set the maximum amplitude of the change
    amplitude = 1

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
            amplitude = amplitude*0.9

        amplitude = amplitude * 1.01
    ###########################################################
    ###########################################################
        # output energy to screen and a file

        #added the loop%often!=0 to stop these from being in the file
        if(loop%often==0 and loop!=0):
            print("{0} {1:.6f}".format(loop,energy))
            out = open(filename,'a')
            out.write("{0} {1:.6f} \n".format(loop,energy))
            out.close()
            looplist.append(loop)
            energylist.append(energy)

    #adding this energy value to the energies_list It is rounded to 6dp
    final_energies = final_energies + [round(energy,6)]

    #store the arrangements of the points in this final energy position to the points_arrangement_list
    points_arrangement_list = points_arrangement_list + [x]


#creates list containing the unique energy solutions
unique_final_energies = list(set(final_energies))

#prints the unique final energies and their frequency
energy_frequency = []
for i in range(0,len(unique_final_energies)):
    energy_frequency = energy_frequency + [0]
    for j in range(0,len(final_energies)):
        if final_energies[j] == unique_final_energies[i]:
            energy_frequency[i] = energy_frequency[i] + 1

for i in range(0,len(unique_final_energies)):
    print("Energy:",unique_final_energies[i],"- Frequency:",(100*energy_frequency[i]/runs),"%")

#prints the minimum energy solution
print("Lowest Energy:", min(unique_final_energies))

#gets index of element with minimum energy
min_energy_index = final_energies.index(min(unique_final_energies))

#gets point arrangement for minimum energy
min_energy_arrangement = points_arrangement_list[min_energy_index]



#REMOVED THIS SECTION AS GRAPHICS SET UP FOR A SINGLE ARRANGEMENT

# # sets x to this value so the visual aspect of the code can now run
# x = min_energy_arrangement
#
# # output final energy to the screen and points to a file
# print("Final energy = {0:.6f} \n".format(energy))
# filename2= "points"+str(n).zfill(3)
# points=open(filename2,'w')
# for i in range(0,n):
#     for j in range(0,3):
#         points.write("{0:.6f} ".format(x[i,j]))
#     points.write('\n')
# points.close()
#
#
# #Create a sphere
# theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
# xs = np.sin(theta)*np.cos(phi)
# ys = np.sin(theta)*np.sin(phi)
# zs = np.cos(theta)
#
# #convert data
# x1=[]
# x2=[]
# x3=[]
# for i in range(0,n):
#     x1.append(x[i,0])
#     x2.append(x[i,1])
#     x3.append(x[i,2])
#
# #Render
# fig = plt.figure()
# ax = fig.add_subplot(211, projection='3d')
# ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.5, linewidth=0)
# ax.scatter(x1,x2,x3,color="black",s=80)
# plt.axis('off')
# ax.view_init(azim=90.0, elev=90.0)
#
# ax.set_xlim([-1.0,1.0])
# ax.set_ylim([-1.0,1.0])
# ax.set_zlim([-1.0,1.0])
# ax.set_aspect("auto")
# ax.set_box_aspect(aspect=(1,1,1))
# ax.set_title("{0} ".format(n)+"points on a sphere")
#
# ax2=fig.add_subplot(212)
# ax2.plot(looplist,energylist)
# ax2.set_xlabel("iteration number")
# ax2.set_ylabel("energy")
#
# plt.show()