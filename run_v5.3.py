#Run V5:
#This moves all particle at once by the the force vector (not a unit vector) scaled by gamma as in the my_functions
#Then generates "minor points" which are coloured according to which major point they are closest too

import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *

#For the number of points for the main
n = 42
x = arrangement_n(n,5000) #If you want to generate the arrangement
# x = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/points042') #If you want to load one (make sure n matches)
x = rotate_to_principle(x) #If you want to roate to principal axis

# #For the minor points
m = 65536 #8192
# y = arrangement_n(m,-1)

# Writing to a text file
# np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/65536sphere.txt', y)

# Reading back from the text file into a new variable
y = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/65536sphere.txt')

#Corresponding list for nearest element
near_p = []
for i in range(0,len(y)):
    near_p = near_p + [nearest_point(y[i], x)[1]]

#Create an empty nearpoint each near_point[i] is the list of minor points nearest to the major point i
near_point = [[]]*n
for i in range(0,len(y)):
    for j in range(0,len(x)):
        if near_p[i] == j:
            near_point[j] = near_point[j] + [y[i]]

#Testing the number of minor points per major point
for i in range(0,len(near_point)):
    print(i,":",len(near_point[i]))

#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)

x1=[]
x2=[]
x3=[]
for i in range(0,n):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])

#Hopefully Final nearpoint- should give all the values closest to point0 then point1 etc
final1=[]
final2=[]
final3=[]
#Cycle through the major points
for j in range(0,len(near_point)):
    #cycle through the minor points nearest this major point
    for i in range(0,len(near_point[j])):
        final1.append(near_point[j][i][0])
        final2.append(near_point[j][i][1])
        final3.append(near_point[j][i][2])

#Render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') #change 211 to 111
# ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.5, linewidth=0) #Comment out to remove the sphere being drawn
# ax.scatter(x1,x2,x3,color="black",s=80)

colors = ['lightpink','crimson','orchid','deeppink','purple','thistle','blueviolet','blue','midnightblue','steelblue','deepskyblue','darkturquoise','cyan','azure','mediumseagreen','seagreen','lime','limegreen','palegreen','lawngreen','olivedrab','yellow','gold','orange','darkorange','bisque','orangered','coral','salmon','red']

#For each of the major points
for i in range(0,len(near_point)):
    f1, f2, f3, = [], [], []
    #For each of the points in this major group
    for j in range(0,len(near_point[i])):
        f1.append(near_point[i][j][0])
        f2.append(near_point[i][j][1])
        f3.append(near_point[i][j][2])
    ax.scatter(f1,f2,f3,s=45,color = colors[i%len(colors)]) #Previously s = 20

plt.axis('off')
ax.view_init(azim=90.0, elev=90.0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("auto")
ax.set_box_aspect(aspect=(1,1,1)) #ax.set_box_aspect(aspect=(1,1,1))
ax.set_title("{0} ".format(n)+"points on a sphere")

plt.subplots_adjust(hspace=0) #Attempting to make the images larger

plt.show()