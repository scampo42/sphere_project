#Run V5:
#This moves all particle at once by the the force vector (not a unit vector) scaled by gamma as in the my_functions
#Used for voronoi tesselation

import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *

#For the number of points for the main
n = 5
x = arrangement_n(n,4000)
# #For the subpoints
m = 100
y = arrangement_n(m,500)

#Corresponding list for nearest element
near_p = []
for i in range(0,len(y)):
    near_p = near_p + [nearest_point(y[i], x)[1]]

near_0 = []
near_1 = []

for i in range(0,len(y)):
    if near_p[i] == 0:
        near_0 = near_0 + [y[i]]
    elif near_p[i] == 1:
        near_1 = near_1 + [y[i]]

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

# y1=[]
# y2=[]
# y3=[]
# for i in range(0,m):
#     y1.append(y[i,0])
#     y2.append(y[i,1])
#     y3.append(y[i,2])

#Near point 0
n01=[]
n02=[]
n03=[]
for i in range(0,len(near_0)):
    n01.append(near_0[i][0])
    n02.append(near_0[i][1])
    n03.append(near_0[i][2])

#Near point 1
n11=[]
n12=[]
n13=[]
for i in range(0,len(near_1)):
    n11.append(near_1[i][0])
    n12.append(near_1[i][1])
    n13.append(near_1[i][2])



#Render
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.5, linewidth=0)
ax.scatter(x1,x2,x3,color="black",s=80)
# ax.scatter(y1,y2,y3,s=40) #New
ax.scatter(n01,n02,n03,s=40) #New
ax.scatter(n11,n12,n13,s=40) #New
plt.axis('off')
ax.view_init(azim=90.0, elev=90.0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("auto")
ax.set_box_aspect(aspect=(1,1,1))
ax.set_title("{0} ".format(n)+"points on a sphere")

plt.show()