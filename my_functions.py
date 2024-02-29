import copy
import random
import matplotlib
import math
import numpy
import numpy as np

# project one or all points to the sphere
def proj(x,j):
    if(j==len(x)):
        for i in range(0,len(x)):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm
    return x

#For 3 Dimensions
def dist(u,v):
    total = 0
    for i in range (0,3):
        total = total + (u[i] - v[i])**2
    return(total**(1/2))

#set direction x=0,y=1,z=2
def force_xyz(xyz, particle_number, particle_set):
    total = 0
    for i in range (0,len(particle_set)):
        #Don't include force on itself
        if i == particle_number:
            total = total
        else:
            d = dist(particle_set[particle_number],particle_set[i])
            total = total + (particle_set[particle_number][xyz] - particle_set[i][xyz])/d**3
    return(total)

###############################################################################
###############################################################################

#unit force vector for individual particle (scaled by gamma)
def unit_force_vector(particle_number, particle_set,gamma):
    vector = [force_xyz(0, particle_number, particle_set),force_xyz(1, particle_number, particle_set),force_xyz(2, particle_number, particle_set)]
    norm = dist(vector,[0,0,0])
    #We will have a divided by 0 error if norm = 0
    if norm == 0:
        #just set norm to any non-zero value, doesn't matter as we are scaling 0
        norm = 1
    return([gamma/norm*force_xyz(0, particle_number, particle_set),gamma/norm*force_xyz(1,particle_number, particle_set),gamma/norm*force_xyz(2, particle_number, particle_set)])

#force vector for individual particle (scaled by gamma)
def force_vector(particle_number, particle_set,gamma):
    vector = [force_xyz(0, particle_number, particle_set),force_xyz(1, particle_number, particle_set),force_xyz(2, particle_number, particle_set)]
    return(gamma*vector[0],gamma*vector[1],gamma*vector[2])

###############################################################################
###############################################################################

#force vector for all the particles moved by a unit force vector (scaled by gamma)
def unit_all_force_vector(particle_set,gamma):
    all_force = []
    for i in range (0, len(particle_set)):
        all_force.append(unit_force_vector(i,particle_set,gamma))
    return(all_force)

#force vector for all the particles moved by a force vector (scaled by gamma)
def all_force_vector(particle_set,gamma):
    all_force = []
    for i in range (0, len(particle_set)):
        all_force.append(force_vector(i,particle_set,gamma))
    return(all_force)

###############################################################################
###############################################################################

#new set of particles if all moved simultaneously by unit force vector (scaled by gamma)
def unit_x_new(particle_set,gamma):
    force = unit_all_force_vector(particle_set,gamma)
    for i in range(0,len(particle_set)):
        for j in range(0, len(particle_set[i])):
            particle_set[i][j] = particle_set[i][j] + gamma*force[i][j]
    return(particle_set)

#new set of particles if all moved simultaneously by force vector (scaled by gamma)
def x_new(particle_set,gamma):
    force = all_force_vector(particle_set,gamma)
    for i in range(0,len(particle_set)):
        for j in range(0, len(particle_set[i])):
            particle_set[i][j] = particle_set[i][j] + gamma*force[i][j]
    return(particle_set)

###############################################################################
###############################################################################
#scales set of particles onto a unit sphere
def x_scaled(particle_set):
    for i in range(0,len(particle_set)):
        norm = dist(particle_set[i],[0,0,0])
        for j in range(0,len(particle_set[i])):
            particle_set[i][j] = particle_set[i][j]/norm
    return(particle_set)

###############################################################################
#Symmetries & Rotations
###############################################################################


#Calculates the inertia tensor for any given arrangement
def inertia_tensor(_set):
    I = numpy.zeros((3, 3))
    for point in _set:
        x, y, z = point  # Assuming each point has x, y, z coordinates
        I[0, 0] += (y**2 + z**2)
        I[1, 1] += (x**2 + z**2)
        I[2, 2] += (x**2 + y**2)
        I[0, 1] -= x * y
        I[1, 0] -= x * y
        I[0, 2] -= x * z
        I[2, 0] -= x * z
        I[1, 2] -= y * z
        I[2, 1] -= y * z
    return(I)

#Returns the index of the unique element in a set of 3 (If all the same returns 2!)
#If 0 returned need to swap x and z coordinate for each point
#If 1 returned need to swap y and z coordinate for each point
def unique_number(set, epsilon):
    x,y,z = set[0], set[1], set[2]
    if abs(x-y) < epsilon and abs(x-z) < epsilon:
        print("All eigenvalues are the same")
        return(999) #All the Same
    elif abs(x-y) < epsilon:
        return(2)
    elif abs(x-z) < epsilon:
        return(1)
    elif abs(y-z) < epsilon:
        return(0)
    else:
        print("All eigenvalues are unique")
        return(998) #All unique

#Sets all the vectors of the parameter to unit vectors
def normalise(points):
    # Makes the vectors all unit vectors
    for i in range(0, len(points)):
        distance = dist(points[i], [0, 0, 0])
        for j in range(0, len(points[i])):
            points[i][j] = points[i][j] / distance
    return(points)

def normalise_point(point):
    distance = dist(point,[0,0,0])
    print("Distance:",distance)
    for i in range(0, 3):
        point[i] = point[i] / distance
    return(point)

#overall function to rotate to principle axis
def rotate_to_principle(point_arrangement):
    #calculate the moment of inertia tensor
    tensor = inertia_tensor(point_arrangement)

    #calculate the eigenvalues, matrix of eigenvectors (and the transpose)
    eigenvalues, eigenvectors = numpy.linalg.eigh(tensor)
    t_eigenvectors = eigenvectors.transpose()

    # used to check that the eigenvectors that were genereated are orthoganal
    # print("eigenvectors:",eigenvectors)
    # for i in range(0,len(eigenvectors)):
    #     for j in range(0,len(eigenvectors)):
    #         print("i",i,"j",j,":",np.dot(eigenvectors[i],eigenvectors[j]))

    #Calculates the unique eigenvalue within epsilon
    flip_value = unique_number(eigenvalues,0.02)

    #rotates each point accordingly
    for i in range(0,len(point_arrangement)):
        point = point_arrangement[i]
        point = t_eigenvectors @ point
        point_arrangement[i] = point

    print("Before flip:",eigenvalues)

    #flips point so unique eigenvalue is last
    for i in range(0,len(point_arrangement)):
        if flip_value != 998 and flip_value != 999:
            point = point_arrangement[i]
            temp = point[2]
            point[2] = point[flip_value]
            point[flip_value] = temp
            point_arrangement[i] = point
        ### Use This for specific cases like 13
        # point = point_arrangement[i]
        # temp = point[2]
        # point[2] = point[0]
        # point[0] = temp
        # point_arrangement[i] = point
        ##
    if flip_value != 998 and flip_value != 999:
        eigenvalues[2], eigenvalues[flip_value] = eigenvalues[flip_value], eigenvalues[2]
        print("Post flip:",eigenvalues)

    return(point_arrangement)


#Rotate around the z axis by an amount (radians)
def point_z_rotate(point,theta):
    c , s = np.cos(theta), np.sin(theta)
    r_matrix = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return(np.matmul(r_matrix, point))

def set_z_rotate(set,theta):
    new_set = copy.deepcopy(set)
    for i in range (0,len(set)):
        new_set[i] = point_z_rotate(set[i],theta)
    return(new_set)

#Rotate around the y axis by an amount (radians)
def point_y_rotate(point,theta):
    c , s = np.cos(theta), np.sin(theta)
    r_matrix = np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return(np.matmul(r_matrix, point))

def set_y_rotate(set,theta):
    new_set = copy.deepcopy(set)
    for i in range (0,len(set)):
        new_set[i] = point_y_rotate(set[i],theta)
    return(new_set)

#Rotate around the x axis by an amount (radians)
def point_x_rotate(point,theta):
    c , s = np.cos(theta), np.sin(theta)
    r_matrix = np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    return(np.matmul(r_matrix, point))

def set_x_rotate(set,theta):
    new_set = copy.deepcopy(set)
    for i in range (0,len(set)):
        new_set[i] = point_x_rotate(set[i],theta)
    return(new_set)

#Checks C symmetry for a specific Cn
def check_c_symmetry(old_set,n):
    n_set =  copy.deepcopy(set_z_rotate(old_set, 2 * np.pi / n))
    for i in range(0,len(n_set)):
        #Assume not the same point - if same point change to true and continue
        same_point = False
        for j in range(0, len(old_set)):
            #Adjust epsilon as necessary
            if dist(n_set[i],old_set[j]) < 0.01:
                same_point = True
        #This point from new_set is not within epsilon of any point in the old_set
        if same_point == False:
            return(False)
    return(True)

#Tests all symmetry up to 12
def all_c_symmetry(_set):
    #List containing all symmetry of Cn
    lst = []
    for i in range (1,13):
        if check_c_symmetry(_set, i):
            lst = lst + [i]
    return(lst)


#Checks if one arrangement is exactly the same as another (given eps)
def same_arrangemet(seta,setb):
    for i in range(0,len(seta)):
        #Assume not the same point - if same point change to true and continue
        same_point = False
        for j in range(0, len(seta)):
            #Adjust epsilon as necessary
            if dist(seta[i],setb[j]) < 0.01:
                same_point = True
        #This point from new_set is not within epsilon of any point in the old_set
        if same_point == False:
            return(False)
    return(True)

#Testing for reflection sym in plane containing z axis
def zaxis_reflection_sym(set_):
    for i in range (0,360):
        rot_set = copy.deepcopy(set_z_rotate(set_, 2 * np.pi * (i/2) / 360))
        flip_set = copy.deepcopy(rot_set)
        for j in range(0,len(flip_set)):
            flip_set[j][0] = -1*flip_set[j][0]
        if same_arrangemet(flip_set,rot_set):
            return(True)
    else:
        return(False)
        # if same_arrangemet(rot_set,flip_set)

def xy_reflection_sym(set_):
    flip_set = copy.deepcopy(set_)
    for j in range(0, len(flip_set)):
        flip_set[j][2] = -1 * flip_set[j][2]
    if same_arrangemet(flip_set, set_):
        return (True)
    return(False)

#Checks the z values to make the foppl arrangements
#NOTE Odd numbers should be chosen for height so that points
# either side of the equator are included at the equator
def foppl(arrangement, heights):
    foppl_lst = []
    eps = 1/heights #2 / heights / 2 (as +- eps)
    for i in range(0,heights):
        # Centre: -1 + (2*i+1)*eps) ; Range:[-1 + (2*i)*eps , -1 + (2*i+2)*eps)
        max = -1 + (2*i+2)*eps
        min = -1 + (2*i)*eps
        count = 0
        for point in arrangement:
            z = point[2]
            # print("z",z,"min:",min,"max:",max)
            if z >= min and z < max:
                count += 1
        if count != 0:
            foppl_lst = foppl_lst + [count]
    #Only point left to check is (for z = 1)
    for point2 in arrangement:
        z = point2[2]
        if 1 <= z and z < 1 + eps/10:
            foppl_lst = foppl_lst +[1]
    return(foppl_lst)



foppl([],200)

###############################################################################
#"Big" functions to do whole simulations as a function
###############################################################################

#Function to calculate the energy of a given number of particles (n)
def energy_n(n):

    #Only works with this inside the loop for some reason
    def proj(x, j):
        if (j == n):
            for i in range(0, n):
                norm = np.sqrt(sum(x[i] ** 2))
                x[i] = x[i] / norm
        else:
            norm = np.sqrt(sum(x[j] ** 2))
            x[j] = x[j] / norm
        return x

    # set the number of loops
    loops=4000

    # set the maximum amplitude of the change
    amplitude=1

    #assign random start points on the sphere
    random.seed()
    x=proj((2.0*np.random.random((n,3))-1.0),n)

    #Calculates this initial energy
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

    return(energy)

#Calculates the minimum energy by multistart (runs)
def min_energy_n(n):
    runs = 4
    lst = []
    for i in range (0,runs):
        lst = lst + [energy_n(n)]
    return(min(lst))

#Generates a low energy arrangement (not lowest guarenteed) for n particles
def arrangement_n(n,loops):

    #Only works with this inside the loop for some reason
    def proj(x, j):
        if (j == n):
            for i in range(0, n):
                norm = np.sqrt(sum(x[i] ** 2))
                x[i] = x[i] / norm
        else:
            norm = np.sqrt(sum(x[j] ** 2))
            x[j] = x[j] / norm
        return x

    # set the maximum amplitude of the change
    amplitude=1

    #assign random start points on the sphere
    random.seed()
    x=proj((2.0*np.random.random((n,3))-1.0),n)

    #Calculates this initial energy
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
            amplitude = amplitude * 0.6 #Recently change from 0.95

    return(x)

###############################################################################
#Voronoi Tesselation Stuff
###############################################################################

#Returns the point and element in the set nearest to the point
def nearest_point(point, set_):
    minimum = dist(point, set_[0])
    element = 0
    for i in range (1, len(set_)):
        if dist(point, set_[i]) < minimum:
            minimum = dist(point, set_[i])
            element = i
    return(set_[element],element)

#Creates points on a unit sphere - Does not work: just generate using normal method
# def sphere_points(number_of_phi):
#     arrangement = []
#     for i in range (0,number_of_phi):
#         phi = i * (2 * np.pi / (number_of_phi-1))
#         for j in range (0,number_of_phi):
#             theta = j * (np.pi / (number_of_phi-1))
#             #Now calculate the corresponding cartaesian point
#             point = (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))
#             print("theta",theta,"phi",phi,"point",point)

###############################################################################
#Test Zone
###############################################################################

# u = [0.5,0,0]
# v = [0,0,0]
# w = [-0.5,0,0]
# set_ = [u,v,w] #not set as this is an inbuilt function



