import copy
import random
import matplotlib
import math
import numpy
import numpy as np
import matplotlib.pyplot as plt

###### Central MoSD Basics ######

#Generate random arrangement
def random_central(n):
    return(2.0*np.random.random((n,3))-1.0)

#For 3 Dimensions
def dist(u,v):
    total = 0
    for i in range (0,3):
        total = total + (u[i] - v[i])**2
    return(total**(1/2))

#Calculates the energy of any arrangement
def energy_arrangement(system):
    energy = 0.0
    for i in range(0, len(system)):
        for j in range(i + 1, len(system)):
            distance = np.sqrt(sum((system[i] - system[j]) ** 2))
            energy += (1.0 / distance)
        energy += (0.5 * sum((system[i] - 0) ** 2))
    return(energy)

#set direction x=0,y=1,z=2
def force_xyz(xyz, particle_number, particle_set):
    total = 0
    for i in range (0,len(particle_set)):
        #Calculates the electrostatic force
        #Don't include force on itself
        if i != particle_number:
            d = dist(particle_set[particle_number],particle_set[i])
            total += (particle_set[particle_number][xyz] - particle_set[i][xyz])/d**3
    #Calculates the central acting force on the particle
    total -= particle_set[particle_number][xyz]
    return(total)

#force vector for individual particle (scaled by gamma)
def force_vector(particle_number, particle_set, gamma):
    vector = [force_xyz(0, particle_number, particle_set),force_xyz(1, particle_number, particle_set),force_xyz(2, particle_number, particle_set)]
    return(gamma*vector[0],gamma*vector[1],gamma*vector[2])

#force vector for all the particles moved by a force vector (scaled by gamma)
def all_force_vector(particle_set, gamma):
    all_force = []
    for i in range (0, len(particle_set)):
        all_force.append(force_vector(i,particle_set,gamma))
    return(all_force)

#new set of particles if all moved simultaneously by force vector (scaled by gamma)
def x_new(particle_set,gamma):
    force = all_force_vector(particle_set,gamma)
    for i in range(0,len(particle_set)):
        for j in range(0, len(particle_set[i])):
            particle_set[i][j] = particle_set[i][j] + gamma*force[i][j]
    return(particle_set)

#scales set of particles onto a unit sphere
def x_scaled(particle_set):
    for i in range(0,len(particle_set)):
        norm = dist(particle_set[i],[0,0,0])
        for j in range(0,len(particle_set[i])):
            particle_set[i][j] = particle_set[i][j]/norm
    return(particle_set)


###### Symmetries & Rotations ######

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

#overall function to rotate to principle axis
def rotate_to_principle(point_arrangement):
    #calculate the moment of inertia tensor
    tensor = inertia_tensor(point_arrangement)

    #calculate the eigenvalues, matrix of eigenvectors (and the transpose)
    eigenvalues, eigenvectors = numpy.linalg.eigh(tensor)
    t_eigenvectors = eigenvectors.transpose()

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

###### Shell Stuff ######

#Returns list of the norm of each point in the arrangemnt
def exact_norm_list(arrangement):
    lst = []
    for points in arrangement:
        lst.append(np.sqrt(sum((points - [0,0,0]) ** 2)))
    return(lst)

def round_norm_list(arrangement):
    norms = exact_norm_list(arrangement)
    return [round(elem, 0) for elem in norms]

def shells(arrangement):
    norm_lst = round_norm_list(arrangement)
    norm_values = list(set(norm_lst))
    norm_value_count = []
    arrangement_shell = [[]]*len(norm_values)
    for i in range(0,len(norm_values)):
        count = 0
        for j in range(0,len(arrangement)):
            if norm_lst[j] == norm_values[i]:
                count += 1
                arrangement_shell[i] = arrangement_shell[i] + [arrangement[j]]
        norm_value_count.append(count)
    return(norm_values, norm_value_count, arrangement_shell)


###### Drawing Functions ######

def draw(arrangement,col):
    #If in the weird numpy array format converts into a 2D NumPy array where each row corresponds to inner arrays in your original data
    draw_arrangement = np.vstack(arrangement)

    # Create a sphere
    theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)

    # convert data
    x1 = []
    x2 = []
    x3 = []
    for i in range(0, len(draw_arrangement)):
        x1.append(draw_arrangement[i, 0])
        x2.append(draw_arrangement[i, 1])
        x3.append(draw_arrangement[i, 2])

    # Render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xs, ys, zs, rstride=4, cstride=4, color='white', alpha=0.5, linewidth=0)
    ax.scatter(x1, x2, x3, color=col, s=80)
    plt.axis('off')
    ax.view_init(azim=90.0, elev=90.0)

    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])
    ax.set_zlim([-2.0, 2.0])
    ax.set_aspect("auto")
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_title("{0} ".format(len(arrangement)) + "points on a sphere")

    plt.subplots_adjust(hspace=0)  # Attempting to make the images larger

    plt.show()

def draw_shells(shell_arrangements):
    colors = ['crimson','midnightblue', 'olivedrab', 'deeppink', 'purple', 'orchid', 'blueviolet', 'blue', 'azure',
              'steelblue', 'deepskyblue', 'darkturquoise', 'lightpink', 'cyan', 'mediumseagreen', 'seagreen', 'lime',
              'limegreen', 'palegreen', 'lawngreen', 'thistle', 'yellow', 'gold', 'orange', 'darkorange', 'bisque',
              'orangered', 'coral', 'salmon', 'red', 'black']

    # Create a sphere
    theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)

    # Render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xs, ys, zs, rstride=4, cstride=4, color='white', alpha=0.5, linewidth=0) #For the sphere to be drawn

    for i in range(0,len(shell_arrangements)):
        draw_arrangement = shell_arrangements[i]
        # convert data
        x1 = []
        x2 = []
        x3 = []
        for j in range(0, len(draw_arrangement)):
            x1.append(draw_arrangement[j][0])
            x2.append(draw_arrangement[j][1])
            x3.append(draw_arrangement[j][2])

        print(i, colors[i])
        ax.scatter(x1, x2, x3, color=colors[i], s=80)
        plt.axis('off')
        ax.view_init(azim=90.0, elev=90.0)

    lim = 3
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_aspect("auto")
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_title("Shell Arrangements")

    plt.subplots_adjust(hspace=0)  # Attempting to make the images larger

    plt.show()


###### Central Major Functions ######

def relax_arrangement(arrangement, loops):
    amplitude = 1
    x = copy.deepcopy(arrangement)

    # calculate the initial energy
    energy = energy_arrangement(x)
    loop = 0

    while amplitude > 1e-5 and loop < loops:

        # temp variable old_energy to see if new arrangement has more or less
        old_energy = energy
        # move all the particles in force*gamma
        x_old = copy.deepcopy(x)
        x = copy.deepcopy(x_new(x, amplitude))

        # calculate new energy
        energy = energy_arrangement(x)

        # only accept move if energy decreases scale gamma down if negative
        if (energy - old_energy > 0.0):
            amplitude = amplitude * 0.9  # For higher values of n we use more loops, use a higher scale factor (0.9)
            x = x_old
            energy = old_energy

        amplitude = amplitude * 1.01 #TEST
        loop += 1

        if loop % 100001 == 0:
            print("Loop",loop,"Gamma:",amplitude, "Energy",energy)

    # print("Final gamma:", amplitude, "Total loops:", loop, "Energy:", energy)

    return(np.vstack(x), energy)


##### Genetic Algorithm Functions ######

#Checks if an energy value is lower than any of the other energy values in the list, returns index of highest energy in current population if higher than child's energy
#Returns 999 if child's energy is higher than all of current population
def check_update_pop(current_pop_energy, child_energy):
    current_max_energy = max(current_pop_energy)
    if child_energy > current_max_energy:
        return(999) #If our child has higher energy than our population we do not include it
    else:
        return(current_pop_energy.index(current_max_energy))

#Currently selects randomly and uniformly
def parent_picker(energy_list):
    return(random.sample(range(len(energy_list)), 2))

#Selects parents biased towards lower energies
def parent_picker_2(energy_list):
    total = sum(energy_list)
    average = total/len(energy_list)
    current_position = 0
    choice1_num = random.random() * total
    choice2_num = random.random() * total
    for i in range(0,len(energy_list)):
        current_position += 2*average - energy_list[i]
        #If we have now exceeded the chosen number then the number was contained in the last one
        if choice1_num < current_position:
            choice1 = i
            choice1_num = 2*total #So that it doesn't select a new choice
        if choice2_num < current_position:
            choice2 = i
            choice2_num = 2 * total  # So that it doesn't select a new choice
    return(choice1,choice2)

#Returns starting_n arrangements and their corresponding energies in a list
def starting_population(starting_n,loops,n):
    energy_pop_list = []
    arrangement_pop_list = []
    for i in range(0,starting_n):
        new_arrangement = relax_arrangement(random_central(n),loops)
        arrangement_pop_list.append(new_arrangement[0])
        energy_pop_list.append(new_arrangement[1])
    return(arrangement_pop_list, energy_pop_list)

# Flips the z values for all the points in an arrangement
def flip_z_arrangement(arrangement):
    flip_arrangement = []
    for point in arrangement:
        flipped_point = copy.deepcopy(point)
        flipped_point[2] = -1 * flipped_point[2]
        flip_arrangement.append(flipped_point)
    return (flip_arrangement)

# Returns the points from an arrangement all above a specificed z value
def above_eq_z_value(arrangement, z):
    new_arrangement = []
    for point in arrangement:
        if point[2] > z:
            new_arrangement = new_arrangement+[point]
    return (new_arrangement)

# Combines 2 parents, rotated randomly to form a child of the same value of n
def breed(parent1, parent2):
    #Cheat way to get the correct value of n
    n = len(parent1)

    #Rotates the parent by a random angle
    parent1 = copy.deepcopy(set_z_rotate(set_y_rotate(set_x_rotate(parent1,2*np.pi*random.random()),2*np.pi*random.random()),2*np.pi*random.random()))
    parent2 = copy.deepcopy(set_z_rotate(set_y_rotate(set_x_rotate(parent2,2*np.pi*random.random()),2*np.pi*random.random()),2*np.pi*random.random()))

    z_value = 0 #The distance above/below
    top = copy.deepcopy(above_eq_z_value(parent1,z_value))
    bottom = copy.deepcopy(flip_z_arrangement(above_eq_z_value(flip_z_arrangement(parent2),z_value)))

    while len(top) + len(bottom) != n:

        if len(top) + len(bottom) < n:
            z_value -= 0.00001
        else:
            z_value += 0.00001

        top = copy.deepcopy(above_eq_z_value(parent1, z_value))
        bottom = copy.deepcopy(flip_z_arrangement(above_eq_z_value(flip_z_arrangement(parent2), z_value)))

    # print("TOP:",len(top),"Bottom:",len(bottom)) #Possible reason for stuck is that we can't find a suitable z_value

    # Once we have obtained our top and bottom half with the correct number of points we combine
    new_arrangement = []
    for point in top:
        new_arrangement.append(point)
    for point in bottom:
        new_arrangement.append(point)
    # print("Total:", len(new_arrangement))
    return (new_arrangement)