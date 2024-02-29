# Main Process Built Upon run_v3

# Create Starting Population of 10 Arrangements - DONE
# Probability Function Determines Parents Based on Energy - (Start with Uniform Distribution) - DONE
# Breeds child from 2 parents - DONE
# Chance of mutation if success- mutation occurs - (First Version no mutation)
# Child Relaxed to Local Minima - DONE
# If lower than any energy of current population, replaces the highest energy in population - DONE
# Can also rotate spheres randomly in breed function
# Can introduce idea that if sphere of same energy already exists don't put in (even if lower than some others)

import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *

### UPDATING THE POPULATION WITH CHILD

#Checks if an energy value is lower than any of the other energy values in the list
#Returns index of highest energy in current population if higher than child's energy
#Returns 999 if child's energy is higher than all of current population
def check_update_pop(current_pop_energy, child_energy):
    current_max_energy = max(current_pop_energy)
    if child_energy > current_max_energy:
        return(999) #If our child has higher energy than our population we do not include it
    else:
        return(current_pop_energy.index(current_max_energy))

#Uses the various other functions to create the next generation from our current population's energy and arrangment list
def next_generation(current_pop_arrangement, current_pop_energy,loops):
    parents = parent_picker(current_pop_energy)
    # print("Parents:",parents)
    starting_child = breed(current_pop_arrangement[parents[0]],current_pop_arrangement[parents[0]])
    child_arrangement, child_energy = relax_arrangement(starting_child,loops)
    # print("Child:",child_energy)
    population_removal = check_update_pop(current_pop_energy,child_energy) #Index of highest energy in current pop
    # print("Removal:",population_removal)
    if population_removal == 999:
        print("Child Energy Too High")
        return (current_pop_arrangement, current_pop_energy, population_removal)
    else:
        new_pop_arrangement, new_pop_energy = copy.deepcopy(current_pop_arrangement), copy.deepcopy(current_pop_energy)
        new_pop_arrangement[population_removal], new_pop_energy[population_removal] = child_arrangement, child_energy  # Updates the new population with child
        return (new_pop_arrangement, new_pop_energy, population_removal)


### SELECTS THE PARENTS

#Currently selects randomly and uniformly
def parent_picker(energy_list):
    return(random.sample(range(len(energy_list)), 2))

def parent_picker_2(energy_list):
    total = sum(energy_list)
    average = total/len(energy_list)
    current_position = 0
    choice1 = random() * total
    for i in range(0,energy_list):
        current_position += 2*average - energy_list[i]


### GENERATING OUR STARTING POPULATION

#Returns starting_n arrangements and their corresponding energies in a list
def starting_population(starting_n,loops):
    energy_pop_list = []
    arrangement_pop_list = []
    for i in range(0,starting_n):
        new_arrangement = relax_arrangement(proj((2.0*np.random.random((n,3))-1.0),n),loops)
        arrangement_pop_list.append(new_arrangement[0])
        energy_pop_list.append(new_arrangement[1])
    return(arrangement_pop_list, energy_pop_list)


### RELAXING POPULATIONS TO LOCAL MINIMA

# Calculates the energy of an arrangement
def energy_arrangement(arrangement):
    energy = 0.0
    for i in range(0, n):
        for j in range(i + 1, n):
            distance = np.sqrt(sum((arrangement[i] - arrangement[j]) ** 2))
            energy = energy + 1.0 / distance
    return(energy)


# Relaxes an arrangement to its local minimum, must have n particles
# Uses loops as an upper bound- stops earlier if low gamma (as likely at minima)
def relax_arrangement(arrangement, loops):
    amplitude = 0.9
    x = copy.deepcopy(arrangement)

    # calculate the initial energy
    energy = energy_arrangement(x)
    loop = 0

    while amplitude > 1e-5 and loop < loops:

        # temp variable old_energy to see if new arrangement has more or less
        old_energy = energy
        # move all the particles in force*gamma
        x = x_new(x, amplitude)
        # project all new points onto a unit sphere
        x = proj(x, n)

        # calculate new energy
        energy = 0.0
        for i in range(0, n):
            for j in range(i + 1, n):
                distance = np.sqrt(sum((x[i] - x[j]) ** 2))
                energy = energy + 1.0 / distance

        # accept move either way but scale gamma down if negative
        if (energy - old_energy > 0.0):
            amplitude = amplitude * 0.9  # For higher values of n we use more loops, use a higher scale factor (0.9)

        amplitude = amplitude * 1.01 #TEST
        loop += 1

        if loop % 10000 == 0:
            print("Gamma:",amplitude)
            print("Energy:",energy)
            print("")

    # print("Final gamma:", amplitude, "Total loops:", loop)

    return(np.vstack(x), energy)

### BREEDING ARRANGEMENTS

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

def breed(parent1, parent2):
    z_value = 0 #The distance above/below
    top = copy.deepcopy(above_eq_z_value(parent1,z_value))
    bottom = copy.deepcopy(flip_z_arrangement(above_eq_z_value(flip_z_arrangement(parent2),z_value)))

    while len(top) + len(bottom) != n:

        if len(top) + len(bottom) < n:
            z_value -= 0.001
        else:
            z_value += 0.001

        top = copy.deepcopy(above_eq_z_value(parent1, z_value))
        bottom = copy.deepcopy(flip_z_arrangement(above_eq_z_value(flip_z_arrangement(parent2), z_value)))

    # Once we have obtained our top and bottom half with the correct number of points we combine
    new_arrangement = []
    for point in top:
        new_arrangement.append(point)
    for point in bottom:
        new_arrangement.append(point)
    # print("Total:", len(new_arrangement))
    return (new_arrangement)

def mutate(arrangement):
    i = random.randint(0,len(arrangement)-1)
    mutated_arrangement = copy.deepcopy(arrangement)
    mutated_arrangement[i] = (2.0*np.random.random(3)-1.0)
    return(mutated_arrangement)

### DRAWING AND GRAPHS

#Creates 3D model of the arrangemnt specified
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

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_aspect("auto")
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_title("{0} ".format(n) + "points on a sphere")

    plt.subplots_adjust(hspace=0)  # Attempting to make the images larger

    plt.show()

def draw_special(arrangement):
    #If in the weird numpy array format converts into a 2D NumPy array where each row corresponds to inner arrays in your original data
    upper = above_eq_z_value(arrangement,0)
    lower = flip_z_arrangement(above_eq_z_value(flip_z_arrangement(arrangement),0))

    draw_arrangement = np.vstack(upper)

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
    ax.scatter(x1, x2, x3, color="red", s=80)

    draw_arrangement = np.vstack(lower)
    # convert data
    x1 = []
    x2 = []
    x3 = []
    for i in range(0, len(draw_arrangement)):
        x1.append(draw_arrangement[i, 0])
        x2.append(draw_arrangement[i, 1])
        x3.append(draw_arrangement[i, 2])

    ax.scatter(x1, x2, x3, color="blue", s=80)
    plt.axis('off')
    ax.view_init(azim=90.0, elev=90.0)

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_aspect("auto")
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_title("{0} ".format(n) + "points on a sphere")

    plt.subplots_adjust(hspace=0)  # Attempting to make the images larger

    plt.show()

def draw_special_mutate(arrangement): #Same as draw_special except the last point is drawn in a different colour
    #If in the weird numpy array format converts into a 2D NumPy array where each row corresponds to inner arrays in your original data
    # print("arrangement:",arrangement)
    mutate_point = arrangement[-1]
    # print("mute point:", mutate_point)
    arrangement = arrangement[0:len(arrangement)-1]
    # print("post arrangement", arrangement)

    upper = above_eq_z_value(arrangement,0)
    lower = flip_z_arrangement(above_eq_z_value(flip_z_arrangement(arrangement),0))

    draw_arrangement = np.vstack(upper)

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
    ax.scatter(x1, x2, x3, color="red", s=80)

    draw_arrangement = np.vstack(lower)
    # convert data
    x1 = []
    x2 = []
    x3 = []
    for i in range(0, len(draw_arrangement)):
        x1.append(draw_arrangement[i, 0])
        x2.append(draw_arrangement[i, 1])
        x3.append(draw_arrangement[i, 2])

    ax.scatter(x1, x2, x3, color="blue", s=80)

    draw_arrangement = mutate_point
    print(draw_arrangement)
    # convert data
    x1 = []
    x2 = []
    x3 = []
    for i in range(0, len(draw_arrangement)):
        x1.append(draw_arrangement[0])
        x2.append(draw_arrangement[1])
        x3.append(draw_arrangement[2])

    ax.scatter(x1, x2, x3, color="green", s=80)

    plt.axis('off')
    ax.view_init(azim=90.0, elev=90.0)



    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_aspect("auto")
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_title("{0} ".format(n) + "points on a sphere")

    plt.subplots_adjust(hspace=0)  # Attempting to make the images larger

    plt.show()

n = 40
# p1 = relax_arrangement(proj((2.0*np.random.random((n,3))-1.0),n),5000)
# draw(above_eq_z_value(p1[0],0),"red")

# p2 = relax_arrangement(proj((2.0*np.random.random((n,3))-1.0),n),5000)
# draw(flip_z_arrangement(above_eq_z_value(flip_z_arrangement(p2[0]),0)),"blue")

# child = breed(p1[0],p2[0])

# np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/draw_temp.txt',child)

child = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/draw_temp.txt') #If you want to load one (make sure n matches)

# draw_special(child) #Embryo before mutation

# draw_special_mutate(child)

draw(relax_arrangement(child,10000)[0],"purple")

# draw(relax_arrangement(proj((2.0*np.random.random((n,3))-1.0),n),20000)[0],"purple")