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

n = 282

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


### SELECTS THE PARENTS

#Currently selects randomly and uniformly
def parent_picker(energy_list):
    return(random.sample(range(len(energy_list)), 2))

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
    for i in range(0, len(arrangement)):
        for j in range(i + 1, n):
            distance = np.sqrt(sum((arrangement[i] - arrangement[j]) ** 2))
            energy = energy + 1.0 / distance
    return(energy)


# Relaxes an arrangement to its local minimum, must have n particles
# Uses loops as an upper bound- stops earlier if low gamma (as likely at minima)
def relax_arrangement(arrangement, loops):
    amplitude = 0.5
    x = copy.deepcopy(arrangement)

    # calculate the initial energy
    energy = energy_arrangement(x)
    loop = 0

    while amplitude > 1e-6 and loop < loops:

        # temp variable old_energy to see if new arrangement has more or less
        old_energy = energy
        # move all the particles in force*gamma
        x_old = copy.deepcopy(x)
        x = copy.deepcopy(x_new(x, amplitude))
        # project all new points onto a unit sphere
        x = proj(x, n)

        # calculate new energy
        energy = 0.0
        for i in range(0, n):
            for j in range(i + 1, n):
                distance = np.sqrt(sum((x[i] - x[j]) ** 2))
                energy = energy + 1.0 / distance

        # only accept move if energy decreases scale gamma down if negative
        if (energy - old_energy > 0.0):
            amplitude = amplitude * 0.9  # For higher values of n we use more loops, use a higher scale factor (0.9)
            x = x_old
            energy = old_energy

        amplitude = amplitude * 1.01 #TEST
        loop += 1

        if loop % 4001 == 0:
            print("Gamma:",amplitude)
            print("Energy:",energy)
            print("")

    #IMPLEMENTED TO TEMP STOP ERROR FOR n = 282
    if energy <37147:
        print("Error in Energy Calc:",energy,"Actual Energy:",energy_arrangement(x))
        np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/error_arrangement.txt',new_pop_arrangement[pop_energy.index(min(pop_energy))])
        energy = 9999999

    print("Final gamma:", amplitude, "Total loops:", loop, "Energy:", energy)

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

### DRAWING AND GRAPHS

#Creates 3D model of the arrangemnt specified
def draw(arrangement):
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
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_surface(xs, ys, zs, rstride=4, cstride=4, color='yellow', alpha=0.5, linewidth=0)
    ax.scatter(x1, x2, x3, color="black", s=80)
    plt.axis('off')
    ax.view_init(azim=90.0, elev=90.0)

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_aspect("auto")
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_title("{0} ".format(n) + "points on a sphere")

    plt.show()
