# Second version of the genetic multistart variation
# Uniform distribution to pick the parents for the child
# Parents chosen proportional to 2*AVG Energy - Parent Energy
# Can select same parent twice

import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *
from gen_functions import *

generations = 15
starting_n = 5

#A mutation that selects a point at random and moves it to a random position
def mutate(arrangement):
    i = random.randint(0,len(arrangement)-1)
    mutated_arrangement = copy.deepcopy(arrangement)
    mutated_arrangement[i] = normalise_point(2.0*np.random.random(3)-1.0) #Added normalise
    return(mutated_arrangement)

#Breed function now rotates each parent by a random amount
def breed(parent1, parent2):
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

#Uses the updated breed function and introduces a mutation 20% of the time
def next_generation(current_pop_arrangement, current_pop_energy,loops):
    parents = parent_picker_2(current_pop_energy)
    # print("Parents selected:",parents)
    starting_child = breed(current_pop_arrangement[parents[0]],current_pop_arrangement[parents[0]])
    # print("Child generated before possible mutation")
    if random.random() > 0.8:
        starting_child = copy.deepcopy(mutate(starting_child))
        print("Mutation Occured")
        print("Energy Post Mutation:",energy_arrangement(starting_child),"n Post Mutation:",len(starting_child))
    # print("Starting Child Complete")
    child_arrangement, child_energy = relax_arrangement(starting_child,loops)
    # print("Relaxed Child Complete")
    population_removal = check_update_pop(current_pop_energy,child_energy) #Index of highest energy in current pop
    if population_removal == 999:
        print("Child Energy Too High")
        return (current_pop_arrangement, current_pop_energy, population_removal)
    else:
        new_pop_arrangement, new_pop_energy = copy.deepcopy(current_pop_arrangement), copy.deepcopy(current_pop_energy)
        new_pop_arrangement[population_removal], new_pop_energy[population_removal] = child_arrangement, child_energy  # Updates the new population with child
        return (new_pop_arrangement, new_pop_energy, population_removal)


pop_arrange, pop_energy = starting_population(starting_n,2000)
gen_list = ["p"]*starting_n #This indicates what generations are in our final population
energy_min_list = ["p"]*(generations+1) #Energy for the min population at each gen
energy_avg_list = ["p"]*(generations+1) #Energy for the avg population at each gen
print("Starting Pop Energy",pop_energy)
print("Generation:",0,"Energy Average:",sum(pop_energy)/len(pop_energy),"Energy Min:",min(pop_energy))
energy_min_list[0], energy_avg_list[0] = min(pop_energy), sum(pop_energy)/len(pop_energy)

for i in range(0,generations):
    pop_arrange, pop_energy, member_removed = next_generation(pop_arrange,pop_energy,2000)
    print("Generation:",i+1,"Energy Average:",sum(pop_energy)/len(pop_energy),"Energy Min:",min(pop_energy))
    energy_min_list[i+1], energy_avg_list[i+1] = min(pop_energy), sum(pop_energy)/len(pop_energy)
    if member_removed != 999:
        gen_list[member_removed] = i+1 #Adds index of generation if added to our population

print("Final Population Energy:",pop_energy)
print("Generation List:",gen_list)

np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/gen_energy_min_1.txt',energy_min_list,fmt='%s')
np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/gen_energy_avg_1.txt',energy_avg_list,fmt='%s')


draw(pop_arrange[pop_energy.index(min(pop_energy))])