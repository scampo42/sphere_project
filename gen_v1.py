# First version of the genetic multistart variation
# Uniform distribution to pick the parents for the child
# Parents chosen from uniform distribution

import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *
from gen_functions import *

generations = 80
starting_n = 20

def breed(parent1, parent2):
    z_value = 0 #The distance above/below
    top = copy.deepcopy(above_eq_z_value(parent1,z_value))
    bottom = copy.deepcopy(flip_z_arrangement(above_eq_z_value(flip_z_arrangement(parent2),z_value)))

    while len(top) + len(bottom) != n:

        if len(top) + len(bottom) < n:
            z_value -= 0.0001 #Previously 0.001
        else:
            z_value += 0.0001

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



pop_arrange, pop_energy = starting_population(starting_n,2000)
gen_list = ["p"]*starting_n #This indicates what generations are in our final population
print("Starting Pop Energy",pop_energy)
print("Generation:",0,"Energy Average:",sum(pop_energy)/len(pop_energy),"Energy Min:",min(pop_energy))

for i in range(0,generations):
    pop_arrange, pop_energy, member_removed = next_generation(pop_arrange,pop_energy,2000)
    print("Generation:",i+1,"Energy Average:",sum(pop_energy)/len(pop_energy),"Energy Min:",min(pop_energy))
    if member_removed != 999:
        gen_list[member_removed] = i+1 #Adds index of generation if added to our population

print("Final Population Energy:",pop_energy)
print("Generation List:",gen_list)

draw(pop_arrange[pop_energy.index(min(pop_energy))])