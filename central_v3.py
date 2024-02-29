#This makes use of genertic algorithms to find the best arrangement. This best arrangement is saved to a text file

import copy
import random
import matplotlib
import math
import numpy
import numpy as np
from central_functions import *

n = 100
generations = 30
starting_n = 10


def mutate(arrangement):
    i = random.randint(0,len(arrangement)-1)
    mutated_arrangement = copy.deepcopy(arrangement)
    mutated_arrangement[i] = (4.0*np.random.random(3)-2.0) #May want to vary
    return(mutated_arrangement)

def next_generation(current_pop_arrangement, current_pop_energy,loops):
    parents = parent_picker_2(current_pop_energy)
    # print("Parents selected:",parents)
    starting_child = breed(current_pop_arrangement[parents[0]],current_pop_arrangement[parents[0]])
    # print("Child generated before possible mutation")
    if random.random() > 0.8:
        starting_child = mutate(starting_child)
        print("Mutation Occured")
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
        np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central3_.txt',new_pop_arrangement[pop_energy.index(min(pop_energy))])
        return (new_pop_arrangement, new_pop_energy, population_removal)

#Generates our starting population & corresponding lists
pop_arrange, pop_energy = starting_population(starting_n,2000, n)
gen_list = ["p"]*starting_n #This indicates what generations are in our final population
print("Starting Pop Energy",pop_energy)
print("Generation:",0,"Energy Average:",sum(pop_energy)/len(pop_energy),"Energy Min:",min(pop_energy))

#Generations of population occur
for i in range(0,generations):
    pop_arrange, pop_energy, member_removed = next_generation(pop_arrange,pop_energy,10000)
    print("Generation:",i+1,"Energy Average:",sum(pop_energy)/len(pop_energy),"Energy Min:",min(pop_energy))
    if member_removed != 999:
        gen_list[member_removed] = i+1 #Adds index of generation if added to our population

print("Final Population Energy:",pop_energy)
print("Generation List:",gen_list)

np.savetxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central3.txt', pop_arrange[pop_energy.index(min(pop_energy))])

best_arrangement = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/central3.txt')

print("Energy:",energy_arrangement(best_arrangement))
print("Shell Details:",shells(best_arrangement)[0:2])
draw_shells(shells(best_arrangement)[2])