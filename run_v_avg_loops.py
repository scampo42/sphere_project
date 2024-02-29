import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from my_functions import *
import timeit
import statistics

#Calculates the energy of an arrangement
def energy_arrangement(arrangement):
    energy = 0.0
    for i in range(0, len(arrangement)):
        for j in range(i + 1, len(arrangement)):
            distance = np.sqrt(sum((arrangement[i] - arrangement[j]) ** 2))
            energy = energy + 1.0 / distance
    return(energy)

#Returns the number of loops before getting below a certain energy - Need to specify target energy
def mosd(arrangement,energy_target,n):
    amplitude = 0.8 #0.7
    x = copy.deepcopy(arrangement)

    # calculate the initial energy
    energy = energy_arrangement(x)

    #Start Timer
    start = timeit.default_timer()

    while energy > energy_target:

        # temp variable old_energy to see if new arrangement has more or less
        old_energy = energy
        x_old = x
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
            x, energy = x_old, old_energy

        amplitude = amplitude * 1.01 #TEST

    stop = timeit.default_timer()
    print(amplitude)
    return(stop-start)

def monte(arrangement,energy_target, n):
    amplitude = 1
    x = copy.deepcopy(arrangement)

    start = timeit.default_timer()

    # calculate the initial energy
    energy = 0.0
    for i in range(0, n):
        for j in range(i + 1, n):
            distance = np.sqrt(sum((x[i] - x[j]) ** 2))
            energy = energy + 1.0 / distance

    # the main loop to reduce the energy

    for loop in range(0, 30000000):

        # randomly choose a point to move
        i = random.randint(0, n - 1)

        # store the old coordinates of this point
        old = np.array(x[i])

        # randomly move this point
        x[i] = x[i] + amplitude * (2.0 * np.random.random(3) - 1.0)
        x = proj(x, i)

        # calculate the difference in energy
        difference = 0.0
        for j in range(0, n):
            if (j != i):
                distance = np.sqrt(sum((x[i] - x[j]) ** 2))
                distanceold = np.sqrt(sum((old - x[j]) ** 2))
                difference = difference + 1.0 / distance - 1.0 / distanceold;

        # accept or reject the move
        if (difference < 0.0):
            energy = energy + difference
            amplitude *= 1.01
        else:
            x[i] = old
            amplitude *= 0.998
            if amplitude < 5e-7:
                amplitude = 3e-6

        if energy < energy_target:
            stop = timeit.default_timer()
            # print("E:",energy,"Gamma:",amplitude)
            return (stop - start)

        if amplitude < 1e-8:
            print("Fail")
            return(1000)

        if loop % 10000 == 0:
            print(loop, energy, amplitude)

    print("Fail Loop")
    return(1000)


def avg_MoSD(n,energy_target,runs):
    loop_list = []
    for i in range(0, runs):
        start = proj((2.0 * np.random.random((n, 3)) - 1.0), n)
        loop_list.append(mosd(start, energy_target, n))
        # print(i)
    # print(loop_list)
    print("MoSD AVG:", statistics.mean(loop_list))
    print("MoSD SD:", statistics.stdev(loop_list))


def avg_MC(n,energy_target,runs):
    loop_list = []
    for i in range(0, runs):
        start = proj((2.0 * np.random.random((n, 3)) - 1.0), n)
        loop_list.append(monte(start, energy_target, n))
        print(i,":",loop_list[-1])
    # print(loop_list)
    print("MC AVG/SD")
    print(statistics.mean(loop_list),statistics.stdev(loop_list))

n = 30
energy_t = 359.60400

# avg_MoSD(n,energy_t,100)
avg_MC(n,energy_t,10)


# energy_t = 150.88156833376
# energy_t = 80.670244114295
# energy_t = 32.716949460149
# energy_t = 6.474691494689
# energy_t = 0.5000000000005