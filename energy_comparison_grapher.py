#Run v2 for the method of steepest descent and v0 for the monte-carlo
#Make sure that the value for n is the same (else the comparison is likely useless)
#The values in v2_loop_list have already been scaled by n (as n particles are moved at once not 1)

import numpy as np
import matplotlib.pyplot as plt


# Reading back from the text file into a new variable
v2_energy_list = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/gen_energy_min.txt')
v0_energy_list = np.loadtxt('/Users/samcampion/Documents/University/Uni Y3/Project/Python Stuff/gen_energy_avg.txt')

for i in range(0, len(v0_energy_list)):
    v0_energy_list[i] = np.log10(v0_energy_list[i] - v2_energy_list[-1])

for i in range(0, len(v2_energy_list)):
    v2_energy_list[i] = np.log10(v2_energy_list[i] - v2_energy_list[-1])

v0_loop_list, v2_loop_list = [], []
for i in range(0, len(v0_energy_list)):
    v0_loop_list.append(i)
for i in range(0,len(v2_energy_list)):
    v2_loop_list.append(i)



#Defining how we are plotting the data
def plot_lines(x1_values, y1_values, x2_values, y2_values, color1='blue', color2='red'):
    # plt.figure(figsize=(8, 6))

    # Plotting the first line
    plt.plot(x1_values, y1_values, color=color1, label='AVG Energy')
    # Plotting the second line
    plt.plot(x2_values, y2_values, color=color2, label='Min Energy')

    # Adding labels and title
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Generation Number', fontsize = 30)
    plt.ylabel('Log(Energy - Final Minimum Energy Value)', fontsize = 30)
    plt.title('Energy over generations®', fontsize = 30)
    plt.legend()

    # Show the plot
    plt.show()

plot_lines(v0_loop_list, v0_energy_list, v2_loop_list, v2_energy_list, color1='green', color2='orange')