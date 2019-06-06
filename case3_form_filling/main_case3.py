__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

'''
##  Case3: GUI
Task is to fill a form by visiting all the states.
A state represents a form item. 
Actions define the transition to a certain state.
'''

import numpy as np

from RL_optimizer import rl_optimizer
from initialParams import initializeParams


###################
# Parameters
input_file = open('input.txt', 'r')
input_data = input_file.read()
input_file.close()

lines = input_data.split('\n')

num_items = np.size(lines)

tags = []
x_coord = [0]*num_items
y_coord = [0]*num_items
widths = [0]*num_items
heights = [0]*num_items

idx = 0
for line in lines:
    item_info = line.split(',')

    tags.append(item_info[0])
    x_coord[idx] = float(item_info[1])
    y_coord[idx] = float(item_info[2])
    widths[idx] = float(item_info[3])
    heights[idx] = float(item_info[4])

    idx += 1

###################
# Initialize parameters
params = initializeParams(x_coord, y_coord, widths, heights)
num_action = params.num_states-1


###################
# Call RL code
best_path, klm_value = rl_optimizer(num_action, params)


###################
# Write to file
output_file = open('best_path.txt', 'w')
for i in range(np.size(best_path)):
    output_file.writelines(tags[int(best_path[i])]+"\n")

output_file.writelines(str(klm_value))
output_file.close()


