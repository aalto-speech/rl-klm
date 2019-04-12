## Version case3: GUI
# Filling a form
# Task: Visit all states.
# A state represents a form item and there is unique action for each item.

import numpy as np

from RL_optimizer import rl_optimizer
from initialParams import initializeParams


###################
# LOAD PARAMETERS
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

# Not needed in case3
UI = np.array([[0,1,1,1], [0,0,1,1], [0,1,0,1], [0,1,1,0]])


###################
# Call RL code
best_path, klm_value = rl_optimizer(UI, num_action, params)


###################
# Write to file
output_file = open('best_path.txt', 'w')
#output_file.writelines(tags[int(best_path[i])]+"\n")
for i in range(np.size(best_path)):
    output_file.writelines(tags[int(best_path[i])]+"\n")
    #print tags[int(best_path[i])]

output_file.writelines(str(klm_value))
output_file.close()


