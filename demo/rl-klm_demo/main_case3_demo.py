## Version case3: GUI
# Filling a form
# Task: Visit all states.
# A state represents a form item and there is unique action for each item.

import sys
import os
import operator as op
import numpy as np
import math
import random
import itertools
from RL_optimizer import rl_optimizer
from initialParams import initializeParams
import time
import logging


############################################
# LOAD PARAMETERS
input_file = open('input.txt', 'r')
input_data = input_file.read()

input_file.close()

lines = input_data.split('\n')

num_items = np.size(lines)

#tags = []*num_items
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

############################################
# Initialize parameters
batch_num = 1
params = initializeParams(x_coord, y_coord, widths, heights)


##############################################
# Saving policy
policies = [([])]*params.num_states

#############################################
# For saving toplist
objective = -1
top_UI = params.top*[] #([([],[],policies,100)])*params.top
top_UI_summary = [] # []*params.top #np.zeros((params.top,3))
multi_obj_topUI = [[],[],[]]


##########################################

# TODO: READ UI from the text file uis/remote.txt
UI = np.array([[0,1,1,1], [0,0,1,1], [0,1,0,1], [0,1,1,0]])


num_action = params.num_states-1 # Number of unique buttons


#############################
# Call RL code
objective, top_UI, best_actions, best_path, klm_value = rl_optimizer(UI, num_action, top_UI, params, batch_num, logging)


if objective == -1: # If not learned, skip all button combinations
    print "Skipping this UI"


###################
# Write to file
output_file = open('best_path.txt', 'w')
#output_file.writelines(tags[int(best_path[i])]+"\n")
for i in range(np.size(best_path)):
    output_file.writelines(tags[int(best_path[i])]+"\n")
    #print tags[int(best_path[i])]

output_file.writelines(str(klm_value))
output_file.close()


