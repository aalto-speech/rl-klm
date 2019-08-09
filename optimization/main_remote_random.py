__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

# Optimizing Remote controller

# Generates all possible binary transition matrices
# Leave out matrices which do not meet requirements
# 1. all states are connected

# Generates all possible button combinations in button_combinations.py

# Binary matrices are expressed with decimal and then transformed to binary.

import sys
import time
import operator as op
import numpy as np
import random
import logging

from RL_optimizer import rl_optimizer
from initialParams import initializeParams
from splitUI import splitUI
from print_output import print_output, save_summary
from button_combinations import find_buttons


############################################
start_time = time.time() 

##########################################
# Parallel programming stuff -- IF NOT USED : batch_tot=1
batch_num = str(sys.argv[1])
batch_tot = str(sys.argv[2])

# Defining Parameters
params = initializeParams(batch_num)
n_start, n_end = splitUI(batch_num, batch_tot, params)

#############################################
# For saving toplist
objective = -1
top_UI = 3*[] #([([],[],policies,100)])*params.top
top_UI_summary = [] # []*params.top #np.zeros((params.top,3))

############################################
## Parameters for generating UIs
n = params.num_states*params.num_states
bit_len = '0'+repr(n)+'b' # binary form, lenght | e.g. format(4, '016b')

############################################
#Time
total_time_rl = 0

##########################################
# Generating transition matrices
# Write to file only once in each outer iteration
for i in range(n_start,n_end+1):
    #Printing progress
    prog = (float(i)-(n_start))/(float(n_end)-float(n_start+1))*100
    sys.stdout.write("\rCurrent progress: %f%% "%prog)
    sys.stdout.flush()

    a_str = format(i,bit_len) # dec to binary | e.g. format(4, '016b')

    a = np.zeros(n)
    for j in range(0,n):
        a[j] = a_str[j] # transforming str bin into array

    UI = np.reshape(a, (params.num_states, params.num_states)) # bin to matix
    
    # Matrix is allowed
    if np.size(np.nonzero(np.sum(UI, axis=0))) + np.size(np.nonzero(np.sum(UI, axis=1))) == 2*params.num_states and np.sum(np.diagonal(UI)) == 0:   

        ##########################
        # Generate button matrices
        ##########################

        #t = int(np.sum(UI)) # num_of_transitions
        #transitions_state = np.sum(UI, 1)
        num_buttons = params.num_states 

        # Number of transitions in each state
        state_trans = []
        for state in range(params.num_states):
            state_trans.append(int(np.sum(UI[state])))

        # Generate all button matrices
        button_matrices = find_buttons(UI)

        if params.random_search_iters >= len(button_matrices) or params.random_search==False:
            search_all = True # No random search
            params.random_search_iters = len(button_matrices)
        else: search_all = False
        search_iter = 0
        search_past_idxes = []
        top_buttons = [] # Saves best of this UI
        while search_iter != params.random_search_iters: # while X: choose 3 random numbers for indexes.

            ##########################
            # Random Search
            ##########################
            if params.random_search == True and search_all==False:
                random_search_idx = random.randint(0, len(button_matrices)-1)
                while random_search_idx in search_past_idxes:
                    random_search_idx = random.randint(0, len(button_matrices)-1)
                search_past_idxes.append(random_search_idx)
                buttonmatrix = button_matrices[random_search_idx]
            else:
                if search_iter == len(button_matrices):
                    break
                buttonmatrix = button_matrices[search_iter]


            ##########################
            # Train RL agent
            ##########################
            start_time_rl = time.time()
            top_buttons, objective = rl_optimizer(UI, buttonmatrix, num_buttons, top_buttons, params, batch_num, logging)

            total_time_rl += (time.time()-start_time_rl)

            if objective == -1: # If not learned, skip all button combinations
                print "Skipping this UI"
                break

            search_iter += 1


        for k in range(len(top_buttons)):
            top_UI.append(top_buttons[k])
            top_UI = sorted(top_UI, key=op.itemgetter(3))[:params.the_top]    
        
                

end_time = time.time()
total_time = (end_time-start_time)/60
if objective != -1:
    print_output(top_UI, top_UI_summary, total_time, (total_time_rl/60), params)





