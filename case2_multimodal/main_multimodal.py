__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

'''
## Case2: Multimodal smart alarm

The task space is defined as to visit all states starting from the all states.
The output is the average KLM estimate over all tasks. The average can be weighted with state propabilities where the more probable goal states are given higher weights.

In this case there is 3 modalities: tactile, gestures and speech. Each modality has a unique command to transit to each state.

It is possible to model a sensor error where the command is not recognized correctly. There are three types of errors
    * Recognition error = Command is not recognized (system does nothing)
    * Confusion error = Command is recognized as another command within the same modality.
    * User error = Error caused by the user. 

'''

import sys
import numpy as np

from RL_optimizer import rl_optimizer
from initialParams import initializeParams
from splitUI import splitUI
from print_output import print_output, save_summary


##########################################
## Optimization
# Running parallel -- IF NOT USED : batch_tot=1
batch_num = str(sys.argv[0])
batch_tot = str(sys.argv[1])

## Defining Parameters
params = initializeParams(batch_num)
# Set sensor errors from arguments if other than initialized ones
#params.setSensorErrors(sys.argv[2], sys.argv[3], sys.argv[4])

# Splitting UIs into batches if parallel running is used. 
n_start, n_end = splitUI(batch_num, batch_tot, params)

# Saving policy
policies = [([])]*params.num_states

# For saving toplist
objective = -1
top_UI = params.top*[] #([([],[],policies,100)])*params.top


##########################################
for i in range(n_start,n_end+1):

    # Define transition matrix for each modality

    # Tactile
    UI1 = np.array([[0,2,3,4,], 
        [1,0,3,4],
        [1,2,0,4], 
        [1,2,3,0]])

    # Gestures
    UI2 = np.array([[0,6,7,8,], 
        [5,0,7,8],
        [5,6,0,8], 
        [5,6,7,0]])

    # Speech
    UI3 = np.array([[0,10,11,12,], 
        [9,0,11,12],
        [9,10,0,12], 
        [9,10,11,0]])


    # Allowed actions in each state
    actionmatrix = np.array([[2,3,4,6,7,8,10,11,12],[1,3,4,5,7,8,9,11,12],[1,2,4,5,6,8,9,10,12],[1,2,3,5,6,7,9,10,11]]) 

    # Number of unique commands
    num_buttons = 16

    #All Uis
    UIs = [UI1, UI2, UI3] 

    # Allowed action in each UI
    actions_in_uis = [[1,2,3,4] ,[5,6,7,8],[9,10,11,12]]

    # Penalty index for each action. The penalty values in params.penalties
    actions_penalty = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

    print "Sensor errors", params.sensor_errors
    print "Confusion errors", params.confusion_error

    
    #############################
    ## Learning policies and calculating KLM esitemate
    
    modality_table_total = np.array([0,0,0])
    klm_total=0

    # If sensor errors are used the number of iterations should be more than 1 to average the effect of errors.
    num_iters = 5

    for j in range(num_iters):
        objective, top_UI, best_actions, modality_table, klm = rl_optimizer(UIs, actionmatrix, actions_in_uis, actions_penalty, num_buttons, top_UI, params)

        # If not policy not learned. Only used in optimization.
        #if objective == -1:
        #    print "skip"
        #    continue

        print modality_table
        modality_table_total += modality_table

        # KLM estimates
        klm_total += klm


    print modality_table_total

    # Average KLM estimate over all iterations
    print "KLM estimate:", klm_total/num_iters




