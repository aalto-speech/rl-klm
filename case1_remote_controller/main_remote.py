__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

## Evaluating remote controller

import numpy as np

from RL_optimizer import rl_optimizer
from initialParams import initializeParams


#Defining Parameters
params = initializeParams()


##########################################
# Transition matrix for each type of buttons.

UI1 = np.array([[0,1,0,0,0,0,0,0,0,0], 
    [1,0,2,0,0,0,0,0,0,3],
    [1,3,0,2,0,0,0,0,0,0], 
    [1,0,3,0,2,0,0,0,0,0], 
    [1,0,0,3,0,2,0,0,0,0], 
    [1,0,0,0,3,0,2,0,0,0], 
    [1,0,0,0,0,3,0,2,0,0], 
    [1,0,0,0,0,0,3,0,2,0], 
    [1,0,0,0,0,0,0,3,0,2], 
    [1,2,0,0,0,0,0,0,3,0]])

# Transition matrix of UI
UI2 = np.array([[0,0,0,0,0,0,0,0,0,0], 
    [0,0,5,6,7,8,9,10,11,12],
    [0,4,0,6,7,8,9,10,11,12], 
    [0,4,5,0,7,8,9,10,11,12], 
    [0,4,5,6,0,8,9,10,11,12], 
    [0,4,5,6,7,0,9,10,11,12], 
    [0,4,5,6,7,8,0,10,11,12], 
    [0,4,5,6,7,8,9,0,11,12], 
    [0,4,5,6,7,8,9,10,0,12], 
    [0,4,5,6,7,8,9,10,11,0]])

# Allowed actions in eac state
actionmatrix = np.array([[1],[1,2,3,5,6,7,8,9,10,11,12],
    [1,2,3,4,6,7,8,9,10,11,12],
    [1,2,3,4,5,7,8,9,10,11,12],
    [1,2,3,4,5,6,8,9,10,11,12],
    [1,2,3,4,5,6,7,9,10,11,12],
    [1,2,3,4,5,6,7,8,10,11,12],
    [1,2,3,4,5,6,7,8,9,11,12],
    [1,2,3,4,5,6,7,8,9,10,12],
    [1,2,3,4,5,6,7,8,9,10,11]]) 
num_buttons = 12 # Number of unique buttons

#All Uis
UIs = [UI1, UI2] 
actions_in_uis = [[1,2,3],[4,5,6,7,8,9,10,11,12]] # [[1,2,3]] # 
actions_penalty = [[],[2,3],[1,4,5,6,7,8,9,10,11,12]]


#############################
# Call RL code
klm = rl_optimizer(UIs, actionmatrix, actions_in_uis, actions_penalty, num_buttons, params)

print "KLM average:", klm


