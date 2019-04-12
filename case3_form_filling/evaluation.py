import pybrain

from scipy import *
import numpy as np
import logging
import os

from UIEnv import UI, UITask
from initialParams import initializeParams


# KLM  Evaluation function
#  getReward returns ID value for Fitts' law. (easy to break, always check)
# Output: klm estimate (seconds), best path

def evaluation(av_table, ui_env, task, goal, params):
    time_klm = 0

    # Set environment parameters
    ui_env.reset()
    task.reset()
    current_state = ui_env.getSensors()

    # Save best path
    best_path = [0] # start with starting point

    steps = 0
    time_klm = 0
    prev_action = -1
    while not(goal):
        action = av_table.getMaxAction(current_state)
        best_path.append(action+1)

        task.performAction(action)
        
        # Remove used action
        allowed_actions =  np.where(np.array(task.env.visited_states) == 0)[0]
        av_table.setAllowedActions(allowed_actions) # Set allowed actions
        
        ID = -task.getReward()
        time_klm = time_klm + params.fitts_a + params.fitts_b*ID + 0.31*100
        current_state = ui_env.getSensors()
        task.prev_action = action

        # For optimization
        if steps > 10: 
            time_klm = -1 # Discard whole UI
            print 'Policy not learned'
            print ui_env.env # Print visited states
            print av_table.params
            logging.warn('Policy not learned or UI is not allowed ')
            logging.warn(ui_env.env)
            logging.warn(av_table.params)
            return -1
        steps = steps+1
        prev_action = action

        if task.isFinished():
            break

    best_path.append(ui_env.num_of_actions) # Last action is the confirmation

    ui_env.reset()
    task.reset()


    return time_klm/100, best_path
