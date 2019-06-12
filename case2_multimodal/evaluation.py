__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

import pybrain

from scipy import *
import numpy as np
import logging

from UIEnv import UI, UITask
from initialParams import initializeParams

# Error log
file_path = "experiments/"+os.environ["folder_name"]
logging.basicConfig(filename=file_path+'/logs/error_test.log',level=logging.DEBUG)


# Modality
def what_modality(action, action_in_uis):
    for i in range(len(action_in_uis)):
        for j in range(len(action_in_uis[i])):
            if int(action+1) == int(action_in_uis[i][j]):
                #print "modality", i
                return i


# Evaluation function
def evaluation(av_table, ui_env, goal, params):
    time_klm = 0
    time_klm_tot = 0
    modality_table = np.array([0,0,0])
    klm_tasks = []

    for initial_state in range(0, ui_env.num_of_states):
        if initial_state == goal: # Skip
            continue

        # Set environment parameters
        ui_env.setInitialState(initial_state)
        current_state = ui_env.getSensors()
        
        steps = 0
        time_klm = 0
        prev_action = -1
        while int(goal) != int(current_state[0]):

            action = av_table.getMaxAction(current_state)
            if steps == 0:
                modality = int(what_modality(action, ui_env.action_in_uis))
                modality_table[modality] += 1
            ui_env.performAction(action)
            time_klm = time_klm + ui_env.getPenalty(action, prev_action)
            current_state = ui_env.getSensors()
            if steps > 30: 
                time_klm = -1 # Discard whole UI
                print 'Policy not learned'
                print ui_env.env
                print ui_env.mods
                print av_table.params
                logging.warn('Policy not learned or UI is not allowed ')
                logging.warn(ui_env.env)
                logging.warn(ui_env.mods)
                logging.warn(av_table.params)
                return -1, -1
            steps = steps+1
            prev_action = action
        klm_tasks.append(time_klm)
        time_klm_tot += time_klm


    return time_klm_tot/4, modality_table 
