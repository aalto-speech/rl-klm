__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

import time
import numpy as np

# Class is used to keep KLM and environment (UI) parameters.
# Todo: Add widths and heights parameters to the fitts' law.
class initializeParams:
    '''
    INPUT:
    x_coord : array of x coordinates of the form items
    y_coord : array of y coordinates of the form items
    widths : array of widths of the form items
    heights : array of heights of the form items
    '''

    def __init__(self, x_coord, y_coord, widths, heights):
        
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

        # UI parameters
        self.num_states = np.size(x_coord)
        self.num_of_mods = 1

        # KLM estiamtes depend on the distance of previous and current item.
        self.penalties = np.array([-1]) # Not used in case 3
        self.widths = np.array([40]*self.num_states) # Width of each item
        self.dimensions = [200, 40] # Width, Height
        self.init_position = [x_coord[0], y_coord[0]] # User's initial position
        self.goal_position = [x_coord[self.num_states-1], y_coord[self.num_states-1]] # The last position user needs to press
        self.grid = [x_coord[1:self.num_states], y_coord[1:self.num_states]] 

        # Fitts' Law parameters
        self.fitts_a = 54.38 
        self.fitts_b = 14.62 

        # Sensor errors 
        self.sensor_errors = [0, 0, 0] # Only single modality, hard coded into perform_action()
        self.confusion_error = [0, 0, 0] # Does not currently work
        
        # State probabilities
        self.state_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] #[0.5, 0.3, 0.2]

        # Initial penalty for moving hand to control the remote
        self.mod_penalty = 0


        #############
        # Optimization
        self.batch_number = 1
        self.start = 1
        self.end = 1 
        self.random_search = False
        self.random_search_iters = 5

        # The number of best UIs that will be saved into text file after the optimization
        self.top = 1
        self.the_top = 1

        # Objective function weights
        self.w_klm = 1 # For KLM
        self.w_const = 1 # For learnability / consistency
        self.w_simpl = 1 # For learnability / simplicity

        self.multi_objective = True
        self.objectives = [[1,1,0],[0,1,1],[1,1,1]] #multi-objective
        


