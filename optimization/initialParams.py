__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

import time
import numpy as np

# Class is used to keep KLM and environment (UI) parameters.
class initializeParams:

    def __init__(self, batch_number):
        self.batch_number = batch_number
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

        # UI parameters
        self.num_states = 3
        self.num_of_mods = 2

        # Limit the search space by giving binary matrix as decimal.
        # Search interval
        # 3 states: - 238
        # 4 states: 4370 - 31710
        # 5 states: - 16510910
        self.start = 1
        self.end = 238 

        # Random Search
        self.random_search = False
        self.random_search_iters = 1

        # Command penalties
        self.penalties = np.array([1.18, 0.13, 0.33])

        # Sensor errors
        self.sensor_errors = [0, 0, 0]
        self.confusion_error = [0, 0, 0]
        
        # State probabilities
        self.state_probs = [0.5, 0.1, 0.1, 0.1, 0.1]

        # The number of best UIs that will be saved into text file after the optimization
        self.top = 3
        self.the_top = 30

        # Objective function weights
        self.w_klm = 1 # For KLM
        self.w_const = 5 # For learnability / consistency
        self.w_simpl = 1 # For learnability / simplicity


    def setSensorErrors(self, e_t, e_g, e_s):
        self.sensor_errors[0] = float(e_t)
        self.sensor_errors[1] = float(e_g)
        self.sensor_errors[2] = float(e_s)

