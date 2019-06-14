__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

import time
import numpy as np

# Class is used to keep KLM and environment (UI) parameters.
class initializeParams:

    def __init__(self):
        # UI parameters
        self.num_states = 10
        self.num__mods = 1

        # Command penalties
        self.penalties = np.array([1.018, 30*0.16, 3*0.39]) 

        # Sensor errors for each modality
        self.sensor_errors = [0, 0, 0] 
        self.confusion_error = [0, 0, 0] 
        
        # State probabilities
        self.state_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] 

        # Initial penalty for moving hand to control the remote
        self.mod_penalty = 0


    def setSensorErrors(self, e_t, e_g, e_s):
        self.sensor_errors[0] = float(e_t)
        self.sensor_errors[1] = float(e_g)
        self.sensor_errors[2] = float(e_s)

