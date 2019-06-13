__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

# Trains RL agent and evaluates the form. 
# Returns best path and the KLM estimate.

import pybrain

from scipy import *
import numpy as np
import math

from UIEnv import UI, UITask
from evaluation import evaluation
from pybrain_rlklm.episodic import EpisodicExperiment

from pybrain.rl.agents import LearningAgent
from pybrain_rlklm.q import Q
from pybrain_rlklm.interface import ActionValueTable


''' 
INPUT:
    * UImatrix : UI matrices for each modality.
    * actionmatrix : Allowed actions in each state.
    * actions_in_uis : Allowed action in each UI.
    * actions_penalty : Penalty index for each action. The penalty values in params.penalties-
    * num_of_actions : Number of unique commands.
    * params : initializeParams class object that holds KLM and environment parameters.

RETURNS:
    * modality_table_total : list indicates how many time each of modalities were used in the learned policy. [tactile, gesture, speech]
    * klm_total : The average task completion time
'''
def rl_optimizer(UImatrix, actionmatrix, actions_in_uis, actions_penalty, num_of_actions, params):
    policies = [([])]*params.num_states

    # Defining UI environment
    goal = 1 # Default goal
    ui_env = UI(UImatrix, actionmatrix, actions_in_uis, actions_penalty, goal, params)
    av_table = ActionValueTable(params.num_states, num_of_actions)

    klm_tot = 0 
    klm_avg = 0
    p_learned = 1
    modality_table_total = np.array([0,0,0])
    klm_total = 0

    # Train agent for each goal
    for g in range(0, ui_env.num_of_states):

        ##############################################
        # Define Q-learning agent
        learner = Q(0.5, 0.9) 
        learner.explorer.epsilon = 0.7 
        learner.explorer.decay = 0.999 
        learner.explorer.env = ui_env
        agent = LearningAgent(av_table, learner)
        
        # Define task and experiment
        task = UITask(ui_env)
        experiment = EpisodicExperiment(task, agent)

        # Initialze av table. Removes not allowed actions.
        av_table.initialize(1., actionmatrix) 

        # Set goal
        experiment.task.env.setGoal(g)

        for j in range(50):

            initial_state = mod(j, ui_env.num_of_states)
            if initial_state == g: continue
            experiment.task.env.setInitialState(initial_state)
    
            runs = 50
            experiment.doEpisodes(runs) 

            agent.learn()
            agent.reset()


        ##############################################
        # Evaluation of UI and policy for the current goal
        # Iterate to get average - use only if errors used
        total_iterations = 10
        klm_tot = 0
        for i in range(total_iterations):
            # KLM value
            klm_g, modality_table = evaluation(av_table, ui_env, g, params)
            
            # Not learned
            if klm_g == -1: 
                klm_tot += 20*5
                p_learned = 0
                return -1, 0,0,0,0

            # Save to total KLM
            klm_tot += klm_g/(params.num_states-1)
            

        klm_avg += klm_tot/total_iterations

        modality_table_total += np.array(modality_table)
        klm_total += klm_avg

        if p_learned == 0: break


    return modality_table_total, klm_total


