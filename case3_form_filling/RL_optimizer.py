__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

# Trains RL agent and evaluates the form. 
# Returns best path and the KLM estimate.

import pybrain

from scipy import *
import numpy as np

from UIEnv import UI, UITask
from evaluation import evaluation
from initialParams import initializeParams

from pybrain_rlklm.episodic import EpisodicExperiment
from pybrain.rl.agents import LearningAgent
from pybrain_rlklm.interface import ActionValueTable
from pybrain_rlklm.q import Q


''' 
INPUT:
    * num_of_actions : Number of form items (states) which can be visited.
    * params : initializeParams class object that holds KLM and environment parameters.

RETURNS:
    * best_path : A list of states in which order they are visited.
    * klm_avg : KLM estimate in average.
'''
def rl_optimizer(num_of_actions, params):

    # Defining UI environment
    actionmatrix = range(num_of_actions)
    goal = False # Default goal
    ui_env = UI(params.num_states, actionmatrix, num_of_actions, goal, params.sensor_errors, params.confusion_error, params.penalties, params.grid, params.dimensions, params.init_position, params.goal_position)
    av_table = ActionValueTable(params.num_states, num_of_actions, ui_env)
    av_table.initialize(1)

    # Train agent for each goal
    klm_tot = 0
    klm_avg = 0
    p_learned = 1


    ##############################################
    # Define Q-learning agent
    learner = Q(0.5, 0.99) 
    learner.explorer.epsilon = 0.7 
    learner.explorer.decay = 0.999
    learner.explorer.env = ui_env
    agent = LearningAgent(av_table, learner)

    #Initialize av table. Give action matrix as an input.
    av_table.initialize(-5., actionmatrix) 

    # Define task and experiment
    task = UITask(ui_env)
    experiment = EpisodicExperiment(task, agent, av_table)

    ##############################################
    # Training Agent
    for j in range(8): # Learning iterations

        runs = 50 # Episodes in one iteration
        experiment.doEpisodes(runs) 
        
        agent.learn()
        agent.reset()


    ##############################################
    # Evaluation of UI and policy for current goal
    # Loop to get average : use only if errors used
    klm_tasks_tot = np.array([0.]*(params.num_states-1))
    total_iterations = 1
    klm_tot = 0
    for i in range(total_iterations):
        # KLM value
        klm_g, best_path = evaluation(av_table, ui_env, task, False, params)
        if klm_g == -1: # Not learned
            klm_tot += 20*5
            p_learned = 0
            print "Policy not learned"
            break
        # Save to total KLM
        klm_tot += klm_g
    # Average KLM estimate
    klm_avg += klm_tot/total_iterations

    return best_path, klm_avg


