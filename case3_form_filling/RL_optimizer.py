# Evaluates and ranks UIs

import pybrain

from scipy import *
import numpy as np
import math
import sys

from copy import copy
import operator as op

from UIEnv import UI, UITask
from evaluation import evaluation
from initialParams import initializeParams

from pybrain_rlklm.episodic import EpisodicExperiment
from pybrain_rlklm.experiment import Experiment

from pybrain.rl.environments.environment import Environment
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import SARSA
from pybrain.rl.learners.learner import EpisodicLearner

from pybrain_rlklm.egreedy import EpsilonGreedyExplorer
from pybrain_rlklm.interface import ActionValueTable
from pybrain_rlklm.q import Q


def rl_optimizer(UImatrix, num_of_actions, top_UI, params, batch_num, logging):

    #global policies
    #global top_UI
    policies = [([])]*params.num_states
    num_states = params.num_states

    # Defining UI environment
    actionmatrix = range(num_of_actions)
    goal = False # Default goal
    ui_env = UI(num_states, actionmatrix, num_of_actions, goal, params.sensor_errors, params.confusion_error, params.penalties, params.grid, params.dimensions, params.init_position, params.goal_position)
    av_table = ActionValueTable(num_states, num_of_actions, ui_env)
    av_table.initialize(1)

    # Train agent for each goal
    klm_tot = 0
    klm_avg = 0
    policies = []
    best_actions = []# [0]*ui_env.num_of_states*(ui_env.num_of_states-1)
    objective = -1
    p_learned = 1
    ii = 0


    #########
    # Define Q-learning agent
    learner = Q(0.5, 0.99) #Q(0.6, 0.99) # 0.5, 0.99
    learner.explorer.epsilon = 0.7 # 0.7 # 0.9
    learner.explorer.decay = 0.999 # 0.99
    learner.explorer.env = ui_env
    agent = LearningAgent(av_table, learner)

    # Define task and experiment
    task = UITask(ui_env)
    experiment = EpisodicExperiment(task, agent, av_table)
    #######

    #Removed bad actions, give action matric as input
    av_table.initialize(-5., actionmatrix) 

    for j in range(8): # Learning iterations

        initial_state = 0 

        runs = 50 # Episodes in one iteration
        experiment.doEpisodes(runs) 
        
        agent.learn()
        agent.reset()

        ##############################################
        # Save policy
        # For optimization
        p = list(av_table.params) # Copies to new memory slot
        policies.append(p)

    ##############################################
    # Evaluation of UI and policy for current goal
    # Loop to get average : use only if errors used

    klm_tasks_tot = np.array([0.]*(params.num_states-1))
    total_iterations = 1
    klm_tot = 0
    for i in range(total_iterations):
        # KLM value
        klm_g, best_path = evaluation(av_table, ui_env, task, False, params, batch_num, logging)
        if klm_g == -1: # Not learned
            klm_tot += 20*5
            p_learned = 0
            print "error"
            break
        # Save to total KLM
        klm_tot += klm_g
    klm_avg += klm_tot/total_iterations

    return top_UI, objective, best_actions, best_path, klm_g


