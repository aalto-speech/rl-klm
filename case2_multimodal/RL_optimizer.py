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
    * top_UI : Saves best UI's. Only for optimization.
    * params : initializeParams class object that holds KLM and environment parameters.

RETURNS:
    * 
'''
def rl_optimizer(UImatrix, actionmatrix, actions_in_uis, actions_penalty, num_of_actions, top_UI, params):
    policies = [([])]*params.num_states

    # Defining UI environment
    goal = 1 # Default goal
    ui_env = UI(UImatrix, actionmatrix, actions_in_uis, actions_penalty, goal, params.sensor_errors, params.confusion_error, params.penalties)
    av_table = ActionValueTable(params.num_states, num_of_actions)
    #av_table.initialize(1)

    # Train agent for each goal
    klm_tot = 0 
    klm_avg = 0
    policies = []
    best_actions = []
    objective = -1
    p_learned = 1
    modality_table_total = np.array([0,0,0])
    klm_total = 0
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

        train_reward=[]
        for j in range(50): # 10, 20 

            initial_state = mod(j, ui_env.num_of_states)
            if initial_state == g: continue
            experiment.task.env.setInitialState(initial_state)
    
            runs = 50 #15
            experiment.doEpisodes(runs) 

            agent.learn()
            agent.reset()
            
        ##############################################
        ## Optimization
        # Save policy
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
            klm_g, modality_table = evaluation(av_table, ui_env, g, params)
            if klm_g == -1: # Not learned
                klm_tot += 20*5
                p_learned = 0
                break

            # Save to total KLM
            klm_tot += klm_g/(params.num_states-1)
            #klm_tasks_tot += klm_tasks
        if klm_g == -1:
            return -1, 0,0,0,0

        #klm_avg += params.state_probs[g]*klm_tot/total_iterations
        klm_avg += klm_tot/total_iterations

        modality_table_total += np.array(modality_table)
        klm_total += klm_avg

        if p_learned == 0: break

        ######### Get best actions from policy for logging
        best_actions_in_state = []
        for s in range(0, ui_env.num_of_states):
            if s != g: 
                best_actions_in_state.append(av_table.getMaxAction(s))
        best_actions.append(best_actions_in_state)




    return top_UI, objective, best_actions, modality_table_total, klm_total


