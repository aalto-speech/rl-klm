__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

# Trains RL agent and evaluates UI. 

import pybrain

from scipy import *
import numpy as np
import math
import operator as op

from UIEnv import UI, UITask
from evaluation import evaluation
from print_output import print_output

from pybrain_rlklm.episodic import EpisodicExperiment
from pybrain.rl.agents import LearningAgent
from pybrain_rlklm.q import Q
from pybrain_rlklm.interface import ActionValueTable



''' 
INPUT:
    * UImatrix : UI matrices for each modality.
    * button : Defines the button combination.
    * num_of_actions : Number of unique commands.
    * top_UI : For saving the best UIs.
    * params : initializeParams class object that holds KLM and environment parameters.
    * batch_num : Batch number if parallel used.
    * logging : log

RETURNS:
    * top_UI : For saving the best UIs.
    * objective : Objective function output
'''
def rl_optimizer(UImatrix, buttonmatrix, num_of_actions, top_UI, params, batch_num, logging):

    # Defining UI environment
    goal = 1 # Default goal
    ui_env = UI(UImatrix, buttonmatrix, goal, params.sensor_errors, params.confusion_error, params.penalties)
    av_table = ActionValueTable(params.num_states, num_of_actions)

    # Check which actions are allowed
    actions_index = [[]]*params.num_states
    bad_actions_idx = []

    for j in range(0,params.num_states):
        for i in range(0,params.num_states):
            if i == j: continue
            if UImatrix[j][i] == 1:
                actions_index[j] = actions_index[j] + [1,1,1]
                bad_actions_idx = bad_actions_idx +[0,0,0]
            else:
                actions_index[j] = actions_index[j] + [0,0,0]
                bad_actions_idx = bad_actions_idx +[1,1,1]


    ui_env.actions_index = actions_index
    bad_actions = np.nonzero(np.array(bad_actions_idx)) 

    # Train agent for each goal
    klm_tot = 0 
    klm_avg = 0
    policies = []
    objective = -1
    p_learned = 1
    ii = 0
    for g in range(0, ui_env.num_of_states):

        ##########################
        # Define Q-learning agent
        ##########################
        learner = Q(0.5, 0.9)
        learner.explorer.epsilon = 0.3
        learner.explorer.decay = 0.999
        learner.explorer.actions_index = actions_index
        learner.explorer.env = ui_env
        agent = LearningAgent(av_table, learner)
        
        ##########################
        # Define task and experiment
        ##########################
        task = UITask(ui_env)
        experiment = EpisodicExperiment(task, agent)

        # Set low values for not allowed actions
        bad_actions = np.ones([ui_env.num_of_states, ui_env.num_of_states])
        for idx_state in range(ui_env.num_of_states):
            for idx_button in range(len(buttonmatrix[idx_state])):
                bad_actions[idx_state, buttonmatrix[idx_state]] = 0
        bad_actions = np.reshape(bad_actions, ui_env.num_of_states*ui_env.num_of_states)
        bad_actions = np.nonzero(bad_actions)

        av_table.initialize(1., bad_actions[0]) #Removed bad actions

        # Initialize saved N av_tables
        convergence_N = 1 # Move to params
        av_tables_save = []*convergence_N

        # Set goal
        experiment.task.env.setGoal(g)

        ##########################
        # Trainin agent
        ##########################
        # Add more iterations and runs if not learning.
        for j in range(10):

            initial_state = mod(j, ui_env.num_of_states)
            if initial_state == g: continue
            experiment.task.env.setInitialState(initial_state)
    
            runs = 15
            experiment.doEpisodes(runs) 

            agent.learn()
            agent.reset()

            
        ##########################
        # Save policy
        ##########################
        p = list(av_table.params) # Copies to new memory slot
        policies.append(p)


        ##############################################
        # Evaluation of UI and policy for current goal
        ##############################################
        # Loop to get average : use only if errors used
        klm_tasks_tot = np.array([0.]*(params.num_states-1))
        total_iterations = 15
        klm_tot = 0
        for i in range(total_iterations):
            # KLM value
            klm_g = evaluation(av_table, ui_env, g, params, batch_num, logging)
            if klm_g == -1: # Not learned
                klm_tot += 20*5
                p_learned = 0
                break
            # Save to total KLM
            klm_tot += klm_g/(params.num_states-1)
        klm_avg += params.state_probs[g]*klm_tot/total_iterations

        if p_learned == 0: break


    if p_learned == 1: # Policy learned
        ##########################
        # Consistency
        ##########################
        consistency = 0
        idx = 0
        transitions_state = np.sum(UImatrix, 0)
        buttons_to_states = np.zeros([params.num_states, params.num_states]) # Which buttons reach to the state goal
        for sr in range(params.num_states):
            for sc in range(params.num_states):
                if UImatrix[sr,sc] == 1:
                    buttons_to_states[sc][buttonmatrix[sr][idx]] += 1
                    idx = idx+1
            idx = 0
        for s in range(params.num_states):
            for act in range(params.num_states):
                if buttons_to_states[s][act] > 0:
                    consistency += math.log(buttons_to_states[s][act]/transitions_state[s])


        ##########################
        # Objective function
        ##########################
        objective = params.w_klm*klm_avg
        objective = objective-1*params.w_const*consistency 
        objective = objective + params.w_simpl*math.log(np.sum(np.sum(UImatrix, 1)))
        objective_func = [klm_avg, consistency, math.log(np.sum(np.sum(UImatrix, 1)))]

        ##########################
        # Save the best 
        ##########################
        top_UI.append([UImatrix, buttonmatrix, policies, objective, objective_func, klm_avg])
        if len(top_UI) > params.top:
            top_UI = sorted(top_UI, key=op.itemgetter(3))[:params.top]

    return top_UI, objective


