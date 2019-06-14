__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

import pybrain

import numpy as np
from scipy import *

class UI():

    '''
    INPUT:
        * UImatrix : UI matrices for each modality.
        * Actionmatrix : Allowed actions in each state.
        * Actions_in_UIs : Allowed action in each UI.
        * Actions_Penalty : Penalty index for each action. The penalty values in params.penalties.
        * Goal : Goal state.
        * params : Class holds KLM and environment parameters
    '''

    def __init__(self, Environment, Actionmatrix, Actions_in_UIs, Actions_Penalty, Goal, params):
    	self.initState = 0 
    	self.current_state = self.initState
        self.env = Environment
        self.mods = Actionmatrix # allowed actions
        self.action_in_uis = Actions_in_UIs
        self.num_of_states = len(self.env[0])
        self.num_of_actions = np.max(Actions_in_UIs)
        self.actions_penalty = Actions_Penalty
        self.goal = Goal
        self.sensor_errors = params.sensor_errors
        self.confusion_error = params.confusion_error
        self.penalties = params.penalties
        self.ui_index = np.nonzero(self.env)
        self.bad_Action = 0
        self.sys_delay = 0
        self.exit_state = False

        self.maxSteps = 20
        self.steps = 0
        

    def setGoal(self, Goal):
        self.goal = int(Goal)

    def setInitialState(self, initial_state):
        self.initState = initial_state
        self.current_state = initial_state

    def getSensors(self):
        """ the currently visible state of the world / Current state """
        return [float(self.current_state),]

    def getGoal(self):
    	return self.goal 

    def performAction(self, action):
        # Performs given action and changes state
        action += 1

        # Goal achieved
        self.steps += 1

        if self.current_state == self.goal: # Go to exit state
            self.current_state = -1
            self.exit_state = True
            return

        # Find correct UI matrix
        ui_idx = -1
        for i in range(len(self.action_in_uis)):
            for j in range(len(self.action_in_uis[i])):
                if int(action) == int(self.action_in_uis[i][j]):
                    ui_idx = i
                    break
            if ui_idx != -1: break

        if ui_idx == -1:
            print "ERROR: Action", action, "not available. Exit"
            exit()


        # Error model
        modality = ui_idx
        action = self.errorModel(action, modality)
        if action == -1: return

        # Find next state
        transition = -1
        for i in range(len(self.env[ui_idx][self.current_state])):
            if self.env[ui_idx][self.current_state][i] == action:
                transition = i
                break

        self.current_state = transition


    def errorModel(self, action, modality):
        # User's error
        if random.uniform(0,1) < 0.04:
            return -1

        # Recognition error: Not recognized
        if random.uniform(0,1) < self.sensor_errors[modality]: 
            return -1

        if random.uniform(0,1) < self.confusion_error[modality]:
            action = int(random.choice(self.action_in_uis[modality]))

        return action

    def reset(self):
        self.current_state = self.initState

    def getPenalty(self, action, prev_action):

        # Get penalty based on action's class 
        for i in range(len(self.actions_penalty)):
            if int(action)+1 in self.actions_penalty[i]:
                penalty = self.penalties[i]

        penalty += self.sys_delay

        return penalty


class UITask():
    #: Discount factor
    discount = True
    batchSize = 1
    

    def __init__(self, environment):
        """ All tasks are coupled to an environment. """
        self.env = environment
        self.cumreward = 0 # tracking cumulative reward
        self.samples = 0 # tracking the number of samples

        self.sensor_limits = None
        self.actor_limits = None
        self.clipping = True

        self.current_action = 0 # Saving current action
        self.prev_action = -1 # Saving previous action

    def reset(self):
        self.env.reset()
        self.cumreward = 0
        self.samples = 0

    def getObservation(self):
        """ A filtered mapping to getSensors of the underlying environment. """
        sensors = self.env.getSensors()
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        return sensors

    def isFinished(self):
        if self.samples >= self.env.maxSteps: 
            return True
        if self.env.current_state == -1:
            return True
        else: return False

    def performAction(self, action):
        act = int(action)
        self.current_action = act
        self.env.performAction(act)
        self.samples += 1
        self.addReward()
        self.prev_action = self.current_action

    def getReward(self):
        reward = 0
        if self.env.bad_Action:
            self.env.bad_Action = 0
            return -15
        else: 
            reward = -1*self.env.getPenalty(self.current_action, self.prev_action)
            if self.isFinished(): reward += 1
        if self.env.maxSteps-1 <= self.samples:
            reward = -15

        return reward
   
    def addReward(self):
        if self.env.current_state == -1: return
        """ A filtered mapping towards performAction of the underlying environment. """
        discount_value = 0.1
        if self.discount:
            self.cumreward += power(discount_value, self.samples) * self.getReward()
        else:
            self.cumreward += self.getReward()

    def getTotalReward(self):
        """ Return the accumulated reward since the start of the episode """
        return self.cumreward




