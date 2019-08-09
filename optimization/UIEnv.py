__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

import pybrain

import numpy as np
from scipy import *
import copy

class UI():


    def __init__(self, Environment, Buttons, Goal, SensorErrors, ConfusionError, Penalties):
    	self.initState = 0 # Muokkaa, satunainen?
    	self.current_state = self.initState
        self.env = Environment
        self.mods = Buttons
        self.num_of_states = len(self.env)
        self.num_of_transitions = int(np.sum(self.env))
        self.num_of_mods = int(np.max(self.num_of_transitions)) # Num of buttons
        self.num_of_actions = int(np.max(self.num_of_transitions)) # Num of buttons
        self.goal = Goal
        self.sensor_errors = SensorErrors
        self.confusion_error = ConfusionError
        self.penalties = Penalties
        self.ui_index = np.nonzero(self.env)
        self.bad_Action = 0
        self.sys_delay = 0
        self.exit_state = False

        self.maxSteps = 27 # Maximum steps before the episode stops
        self.steps = 0
        


    def setGoal(self, Goal):
        self.goal = int(Goal)

    def setInitialState(self, initial_state):
        self.initState = initial_state
        self.current_state = initial_state

    def setModalities(self, Buttons):
        print "setModalities: Not available"

    def getSensors(self):
        """ the currently visible state of the world / Current state """
        return [float(self.current_state),]

    def getGoal(self):
    	return self.goal 

    def performAction(self, action):
        # Actions is one of the buttons.
        self.steps += 1

        if self.current_state == self.goal: # Go to exit state
            self.current_state = -1
            self.exit_state = True
            return

        # Current state
        actions_in_state = self.mods[self.current_state]
        trans_in_state = self.env[self.current_state]
        index_trans_in_state = np.nonzero(trans_in_state)[0]

        # Error model
        modality = 0
        action = self.errorModel(action, modality)
        if action == -1: return

        current_transition = 0
        for i in range(len(self.mods[self.current_state])):
            if action == self.mods[self.current_state][i]:
                break
            if i == (len(self.mods[self.current_state])-1): # Action does nothing
                return
            current_transition += 1


        # Update current state
        self.current_state = index_trans_in_state[current_transition]

    def errorModel(self, action, modality):
        # User's error
        if random.uniform(0,1) < 0.04:
            return -1

        # Recognition error: Not recognized
        if random.uniform(0,1) < self.sensor_errors[modality]: 
            return -1

        # Confusion error 
        if random.uniform(0,1) < self.confusion_error[modality]:
            action_list = copy.deepcopy(self.mods[self.current_state])
            action_list.remove(float(action)) # Remove original action from the list
            if len(action_list) > 0:
                act = int(random.choice(action_list))
                return act

        return action

    def reset(self):
        self.current_state = self.initState

    def getPenalty(self, action, prev_action):
        penalty = 0
        if prev_action == -1: # First time pressing button in episode
            penalty = self.penalties[0]
            prev_action = True
        elif prev_action == int(action): # Press same button as before
            penalty = self.penalties[1]
        else: # Press different button
            penalty = self.penalties[2]
        # System delay + feedback
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
        return reward
   
    def addReward(self):
        if self.env.current_state == -1: return
        """ A filtered mapping towards performAction of the underlying environment. """
        # by default, the cumulative reward is just the sum over the episode
        discount_value = 0.1
        if self.discount:
            self.cumreward += power(discount_value, self.samples) * self.getReward()
        else:
            self.cumreward += self.getReward()

    def getTotalReward(self):
        """ Return the accumulated reward since the start of the episode """
        return self.cumreward




