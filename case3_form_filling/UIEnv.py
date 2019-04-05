import pybrain

import numpy as np
from scipy import *
import math
import copy

class UI():


    def __init__(self, NumStates, Actionmatrix, NumActions, Goal, SensorErrors, ConfusionError, Penalties, ItemGrid, ItemDimensions, InitPos, GoalPos):
    	self.initState = 0 
    	self.current_state = self.initState
        self.actionmatrix = Actionmatrix # allowed actions
        self.num_of_states = NumStates
        self.num_of_actions = NumActions # assumes all numbers are used
        self.visited_states = [0]*self.num_of_actions # chance also reset
        self.visited_states[self.num_of_actions-1] = 1
        self.used_actions = [0]*self.num_of_actions
        
        self.goal = Goal # True/False
        self.sensor_errors = SensorErrors
        self.confusion_error = ConfusionError
        self.penalties = Penalties
        self.bad_Action = 0
        self.sys_delay = 0
        self.exit_state = False

        self.width = 6.
        self.grid = ItemGrid
        self.dimensions = ItemDimensions
        self.initial_position = InitPos
        self.goal_position = GoalPos

        self.maxSteps = 20 # Maximum steps before the episode stops
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
    	return self.goal # [float(self.goal),] 

    def performAction(self, action):
        # Performs given action and changes state
        self.steps += 1
        

        # Error model
        #modality = 0 # for remote controller there is only a single modality
        #action = self.errorModel(action, modality)
        #if action == -1: return


        self.current_state = action+1
        self.visited_states[action] = 1

        if np.sum(self.visited_states) == self.num_of_actions:
            self.visited_states[self.num_of_actions-1] = 0
            self.goal = True

    def errorModel(self, action, modality):
        # TODO: Does not work
        # User's error
        if random.uniform(0,1) < 0.04:
            return -1

        # Recognition error: Not recognized
        if random.uniform(0,1) < self.sensor_errors[modality]: 
            return -1

        # Confusion error  # TODO: Not working, fix
        if random.uniform(0,1) < self.confusion_error[modality]:
            action_list = copy.deepcopy(self.mods[self.current_state])
            action_list.remove(float(action)) # Remove original action from the list
            if len(action_list) > 0:
                act = int(random.choice(action_list))
                return act

        return action

    def reset(self):
        self.current_state = self.initState
        self.visited_states = [0]*self.num_of_actions
        self.visited_states[self.num_of_actions-1] = 1
        self.used_actions = [0]*self.num_of_actions
        self.goal = False

    def getPenalty(self, action, prev_action):
        # Compute penalty

        penalty = 0
        

        if int(self.used_actions[action]) > 1: # Uses same action more than once. Should not be possible
            return 15

        # Calculating distance between items.
        if prev_action == -1: #Previous state was the initial state
            distance_x = np.abs(self.grid[0][int(action)]-self.initial_position[0])
            distance_y = np.abs(self.grid[1][int(action)]-self.initial_position[1])
            distance = math.sqrt(distance_x**2 + distance_y**2)
        elif np.sum(self.visited_states) == self.num_of_states: # Last command
            distance_x = np.abs(self.grid[0][int(action)]-self.goal_position[0])
            distance_y = np.abs(self.grid[1][int(action)]-self.goal_position[1])
            distance = math.sqrt(distance_x**2 + distance_y**2)
            penalty += 5
        else:
            distance_x = np.abs(self.grid[0][int(action)]-self.grid[0][int(prev_action)])
            distance_y = np.abs(self.grid[1][int(action)]-self.grid[1][int(prev_action)])
            distance = math.sqrt(distance_x**2 + distance_y**2)

        if distance_x== 0: # Approach from above or below
            width = self.dimensions[1]
        elif distance_y == 0: # Approach from either side
            width = self.dimensions[0]
        elif math.tan(distance_y/distance_x) < math.pi/4: # Approach from side
            width = self.dimensions[0]
        else: 
            width = self.dimensions[1]


        #Fitts' Law
        penalty += math.log((2.*distance/width), 2)
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
        self.prev_action = -1 

    def getObservation(self):
        """ A filtered mapping to getSensors of the underlying environment. """
        sensors = self.env.getSensors()
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        return sensors

    def isFinished(self):
        if self.samples >= self.env.maxSteps: 
            return True

        if np.sum(self.env.visited_states) == self.env.num_of_actions: # action list is empty
            return True

        if self.env.goal:
            return True

        else: return False

    def performAction(self, action):
        act = int(action)
        self.current_action = act
        self.env.performAction(act)
        self.samples += 1

    def getReward(self):      
        reward = -1*self.env.getPenalty(self.current_action, self.prev_action)
        return reward
   
    def addReward(self):
        # Is this even needed?
        if self.env.current_state == -1: return
        """ A filtered mapping towards performAction of the underlying environment. """
        # by default, the cumulative reward is just the sum over the episode

        discount_value = 0.9
        if self.discount:
            self.cumreward += power(discount_value, self.samples) * self.getReward()
        else:
            self.cumreward += self.getReward()

    def getTotalReward(self):
        """ Return the accumulated reward since the start of the episode """
        return self.cumreward




