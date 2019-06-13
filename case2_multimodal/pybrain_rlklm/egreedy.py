__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, array
import numpy as np

from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.rl.environments.environment import Environment


# Add params

class EpsilonGreedyExplorer(DiscreteExplorer):
    """ A discrete explorer, that executes the original policy in most cases,
        but sometimes returns a random action (uniformly drawn) instead. The
        randomness is controlled by a parameter 0 <= epsilon <= 1. The closer
        epsilon gets to 0, the more greedy (and less explorative) the agent
        behaves.
    """

    def __init__(self, epsilon = 0.3, decay = 0.9999):
        DiscreteExplorer.__init__(self)
        self.epsilon = epsilon
        self.decay = decay
        self.env = []

    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module

        #print np.nonzero(self.env.actions_index[self.env.current_state])[0]

        # Choose action from allowed actions. 
        if random.random() < self.epsilon:
            act = array([random.choice(self.env.mods[self.env.current_state])-1])
            outbuf[:] = act
            #print "current_state", self.env.current_state, "exploration action", act
            #print "allowed actions", self.env.mods[self.env.current_state]
            #outbuf[:] = array([random.randint(self.module.numActions)])
        else:
            outbuf[:] = inbuf

        self.epsilon *= self.decay


