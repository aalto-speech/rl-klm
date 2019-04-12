__author__ = 'Tom Schaul, tom@idsia.ch'
import numpy as np
# EDITED: Removing used actions


class Experiment(object):
    """ An experiment matches up a task with an agent and handles their interactions.
    """

    def __init__(self, task, agent, avtable):
        self.task = task
        self.agent = agent
        self.avtable = avtable
        self.stepid = 0

    def doInteractions(self, number = 1):
        """ The default implementation directly maps the methods of the agent and the task.
            Returns the number of interactions done.
        """
        for _ in range(number):
            # Set goal?
            #print "Goal is", np.random.random_integers(0,4)
            self._oneInteraction()
        return self.stepid

    def _oneInteraction(self):
        # MODIFIED
        """ Give the observation to the agent, takes its resulting action and returns
            it to the task. Then gives the reward to the agent again and returns it.
        """
        self.stepid += 1
        self.agent.integrateObservation(self.task.getObservation())
        action = self.agent.getAction()
        self.task.performAction(action)

        # Remove used action
        allowed_actions =  np.where(np.array(self.task.env.visited_states) == 0)[0]
        self.avtable.setAllowedActions(allowed_actions) # Set allowed actions

        reward = self.task.getReward()
        self.agent.giveReward(reward)
        self.task.prev_action = action
        return reward
