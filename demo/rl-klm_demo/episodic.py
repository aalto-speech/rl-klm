__author__ = 'Tom Schaul, tom@idsia.ch'


from experiment import Experiment
from pybrain.rl.agents.optimization import OptimizationAgent

import numpy as np


class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. """

    doOptimization = False

    def __init__(self, task, agent, avtable):
        if isinstance(agent, OptimizationAgent):
            self.doOptimization = True
            self.optimizer = agent.learner
            self.optimizer.setEvaluator(task, agent.module)
            self.optimizer.maxEvaluations = self.optimizer.numEvaluations
            self.task = task
            self.avtable = avtable
        else:
            Experiment.__init__(self, task, agent, avtable)

    def _oneInteraction(self):
        """ Do an interaction between the Task and the Agent. """
        if self.doOptimization:
            raise Exception('When using a black-box learning algorithm, only full episodes can be done.')
        else:
            return Experiment._oneInteraction(self)

    def doEpisodes(self, number = 1):
        """ Do one episode, and return the rewards of each step as a list. """
        #print "Starting episodes"

        if self.doOptimization:
            self.optimizer.maxEvaluations += number
            self.optimizer.learn()
        else:
            all_rewards = []
            for dummy in range(number):
                self.agent.newEpisode()
                rewards = []
                self.stepid = 0
                self.task.reset()
                while not self.task.isFinished():
                    r = self._oneInteraction()
                    rewards.append(r)
                all_rewards.append(rewards)


            return all_rewards
