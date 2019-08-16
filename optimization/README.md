# Optimization

Optimizes a simple remote controller similar to case1. The objective function of the optimization maximizes task completion time, consistency and, simplicity. The code creates all possible transition matrices and button combinations for the given amount of states, and evaluates the designs with RL-KLM. The best designs are reported in a text file located in ```experiments/results/```.

## Background

The UI is defined by binary transition matrix which indicates which transitions between states are allowed. Button matrix defines which buttons are used for these transitions. 

Weights of the objective function can be chaged depending on whether the speed or simplisity is wanted more from the best design.

## Code structure

Run with ```./run_random_search.sh TEST_NAME```

Change parameters in ```initialParams.py```.

Result is written in ```experiments/results/```.
