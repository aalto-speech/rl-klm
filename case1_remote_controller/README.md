# Case 1: Remote Controller

Estimates the KLM value for remote controller. RL-KLM learns the fastest (possible) policy, which is to utilize different button types.

# Background

RL-KLM is trained to control a television with a remote controller. 
There are ten states in the television: nine channels and the switched-off state. 
* Channels can be changed directly with number buttons. 
* With arrow buttons it is possible to move to next or previous channels. 
* Off/on button turns tv off/on.

# Code Structure
Run ```python main_remote.py```.

## Input and Output

Prints to terminal KLM estimates for each goal state and the average KLM estimate.

## Codes
- ```initialParams.py``` : Defines the KLM parameters.
- ```main_remote.py``` : Main file. 
- ```RL_optimizer.py``` : Defines RL parameters and trains the RL agent for the task.
- ```UIEnv.py``` : Defines the environment and actions.
- ```evaluation.py``` : Returns the KLM estimate based on the given policy.
