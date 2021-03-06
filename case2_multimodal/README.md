# Case 2: Multimodal Smart Alarm

In the second case multimodal smart alarm is evaluated by computing the average KLM estimate. The alarm can be controlled with three types of modalities: tactile, hand gestures and speech. Sensor error which affect state transitions can be set for each modality.

There are three types of errors: 
* Recognition error : Command is not recognized (system does nothing)
* Confusion error : Command is recognized as another command within the same modality.
* User error : Error caused by the user. 
    
# Background

This case looks at the effect of input/output recognition errors. The agent can turn the light on and off on the device, turn on the sleeping mode or turn the alarm on. Each command can be given using any of three modalities: tactile, gesture, and speech. 

The task space is defined as to visit all states starting from the all states.
The output is the average KLM estimate over all tasks. The average can be weighted with state propabilities where the more probable goal states are given higher weights.

See more information from the paper.

# Code Structure
Run ```python main_multimodal.py```.

## Input and Output
Define transition matrix for each modality in ```main_multimodal.py```. The transition matrix element indicates the command which causes the transition from s_{current_state} to s_{next_state}. 

Prints to terminal the average KLM estimate and the times agent used the modalities.

## Codes
- ```initialParams.py``` : Defines the KLM parameters and sensor errors.
  - note: some parameters are used only in other cases.
- ```main_multimodal.py``` : Main file. 
- ```RL_optimizer.py``` : Defines RL parameters and trains the RL agent for the task.
- ```UIEnv.py``` : Defines the environment and actions.
- ```evaluation.py``` : Returns the KLM estimate based on the given policy.

## PyBrain library modifications
Modified PyBrain files are located in ```pybrain_rlklm``` directory.
