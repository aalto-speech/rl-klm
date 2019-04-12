# Case 3: Form Filling

Codes for the case 3 where the task is to evaluate the form with KLM model. The RL-KLM agent learns the fastest path according to the KLM operators between the form items and reports the task completion time for visiting all form items.

# Background

In the form filling task, the task is not only to reach a certain state but to visit all states at least once and end by pressing ’Confirm’. The agent is able to visit any unvisited state at any given time. The policy agent learns defines the path between the form items. Learning agent is given a penalty after each action based on the time consumption of the action. The costs are defined with Fitts' Law. 

The KLM task completion time is estimated from the policies. The operators which are accounted for are the finger movement and the tapping operator. We do not include the cost of entering text as an operator, as this is inconsequential to policies.

See more information from the paper.

# Code structure

## Input and Output
- input.txt defines the locations of the form item: 
  - item_id x_coordinate y_coordinate width height
  - note: width and height have not yet been implemented to the model.
- RL-KLM code returns best path to file "best_path.txt"
    - gives item ids in the path order.
    - task completion time is in the last row.

## Codes

- **initialParams.py*** : 
- **main_case3.py** : 
- **RL_optimizer.py** : 
- **UIEnv.py** :
- **evaluation.py** : 

## PyBrain library modifications
Modified PyBrain files are located in pybrain_rlklm directory.
