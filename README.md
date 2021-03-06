# KLM modeling with Reinforcement Learning

RL-KLM automates KLM modeling with Reinforcement Learning for User Interface evaluation. KLM models are suited to evaluate point-and-click type of the user interfaces. In our approach Reinforcement Learning agent learns task policies which minimize the task completion time. The learned task policies are then used to form a KLM model to estimate the total task completion time for the user interface.

The currect version of the code assumes that the user interface can be modeled with Finite State Machine (FSM). RL-KLM can learn a policy for a task when the initial and goal states are defined. With FSM, it is possible to generate the tasks automatically.

This repository includes documented codes for all experiments in the paper "RL-KLM: Automating Keystroke-level Modeling with Reinforcement Learning" url: http://doi.org/10.1145/3301275.3302285.

RL-KLM demo evaluates form templates and is located in demo directory. Please read README.md in the demo directory for more details. 

# Requirements
* Python 2.7
* PyBrain library (http://pybrain.org)

# Codes
* Demo for form evaluation.
* Case 1: Computing the average KLM estimate for tactile remote controller with alternative commands.
* Case2: Computing the average KLM estimate for a multimodal smart alarm and reporting which modalities were used.
* Case3: Computing KLM for a form and reporting the best path between form items.
* Optimization: Optimizing simple remote controller.

See README files in each directory for more information.

Contact: Katri Leino ( katri.k.leino a aalto.fi )
