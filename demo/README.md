# RL-KLM DEMO: FORM FILLING

This demonstration shows how RL-KLM can be used to predict task completion time for form filling task. Reinforcement Learning agent learns a policy to complete the form filling task by minimizing the task completion time. The learned policy is then used to form a KLM model to estimate the task completion time.

Demo app works only for MacOS.

RL-KLM code author: Katri Leino (katri.k.leino a aalto.fi), Aalto university.
The demo app author: Kashyap Todi, Aalto university.



## FILES
* Application KLMForms.app (Demo app)
* Directory: rl-klm_demo (Codes for RL-KLM)

## REQUIREMENTS
* PyBrain library (url: pybrain.org )
* Python 2.7
* MacOS

## INSTALL
1. Install PyBrain (url: pybrain.org )
2. Set KLMForms app and rl-klm_demo directory into Documents directory


---------------------------
## HOW TO USE
- Run KLMForms.app

### CHANGING LAYOUT
- Start item indicates the item user starts with
- Confirm item is the last item user selects
- Right click to add item
- Left click to delete item
- Drag item to move it to different location

### CALCULATING TASK COMPLETION TIME
- Click compute from right bottom corner.
- Red line shows the path RL-KLM generated for form filling.
- Task completion time is shown at the bottom. Formed KLM model sums the time estimates of the point and click operators required to estimate time completion time.


----------------------------
## HOW IT WORKS
- KLMForms calls RL-KLM code and gives it "input.txt" as an input.
- input.txt defines the locations of the form item:
    item_id x_coordinate y_coordinate width height
    * note: width and height have not yet been implemented to the model.
- RL-KLM code returns best path to file "best_path.txt"
    * gives item ids in the path order.
    * task completion time is in the last row.
    * Fitts' law parameters defined in "initialParams.py": fitts_a, fitts_b

