__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2018, Aalto Speech Research"

# Prints and writes them into textfile out results
import numpy as np
import operator as op
import os

file_path = "experiments/"+os.environ["folder_name"]

def save_summary(top_UI_summary, i, k, objective, params, best_actions):
    top_UI_summary.append([i, k, objective, best_actions])
    if len(top_UI_summary) > params.top:
        top_UI_summary = sorted(top_UI_summary, key=op.itemgetter(2))[:params.top]
    return top_UI_summary

def print_output(top_UI, top_UI_summary, total_time, total_time_rl, params):
    # Time stamp
    filename = "result_"+params.timestr+"_"+params.batch_number
    filename_summary = "result_summary_"+params.timestr+"_"+params.batch_number
    # File
    f = open(file_path+'/results/'+filename, 'w')

    goals = params.num_states

    f.write("Testing\n")
    f.write("Parameters: \n")
    f.write("Number of states: "+str(params.num_states)+"\n")
    f.write("Sensor errors: "+str(params.sensor_errors)+"\n")
    f.write("Penalties: "+str(params.penalties)+"\n")
    f.write("State Probabilities: "+str(params.state_probs)+"\n")
    f.write("Total computational time:"+str(total_time)+"\n")
    f.write("Time used to RL:"+str(total_time_rl)+"\n")
    f.write("Objective function weights: KLM:"+str(params.w_klm)+", Consist:"+str(params.w_const)+", Simpl:"+str(params.w_simpl)+"\n")

    print top_UI[0][5]

    f.write("TOP"+str(params.top)+"\n")
    for i in range(0, len(top_UI)): #range(0,params.top): 
        f.write("UI\n")
        f.write(str(top_UI[i][0])+"\n")

        f.write("Buttons\n")
        f.write(str(top_UI[i][1])+"\n")

        f.write("Policies\n")
        for g in range(0,goals):
            f.write("Goal "+str(g)+"\n")
            P = np.reshape((top_UI[i][2][g]), (params.num_states,len(top_UI[i][2][g])/params.num_states))
            f.write(str(P)+"\n")

        f.write("Objective function\n")
        f.write(str(top_UI[i][3])+"\n")

        f.write("Best actions\n")
        f.write(str(top_UI[i][4])+"\n")

        f.write("Avg task completion time\n")
        f.write(str(top_UI[i][5])+"\n")

        f.write("############################\n")
        f.write("############################\n")

    f.close()
