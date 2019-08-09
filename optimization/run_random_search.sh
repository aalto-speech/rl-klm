#!/bin/bash

folder_name="$1"
export folder_name

path_name="experiments/"$folder_name
mkdir $path_name
mkdir $path_name/logs
mkdir $path_name/results

# ARG: Batch_num, batch_total
python main_remote_random.py $SLURM_ARRAY_TASK_ID 1 1

