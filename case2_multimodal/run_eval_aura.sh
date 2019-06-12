#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=100
#SBATCH --array=0-19
#SBATCH -o out/"$1_"%A_%a.out

folder_name="$1"
export folder_name

path_name="experiments/"$folder_name
mkdir $path_name
mkdir $path_name/logs
mkdir $path_name/results

# Args: Batch_number, total number of batches, sensor errors 1, 2, 3
python main_aura_evaluation.py $SLURM_ARRAY_TASK_ID 1 1 0.1 0.3 0.1
#srun python main.py $SLURM_ARRAY_TASK_ID 20 5 $2 $3 $4

