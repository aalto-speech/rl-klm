

folder_name="$1"
export folder_name

path_name="experiments/"$folder_name
mkdir $path_name
mkdir $path_name/logs
mkdir $path_name/results

# Args: Batch_number, total number of batches, sensor errors 1, 2, 3
python main_multimodal.py 1 1 0.1 0.3 0.1
#srun python main.py $SLURM_ARRAY_TASK_ID 20 5 $2 $3 $4

