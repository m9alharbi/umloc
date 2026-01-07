#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
export PROJECT_DIR="$PWD"

# path to the Conda environment
# ENV_PREFIX="$PROJECT_DIR"/env

# project should have a data directory
DATA_DIR="$PROJECT_DIR"/data

# creates a separate directory for each job
JOB_NAME=ronin_calib_oritransform_smooth_st100_ws400
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src/main.py \
        --dataset_directory "$DATA_DIR"/datasets/ronin/train_dataset_1 \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'disp' \
    --dataset 'ronin' \
	--model_type 'single_task' \
    --step_size 100 \
    --window_size 400 \
    --layers 3 \
    --h_dim 100 \
	test \
	--test_list "$PROJECT_DIR"/lists/ronin/list_val.txt \
	--model_path "$JOB_RESULTS_DIR"/checkpoints/checkpoint_996.pt
