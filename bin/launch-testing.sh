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
JOB_NAME=quantile
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src/main.py \
        --dataset_directory "$DATA_DIR"/datasets/seq_data \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'disp' \
	--model_type 'Q' \
	test \
	--test_list "$PROJECT_DIR"/lists/our/list_train.txt \
	--model_path "$JOB_RESULTS_DIR"/checkpoints/checkpoint_831.pt
