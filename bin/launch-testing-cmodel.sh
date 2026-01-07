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
JOB_NAME=risc_qlstm_gan_nopenalty
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/test.sbatch \
    "$PROJECT_DIR"/src_1/main.py \
        --dataset_directory "$DATA_DIR"/datasets/seq_data \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'disp' \
	--use_scheduler \
	--model_type 'GAN' \
    --use_map \
	test \
	--lstm_path "$JOB_RESULTS_DIR"/checkpoints/lstm_checkpoint_740.pt \
	--gan_path "$JOB_RESULTS_DIR"/checkpoints/gan_checkpoint_310.pt \
	--test_list "$PROJECT_DIR"/lists/our/test_list.txt