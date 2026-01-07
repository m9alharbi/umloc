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
JOB_NAME=umgloc_nomap_kaust_dataset_5sec_fold5_2
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src_1/main.py \
        --dataset_directory "$DATA_DIR"/datasets/umgloc_dataset \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'global_vel' \
	--use_scheduler \
    --feat_sigma 1.0 \
    --targ_sigma 10.0 \
	--model_type 'GAN' \
    --window_size 300 \
    --seq_len 300 \
    --step_size 30 \
    --dataset 'our' \
	train \
	--epochs 500 \
	--train_list "$PROJECT_DIR"/lists/our/train_fold5.txt \
	--val_list "$PROJECT_DIR"/lists/our/val_fold5.txt \
    --lstm_path "$JOB_RESULTS_DIR"/checkpoints/lstm_checkpoint_45.pt