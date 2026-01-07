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
JOB_NAME=umgloc_model_idol_dataset_map_building1_2
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src_1/main.py \
        --dataset_directory "$DATA_DIR"/datasets/idol_dataset \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'global_vel' \
	--use_scheduler \
    --use_map \
    --map_size 256 \
    --feat_sigma 10.0 \
    --targ_sigma 0.0 \
	--model_type 'GAN' \
    --window_size 200 \
    --seq_len 200 \
    --step_size 50 \
    --batch_size 32 \
    --dataset 'idol' \
	train \
	--epochs 125 \
    --lr 0.0003 \
	--train_list "$PROJECT_DIR"/lists/idol/known_building1.txt \
	--val_list "$PROJECT_DIR"/lists/idol/unknown_building1.txt \
    --lstm_path "$JOB_RESULTS_DIR"/checkpoints/lstm_checkpoint_20.pt