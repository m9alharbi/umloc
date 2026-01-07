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
JOB_NAME=neurit_umgloc_model_paper_smoothing_trans_3
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src_1/main.py \
        --dataset_directory "$DATA_DIR"/datasets/neurit \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'global_vel' \
	--use_scheduler \
	--model_type 'Q_LSTM_GAN' \
    --window_size 2000 \
    --seq_len 2000 \
    --step_size 200 \
    --feat_sigma 2.0 \
    --targ_sigma 30.0 \
    --dataset 'neurit' \
	train \
	--epochs 500 \
	--train_list "$PROJECT_DIR"/lists/neurit/list_train.txt \
	--val_list "$PROJECT_DIR"/lists/neurit/list_val.txt 