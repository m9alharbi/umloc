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
JOB_NAME=umgloc_model_ronin_dataset_90_oldmodel_2
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src_1/main.py \
        --dataset_directory "$DATA_DIR"/datasets/ronin \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'global_vel' \
	--use_scheduler \
	--model_type 'Q_LSTM_GAN' \
    --window_size 400 \
    --seq_len 400 \
    --step_size 100 \
    --batch_size 32 \
    --feat_sigma 0.001 \
    --targ_sigma 0.0 \
    --dataset 'ronin' \
	train \
    --lr 0.0003 \
	--epochs 200 \
	--train_list "$PROJECT_DIR"/lists/ronin/list_train.txt \
	--val_list "$PROJECT_DIR"/lists/ronin/list_val.txt 