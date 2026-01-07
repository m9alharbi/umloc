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
JOB_NAME=umgloc_map_kaust_dataset_fold8
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# ─────────────────────────  Parse CLI flags  ─────────────────────────
GAN_FILE=""          # e.g. checkpoints/gan_checkpoint_last.pt
LSTM_FILE=""         # e.g. checkpoints/lstm_checkpoint_last.pt

usage() {
  echo "Usage: $0 --gan_file <relative_path_inside_JOB_RESULTS_DIR> \\"
  echo "          --lstm_file <relative_path_inside_JOB_RESULTS_DIR>"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
      --gan_file)  GAN_FILE="$2";  shift 2 ;;
      --lstm_file) LSTM_FILE="$2"; shift 2 ;;
      *)           echo "Unknown option: $1"; usage ;;
  esac
done

# ─────────────────────────  Require both flags  ──────────────────────
if [[ -z "$GAN_FILE" || -z "$LSTM_FILE" ]]; then
  echo "Error: both --gan_file and --lstm_file must be supplied."
  usage
fi

# Final absolute paths (always under JOB_RESULTS_DIR)
GAN_PATH="$JOB_RESULTS_DIR/$GAN_FILE"
LSTM_PATH="$JOB_RESULTS_DIR/$LSTM_FILE"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/test.sbatch \
    "$PROJECT_DIR"/src_1/main.py \
        --dataset_directory "$DATA_DIR"/datasets/umgloc_dataset \
	--output_directory "$JOB_RESULTS_DIR" \
	--target_type 'global_vel' \
	--use_scheduler \
    --use_map \
	--model_type 'Q_LSTM_GAN' \
    --window_size 600 \
    --feat_sigma 1.0 \
    --targ_sigma 10.0 \
    --seq_len 600 \
    --step_size 60 \
    --dataset 'our' \
	test \
	--test_list "$PROJECT_DIR"/lists/our/test_fold8.txt \
	--lstm_path "$LSTM_PATH" \
    --gan_path "$GAN_PATH"