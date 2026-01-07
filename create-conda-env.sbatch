#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=debug
#SBATCH --job-name=create-conda-env
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# create the conda environment
./create-conda-env.sh
