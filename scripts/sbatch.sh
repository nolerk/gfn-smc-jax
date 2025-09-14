#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH -J gfn-smc-jax
#SBATCH -o /home/mila/c/chois/workspace/gfn-smc-jax/slurm_logs/%x-%j.out

cd /home/mila/c/chois/workspace/gfn-smc-jax/
export PYTHONPATH=".:$PYTHONPATH"

source /home/mila/c/chois/.bashrc
conda activate sampling_bench

# Make wandb directory if it doesn't exist
export WANDB_DIR=/home/mila/c/chois/workspace/gfn-smc-jax/
mkdir -p $WANDB_DIR
wandb login --relogin 909a41a8ff4a89ae006840ffaffe1463dcf2d12f

# Run the experiment
ARGS=$1
python run.py $ARGS
