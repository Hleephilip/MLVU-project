#!/bin/bash

#SBATCH --job-name=train_latent_ddim_v14
#SBATCH --partition=mlvu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32000MB
#SBATCH --cpus-per-task=16

source /home/n0/mlvu019/.bashrc
eval "$(conda shell.bash hook)"
conda activate mlvu

srun python train_latent_ddim.py --epochs 100 --log_version "v14" --cfg_prob 0.05 --lambda_2 2.0 --gamma 1.0 --use_default_init