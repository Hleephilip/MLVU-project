#!/bin/bash

#SBATCH --job-name=train_latent_ddim_v1
#SBATCH --partition=mlvu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=24000MB
#SBATCH --cpus-per-task=8

source /home/n0/mlvu019/.bashrc
eval "$(conda shell.bash hook)"
conda activate mlvu

srun python train_latent_ddim.py