#!/bin/bash

#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=gpu
#SBATCH -o outputs/%j
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=128GB

python train.py model=dedmv2 dataset=mesh_dataset
