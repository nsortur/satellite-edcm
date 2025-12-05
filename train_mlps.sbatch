#!/bin/bash

#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=gpu
#SBATCH -o outputs/%j
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=128GB

python train.py --multirun model_data=invariant syms=no_sym stl_data=cube_H,cube_He,cube_N,cube_N2,cube_O,cube_O2 syms.hidden_dim=128
# python train.py --multirun model_data=invariant syms=no_sym stl_data=cube_O syms.hidden_dim=128
