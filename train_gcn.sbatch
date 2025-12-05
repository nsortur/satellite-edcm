#!/bin/bash

#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=gpu
#SBATCH -o outputs/%j
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=128GB

# python train.py model_data=equivariant_dedm stl_data=cube_H n_epochs=25 weights.save_epoch_modulus=5 model_data.model.encoder_layers=3 model_data.model.lmax=3 model_data.model.decoder_layers=3 model_data.model.decoder_layer_hiddens="[64, 128, 256]"
# python train.py model_data=equivariant_dedm stl_data=cube_H n_epochs=25 weights.save_epoch_modulus=5 model_data.model.encoder_layers=2 model_data.model.f_out=32
python train.py --multirun model_data=gcn stl_data=cube_H,cube_He,cube_N,cube_N2,cube_O,cube_O2 n_epochs=25 weights.save_epoch_modulus=5 
