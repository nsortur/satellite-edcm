# satellite-edcm

Code for [Sample Efficient Modeling of Drag Coefficients for Satellites with Symmetry](https://openreview.net/forum?id=u7r2160QiP)

## Training
1) `pip3 install -r requirements.txt`
    - `py3nj` might cause issues, make sure you have OpenMP installed
2) Generate dataset with [RSM toolkit](https://github.com/ASSISTLaboratory/WVU_RSM_Suite)

3) Configure correct symmetry, path to dataset, and other hyperparameters in `conf/config.yaml` and `conf/stl_data`. This project uses [Hydra](https://hydra.cc/) for configuration

4) Run `python3 train.py <symmetry> <satellite>`
    - example: `python3 train.py c2c2c2 cube`
    - `python3 train.py c2 grace`

5) Hydra automatically makes a directory for each run at `outputs/YYYY:MM:DD/HH:MM:SS`. The weights are saved here every configurable amount of epochs (change with `python3 train.py <symmetry> <satellite> weights.save_epoch_modulus=<num>`).

## Development

1) To add a data source path and/or a new satellite, create a config file in `cfg/stl_data`. 
2) To add a symmetry, create a config file in `cfg/syms` and add the necessary group in `groups.py`
