# satellite-edcm

Code for [Sample Efficient Modeling of Drag Coefficients for Satellites with Symmetry](https://openreview.net/forum?id=u7r2160QiP)

## Training
1) Generate dataset with [RSM toolkit](https://github.com/ASSISTLaboratory/WVU_RSM_Suite)

2) Configure correct symmetry, path to dataset, and other hyperparameters in `conf/config.yaml`. This project uses [Hydra](https://hydra.cc/) for configuration

3) Run `python3 train.py`
