# satellite-edcm

Code for [Sample Efficient Modeling of Drag Coefficients for Satellites with Symmetry](https://openreview.net/forum?id=u7r2160QiP).

## Training

This code can be used to reproduce experimental results or train on new satellite geometries / drag data.

1) `pip3 install -r requirements.txt`
    - `py3nj` might cause issues, make sure you have OpenMP installed

2) Generate dataset with [RSM toolkit](https://github.com/ASSISTLaboratory/WVU_RSM_Suite) for the satellite. An example CSV for training the invariant MLPs can be found at `data/cube50k.dat`, and `data/cube50k_mesh.dat` for training DEDM (single mesh). 

3) If using DEDM, obtain an .STL file that represents the satellite geometry. An example 1m cubesat can be found at `data/STLs/Cube_38_1m.stl`. 

This project uses [Hydra](https://hydra.cc/) for configuration, and PyTorch Lightning for handling training.

### Invariant MLPs
```
python src/main.py model=equi_mlp dataset=tensor_dataset
```

### DEDM
```
python src/main.py model=invariant dataset=mesh_dataset
```

5) Hydra automatically makes a directory for each run at `outputs/YYYY:MM:DD/HH:MM:SS`. Model checkpoints with the top 3 validation RMSEs are saved in `lightning_logs`, as well as the last when training finishes.

## Development

1) To add a data source path and/or a new satellite shape (as an STL file), change the data directory in `configs/dataset/mesh_dataset` or `tensor_dataset`. 
2) To add a symmetry for invariant MLPs, create a config file in `configs/syms` and add the necessary group in the `GSpaceInfo` class at `src/models/equi_mlp.py`.
