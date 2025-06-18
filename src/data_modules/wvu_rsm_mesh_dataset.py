import pandas as pd
import torch
import numpy as np
from hydra import utils
import trimesh
from torch_geometric.data import Data
from e3nn import o3
import torch_geometric.transforms as T
from wvu_rsm_tensor_dataset import DragDataset, WVURSMDataModule
import os
from torch_geometric.data import Batch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GeometricDataLoader


class DragMeshDataset(DragDataset):
    def __init__(
        self, df, feature_min=None, feature_max=None, mesh_dir="", device="cpu"
    ):
        super().__init__(df, feature_min, feature_max)
        self.mesh_dir = mesh_dir
        self.device = device

        self.mesh_data = {}
        self._preload_meshes()

    def _preload_meshes(self):
        """Load all unique meshes into memory at initialization"""
        unique_mesh_files = self.df.iloc[:, -1].unique()
        print(f"Preloading {len(unique_mesh_files)} unique meshes...")

        for filename in unique_mesh_files:
            mesh_path = os.path.join(self.mesh_dir, filename)
            mesh = trimesh.load(mesh_path, file_type="stl", force="mesh")

            vertices = torch.tensor(mesh.vertices, dtype=torch.get_default_dtype()).to(
                self.device
            )
            edges = torch.tensor(mesh.edges_unique).to(self.device)

            # Calculate edge vectors
            edge_vec = vertices[edges[:, 1]] - vertices[edges[:, 0]]

            self.mesh_data[filename] = {
                "vertices": vertices,
                "edges": edges,
                "edge_vec": edge_vec,
            }

        print("Mesh preloading complete.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get input features and normalize if needed
        in_vars = row.iloc[:5].values.astype(np.float32)
        if self.norm_features:
            in_vars = (in_vars - self.feature_min) / (
                self.feature_max - self.feature_min
            )

        # Get orientation
        orientation = torch.tensor(row.iloc[5:7].values.astype(np.float32), dtype=torch.float32).to(
            self.device
        )

        # Get mesh filename from the last column
        mesh_filename = row.iloc[-1]

        # Get preloaded mesh data
        mesh_data = self.mesh_data[mesh_filename]

        # Create feature tensor to repeat for each vertex
        features = torch.tensor(in_vars, dtype=torch.float32).to(self.device)

        # Create Data object
        data = Data(
            pos=mesh_data["vertices"],
            x=features.repeat(mesh_data["vertices"].size(0), 1),
            edge_index=mesh_data["edges"].T,
            edge_vec=mesh_data["edge_vec"],
            orientation=orientation.unsqueeze(0),
        )

        # Get the label
        y = torch.tensor(row.iloc[7], dtype=torch.float32).to(self.device)

        return data, y

    def _rotate(self, data, orientation):
        vertices = data.pos
        vertices_rot = vertices @ o3.Irrep("1e").D_from_angles(
            orientation[0], orientation[1], torch.tensor(0)
        )
        data_rot = data.clone()
        data_rot.pos = vertices_rot
        return data_rot


class DragMeshDataModule(WVURSMDataModule):
    def __init__(
        self,
        data_dir: str = "data/cube50k_mesh.dat",
        mesh_dir: str = "data/STLs",
        train_val_test_split: tuple[int, int] = (40000, 5000, 4999),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        norm_features: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            data_dir=data_dir,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            norm_features=norm_features,
        )
        # Add mesh-specific parameters
        self.hparams.mesh_dir = mesh_dir
        self.hparams.device = device

    def setup(self, stage: str = None):
        if self.data_train:
            return

        df = pd.read_csv(
            utils.to_absolute_path(self.hparams.data_dir), sep="\s+", header=None
        )
        train_df, val_df, test_df = self._train_test_split(df)

        self.data_train = DragMeshDataset(
            train_df,
            self.feature_min,
            self.feature_max,
            mesh_dir=self.hparams.mesh_dir,
            device=self.hparams.device,
        )
        self.data_val = DragMeshDataset(
            val_df,
            self.feature_min,
            self.feature_max,
            mesh_dir=self.hparams.mesh_dir,
            device=self.hparams.device,
        )
        self.data_test = DragMeshDataset(
            test_df,
            self.feature_min,
            self.feature_max,
            mesh_dir=self.hparams.mesh_dir,
            device=self.hparams.device,
        )

    def train_dataloader(self):
        return GeometricDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return GeometricDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return GeometricDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    # Test the DataModule
    batch_size = 16
    dm = DragMeshDataModule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    print(f"Size of train set: {len(dm.data_train)}")
    print(f"Size of val set: {len(dm.data_val)}")
    print(f"Size of test set: {len(dm.data_test)}")

    # Check a batch from the train loader
    train_batch = next(iter(dm.train_dataloader()))
    x, y = train_batch
    print("\nTrain batch shapes:")
    print(f"pos shape:: {x.pos.shape}") # [batch_size*nodes, 3]
    print(f"x shape: {x.x.shape}")  # Should be [batch_size*nodes, num_features]
    print(f"y shape: {y.shape}")  # Should be [batch_size, 1]