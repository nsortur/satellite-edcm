from typing import Any

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms as T
from hydra import utils
import pandas as pd
import torch
import math
import numpy as np
import os


class DragDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_min=None, feature_max=None):
        self.df = df
        self.norm_features = (feature_min is not None) and (feature_max is not None)
        if self.norm_features:
            self.feature_min = feature_min
            self.feature_max = feature_max

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        in_vars = row.iloc[:5].values.astype(np.float32)
        if self.norm_features:
            # min-max normalization of non orientation features
            in_vars = (in_vars - self.feature_min) / (
                self.feature_max - self.feature_min
            )

        yaw, pitch = row.iloc[5:7].values
        dir_x = math.cos(yaw) * math.cos(pitch)
        dir_y = math.sin(yaw) * math.cos(pitch)
        dir_z = math.sin(pitch)
        dir_vec = np.array([dir_y, dir_x, dir_z], dtype=np.float32)
        x_features = np.concatenate((dir_vec, in_vars))
        y_label = np.array(row.iloc[7], dtype=np.float32)
        return torch.from_numpy(x_features).float(), torch.from_numpy(y_label)


class WVURSMDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/cube50k.dat",
        train_val_test_split: tuple[int, int] = (40000, 5000, 4999),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        norm_features: bool = True,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.feature_min = None
        self.feature_max = None

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def prepare_data(self):
        # assume already downloaded from simulation
        assert os.path.exists(self.hparams.data_dir), "Data directory does not exist!"

    def setup(self, stage: str = None):
        if self.data_train:
            return

        df = pd.read_csv(
            utils.to_absolute_path(self.hparams.data_dir), sep="\s+", header=None
        )
        train_df, val_df, test_df = self._train_test_split(df)

        self.data_train = DragDataset(train_df, self.feature_min, self.feature_max)
        self.data_val = DragDataset(val_df, self.feature_min, self.feature_max)
        self.data_test = DragDataset(test_df, self.feature_min, self.feature_max)

    def _train_test_split(self, df: pd.DataFrame):
        df = df.iloc[:-1].reset_index(drop=True)

        # Split the DataFrame indices first
        full_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_size, val_size, _ = self.hparams.train_val_test_split
        train_df = full_df[:train_size]
        val_df = full_df[train_size : train_size + val_size]
        test_df = full_df[train_size + val_size :]

        # calculate stats only on training set
        if self.hparams.norm_features:
            # Check if we are loading from a checkpoint
            if self.feature_min is None or self.feature_max is None:
                print("Calculating normalization stats from the training set.")
                train_in_vars_df = train_df.iloc[:, :5]
                self.feature_min = train_in_vars_df.min(axis=0).values
                self.feature_max = train_in_vars_df.max(axis=0).values

        return train_df, val_df, test_df

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def state_dict(self):
        # save train set stats
        if self.hparams.norm_features:
            return {"feature_min": self.feature_min, "feature_max": self.feature_max}
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        # load train set stats if loading from checkpoint
        if self.hparams.norm_features:
            self.feature_min = state_dict["feature_min"]
            self.feature_max = state_dict["feature_max"]


if __name__ == "__main__":
    # Test the DataModule
    dm = WVURSMDataModule()
    dm.prepare_data()
    dm.setup()

    print(f"Size of train set: {len(dm.data_train)}")
    print(f"Size of val set: {len(dm.data_val)}")
    print(f"Size of test set: {len(dm.data_test)}")

    # Check a batch from the train loader
    train_batch = next(iter(dm.train_dataloader()))
    x, y = train_batch
    print("\nTrain batch shapes:")
    print(f"x shape: {x.shape}")  # Should be [batch_size, num_features]
    print(f"y shape: {y.shape}")  # Should be [batch_size, 1]
