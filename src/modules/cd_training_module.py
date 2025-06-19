# src/modules/cd_training_module.py

from typing import Any, Dict
import hydra
import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError, MaxMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CDTrainingModule(pl.LightningModule):
    """
    A generic LightningModule for REGRESSION tasks.
    """

    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        lr_scheduler_config: Dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.net = model_config
        self.loss_fn = torch.nn.MSELoss()

        # --- Setup Metrics for Regression ---
        # We calculate RMSE. squared=False gives us RMSE directly from MSE.
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        # For tracking the best validation RMSE (lower is better)
        # We use MaxMetric on the negative RMSE.
        self.val_rmse_best = MaxMetric()

    def forward(self, x: Any):
        return self.net(x)

    def _shared_step(self, batch: Any):
        x, y = batch
        # Ensure preds and y are the same shape for loss calculation
        preds = self(x).squeeze()
        y = y.squeeze()
        loss = self.loss_fn(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        self.train_rmse(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        self.val_rmse(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        rmse = self.val_rmse.compute()

        # Track the negative RMSE so that MaxMetric finds the minimum RMSE
        self.val_rmse_best(-rmse)
        self.log("val/rmse_best", -self.val_rmse_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        self.test_rmse(preds, targets)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True)
        self.val_rmse.reset()

    def configure_optimizers(self):
        """
        Manually instantiates the optimizer and scheduler from the config dictionaries.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.optimizer_config.lr,
            weight_decay=self.hparams.optimizer_config.weight_decay,
        )

        if self.hparams.lr_scheduler_config is None:
            return optimizer

        # Step 2: Manually instantiate the scheduler.
        scheduler = ReduceLROnPlateau(
            mode=self.hparams.lr_scheduler_config.mode,
            factor=self.hparams.lr_scheduler_config.factor,
            patience=self.hparams.lr_scheduler_config.patience,
            optimizer=optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
