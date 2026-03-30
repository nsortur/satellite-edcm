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
        if len(batch) == 3:
            x, geo_x, y = batch
            preds = self((x, geo_x)).squeeze()
        else:
            targets = batch.y.squeeze()
            preds = self(batch).squeeze()
            # x, y = batch
            # Ensure preds and y are the same shape for loss calculation
            # preds = self(x).squeeze()
            
        y = targets
        loss = self.loss_fn(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        b_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch[0].shape[0]

        self.train_rmse(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=b_size)
        self.log(
            "train/rmse", self.train_rmse, on_step=True, on_epoch=True, prog_bar=False, batch_size=b_size
        )

        # --- Diagnostic Logging (every N steps to reduce overhead) ---
        _log_interval = 50
        if self.global_step % _log_interval == 0:
            with torch.no_grad():
                # Prediction & Target Statistics
                self.log("debug/pred_mean", preds.mean(), on_step=True, on_epoch=False)
                self.log("debug/pred_std", preds.std(), on_step=True, on_epoch=False)
                self.log("debug/pred_min", preds.min(), on_step=True, on_epoch=False)
                self.log("debug/pred_max", preds.max(), on_step=True, on_epoch=False)
                self.log("debug/target_mean", targets.mean(), on_step=True, on_epoch=False)
                self.log("debug/target_std", targets.std(), on_step=True, on_epoch=False)
                self.log("debug/target_min", targets.min(), on_step=True, on_epoch=False)
                self.log("debug/target_max", targets.max(), on_step=True, on_epoch=False)

                # Residual (pred - target) stats
                residuals = preds - targets
                self.log("debug/residual_mean", residuals.mean(), on_step=True, on_epoch=False)
                self.log("debug/residual_std", residuals.std(), on_step=True, on_epoch=False)
                self.log("debug/residual_absmax", residuals.abs().max(), on_step=True, on_epoch=False)

                # Graph structure stats (if PyG batch)
                if hasattr(batch, 'edge_index') and batch.edge_index is not None:
                    num_edges = batch.edge_index.shape[1]
                    num_nodes = batch.pos.shape[0] if hasattr(batch, 'pos') else batch.x.shape[0]
                    self.log("debug/edges_per_node", float(num_edges) / max(num_nodes, 1), on_step=True, on_epoch=False)
                    self.log("debug/num_nodes", float(num_nodes), on_step=True, on_epoch=False)
                    self.log("debug/num_edges", float(num_edges), on_step=True, on_epoch=False)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before the optimizer step — critical for diagnosing instability."""
        _log_interval = 50
        if self.global_step % _log_interval != 0:
            return

        # Total gradient norm across all parameters
        total_norm = 0.0
        layer_norms = {}
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                # Group by top-level module (e.g. 'embedding_layer', 'encoder.0', 'decoder')
                group = name.split('.')[0]
                if len(name.split('.')) > 1:
                    group = '.'.join(name.split('.')[:2])
                layer_norms[group] = layer_norms.get(group, 0.0) + param_norm ** 2

        total_norm = total_norm ** 0.5
        self.log("debug/grad_norm_total", total_norm, on_step=True, on_epoch=False)

        # Per-layer-group gradient norms
        for group, norm_sq in layer_norms.items():
            self.log(f"debug_grads/{group}", norm_sq ** 0.5, on_step=True, on_epoch=False)

        # Log if any grads are NaN or Inf
        has_nan = any(torch.isnan(p.grad).any().item() for p in self.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any().item() for p in self.parameters() if p.grad is not None)
        self.log("debug/grad_has_nan", float(has_nan), on_step=True, on_epoch=False)
        self.log("debug/grad_has_inf", float(has_inf), on_step=True, on_epoch=False)

    def on_train_epoch_start(self):
        """Log weight norms at the start of each epoch."""
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                group = name.split('.')[0]
                if len(name.split('.')) > 1:
                    group = '.'.join(name.split('.')[:2])
                self.log(f"debug_weights/{group}_norm", param.data.norm(2).item(), on_step=False, on_epoch=True)

        # Log current learning rate
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]['lr']
            self.log("debug/learning_rate", current_lr, on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        b_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch[0].shape[0]

        self.val_rmse(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=False, batch_size=b_size)

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
