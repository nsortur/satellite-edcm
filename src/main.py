import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pyrootutils
from typing import List
from hydra.core.hydra_config import HydraConfig
import os

# Set up paths so that imports work correctly from anywhere
# This allows 'from src.modules...' to work in notebooks and scripts
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------
# A couple of optional utilities for a more robust setup
# ------------------------------------------------------------------------------------


def print_config(cfg: DictConfig) -> None:
    """Prints the configuration to the console."""
    print("----------------- CONFIGURATION -----------------")
    print(OmegaConf.to_yaml(cfg))
    print("-------------------------------------------------")


def get_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Instantiates callbacks from the configuration."""
    callbacks: List[pl.Callback] = []
    print(cfg.get("callbacks"))
    if not cfg.get("callbacks"):
        return callbacks

    callbacks_cfg = cfg.callbacks

    if isinstance(callbacks_cfg, DictConfig) and "_target_" in callbacks_cfg:
        callbacks.append(hydra.utils.instantiate(callbacks_cfg))

    else:
        for _, cb_conf in cfg.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


# ------------------------------------------------------------------------------------
# The Main Training Function
# ------------------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> float:
    """
    The main training function, orchestrated by Hydra.

    Args:
        cfg: The configuration object composed by Hydra.

    Returns:
        The best validation metric score, useful for hyperparameter optimization.
    """
    # 1. Print and validate the configuration
    print_config(cfg)

    # 2. Set the seed for reproducibility
    pl.seed_everything(cfg.seed)

    # 3. Instantiate the DataModule from the config
    print(f"--> Instantiating DataModule <{cfg.dataset._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    # 4. Instantiate the LightningModule (your CDTrainingModule) from the config
    print(f"--> Instantiating Model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # 5. Instantiate callbacks and loggers
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    callbacks = get_callbacks(cfg)

    # logger: pl.loggers.WandbLogger = hydra.utils.instantiate(
    #     cfg.logger,
    #     save_dir=hydra_output_dir
    # )

    # 6. Instantiate the Trainer
    print(f"--> Instantiating Trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        default_root_dir=hydra_output_dir
        # logger=logger
    )

    # 7. Start the training process
    print("--> Starting Training!")
    trainer.fit(model=model, datamodule=datamodule)

    # 8. (Optional) Test the model after training
    # The 'ckpt_path="best"' command tells Lightning to automatically load the
    # best checkpoint saved by the ModelCheckpoint callback.
    if cfg.get("test_after_fit"):
        print("--> Starting Testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # 9. Log the best score and finish
    best_score = trainer.checkpoint_callback.best_model_score
    if best_score:
        print(f"Best validation score: {best_score:.4f}")

    # Return the best score for Hydra's hyperparameter sweepers
    return best_score if best_score else 0.0


if __name__ == "__main__":
    main()
