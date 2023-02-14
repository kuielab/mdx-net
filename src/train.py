from typing import List, Optional

import hydra
import pytorch_lightning as pl
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import WandbLogger

from src import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    try:
        if "seed" in cfg:
            # set seed for random number generators in pytorch, numpy and python.random
            if cfg.get("seed"):
                pl.seed_everything(cfg.seed, workers=True)

        else:
            raise ModuleNotFoundError

    except ModuleNotFoundError:
        print('[Error] seed should be fixed for reproducibility \n=> e.g. python run.py +seed=$SEED')
        exit(-1)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init Lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

        for wandb_logger in [l for l in logger if isinstance(l, WandbLogger)]:
            utils.wandb_login(key=cfg.wandb_api_key)
            # utils.wandb_watch_all(wandb_logger, model) # TODO buggy
            break

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        dict(
            cfg=cfg,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if not cfg.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
