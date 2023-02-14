from os import listdir
from pathlib import Path
from typing import Optional, List

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from tqdm import tqdm
import numpy as np
from src.callbacks.wandb_callbacks import get_wandb_logger
from src.evaluation.separate import separate_with_onnx, separate_with_ckpt
from src.utils import utils
from src.utils.utils import load_wav, sdr

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def evaluation(config: DictConfig):

    assert config.split in ['train', 'valid', 'test']

    data_dir = Path(config.get('data_dir')).joinpath(config['split'])
    assert data_dir.exists()

    # Init Lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

        if any([isinstance(l, WandbLogger) for l in loggers]):
            utils.wandb_login(key=config.wandb_api_key)

    model = hydra.utils.instantiate(config.model)
    target_name = model.target_name
    ckpt_path = Path(config.ckpt_dir).joinpath(config.ckpt_path)

    scores = []
    num_tracks = len(listdir(data_dir))
    for i, track in tqdm(enumerate(sorted(listdir(data_dir)))):
        track = data_dir.joinpath(track)
        mixture = load_wav(track.joinpath('mixture.wav'))
        target = load_wav(track.joinpath(target_name + '.wav'))
        #target_hat = {source: separate(config['batch_size'], models[source], onnxs[source], mixture) for source in sources}
        target_hat = separate_with_ckpt(config.batch_size, model, ckpt_path, mixture, config.device)
        score = sdr(target_hat, target)
        scores.append(score)

        for logger in loggers:
            logger.log_metrics({'sdr': score}, i)

        for wandb_logger in [logger for logger in loggers if isinstance(logger, WandbLogger)]:
            mid = mixture.shape[-1] // 2
            track = target_hat[:, mid - 44100 * 3:mid + 44100 * 3]
            wandb_logger.experiment.log(
                {f'track={i}_target={target_name}': [wandb.Audio(track.T, sample_rate=44100)]})

    for logger in loggers:
        logger.log_metrics({'mean_sdr_' + target_name: sum(scores)/num_tracks})
        logger.close()

    if any([isinstance(logger, WandbLogger) for logger in loggers]):
        wandb.finish()
