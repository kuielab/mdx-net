from os import listdir
from pathlib import Path
from typing import Optional, List

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from tqdm import tqdm

from src.callbacks.wandb_callbacks import get_wandb_logger
from src.evaluation.onnx_wrapper import separate, separate_with_ckpt
from src.utils import utils
from src.utils.utils import load, sdr

log = utils.get_logger(__name__)


def evaluation(config: DictConfig):
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    if config.get('split') in ['validation', 'valid']:
        # dummy to crate a validation directory
        log.info(f"Instantiating datamodule <{config.datamodule._target_}> for dummy")
        dummy: LightningDataModule = hydra.utils.instantiate(config.datamodule)
        split_name = 'valid'
    elif config.get('split') == 'test':
        split_name = 'test'
    else:
        raise NotImplementedError

    log.info("check files")

    data_dir = Path(config.get('data_dir')).joinpath(split_name)
    onnx_dir = Path(config.get('onnx_dir'))
    batch_size = config.get('batch_size')
    assert data_dir.exists() and onnx_dir.exists()

    # Init Lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

        if any([isinstance(l, WandbLogger) for l in loggers]):
            utils.wandb_login(key=config.wandb_api_key)

    models = {'vocals': hydra.utils.instantiate(config.eval_config.vocals)}
    onnxs = {'vocals': onnx_dir.joinpath('vocals.onnx')}
    ckpts = {'vocals': onnx_dir.joinpath('vocals.ckpt')}

    sources = config.eval_config['sources']  # TODO:, 'drums', 'bass', 'other']
    assert len(set(sources) - {'vocals', 'drums', 'bass', 'other'}) == 0

    score_per_source = {source: [] for source in sources}
    num_tracks = len(listdir(data_dir))
    for i, track in tqdm(enumerate(sorted(listdir(data_dir)))):
        track = data_dir.joinpath(track)
        mixture = load(track.joinpath('mixture.wav'))
        # target_hat = {source: separate_with_ckpt(batch_size, models[source], onnxs[source], mixture) for source in sources}
        target_hat = {source: separate_with_ckpt(batch_size, models[source], ckpts[source], mixture) for source in sources}

        scores = {source: sdr(load(track.joinpath(source + '.wav')), target_hat[source]) for source in sources}

        for source in sources:
            score_per_source[source].append(scores[source])

        for logger in loggers:
            logger.log_metrics(scores, i)

        for wandb_logger in [logger for logger in loggers if isinstance(logger, WandbLogger)]:
            for source in scores.keys():
                mid = mixture.shape[-1] // 2
                track = target_hat[source][:, mid - 44100 * 3:mid + 44100 * 3]
                wandb_logger.experiment.log(
                    {'track={}_target={}'.format(i, source): [wandb.Audio(track.T, sample_rate=44100)]})

    for logger in loggers:
        for source in sources:
            logger.log_metrics({'mean_' + source: sum(score_per_source[source])/num_tracks})
        logger.close()

    if any([isinstance(l, WandbLogger) for l in loggers]):
        wandb.finish()
