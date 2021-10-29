import os.path
from typing import Dict, Any

import torch
from pytorch_lightning import Callback
import pytorch_lightning as pl
import inspect
from src.models.mdxnet import AbstractMDXNet


class MakeONNXCallback(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, dirpath: str):
        self.dirpath = dirpath
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)

    def on_save_checkpoint(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule',
                           checkpoint: Dict[str, Any]) -> dict:
        res = super().on_save_checkpoint(trainer, pl_module, checkpoint)

        var = inspect.signature(pl_module.__init__).parameters
        model = pl_module.__class__(**dict((name, pl_module.__dict__[name]) for name in var))
        model.load_state_dict(pl_module.state_dict())

        target_dir = '{}epoch_{}'.format(self.dirpath, pl_module.current_epoch)

        try:
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            with torch.no_grad():
                torch.onnx.export(model,
                                  torch.zeros(model.inference_chunk_shape),
                                  '{}/{}.onnx'.format(target_dir, model.target_name),
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=13,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  input_names=['input'],  # the model's input names
                                  output_names=['output'],  # the model's output names
                                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                                'output': {0: 'batch_size'}})
        except:
            print('onnx error')
        finally:
            del model

        return res
