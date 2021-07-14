from abc import ABCMeta
from typing import Optional, Union, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.functional import mse_loss

from src.models.modules import Conv_TDF
from src.utils.utils import sdr


class AbstractMDXNet(LightningModule):
    __metaclass__ = ABCMeta

    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length):
        super().__init__()
        self.target_name = target_name
        self.lr = lr
        self.optimizer = optimizer
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.trim = n_fft // 2
        self.hop_length = hop_length

        self.n_bins = self.n_fft // 2 + 1
        self.sampling_size = hop_length * (self.dim_t - 1)
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, dim_c, self.n_bins - self.dim_f, self.dim_t]), requires_grad=False)
        self.input_sample_shape = (self.stft(torch.zeros([1, 2, self.sampling_size]))).shape

    def configure_optimizers(self):
        if self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), self.lr)

    def on_train_start(self) -> None:
        pass

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        mixture_wav, target_wav = args[0]
        mix_spec = self.stft(mixture_wav)
        tar_spec = self.stft(target_wav)
        tar_spec_hat = self(mix_spec)
        loss = mse_loss(tar_spec_hat, tar_spec)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        num_tracks, batch_size, index, mixture_wav_batched, target_wav = args[0]
        num_tracks, batch_size, index = num_tracks.item(), batch_size.item(), index.item()
        if num_tracks < 0:
            self.log("val/sdr", 0, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                     reduce_fx=torch.sum,
                     sync_dist=True,
                     sync_dist_op="sum")
            return None
        else:
            target_wav_hats = []

            for mixture_wav in mixture_wav_batched[0].split(batch_size):
                mix_spec = self.stft(mixture_wav)
                spec_hat = self(mix_spec)
                target_wav_hat = self.istft(spec_hat)
                target_wav_hat = target_wav_hat.cpu().detach().numpy()
                target_wav_hats.append(target_wav_hat)

            target_wav_hat = np.vstack(target_wav_hats)[:, :, self.trim:-self.trim]
            target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :target_wav.shape[-1]]
            loss = sdr(target_wav[0].cpu().detach().numpy(), target_wav_hat) / num_tracks

            self.log("val/sdr", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                     reduce_fx=torch.sum,
                     sync_dist=True,
                     sync_dist_op="sum")

            return {'track_id': index, 'track': target_wav_hat}

    def stft(self, x):
        x = x.reshape([-1, self.sampling_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, spec):
        spec = torch.cat([spec, self.freq_pad.repeat([spec.shape[0], 1, 1, 1])], -2)
        spec = spec.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        spec = spec.permute([0, 2, 3, 1])
        spec = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        return spec.reshape([-1, 2, self.sampling_size])


class ConvTDFNet(AbstractMDXNet):
    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length,
                 num_blocks, l, g, k, bn, bias):

        super(ConvTDFNet, self).__init__(target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length)
        self.save_hyperparameters()

        # Important!: Required!
        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        self.n = num_blocks // 2
        t_scale = np.arange(self.n)

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c, out_channels=g, kernel_size=(1, 1)),
            nn.BatchNorm2d(g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()
        for i in range(self.n):
            self.encoding_blocks.append(Conv_TDF(c, l, f, k, bn, bias=bias))
            scale = (2, 2) if i in t_scale else (1, 2)
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.mid_dense = Conv_TDF(c, l, f, k, bn, bias=bias)

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            scale = (2, 2) if i in self.n - 1 - t_scale else (1, 2)
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c - g),
                    nn.ReLU()
                )
            )
            f = f * 2
            c -= g

            self.decoding_blocks.append(Conv_TDF(c, l, f, k, bn, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c, kernel_size=(1, 1)),
        )

    def forward(self, x):

        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.mid_dense(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x
