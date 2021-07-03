from abc import ABCMeta
from itertools import groupby
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn.functional import mse_loss

from src.models.fourier import multi_channeled_STFT
from src.models.modules import Conv_TDF
from src.utils.utils import sdr

dim_c = 4  # CaC


class AbstractMDXNet(LightningModule):
    __metaclass__ = ABCMeta

    def __init__(self, target_name, lr, optimizer, dim_f, dim_t, n_fft, hop_length):
        super().__init__()
        self.target_name = target_name
        self.lr = lr
        self.optimizer = optimizer
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
        if self.current_epoch > 0:
            pass

        # Initialization TODO: check resume from checkpoint (epoch>0 checked)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        mixture_wav, target_wav = args[0]

        batch_size = mixture_wav.shape[0]

        mix_spec = self.stft(mixture_wav)[:, :, :self.dim_f]
        spec_hat = self(mix_spec)
        pad = torch.zeros([batch_size, dim_c, self.n_bins - self.dim_f, self.dim_t],
                          #dtype=spec_hat.dtype,
                          device=spec_hat.device)

        target_wav_hat = self.istft(torch.cat([spec_hat, pad], -2))

        loss = mse_loss(target_wav_hat, target_wav)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        track_idx, chunk_idx, mixture_wav = args[0]

        batch_size = mixture_wav.shape[0]

        mix_spec = self.stft(mixture_wav)[:, :, :self.dim_f]
        spec_hat = self(mix_spec)
        pad = torch.zeros([batch_size, dim_c, self.n_bins - self.dim_f, self.dim_t],
                          #dtype=spec_hat.dtype,
                          device=spec_hat.device)

        target_wav_hat = self.istft(torch.cat([spec_hat, pad], -2))

        track_idx = track_idx.cpu().detach().numpy()
        chunk_idx = chunk_idx.cpu().detach().numpy()
        target_wav_hat = target_wav_hat.cpu().detach().numpy()

        return {"track_idx": track_idx, "chunk_idx": chunk_idx, "target_wav_hat":target_wav_hat}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:

        self.val_track_hats = {}
        sdr_dict = {}

        track_array = np.concatenate([output["track_idx"] for output in outputs])
        chunk_array = np.concatenate([output["chunk_idx"] for output in outputs])
        target_array = np.concatenate([output["target_wav_hat"] for output in outputs])

        result_list = zip(track_array,chunk_array,target_array)
        for track_id, chunks in groupby(result_list, lambda x: x[0]):
            chunks = sorted(chunks, key=lambda x: x[1])
            target_hat = np.concatenate(np.stack([chunk[-1] for chunk in chunks])[:,:,self.trim:-self.trim], axis=-1)
            target = self.val_dataloader().dataset.get_reference(track_id)

            if target_hat.shape[-1] < target.shape[-1]:
                sdr_dict[track_id] = float('NaN')
            else:
                sdr_dict[track_id] = sdr(target, target_hat[:, :target.shape[-1]])
                self.val_track_hats[track_id] = target_hat.copy()

        self.log('val/loss', sum(sdr_dict.values())/len(sdr_dict))

        for key in sorted(sdr_dict.keys()):
            self.log('val/loss_{}'.format(key), sdr_dict[key])

    def stft(self, x):
        x = x.reshape([-1, self.sampling_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, spec):
        spec = spec.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        spec = spec.permute([0, 2, 3, 1])
        spec = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        return spec.reshape([-1, 2, self.sampling_size])


class ConvTDFNet(AbstractMDXNet):
    def __init__(self, target_name, lr, optimizer, dim_f, dim_t, n_fft, hop_length,
                 num_blocks, l, g, k, t_scale, bn, bias, mid_tdf):

        super(ConvTDFNet, self).__init__(target_name, lr, optimizer, dim_f, dim_t, n_fft, hop_length)
        self.save_hyperparameters()

        # Important!: Required!
        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.t_scale = t_scale
        self.bn = bn
        self.bias = bias
        self.mid_tdf = mid_tdf

        self.n = num_blocks // 2
        if t_scale is None:
            t_scale = np.arange(self.n)

            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=dim_c, out_channels=g, kernel_size=(1, 1)),
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

            # TODO: no mid_dense default : deprecated ~128
            self.mid_dense = Conv_TDF(c, l, f, k, bn, bias=bias)
            if bn is None and mid_tdf:
                self.mid_dense = Conv_TDF(c, l, f, k, bn=0, bias=False)

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
                nn.Conv2d(in_channels=c, out_channels=dim_c, kernel_size=(1, 1)),
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
