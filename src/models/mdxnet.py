from abc import ABCMeta
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.functional import mse_loss

from src.models.modules import TFC_TDF
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
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length

        self.chunk_size = hop_length * (self.dim_t - 1)
        self.overlap = n_fft // 2
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, dim_c, self.n_bins - self.dim_f, self.dim_t]), requires_grad=False)
        self.input_sample_shape = (self.stft(torch.zeros([1, 2, self.chunk_size]))).shape

    def configure_optimizers(self):
        if self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), self.lr)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        mix_wave, target_wave = args[0]
        mix_spec = self.stft(mix_wave)

        target_wave_hat = self.istft(self(mix_spec))
        loss = mse_loss(target_wave_hat, target_wave)
        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        mix_chunk_batches, target = args[0]

        # remove data_loader batch dimension
        mix_chunk_batches, target = [batch[0] for batch in mix_chunk_batches], target[0]

        # process whole track in batches of chunks
        target_hat_chunks = []
        for batch in mix_chunk_batches:
            mix_spec = self.stft(batch)
            target_hat_chunks.append(self.istft(self(mix_spec))[..., self.overlap:-self.overlap])
        target_hat_chunks = torch.cat(target_hat_chunks)

        # concat all output chunks
        target_hat = target_hat_chunks.transpose(0, 1).reshape(2, -1)[:, :target.shape[-1]]

        score = sdr(target_hat.detach().cpu().numpy(), target.cpu().numpy())
        self.log("val/sdr", score, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        return {'loss': score}

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, spec):
        spec = torch.cat([spec, self.freq_pad.repeat([spec.shape[0], 1, 1, 1])], -2)
        spec = spec.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        spec = spec.permute([0, 2, 3, 1])
        spec = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        return spec.reshape([-1, 2, self.chunk_size])


class ConvTDFNet(AbstractMDXNet):
    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length,
                 num_blocks, l, g, k, bn, bias):

        super(ConvTDFNet, self).__init__(target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length)
        self.save_hyperparameters()

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        self.n = num_blocks // 2
        scale = (2, 2)

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
            self.encoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block = TFC_TDF(c, l, f, k, bn, bias=bias)

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c - g),
                    nn.ReLU()
                )
            )
            f = f * 2
            c -= g

            self.decoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))

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

        x = self.bottleneck_block(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x


class Mixer(LightningModule):
    def __init__(self, model_cfg_dir, separator_configs, lr, optimizer, dim_t, hop_length, target_name='all'):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained separators per source
        self.separators = []
        for cfg in separator_configs:
            model_config = OmegaConf.load(model_cfg_dir + cfg)
            assert 'ConvTDFNet' in model_config._target_
            separator = ConvTDFNet(**{key: model_config[key] for key in dict(model_config) if key !='_target_'})
            self.separators.append(separator)

        # Freeze
        for sep in self.separators:
            with torch.no_grad():
                for param in sep.parameters():
                    param.requires_grad = False

        self.lr = lr
        self.optimizer = optimizer

        self.chunk_size = hop_length * (dim_t - 1)
        self.dim_s = len(separator_configs)
        self.mixing_layer = nn.Linear((self.dim_s+1) * 2, self.dim_s * 2, bias=False)

    def configure_optimizers(self):
        if self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), self.lr)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        mix_wave, target_waves = args[0]

        with torch.no_grad():
            target_wave_hats = []
            for S in self.separators:
                target_wave_hat = S.istft(S(S.stft(mix_wave)))
                target_wave_hats.append(target_wave_hat)  # shape = [source, batch, channel, time]

            target_wave_hats = torch.stack(target_wave_hats).transpose(0, 1)

        mixer_output = self(torch.cat([target_wave_hats, mix_wave.unsqueeze(1)], 1))

        loss = mse_loss(mixer_output, target_waves)
        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    # data_loader batch_size should always be 1
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        mix_chunk_batches, target = args[0]

        # remove data_loader batch dimension
        mix_chunk_batches, target = [batch[0] for batch in mix_chunk_batches], target[0]

        # process whole track in batches of chunks
        target_hat_chunks = []
        for batch in mix_chunk_batches:
            mix_spec = self.stft(batch)
            target_hat_chunks.append(self.istft(self(mix_spec))[..., self.overlap:-self.overlap])
        target_hat_chunks = torch.cat(target_hat_chunks)

        # concat all output chunks
        target_hat = target_hat_chunks.transpose(0, 1).reshape(2, -1)[:, :target.shape[-1]]

        score = sdr(target_hat.detach().cpu().numpy(), target.cpu().numpy())
        self.log("val/sdr", score, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        return {'loss': score}

    def forward(self, x):
        x = x.reshape(-1, (self.dim_s + 1) * 2, self.chunk_size).transpose(-1, -2)
        x = self.mixing_layer(x)
        return x.transpose(-1, -2).reshape(-1, self.dim_s, 2, self.chunk_size)
