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

    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length, overlap):
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
        self.inference_chunk_size = hop_length * (self.dim_t*2 - 1)
        self.overlap = overlap
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, dim_c, self.n_bins - self.dim_f, 1]), requires_grad=False)
        self.inference_chunk_shape = (self.stft(torch.zeros([1, 2, self.inference_chunk_size]))).shape

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

    # Validation SDR is calculated on whole tracks and not chunks since
    # short inputs have high possibility of being silent (all-zero signal)
    # which leads to very low sdr values regardless of the model.
    # A natural procedure would be to split a track into chunk batches and
    # load them on multiple gpus, but aggregation was too difficult.
    # So instead we load one whole track on a single device (data_loader batch_size should always be 1)
    # and do all the batch splitting and aggregation on a single device.
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
        target_hat = target_hat_chunks.transpose(0, 1).reshape(2, -1)[..., :target.shape[-1]]

        score = sdr(target_hat.detach().cpu().numpy(), target.cpu().numpy())
        self.log("val/sdr", score, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        return {'loss': score}

    def stft(self, x):
        dim_b = x.shape[0]
        x = x.reshape([dim_b * 2, -1])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([dim_b, 2, 2, self.n_bins, -1]).reshape([dim_b, self.dim_c, self.n_bins, -1])
        return x[:, :, :self.dim_f]

    def istft(self, x):
        dim_b = x.shape[0]
        x = torch.cat([x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])], -2)
        x = x.reshape([dim_b, 2, 2, self.n_bins, -1]).reshape([dim_b * 2, 2, self.n_bins, -1])
        x = x.permute([0, 2, 3, 1])
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        return x.reshape([dim_b, 2, -1])


class ConvTDFNet(AbstractMDXNet):
    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length,
                 num_blocks, l, g, k, bn, bias, overlap):

        super(ConvTDFNet, self).__init__(
            target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length, overlap)
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
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x


class Mixer(LightningModule):
    def __init__(self, separator_configs, separator_ckpts, lr, optimizer, dim_t, hop_length, overlap, target_name='all'):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained separators per source
        self.separators = nn.ModuleDict()
        for ckpt in separator_ckpts.values():
            # if failed here, then fill valid ckpt pahts in the given yaml for Mixer training
            assert ckpt is not None

        for source in separator_configs.keys():
            model_config = OmegaConf.load(separator_configs[source])
            assert 'ConvTDFNet' in model_config._target_
            separator = ConvTDFNet(**{key: model_config[key] for key in dict(model_config) if key !='_target_'})
            separator.load_from_checkpoint(separator_ckpts[source])
            self.separators[source] = separator

        # Freeze
        with torch.no_grad():
            for param in self.separators.parameters():
                param.requires_grad = False

        self.lr = lr
        self.optimizer = optimizer

        self.chunk_size = hop_length * (dim_t - 1)
        self.overlap = overlap
        self.dim_s = len(separator_configs)
        self.mixing_layer = nn.Linear((self.dim_s+1) * 2, self.dim_s * 2, bias=False)

    def configure_optimizers(self):
        if self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), self.lr)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        mix_wave, target_waves = args[0]

        with torch.no_grad():
            target_wave_hats = []
            for source in ['bass', 'drums', 'other', 'vocals']:
                S = self.separators[source]
                target_wave_hat = S.istft(S(S.stft(mix_wave)))
                target_wave_hats.append(target_wave_hat)  # shape = [source, batch, channel, time]

            target_wave_hats = torch.stack(target_wave_hats).transpose(0, 1)

        mixer_output = self(torch.cat([target_wave_hats, mix_wave.unsqueeze(1)], 1))

        loss = mse_loss(mixer_output[..., self.overlap:-self.overlap],
                        target_waves[..., self.overlap:-self.overlap])
        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    # data_loader batch_size should always be 1
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        mix_chunk_batches, target_waves = args[0]

        # remove data_loader batch dimension
        mix_chunk_batches, target_waves = [batch[0] for batch in mix_chunk_batches], target_waves[0]

        # process whole track in batches of chunks
        target_hat_chunks = []
        for mix_wave in mix_chunk_batches:
            target_wave_hats = []
            for source in ['bass', 'drums', 'other', 'vocals']:
                S = self.separators[source]
                target_wave_hat = S.istft(S(S.stft(mix_wave)))
                target_wave_hats.append(target_wave_hat)  # shape = [source, batch, channel, time]
            target_wave_hats = torch.stack(target_wave_hats).transpose(0, 1)
            mixer_output = self(torch.cat([target_wave_hats, mix_wave.unsqueeze(1)], 1))
            target_hat_chunks.append(mixer_output[..., self.overlap:-self.overlap])

        target_hat_chunks = torch.cat(target_hat_chunks)

        # concat all output chunks
        target_hat = target_hat_chunks.permute(1,2,0,3).reshape(self.dim_s, 2, -1)[..., :target_waves.shape[-1]]

        score = sdr(target_hat.detach().cpu().numpy(), target_waves.cpu().numpy())
        self.log("val/sdr", score, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        return {'loss': score}

    def forward(self, x):
        x = x.reshape(-1, (self.dim_s + 1) * 2, self.chunk_size).transpose(-1, -2)
        x = self.mixing_layer(x)
        return x.transpose(-1, -2).reshape(-1, self.dim_s, 2, self.chunk_size)
