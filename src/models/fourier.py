import math

import torch
import pytorch_lightning as pl
from torch import Tensor


def get_trim_length(hop_length, min_trim=5000):
    trim_per_hop = math.ceil(min_trim / hop_length)
    trim_length = trim_per_hop * hop_length
    assert trim_per_hop > 1
    return trim_length


def complex_norm(spec_complex, power=1.0):
    return spec_complex.pow(2.).sum(-1).pow(0.5 * power)


def complex_angle(spec_complex):
    return torch.atan2(spec_complex[..., 1], spec_complex[..., 0])


def mag_phase_to_complex(mag, phase, power=1.0):
    """
    input_signal: mag(*, N, T) , phase(*, N, T), power is optional
    output: *, N, T, 2
    """
    mag_power_1 = mag.pow(1 / power)
    spec_real = mag_power_1 * torch.cos(phase)
    spec_imag = mag_power_1 * torch.sin(phase)
    spec_complex = torch.stack([spec_real, spec_imag], dim=-1)
    return spec_complex


class STFT(pl.LightningModule):

    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.nn.Parameter(torch.hann_window(n_fft))
        self.freeze()

    def forward(self, input_signal):
        return self.to_spec_complex(input_signal)

    def to_spec_complex(self, input_signal: torch.Tensor):
        """
        input_signal: *, signal
        output: *, N, T, 2
        """
        # if input_signal.dtype != self.window.dtype or input_signal.device != self.window.device :
        #     self.window = torch.as_tensor(self.window, dtype=input_signal.dtype, device=input_signal.device)
        # else:
        #     window = self.window

        return torch.stft(input_signal, self.n_fft, self.hop_length, window=self.window)

    def to_mag(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal), power is optional
        output: *, N, T
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_norm(spec_complex, power)

    def to_phase(self, input_signal):
        """
        input_signal: *, signal
        output: *, N, T
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_angle(spec_complex)

    def to_mag_phase(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal), power is optional
        output: tuple (mag(*, N, T) , phase(*, N, T))
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_norm(spec_complex, power), complex_angle(spec_complex)

    def restore_complex(self, spec_complex):
        """
        input_signal:  *, N, T, 2
        output: *, signal
        """
        if spec_complex.dtype != self.window.dtype:
            window = torch.as_tensor(self.window, dtype=spec_complex.dtype)
        else:
            window = self.window

        if spec_complex.device != self.window.device:
            window = window.to(spec_complex.device)
        else:
            window = self.window

        return torch.istft(spec_complex, self.n_fft, self.hop_length, window=window)

    def restore_mag_phase(self, mag, phase, power=1.):
        """
        input_signal: mag(*, N, T), phase(*, N, T), power is optional
        output: *, signal
        """
        spec_complex = mag_phase_to_complex(mag, phase, power)
        return self.restore_complex(spec_complex)


class multi_channeled_STFT(pl.LightningModule):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = STFT(n_fft, hop_length)

    def forward(self, input_signal):
        return self.to_spec_complex(input_signal)

    def to_spec_complex(self, input_signal) -> Tensor:
        """
        input_signal: *, signal, ch
        output: *, N, T, 2, ch
        """
        num_channels = input_signal.shape[-1]
        spec_complex_ch = [self.stft.to_spec_complex(input_signal[..., ch_idx])
                           for ch_idx in range(num_channels)]
        return torch.stack(spec_complex_ch, dim=-1)

    def to_mag(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal, ch), power is optional
        output: *, N, T, ch
        """
        num_channels = input_signal.shape[-1]
        mag_ch = [self.stft.to_mag(input_signal[..., ch_idx], power)
                  for ch_idx in range(num_channels)]
        return torch.stack(mag_ch, dim=-1)

    def to_phase(self, input_signal):
        """
        input_signal: *, signal, ch
        output: *, N, T, ch
        """
        num_channels = input_signal.shape[-1]
        phase_ch = [self.stft.to_phase(input_signal[..., ch_idx])
                    for ch_idx in range(num_channels)]
        return torch.stack(phase_ch, dim=-1)

    def to_mag_phase(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal, ch), power is optional
        output: tuple (mag(*, N, T, ch) , phase(*, N, T, ch))
        """
        num_channels = input_signal.shape[-1]
        mag_ch = [self.stft.to_mag(input_signal[..., ch_idx], power)
                  for ch_idx in range(num_channels)]
        phase_ch = [self.stft.to_phase(input_signal[..., ch_idx])
                    for ch_idx in range(num_channels)]
        return torch.stack(mag_ch, dim=-1), torch.stack(phase_ch, dim=-1)

    def restore_complex(self, spec_complex):
        """
        input_signal:  *, N, T, 2, ch
        output: *, signal, ch
        """
        num_channels = spec_complex.shape[-1]
        signal_ch = [self.stft.restore_complex(spec_complex[..., ch_idx])
                     for ch_idx in range(num_channels)]
        return torch.stack(signal_ch, dim=-1)

    def restore_mag_phase(self, mag, phase, power=1.):
        """
        input_signal: mag(*, N, T, ch), phase(*, N, T, ch), power is optional
        output: *, signal
        """
        num_channels = mag.shape[-1]
        signal_ch = [self.stft.restore_mag_phase(mag[..., ch_idx], phase[..., ch_idx], power)
                     for ch_idx in range(num_channels)]
        return torch.stack(signal_ch, dim=-1)