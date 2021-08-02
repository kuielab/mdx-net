from os import listdir
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort


def separate_with_onnx(batch_size, model, onnx_path: Path, mix):
    n_sample = mix.shape[1]

    trim = model.n_fft // 2
    gen_size = model.sampling_size - 2 * trim
    pad = gen_size - n_sample % gen_size
    mix_p = np.concatenate((np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))), 1)

    mix_waves = []
    i = 0
    while i < n_sample + pad:
        waves = np.array(mix_p[:, i:i + model.sampling_size], dtype=np.float32)
        mix_waves.append(waves)
        i += gen_size
    mix_waves_batched = torch.tensor(mix_waves, dtype=torch.float32).split(batch_size)

    tar_signals = []

    with torch.no_grad():
        _ort = ort.InferenceSession(str(onnx_path))
        for mix_waves in mix_waves_batched:
            tar_waves = model.istft(torch.tensor(
                _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
            ))
            tar_signals.append(tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy())
        tar_signal = np.concatenate(tar_signals, axis=-1)[:, :-pad]

    return tar_signal


def separate_with_ckpt(batch_size, model, ckpt_path: Path, mix, device):
    model = model.load_from_checkpoint(ckpt_path).to(device)
    true_samples = model.sampling_size - 2 * model.trim

    right_pad = true_samples + model.trim - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((2, model.trim), dtype='float32'),
                              mix,
                              np.zeros((2, right_pad), dtype='float32')),
                             1)
    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = [mixture[:, i * true_samples: i * true_samples + model.sampling_size] for i in
                         range(num_chunks)]
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

    target_wav_hats = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_spec = model.stft(mixture_wav.to(device))
            spec_hat = model(mix_spec)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat)

        target_wav_hat = np.vstack(target_wav_hats)[:, :, model.trim:-model.trim]
        target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :mix.shape[-1]]
    return target_wav_hat
