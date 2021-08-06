import io
import os
import subprocess as sp
import tempfile
import warnings

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=Warning)

data_root = '../../data/'
musdb_train_path = data_root + 'musdbHQ/train/'
musdb_test_path = data_root + 'musdbHQ/test/'
musdb_valid_path = data_root + 'musdbHQ/valid/'
slakh_path = data_root + 'slakh/2100/'
vocalset_path = data_root + 'vocalset/'

mix_name = 'mixture'
source_names = ['vocals', 'drums', 'bass', 'other']

sample_rate = 44100


def shift(wav, pitch, tempo, voice=False, quick=False, samplerate=44100):
    def i16_pcm(wav):
        if wav.dtype == np.int16:
            return wav
        return (wav * 2 ** 15).clamp_(-2 ** 15, 2 ** 15 - 1).short()

    def f32_pcm(wav):
        if wav.dtype == np.float:
            return wav
        return wav.float() / 2 ** 15

    """
    tempo is a relative delta in percentage, so tempo=10 means tempo at 110%!
    pitch is in semi tones.
    Requires `soundstretch` to be installed, see
    https://www.surina.net/soundtouch/soundstretch.html
    """
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")
    in_ = io.BytesIO()
    wavfile.write(in_, sample_rate=samplerate, audio_data=i16_pcm(wav).t().numpy())
    command = [
        "soundstretch",
        "stdin",
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        sp.run(command, capture_output=True, input=in_.getvalue(), check=True)
    except sp.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    wav, sr, _ = wavfile.read(outfile.name)
    wav = wav.copy()
    wav = f32_pcm(torch.from_numpy(np.array(wav)).t())
    assert sr == samplerate
    return wav


def save_shifted_dataset(delta_pitch, delta_tempo, data_path):
    out_path = data_path[:-1] + f'_p={delta_pitch}_t={delta_tempo}/'
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass
    track_names = list(filter(lambda x: os.path.isdir(f'{data_path}/{x}'), sorted(os.listdir(data_path))))
    for track_name in tqdm(track_names):
        try:
            os.mkdir(f'{out_path}/{track_name}')
        except FileExistsError:
            pass
        for s_name in source_names:
            source = load_wav(f'{data_path}/{track_name}/{s_name}.wav')
            shifted = shift(
                torch.tensor(source),
                delta_pitch,
                delta_tempo,
                voice=s_name == 'vocals').T
            sf.write(f'{out_path}/{track_name}/{s_name}.wav', shifted, samplerate=sample_rate)


def load_wav(path, sr=None):
    return sf.read(path, samplerate=sr, dtype='float32')[0].T


P = [-3, -2, -1, 0, 1, 2, 3]   # pitch shift amounts (in semitones)
T = [-30, -20, -10, 0, 10, 20, 30]   # time stretch amounts (10 means 10% slower)

for p in P:
    for t in T:
        if not (p==0 and t==0):
            save_shifted_dataset(p, t, musdb_train_path)