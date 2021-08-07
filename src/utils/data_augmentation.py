import os
import subprocess as sp
import tempfile
import warnings
from argparse import ArgumentParser

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=Warning)
source_names = ['vocals', 'drums', 'bass', 'other']
sample_rate = 44100

def main (args):
    data_root = args.data_dir
    train = args.train
    test = args.test
    valid = args.valid

    musdb_train_path = data_root + 'train/'
    musdb_test_path = data_root + 'test/'
    musdb_valid_path = data_root + 'valid/'

    mix_name = 'mixture'

    P = [-3, -2, -1, 0, 1, 2, 3]   # pitch shift amounts (in semitones)
    T = [-30, -20, -10, 0, 10, 20, 30]   # time stretch amounts (10 means 10% slower)

    for p in P:
        for t in T:
            if not (p==0 and t==0):
                if train:
                    save_shifted_dataset(p, t, musdb_train_path)
                if valid:
                    save_shifted_dataset(p, t, musdb_valid_path)
                if test:
                    save_shifted_dataset(p, t, musdb_test_path)


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

    inputfile = tempfile.NamedTemporaryFile(suffix=".wav")
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")

    sf.write(inputfile.name, data=i16_pcm(wav).t().numpy(), samplerate=samplerate, format='WAV')
    command = [
        "soundstretch",
        inputfile.name,
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        sp.run(command, capture_output=True, check=True)
    except sp.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    wav, sr = sf.read(outfile.name, dtype='float32')
    # wav = np.float32(wav)
    # wav = f32_pcm(torch.from_numpy(wav).t())
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
                voice=s_name == 'vocals')
            sf.write(f'{out_path}/{track_name}/{s_name}.wav', shifted, samplerate=sample_rate, format='WAV')


def load_wav(path, sr=None):
    return sf.read(path, samplerate=sr, dtype='float32')[0].T


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--valid', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)

    main(parser.parse_args())