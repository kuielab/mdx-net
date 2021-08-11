import os
import subprocess as sp
import tempfile
import warnings
from argparse import ArgumentParser
from ast import literal_eval

import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=Warning)
source_names = ['vocals', 'drums', 'bass', 'other']
sample_rate = 44100

def main (args):
    data_dir = Path(args.data_dir)
    train = args.train
    test = args.test

    P = [-2, -1, 0, 1, 2]   # pitch shift amounts (in semitones)
    T = [-20, -10, 0, 10, 20]   # time stretch amounts (10 means 10% slower)

    for p in P:
        for t in T:
            if not (p==0 and t==0):
                if train:
                    save_shifted_dataset(p, t, data_dir, 'train')
                if test:
                    save_shifted_dataset(p, t, data_dir, 'test')


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


def save_shifted_dataset(delta_pitch, delta_tempo, data_dir, split):
    aug_split = split + f'_p={delta_pitch}_t={delta_tempo}'
    in_dir = data_dir.joinpath(split)
    out_dir = data_dir.joinpath(aug_split)
    if not out_dir.exists():
        os.mkdir(out_dir)
    track_names = os.listdir(in_dir)
    for track_name in tqdm(track_names):
        in_path = in_dir.joinpath(track_name)
        out_path = out_dir.joinpath(track_name)
        if not out_path.exists():
            os.mkdir(out_path)
        for s_name in source_names:
            source = load_wav(in_path.joinpath(s_name+'.wav'))
            shifted = shift(
                torch.tensor(source),
                delta_pitch,
                delta_tempo,
                voice=s_name == 'vocals')
            sf.write(out_path.joinpath(s_name+'.wav'), shifted, samplerate=sample_rate, format='WAV')


def load_wav(path, sr=None):
    return sf.read(path, samplerate=sr, dtype='float32')[0].T


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train', default=True, type=literal_eval)
    parser.add_argument('--test', default=False, type=literal_eval)

    main(parser.parse_args())