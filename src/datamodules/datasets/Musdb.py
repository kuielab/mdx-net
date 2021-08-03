import os
from abc import ABCMeta, ABC
from pathlib import Path

import soundfile
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from tqdm import tqdm

from src.utils.utils import load_wav


def check_target_name(target_name, source_names):
    try:
        assert target_name is not None
    except AssertionError:
        print('[ERROR] please identify target name. ex) +datamodule.target_name="vocals"')
        exit(-1)
    try:
        assert target_name in source_names
    except AssertionError:
        print('[ERROR] target name should one of "bass", "drums", "other", "vocals"')
        exit(-1)


def check_sampling_rate(sr, sample_track):
    try:
        sampling_rate = soundfile.read(sample_track)[1]
        assert sampling_rate == sr
    except AssertionError:
        sampling_rate = soundfile.read(sample_track)[1]
        print('[ERROR] sampling rate mismatched')
        print('\t=> sr in Config file: {}, but sr of data: {}'.format(sr, sampling_rate))
        exit(-1)


class MusdbDataset(Dataset):

    def __init__(self, data_dir, split, aug_params, target_name, sampling_size, external_datasets):
        super(MusdbDataset, self).__init__()

        self.source_names = ['bass', 'drums', 'other', 'vocals']
        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        self.sampling_size = sampling_size

        musdb_path = Path(data_dir)

        if not musdb_path.joinpath('metadata').exists():
            os.mkdir(musdb_path.joinpath('metadata'))

        # create lists of paths for datasets and metadata (track names and duration)
        dataset_names = [musdb_path.joinpath(split)]
        metadata_caches = [musdb_path.joinpath('metadata').joinpath(split+'.pkl')]
        max_pitch, max_tempo = aug_params
        for p in range(-max_pitch, max_pitch+1):
            for t in range(-max_tempo, max_tempo+1, 10):
                if (p, t) == (0, 0):  # no aug
                    pass
                else:
                    aug_split = split + f'_p={p}_t={t}'
                    dataset_names.append(musdb_path.joinpath(aug_split))
                    metadata_caches.append(musdb_path.joinpath('metadata').joinpath(aug_split + '.pkl'))

        # collect all track names and their duration
        self.metadata = []
        for i, (dataset, metadata_cache) in enumerate(tqdm(zip(dataset_names, metadata_caches))):
            try:
                metadata = torch.load(metadata_cache)
            except FileNotFoundError:
                print('creating metadata for', dataset)
                metadata = []
                for track_name in sorted(os.listdir(dataset)):
                    track_path = dataset.joinpath(track_name)
                    track_length = load_wav(track_path.joinpath('vocals.wav')).shape[-1]
                    metadata.append((track_path, track_length))
                torch.save(metadata, metadata_cache)

            self.metadata += metadata

            if i == 0:  # get epoch size
                lengths = [length for path, length in self.metadata]
                self.num_iter = sum(lengths) // self.sampling_size + 1

    def __getitem__(self, _):
        sources = []
        for s_name in self.source_names:
            track_path, track_length = random.choice(self.metadata)   # random mixing between tracks
            source = load_wav(track_path.joinpath(s_name + '.wav'),
                              track_length=track_length, chunk_size=self.sampling_size)
            sources.append(source)
        mix = sum(sources)
        target = sources[self.source_names.index(self.target_name)]
        return torch.from_numpy(mix), torch.from_numpy(target)

    def __len__(self):
        return self.num_iter


class MusdbValidDataset(Dataset):

    def __init__(self, data_dir, target_name, sampling_size, trim, batch_size):
        super(MusdbValidDataset, self).__init__()

        self.source_names = ['bass', 'drums', 'other', 'vocals']
        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        self.sampling_size = sampling_size
        self.trim = trim
        self.batch_size = batch_size

        musdb_valid_path = Path(data_dir).joinpath('valid')

        self.track_paths = [musdb_valid_path.joinpath(track_name)
                            for track_name in os.listdir(musdb_valid_path)]

    def __getitem__(self, index):
        mix = load_wav(self.track_paths[index].joinpath('mixture.wav'))
        target = load_wav(self.track_paths[index].joinpath(self.target_name + '.wav'))

        chunk_output_size = self.sampling_size - 2 * self.trim
        left_pad = np.zeros([2, self.trim])
        right_pad = np.zeros([2, chunk_output_size + self.trim - (mix.shape[-1] % chunk_output_size)])
        mix_padded = np.concatenate([left_pad, mix, right_pad], 1)

        num_chunks = mix_padded.shape[-1] // chunk_output_size
        mix_chunks = [mix_padded[:, i * chunk_output_size: i * chunk_output_size + self.sampling_size]
                      for i in range(num_chunks)]
        mix_chunk_batches = torch.tensor(mix_chunks, dtype=torch.float32).split(self.batch_size)
        return mix_chunk_batches, torch.from_numpy(target)

    def __len__(self):
        return len(self.track_paths)
