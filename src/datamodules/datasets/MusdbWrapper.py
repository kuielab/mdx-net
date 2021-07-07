import os
from abc import ABCMeta, ABC
from pathlib import Path

import soundfile
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data.dataset import T_co

from src.utils.utils import load, load_from_start_position


def check_target_valid(target_name):
    try:
        assert target_name is not None
    except AssertionError:
        print('[ERROR] please identify target name. ex) +datamodule.target_name="vocals"')
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


class MusdbWrapperDataset(Dataset, ABC):
    __metaclass__ = ABCMeta

    def __init__(self, musdb_path):
        self.musdb_path = musdb_path
        self.source_names = ['vocals', 'drums', 'bass', 'other']
        self.source_wav_names = [s_name + '.wav' for s_name in self.source_names]


class MusdbTrainDataset(MusdbWrapperDataset):

    def __init__(self, data_dir, augmentation: bool, external_datasets, target_name, sampling_size):
        super(MusdbTrainDataset, self).__init__(data_dir)

        check_target_valid(target_name)
        self.target_name = target_name
        self.sampling_size = sampling_size

        musdb_path = Path(data_dir)

        dataset_names = [musdb_path.joinpath('train')]
        metadata_caches = [musdb_path.joinpath('metadata').joinpath('train.pkl')]
        if not musdb_path.joinpath('metadata').exists():
            os.mkdir(musdb_path.joinpath('metadata'))

        if augmentation:
            for p in range(-3, 4):
                for t in range(-30, 40, 10):
                    if (p, t) == (0, 0):  # original setting
                        pass
                    else:
                        split = f'train_p={p}_t={t}'
                        dataset_names.append(musdb_path.joinpath(split))
                        metadata_caches.append(musdb_path.joinpath('metadata').joinpath(split + '.pkl'))

        self.metadata = dict([(s_name, []) for s_name in self.source_names])

        for i, (dataset, metadata_cache) in enumerate(tqdm(zip(dataset_names, metadata_caches))):

            # Check cached
            try:
                print('try to load metadata cache')
                metadata = torch.load(metadata_cache)
            except FileNotFoundError:
                print('creating metadata for', dataset)
                metadata = {s_name: [] for s_name in self.source_names}
                for track_name in sorted(os.listdir(dataset)):
                    track_path = dataset.joinpath(track_name)
                    track_length = load(track_path.joinpath('vocals.wav')).shape[-1]
                    for s_name in os.listdir(track_path):
                        s_name = s_name[:-4]
                        if s_name in self.source_names:
                            metadata[s_name].append((track_path, track_length))
                torch.save(metadata, metadata_cache)

            for key in metadata.keys():
                self.metadata[key] += metadata[key]

            if i == 0:
                lengths = [length for path, length in self.metadata['vocals']]
                self.num_iter = sum(lengths) // self.sampling_size + 1

    def __getitem__(self, _):
        sources = []
        for s_name in self.source_names:
            track_path, track_length = random.choice(self.metadata[s_name])
            source = load(track_path.joinpath(s_name + '.wav'),
                          max_length=track_length, sampling_size=self.sampling_size)
            sources.append(source)
        mix = sum(sources)
        target = sources[self.source_names.index(self.target_name)]
        return torch.from_numpy(mix), torch.from_numpy(target)

    def __len__(self):
        return self.num_iter  # 3241 ~ compatible to check_val_n_epoch = 25


class MusdbValidationDataset(MusdbWrapperDataset):

    def __init__(self, batch_size, data_dir, target_name, sampling_size, n_fft):
        super().__init__(data_dir)

        check_target_valid(target_name)
        self.target_name = target_name
        self.target_wav_name = target_name + '.wav'

        self.batch_size = batch_size

        path = Path(data_dir)
        musdb_valid_path = path.joinpath('valid')
        self.track_names = sorted(set(os.listdir(musdb_valid_path)))
        self.file_paths = [musdb_valid_path.joinpath(track_name) for track_name in self.track_names]

        self.sampling_size = sampling_size

        self.trim = n_fft // 2
        self.true_samples = sampling_size - 2 * self.trim

        self.tracks = [load(track.joinpath('vocals.wav')) for track in self.file_paths]
        self.num_iter = len(self.track_names)

    def __getitem__(self, index):
        files = [self.file_paths[index].joinpath(s_name) for s_name in ['mixture.wav', self.target_wav_name]]
        mixture, target = [load(file) for file in files]

        right_pad = self.true_samples + self.trim - ((mixture.shape[-1]) % self.true_samples)
        mixture = np.concatenate((np.zeros((2, self.trim), dtype='float32'),
                                  mixture,
                                  np.zeros((2, right_pad), dtype='float32')),
                                 1)
        num_chunks = mixture.shape[-1] // self.true_samples
        batches = [mixture[:, i * self.true_samples: i * self.true_samples + self.sampling_size] for i in
                   range(num_chunks)]

        return self.batch_size, index, np.stack(batches), target

    def __len__(self):
        return self.num_iter

    def get_reference(self, index):
        return load(self.file_paths[index].joinpath(self.target_wav_name))
