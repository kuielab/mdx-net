import os
from abc import ABCMeta, ABC
from pathlib import Path
from typing import Set, List

import soundfile
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from tqdm import tqdm

from src.utils import utils
from src.utils.utils import load_wav


from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def check_target_name(target_name, source_names):
    try:
        assert target_name is not None
    except AssertionError:
        print('[ERROR] please identify target name. ex) +datamodule.target_name="vocals"')
        exit(-1)
    try:
        assert target_name in source_names or target_name == 'all'
    except AssertionError:
        print('[ERROR] target name should one of "bass", "drums", "other", "vocals", "all"')
        exit(-1)


def check_sample_rate(sr, sample_track):
    try:
        sample_rate = soundfile.read(sample_track)[1]
        assert sample_rate == sr
    except AssertionError:
        sample_rate = soundfile.read(sample_track)[1]
        print('[ERROR] sampling rate mismatched')
        print('\t=> sr in Config file: {}, but sr of data: {}'.format(sr, sample_rate))
        exit(-1)


class MusdbDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, data_dir, chunk_size):
        self.source_names = ['bass', 'drums', 'other', 'vocals']
        self.chunk_size = chunk_size
        self.musdb_path = Path(data_dir)


class MusdbTrainDataset(MusdbDataset):
    def __init__(self, data_dir, valid_track_names, chunk_size, target_name, aug_params, external_datasets):
        super(MusdbTrainDataset, self).__init__(data_dir, chunk_size)

        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        metadata_cache_path = self.musdb_path.joinpath('metadata').joinpath('raw_data.pkl')

        metadata = []
        if self.musdb_path.joinpath('metadata').exists():
            if metadata_cache_path.exists():
                metadata = torch.load(metadata_cache_path)
        else:
            os.mkdir(self.musdb_path.joinpath('metadata'))

        tracks: Set[str] = set(os.listdir(self.musdb_path)).difference({'metadata', 'train'})
        train_data: Set[str] = tracks.difference(set(valid_track_names))
        assert len(set(valid_track_names)) + len(train_data) == len(set(tracks))

        train_data: List[str] = sorted(train_data)
        if metadata is not None:
            if [track.name for track, length in metadata] == train_data:
                log.info('cached metadata loaded.')
            else:
                log.warning('cached metadata does not look valid. previous one will removed, and be re-generated.')
                metadata = None

        if metadata is None:
            # collect all track names and their duration
            self.metadata = []
            raw_track_lengths = []   # for calculating epoch size

            metadata = []
            log.info('analyzing train dataset to generate metadata')
            for track_name in tqdm(train_data):
                track_path = self.musdb_path.joinpath(track_name)
                track_length = load_wav(track_path.joinpath('vocals.wav')).shape[-1]
                metadata.append((track_path, track_length))

            torch.save(metadata, metadata_cache_path)

        self.metadata = metadata
        raw_track_lengths = [length for path, length in metadata]

        self.epoch_size = sum(raw_track_lengths) // self.chunk_size

    def __getitem__(self, _):
        sources = []
        for source_name in self.source_names:
            track_path, track_length = random.choice(self.metadata)   # random mixing between tracks
            source = load_wav(track_path.joinpath(source_name + '.wav'),
                              track_length=track_length, chunk_size=self.chunk_size)
            sources.append(source)

        mix = sum(sources)

        if self.target_name == 'all':
            # Targets for models that separate all four sources (ex. Demucs).
            # This adds additional 'source' dimension => batch_shape=[batch, source, channel, time]
            target = sources
        else:
            target = sources[self.source_names.index(self.target_name)]

        return torch.tensor(mix), torch.tensor(target)

    def __len__(self):
        return self.epoch_size


class MusdbValidDataset(MusdbDataset):

    def __init__(self, data_dir, valid_track_names, chunk_size, target_name, overlap, batch_size):
        super(MusdbValidDataset, self).__init__(data_dir, chunk_size)

        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        self.overlap = overlap
        self.batch_size = batch_size

        self.track_paths = [data_dir.joinpath(track_name) for track_name in valid_track_names]

    def __getitem__(self, index):
        # mix = load_wav(self.track_paths[index].joinpath('mixture.wav'))

        sources = {source: load_wav(self.track_paths[index].joinpath(source + '.wav')) for source in self.source_names}
        mix = sum(sources.values())
        # if self.target_name == 'all':
        #     # Targets for models that separate all four sources (ex. Demucs).
        #     # This adds additional 'source' dimension => batch_shape=[batch, source, channel, time]
        #     target = [load_wav(self.track_paths[index].joinpath(source_name + '.wav'))
        #               for source_name in self.source_names]
        # else:
        #     target = load_wav(self.track_paths[index].joinpath(self.target_name + '.wav'))
        target = sources[self.target_name]

        chunk_output_size = self.chunk_size - 2 * self.overlap
        left_pad = np.zeros([2, self.overlap])
        right_pad = np.zeros([2, self.overlap + chunk_output_size - (mix.shape[-1] % chunk_output_size)])
        mix_padded = np.concatenate([left_pad, mix, right_pad], 1)

        num_chunks = mix_padded.shape[-1] // chunk_output_size
        mix_chunks = [mix_padded[:, i * chunk_output_size: i * chunk_output_size + self.chunk_size]
                      for i in range(num_chunks)]
        mix_chunk_batches = torch.tensor(mix_chunks, dtype=torch.float32).split(self.batch_size)
        return mix_chunk_batches, torch.tensor(target)

    def __len__(self):
        return len(self.track_paths)