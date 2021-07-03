import os
from abc import ABCMeta, ABC
from pathlib import Path

import soundfile
from torch.utils.data import Dataset
import torch
import numpy as np
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
        self.source_wav_names = [source + '.wav' for source in self.source_names]


class MusdbTrainDataset(MusdbWrapperDataset):

    def __init__(self, musdb_path, target_name, validation_set, sr, sampling_size):
        super(MusdbTrainDataset, self).__init__(musdb_path)

        check_target_valid(target_name)
        self.target_name = target_name
        self.target_wav_name = target_name + '.wav'

        path = Path(musdb_path)
        self.track_names = sorted(set(os.listdir(path.joinpath('train'))) - set(validation_set))
        self.file_paths = [path.joinpath('train', track_name) for track_name in self.track_names]

        # integrity check 1: number of files
        try:
            assert len(set(validation_set) - set(os.listdir(path.joinpath('train')))) == 0
        except AssertionError:
            if len(self.track_names) == 0:
                print('[ERROR] it seems the musdb path is not valid.')
            else:
                print('[ERROR] it seems that invalid track names are included in validation set')
            exit(-1)

        # integrity check 2: sampling rate
        check_sampling_rate(sr, self.file_paths[0].joinpath('vocals.wav'))
        self.sampling_size = sampling_size

        self.lengths = [load(track.joinpath('vocals.wav')).shape[1] for track in self.file_paths]
        self.num_tracks = len(self.lengths)
        self.num_iter = sum(self.lengths) // self.sampling_size + 1

    def __getitem__(self, index):

        # target
        index = index % self.num_tracks
        target = load(self.file_paths[index].joinpath(self.target_wav_name), self.lengths[index], self.sampling_size)

        # mixing different songs
        mix = target
        for t_wav_name in self.source_wav_names:
            if t_wav_name != self.target_wav_name:
                index2 = np.random.randint(self.num_tracks)
                target2 = load(self.file_paths[index2].joinpath(t_wav_name), self.lengths[index2], self.sampling_size)
                mix = mix + target2

        return torch.from_numpy(mix), torch.from_numpy(target)

    def __len__(self):
        return self.num_iter  # 3241 ~ compatible to check_val_n_epoch = 25


class MusdbValidationDataset(MusdbWrapperDataset):

    def __init__(self, musdb_path, target_name, validation_set, sr, sampling_size, n_fft):
        super().__init__(musdb_path)

        check_target_valid(target_name)
        self.target_name = target_name
        self.target_wav_name = target_name + '.wav'

        path = Path(musdb_path)
        self.track_names = sorted(set(os.listdir(path.joinpath('train'))).intersection(set(validation_set)))
        self.file_paths = [path.joinpath('train', track_name) for track_name in self.track_names]

        # integrity check 1: number of files
        try:
            assert len(set(validation_set)) == len(self.track_names)
        except AssertionError:
            if len(self.track_names) == 0:
                print('[ERROR] it seems the musdb path is not valid.')
            else:
                print('[ERROR] it seems that invalid track names are included in validation set')
            exit(-1)

        # integrity check 2: sampling rate
        check_sampling_rate(sr, self.file_paths[0].joinpath('vocals.wav'))
        self.sampling_size = sampling_size

        self.trim = n_fft // 2
        self.true_samples = sampling_size - 2 * self.trim

        self.tracks = [load(track.joinpath('vocals.wav')) for track in self.file_paths]
        self.lengths = [track.shape[1] for track in self.tracks]
        num_chunks = [length // self.true_samples for length in self.lengths]
        id_pad_start_frame_pad = [
            [
                (track_i,
                 chunk_i,
                 0,
                 chunk_i * self.true_samples - self.trim,
                 sampling_size,
                 0)
                for chunk_i
                in range(num_chunk + 1)
            ]
            for track_i, num_chunk
            in enumerate(num_chunks)
        ]

        for items in id_pad_start_frame_pad:
            track_i, chunk_i, left_pad, s, frame, right_pad = items[0]
            items[0] = (track_i, chunk_i, self.trim, 0, self.true_samples + self.trim, right_pad)

            track_i, chunk_i, left_pad, s, frame, right_pad = items[-1]
            if self.lengths[track_i] % self.true_samples != 0:
                items[-1] = (track_i,
                             chunk_i,
                             left_pad,
                             s,
                             -1,
                             self.true_samples + self.trim - (self.lengths[track_i] % self.true_samples)
                             )
            else:
                items.remove(items[-1])

        self.idx_pad_s_frame_pads = [item for items in id_pad_start_frame_pad for item in items]
        self.num_iter = len(self.idx_pad_s_frame_pads)

    def __getitem__(self, index):
        track_idx, chunk_idx, left_pad, start_pos, frame, right_pad = self.idx_pad_s_frame_pads[index]
        track_mix = load_from_start_position(self.file_paths[track_idx].joinpath('mixture.wav'), start_pos, frame)

        if left_pad > 0 and right_pad > 0:
            track_mix = np.concatenate((np.zeros((2, left_pad), dtype='float32'),
                                        track_mix,
                                        np.zeros((2, right_pad), dtype='float32')), 1)
        elif left_pad > 0:
            track_mix = np.concatenate((np.zeros((2, left_pad), dtype='float32'), track_mix), 1)
        elif right_pad > 0:
            track_mix = np.concatenate((track_mix, np.zeros((2, right_pad), dtype='float32')), 1)

        assert track_mix.shape[-1] == self.sampling_size
        return track_idx, chunk_idx, torch.from_numpy(track_mix)

    def __len__(self):
        return self.num_iter

    def get_reference(self, index):
        return load(self.file_paths[index].joinpath(self.target_wav_name))


#
# def preprocess_track(y):
#     n_sample = y.shape[1]
#
#     gen_size = sampling_size - 2 * trim
#     pad = gen_size - n_sample % gen_size
#     y_p = np.concatenate((np.zeros((2, trim)), y, np.zeros((2, pad)), np.zeros((2, trim))), 1)
#
#     all_waves = []
#     i = 0
#     while i < n_sample + pad:
#         waves = np.array(y_p[:, i:i + sampling_size], dtype=np.float32)
#         all_waves.append(waves)
#         i += gen_size
#
#     return torch.tensor(all_waves), pad
#
#
# def separate(model, mix):
#     model.eval()
#
#     mix_waves, pad_len = preprocess_track(mix)
#
#     # create batches
#     batch_size = 16
#     i = 0
#     num_intervals = mix_waves.shape[0]
#     batches = []
#     while i < num_intervals:
#         batches.append(mix_waves[i:i + batch_size])
#         i = i + batch_size
#
#     # obtain estimated target spectrograms
#     tar_signal = np.array([[], []])
#     with torch.no_grad():
#         for batch in tqdm_notebook(batches):
#             tar_specs = model(S.stft(batch.to(device))[:, :, :dim_f])
#             pad = torch.zeros([batch.shape[0], dim_c, n_bins - dim_f, dim_t]).to(device)
#             tar_waves = S.istft(torch.cat([tar_specs, pad], -2))
#             est_interval = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
#             tar_signal = np.concatenate((tar_signal, est_interval), 1)
#
#     return tar_signal[:, :-pad_len]
#
#
# def median_nan(a):
#     return np.median(a[~np.isnan(a)])
#
#
# def musdb_sdr(ref, est, sr=sampling_rate):
#     sdr, isr, sir, sar, perm = eval4.metrics.bss_eval(ref, est, window=sr, hop=sr)
#     return median_nan(sdr[0]), sdr
#
#
# def sdr_(ref, est):
#     ratio = np.sum(ref ** 2) / np.sum((ref - est) ** 2)
#     return 10 * np.log10(ratio)
#
#
# def L2(ref, est):
#     return ((ref - est) ** 2).mean()
#
#
# def L1(ref, est):
#     return np.abs(ref - est).mean()
#
#
# max_epoch = 32000
# min_epoch = 20000
# cs = 1000
#
# num_ckpts = (max_epoch - min_epoch) // cs + 1
#
# scores_mean = []
# score_path = '{0}/{1}/lr{2}_valid_{3}.npy'.format(model_path, model_name, lr, target_name)
#
# try:
#     scores_mean = list(np.load(score_path))
# except Exception:
#     pass
#
# file_names = train_set.valid_file_names
#
# for c in range(num_ckpts):
#     ckpt_scores = []
#     init_weights(model, lr=lr, epoch=min_epoch + c * cs)
#     # init_weights(model, lr=lr, epoch=None)
#     for i in tqdm_notebook(range(len(file_names))):
#         mix = load('{0}/{1}/{2}.wav'.format(musdb_train_path, file_names[i], mix_name))
#         est = separate(model, mix)
#         ref = load('{0}/{1}/{2}.wav'.format(musdb_train_path, file_names[i], target_name))
#
#         score = sdr_(ref, est)
#         print(score)
#
#         ckpt_scores.append(score)
#
#     ckpt_score_mean = np.array(ckpt_scores).mean()
#     scores_mean.append(ckpt_score_mean)
#
#     ipd.clear_output(wait=True)
#     print(ckpt_score_mean)
#     plt.plot(scores_mean)
#     plt.show()
#
# np.save(score_path, np.array(scores_mean))
