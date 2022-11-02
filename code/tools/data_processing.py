from itertools import compress

import numpy as np
import torch
from mne.time_frequency import tfr_array_morlet
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from .feature_extraction import FeatureExtractor


def wavelet_tf(timeseries: np.ndarray, fs: float) -> np.ndarray:
    """Wavelet-based time-frequency decomposition of epoched waveforms

    Args:
        timeseries (np.ndarray): Shape (n_epochs, n_channels, n_times)
        fs (float): Sampling frequency

    Returns:
        np.ndarray: Shape (n_epochs, n_channels, n_freqs, n_times_decimated)
    """

    frequencies = np.arange(8, 201, 3)
    n_cycles = frequencies / 6

    power = tfr_array_morlet(
        timeseries,
        n_cycles=n_cycles,
        freqs=frequencies,
        decim=4,
        output="power",
        sfreq=fs,
    )

    return power


class Dataset:
    def __init__(
        self,
        timeseries: np.ndarray,
        label: np.ndarray,
        Fs: float,
        window_length: float = 0.250,
        hop_size: float = 0.250,
        batch_size: int = 64,
        tf_transform: bool = False,
        quantile_transform: bool = False,
        ar_len: int | None = 10,
        n_folds: int = 5,
        fold_id: int = 4,
    ):

        self.bs = batch_size
        N = len(label)
        self.Fs = Fs

        timeseries = timeseries.astype(np.float32)
        if timeseries.ndim < 2:
            timeseries = timeseries[np.newaxis, ...]

        self.timeseries = timeseries

        # Epoch data:
        idx_start = np.arange(0, N - 1, int(Fs * hop_size))
        idx_end = idx_start + int(Fs * window_length)
        while idx_end[-1] >= N:
            idx_end = idx_end[:-1]
            idx_start = idx_start[:-1]

        self.X = [
            timeseries[:, id_start:id_end]
            for id_start, id_end in zip(idx_start, idx_end)
        ]
        self.labels = [
            np.mean(label[id_start:id_end]) > 0.25
            for id_start, id_end in zip(idx_start, idx_end)
        ]

        # Train / Validation split:
        assert fold_id < n_folds, "fold_id is greater than number of folds"
        idx = np.ones(len(self.X))
        idx[
            int(fold_id * len(self.X) // n_folds) : int(
                (fold_id + 1) * len(self.X) // n_folds
            )
        ] = 0
        is_train_idx = idx.astype(bool)

        # Prepare epoched timeseries:
        self.X_train = list(compress(self.X, is_train_idx))
        self.X_valid = list(compress(self.X, np.logical_not(is_train_idx)))

        assert all(
            el.shape == self.X_valid[0].shape for el in self.X_valid
        ), "Valid data with unexpected shape encountered"

        self.y_train = list(compress(self.labels, is_train_idx))
        self.y_valid = list(compress(self.labels, np.logical_not(is_train_idx)))

        if quantile_transform:
            x_train = np.concatenate(self.X_train, axis=-1)
            qt = QuantileTransformer(n_quantiles=4096).fit(x_train.T)
            self.X_train = [qt.transform(x_.T).T for x_ in self.X_train]
            self.X_valid = [qt.transform(x_.T).T for x_ in self.X_valid]

        # z-score timeseries:
        mean = np.mean([np.mean(x_, axis=-1) for x_ in self.X_train], axis=0)[..., None]
        std = np.mean([np.std(x_, axis=-1) for x_ in self.X_train], axis=0)[..., None]

        self.X_train = [(x_ - mean) / std for x_ in self.X_train]
        self.X_valid = [(x_ - mean) / std for x_ in self.X_valid]

        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.valid_data = [(x, y) for x, y in zip(self.X_valid, self.y_valid)]

        # Extract and prepare features:
        self.X_features = FeatureExtractor(Fs).extract_features(np.stack(self.X))

        self.X_features_train = self.X_features[is_train_idx, :]
        self.X_features_valid = self.X_features[np.logical_not(is_train_idx), :]

        self.scaler = StandardScaler().fit(self.X_features_train)
        self.X_features_train_scaled = self.scaler.transform(self.X_features_train)
        self.X_features_valid_scaled = self.scaler.transform(self.X_features_valid)

        if (
            ar_len is not None
        ):  # TODO: For folds that are not first or last, avoid breaks in continuity for AR data
            train_with_feats = [
                (x, feats, y)
                for (x, y), feats in zip(self.train_data, self.X_features_train_scaled)
            ]
            valid_with_feats = [
                (x, feats, y)
                for (x, y), feats in zip(self.valid_data, self.X_features_valid_scaled)
            ]
            self.train_ar, self.valid_ar = [], []
            for jj in range(len(train_with_feats) - ar_len):
                self.train_ar.append(train_with_feats[jj : jj + ar_len])

            for jj in range(len(valid_with_feats) - ar_len):
                self.valid_ar.append(valid_with_feats[jj : jj + ar_len])

        if tf_transform:
            stacked = np.stack(self.X)
            power = wavelet_tf(stacked, Fs).astype(
                np.float32
            )  # (n_samples, n_chans, n_freqs, n_times)

            self.X_tf_train = power[is_train_idx, ...]
            self.X_tf_valid = power[np.logical_not(is_train_idx), ...]

            # Z-score each frequency (extract stats from training data only):
            mean = np.expand_dims(
                np.expand_dims(
                    np.nanmean(self.X_tf_train, axis=(0, -1), keepdims=False), 0
                ),
                -1,
            )
            std = np.expand_dims(
                np.expand_dims(
                    np.nanstd(self.X_tf_train, axis=(0, -1), keepdims=False), 0
                ),
                -1,
            )

            self.X_tf_train = (self.X_tf_train - mean) / std
            self.X_tf_valid = (self.X_tf_valid - mean) / std

            self.tf_train_data = [(x, y) for x, y in zip(self.X_tf_train, self.y_train)]
            self.tf_valid_data = [(x, y) for x, y in zip(self.X_tf_valid, self.y_valid)]

    def train_ar_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ar, self.bs, shuffle=True)

    def valid_ar_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_ar, self.bs, shuffle=False)

    def train_tf_dataloader(self):
        return torch.utils.data.DataLoader(self.tf_train_data, self.bs, shuffle=True)

    def valid_tf_dataloader(self):
        return torch.utils.data.DataLoader(self.tf_valid_data, self.bs, shuffle=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, self.bs, shuffle=True)

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_data, self.bs, shuffle=False)
