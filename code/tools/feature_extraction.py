import numpy as np
from scipy.signal import periodogram
from numba import njit


@njit
def fill_feat_mat(
    feat_mat: np.ndarray,
    win_id: int,
    Pxx: np.ndarray,
    n_chan: int,
    freq_idx: np.ndarray,
):

    for ch in range(n_chan):
        for kk, freq_lims in enumerate(freq_idx):
            feat_mat[win_id, (ch * freq_idx.shape[0]) + kk] = np.mean(
                Pxx[ch, freq_lims[0] : freq_lims[1]]
            )
    return feat_mat


class FeatureExtractor:
    def __init__(self, fs):

        self.fs = fs
        self.freq_ranges = [
            [2, 7],
            [8, 12],
            [13, 20],
            [21, 30],
            [31, 45],
            [46, 55],  # Line noise
            [56, 75],
            [76, 95],
            [95, 105],  # (Potential) Line noise
            [106, 145],
            [146, 155],  # (Potential) Line noise
            [156, 195],
        ]

    def prepare_bands(self, data_len: int) -> np.ndarray:
        f, _ = periodogram(np.random.rand((data_len)), fs=self.fs)
        self.n_samp = data_len

        self.freq_idx = np.zeros_like(np.array(self.freq_ranges))
        for jj, freq_range in enumerate(self.freq_ranges):
            for kk in range(2):
                self.freq_idx[jj, kk] = np.argmin(np.abs(freq_range[kk] - f))

        return self.freq_idx

    def extract_features(self, X: np.ndarray):
        """Extract spectral features from multi-channel timeseries.

        Args:
            X (np.ndarray): Data of shape (n_batch/n_win, n_chan, n_samp) or (n_chan, n_samp), in which case an additional first dimension will be added.
        """
        if X.ndim < 3:
            X = X[np.newaxis, ...]

        # Determine periodogram indices corresponding to canonical frequency bands:
        self.prepare_bands(X.shape[-1])

        # Preallocate feature matrix:
        feat_mat = np.zeros((X.shape[0], X.shape[1] * self.freq_idx.shape[0])) * np.nan

        # Compute features and add to feature matrix:
        for ii, this_data in enumerate(X):
            _, Pxx = periodogram(
                this_data,
                fs=self.fs,
            )
            feat_mat = fill_feat_mat(feat_mat, ii, Pxx, X.shape[1], self.freq_idx)

        assert np.isnan(feat_mat).any() == False, "NaNs found after feature extraction"

        return feat_mat
