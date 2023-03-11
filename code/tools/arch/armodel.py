import torch
import torch.nn as nn

from .cnn1d import CNN1d
from .cnn2d import resnet
from .dotfft import ComplexFFTNet


class Conv2dEncoder(nn.Module):
    def __init__(self, n_channels: int = 1, n_hidden: int = 45):

        super().__init__()
        self.cnn = resnet(n_channels, n_hidden)

    def forward(self, x):
        return self.cnn(x)


class ConvEncoder(nn.Module):
    def __init__(self, n_channels: int = 1, n_hidden: int = 45):

        super().__init__()
        self.cnn = CNN1d(n_channels, n_hidden, n_hidden, compress=True, depth=5)

    def forward(self, x):
        # x (n_batch, n_chan, n_times)
        return self.cnn(x)  # (n_batch, n_out_feats)


class FFTEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_hidden: int = 45,
        n_hidden_int: int = 15,
        depth: int = 10,
    ):

        super().__init__()
        self.model = ComplexFFTNet(
            n_channels, n_samples, n_out=n_hidden, n_hidden=n_hidden_int, depth=depth
        )

    def forward(self, x):
        # x (n_batch, n_chan, n_times)
        return self.model(x)  # (n_batch, n_out_feats)


class EnsembleAR(nn.Module):
    def __init__(self, n_channels, n_feats, n_samples):

        super().__init__()

        self.conv1d = ConvEncoder(n_channels, 35)
        # self.conv2d = Conv2dEncoder(n_channels, 35)
        self.fft = FFTEncoder(n_channels, n_samples, 35)
        self.feat_encoder = nn.Sequential(
            nn.Linear(n_feats, 35), nn.BatchNorm1d(35), nn.SiLU(), nn.Linear(35, 35)
        )

        self.ensembler = nn.Sequential(
            nn.Linear(3 * 35, 50),
            nn.BatchNorm1d(50),
            nn.SiLU(),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.SiLU(),
            nn.Linear(10, 10),
        )

        self.AR = nn.LSTM(10, 3, proj_size=1)

    def forward(self, signals, feats):
        # signals (n_seq, n_batch, n_chan, n_times)
        # feats (n_seq, n_batch, n_feats)

        signal_feats = []
        for model in [self.conv1d, self.fft]:
            l_ = []
            for x in torch.unbind(signals, 0):
                l_.append(model(x))
            signal_feats.append(torch.stack(l_))  # (n_seq, n_batch, n_conv_feats)

        l_ = []
        for x in torch.unbind(feats, 0):
            l_.append(self.feat_encoder(x))
        features_encoded = torch.stack(l_)  # (n_seq, n_batch, n_feats_encoded)

        feats_all = torch.cat((*signal_feats, features_encoded), -1)

        l_ = []
        for x in torch.unbind(feats_all, 0):
            l_.append(self.ensembler(x))
        feats_all = torch.stack(l_)  # (n_seq, n_batch, n_feats_combined)

        out, _ = self.AR(feats_all)  # out (n_seq, n_batch, 1)
        out = out.squeeze()

        return out


class ARModel(nn.Module):
    def __init__(self, n_channels, n_feats):

        super().__init__()

        self.convs = ConvEncoder(n_channels, 45)
        self.feat_encoder = nn.Sequential(
            nn.BatchNorm1d(n_feats), nn.Linear(n_feats, 5)
        )
        self.combiner = nn.Sequential(
            nn.Linear(45 + 5, 10), nn.BatchNorm1d(10), nn.SiLU()
        )

        self.AR = nn.LSTM(10, 3, proj_size=1)

    def forward(self, signals, feats):
        # signals (n_seq, n_batch, n_chan, n_times)
        # feats (n_seq, n_batch, n_feats)

        l_ = []
        for x in torch.unbind(signals, 0):
            l_.append(self.convs(x))
        conved = torch.stack(l_)  # (n_seq, n_batch, n_conv_feats)

        l_ = []
        for x in torch.unbind(feats, 0):
            l_.append(self.feat_encoder(x))
        features_encoded = torch.stack(l_)  # (n_seq, n_batch, n_feats_encoded)

        feats_both = torch.cat((conved, features_encoded), -1)

        l_ = []
        for x in torch.unbind(feats_both, 0):
            l_.append(self.combiner(x))
        feats_all = torch.stack(l_)  # (n_seq, n_batch, n_feats_combined)

        out, _ = self.AR(feats_all)  # out (n_seq, n_batch, 1)
        out = out.squeeze()

        return out
