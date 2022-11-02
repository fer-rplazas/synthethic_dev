import torch
import torch.nn as nn

from .cnn1d import CNN1d


class ConvEncoder(nn.Module):
    def __init__(self, n_channels: int = 1, n_hidden: int = 45):

        super().__init__()
        self.cnn = CNN1d(n_channels, n_hidden, n_hidden, compress=True, depth=5)

    def forward(self, x):
        # x (n_batch, n_chan, n_times)
        return self.cnn(x)  # (n_batch, n_out_feats)


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
