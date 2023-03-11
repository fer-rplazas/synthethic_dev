from torch import nn
import torch


class Compressor(nn.Module):
    def __init__(self, n_channels: int, amp: float = 0.0):

        super().__init__()
        self.slope =  nn.Parameter(amp * torch.randn(1, n_channels, 1))

    def forward(self, x):
        eps = 1e-8
        slope_ = torch.exp(self.slope)

        return (
            torch.sign(x)
            * torch.log(torch.abs(x) * slope_ + 1.0)
            / (torch.log(slope_ + 1.0) + eps)
        )

class CNN1d(nn.Module):
    """Small network with 2 convolutional layers"""

    def __init__(
        self,
        n_channels: int = 1,
        n_out: int = 1,
        n_hidden: int = 45,
        depth: int = 7,
        ks: int = 35,
        stride: int = 1,
        compress: bool = False,
    ):

        super().__init__()

        self.n_hidden = n_hidden

        norm = nn.BatchNorm1d
        affine = True
        self.convolutional_layers = [
            norm(n_channels, affine=affine)
            if compress
            else nn.Identity(),  # Initial normalization layer
            Compressor(n_channels) if compress else nn.Identity(),
            norm(n_channels, affine=affine),  # Initial normalization layer
            nn.Conv1d(n_channels, self.n_hidden, ks, padding="same"),  # Convolution 1
            nn.SiLU(),  # Nonlinearity
            norm(self.n_hidden, affine=affine),  # Normalization
        ]

        for _ in range(depth - 1):
            self.convolutional_layers.extend(
                [
                    Compressor(self.n_hidden) if compress else nn.Identity(),
                    nn.Conv1d(
                        self.n_hidden,
                        self.n_hidden,
                        ks,
                        stride=stride,
                        padding="same" if stride == 1 else 0,
                    ),  # Convolution
                    nn.SiLU(),  # Nonlinearity
                    nn.AvgPool1d(2),
                    norm(self.n_hidden, affine=affine),  # Normalization
                ]
            )

        self.convolutions = nn.Sequential(*self.convolutional_layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.n_hidden, n_out),
        )

    def forward(self, x):
        """the forward function defines how the input x is processed throughout the network layers

        Input:
            x: Tensor of shape (elements_in_batch x channels [here only one] x samples_per_segment [here 512 samples = 0.250 ms at 2048 Hz])
        """
        return self.classifier(self.convolutions(x))
