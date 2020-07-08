import torch
import torch.nn as nn

from neural_bezier.utils import weights_init_normal


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class CNNDrawerV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_input_params = 10

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.n_input_params, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=4096),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.Sequential(
            Up(in_channels=64, out_channels=32),  # 16
            Up(in_channels=32, out_channels=16),  # 32
            Up(in_channels=16, out_channels=8),  # 64
            Up(in_channels=8, out_channels=4),  # 128
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        weights_init_normal(self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emedding = self.fc(x)
        emedding = emedding.view(-1, 64, 8, 8)
        canvas = self.convs(emedding)
        canvas = torch.sigmoid(canvas)
        return canvas
