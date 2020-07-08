import torch
from torch import nn


class PatchGanDiscriminator(nn.Module):
    def __init__(self, in_channels1: int, in_channels2: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, 64, 4, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # PatchGAN. See: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
            nn.Conv2d(512, 1, 4, stride=1, padding=2),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.model(x)
        x = torch.sigmoid(x)
        # Shape is [batch_size, 1, N, N] where N is a patch size
        return x
