"""This is a simple conv neural net to emulate Bezier drawing function.
The idea behind is to have a differentiable function for drawing,
so later we can use it as a part of big neural network to draw full images.

The NN draws single curve parametrized by P0, P1, P2, Radius and Color.
Color is the line's Transparency. Radius is the line's Thickness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class CNNDrawer2(nn.Module):
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
            Up(in_channels=16, out_channels=8),   # 64
            Up(in_channels=8, out_channels=4),    # 128
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        weights_init_normal(self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emedding = self.fc(x)
        emedding = emedding.view(-1, 64, 8, 8)
        canvas = self.convs(emedding)
        canvas = torch.sigmoid(canvas)
        return canvas


class CNNDrawer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_input_params = 10

        self.fc1 = nn.Linear(in_features=self.n_input_params, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=2048)
        self.fc4 = nn.Linear(in_features=2048, out_features=4096)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = torch.sigmoid(x)
        return 1 - x  # predict inversion of the image, helps to avoid zeroing


def summary():
    from torchsummary import summary
    model = CNNDrawer()
    summary(model, input_size=(10,))


if __name__ == '__main__':
    summary()
