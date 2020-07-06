"""This is a simple conv neural net to emulate Bezier drawing function.
The idea behind is to have a differentiable function for drawing,
so later we can use it as a part of big neural network to draw full images.

The NN draws single curve parametrized by P0, P1, P2, Radius and Color.
Color is the line's Transparency. Radius is the line's Thickness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return x


def summary():
    from torchsummary import summary
    model = CNNDrawer()
    summary(model, input_size=(10,))


if __name__ == '__main__':
    summary()
