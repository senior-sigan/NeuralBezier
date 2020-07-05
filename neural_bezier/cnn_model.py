"""This is a simple conv neural net to emulate Bezier drawing function.
The idea behind is to have a differentiable function for drawing,
so later we can use it as a part of big neural network to draw full images.

The NN draws single curve parametrized by P0, P1, P2, Radius and Color.
Color is the line's Transparency. Radius is the line's Thickness.
"""
import torch
import torch.nn as nn


class CNNDrawer(nn.Module[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()
