import unittest

import torch

from neural_bezier.models import CNNDrawerV1, CNNDrawerV2
from neural_bezier.models.cnn_drawer_v2 import Up


class CNNDrawerTest(unittest.TestCase):
    def test_forward(self):
        model = CNNDrawerV1()
        batch_size = 32
        parameters = 10  # number of params from draw function
        x = torch.zeros(batch_size, parameters, dtype=torch.float32)
        y = model.forward(x)
        channels = 1
        width = height = 128
        size = (batch_size, channels, width, height)
        self.assertEqual(size, y.size())

    def test_forward2(self):
        model = CNNDrawerV2()
        batch_size = 32
        parameters = 10  # number of params from draw function
        x = torch.zeros(batch_size, parameters, dtype=torch.float32)
        y = model.forward(x)
        channels = 1
        width = height = 128
        size = (batch_size, channels, width, height)
        self.assertEqual(size, y.size())

    def test_up(self):
        up = Up(4, 8)
        batch_size = 32
        x = torch.randn(batch_size, 4, 8, 8)
        y = up(x)
        self.assertEqual((batch_size, 8, 16, 16), y.shape)
