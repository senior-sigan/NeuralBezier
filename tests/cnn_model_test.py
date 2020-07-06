import unittest

import torch

from neural_bezier.cnn_model import CNNDrawer


class CNNDrawerTest(unittest.TestCase):
    def test_forward(self):
        model = CNNDrawer()
        batch_size = 32
        parameters = 10  # number of params from draw function
        x = torch.zeros(batch_size, parameters, dtype=torch.float32)
        y = model.forward(x)
        channels = 1
        width = height = 128
        size = (batch_size, channels, width, height)
        self.assertEqual(size, y.size())
