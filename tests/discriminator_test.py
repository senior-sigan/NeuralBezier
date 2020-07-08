import unittest

import torch

from neural_bezier.models.patch_gan_discriminator import Discriminator


class DiscriminatorTest(unittest.TestCase):
    def test_forward(self):
        model = Discriminator(in_channels1=1, in_channels2=1)
        bs = 1
        x1 = torch.rand(bs, 1, 128, 128)
        x2 = torch.rand(bs, 1, 128, 128)
        y = model.forward(x1=x1, x2=x2)
        self.assertEqual((bs, 1, 19, 19), y.shape)
