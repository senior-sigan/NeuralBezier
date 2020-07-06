import unittest

import numpy as np
from numpy.random.mtrand import RandomState

from neural_bezier.dataset import BezierDataset, gen_random_canvases


class BezierDatasetTest(unittest.TestCase):
    def test_iter_seed(self):
        dataset1 = iter(BezierDataset(size=256, seed=42))
        dataset2 = iter(BezierDataset(size=256, seed=42))
        for i in range(10):
            params1, img1 = next(dataset1)
            params2, img2 = next(dataset2)
            self.assertEqual((10,), params1.shape)
            self.assertEqual((10,), params2.shape)
            self.assertEqual((256, 256), img1.shape)
            self.assertEqual((256, 256), img2.shape)
            self.assertTrue((np.abs(img1 - img2) < 0.0001).all())
            self.assertTrue((np.abs(params1 - params2) < 0.0001).all())
            self.assertTrue(((params1 >= 0) | (params1 <= 1)).all())
