import unittest

from neural_bezier.dataset import BezierDataset


class BezierDatasetTest(unittest.TestCase):
    def test_iter_seed(self):
        dataset = iter(BezierDataset(length=16, size=128))
        params, img = next(dataset)
        self.assertEqual((10,), params.shape)
        self.assertEqual((128, 128), img.shape)
