from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from neural_bezier.draw import draw_params


class BezierDataset(Dataset):
    def __init__(self, length: int, image_transform: Optional[Callable] = None,
                 params_transform: Optional[Callable] = None,
                 size: int = 128) -> None:
        self.size = size
        self.image_transform = image_transform
        self.params_transform = params_transform
        assert length > 0, 'Length should be > 0'
        self.length = length

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        params = np.random.random_sample(size=10).astype(np.float32)
        canvas = draw_params(params, size=self.size)
        canvas = canvas.astype(np.float32) / 255.0

        if self.params_transform is not None:
            params = self.params_transform(params)
        if self.image_transform is not None:
            canvas = self.image_transform(canvas)
        return params, canvas

    def __len__(self) -> int:
        return self.length


class BezierDatasetStatic(Dataset):
    def __init__(self, length: int, image_transform: Optional[Callable] = None,
                 params_transform: Optional[Callable] = None,
                 size: int = 128) -> None:
        self.size = size
        self.image_transform = image_transform
        self.params_transform = params_transform
        assert length > 0, 'Length should be > 0'
        self.length = length
        self.params = np.random.random_sample(size=(length, 10)).astype(np.float32)

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        params = self.params[index]
        canvas = draw_params(params, size=self.size)
        canvas = canvas.astype(np.float32) / 255.0

        if self.params_transform is not None:
            params = self.params_transform(params)
        if self.image_transform is not None:
            canvas = self.image_transform(canvas)
        return params, canvas

    def __len__(self) -> int:
        return self.length
