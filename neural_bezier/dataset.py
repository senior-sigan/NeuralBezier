from typing import Callable, Optional, Tuple

import numpy as np
from numpy.random.mtrand import RandomState
from torch.utils.data import Dataset

from neural_bezier.draw import draw_params


def random_canvas(size: int = 128, random_state: RandomState = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is None:
        random_state = RandomState()
    params = random_state.uniform(low=0, high=1, size=10).astype(np.float32)
    canvas = draw_params(params, size=size)
    canvas = canvas.astype(np.float32) / 255.0
    return params, canvas


class BezierDataset(Dataset):
    def __init__(self, length: int, image_transform: Optional[Callable] = None,
                 params_transform: Optional[Callable] = None,
                 size: int = 128, seed: int = 42) -> None:
        self.size = size
        self.rs = RandomState(seed=seed)
        self.image_transform = image_transform
        self.params_transform = params_transform
        assert length > 0, 'Length should be > 0'
        self.length = length

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        params, canvas = random_canvas(size=self.size, random_state=self.rs)
        if self.params_transform is not None:
            params = self.params_transform(params)
        if self.image_transform is not None:
            canvas = self.image_transform(canvas)
        return params, canvas

    def __len__(self) -> int:
        return self.length
