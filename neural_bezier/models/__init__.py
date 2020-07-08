from typing import Dict

from omegaconf import DictConfig
from torch import nn

from neural_bezier.models.cnn_drawer_v1 import CNNDrawerV1
from neural_bezier.models.cnn_drawer_v2 import CNNDrawerV2

MODELS: Dict[str, nn.Module] = {
    'CNNDrawerV2': CNNDrawerV2,
    'CNNDrawerV1': CNNDrawerV1
}


def get_model(config: DictConfig) -> nn.Module:
    key = config.experiment.model.name
    factory = MODELS.get(key)
    if factory is None:
        raise NotImplementedError(f"Model {key} is not found. May be you mean one of them: {MODELS.keys()}")

    return factory()
