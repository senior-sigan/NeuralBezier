from typing import Dict

from omegaconf import DictConfig
from torch import nn

from neural_bezier.models.cnn_drawer_v1 import CNNDrawerV1
from neural_bezier.models.cnn_drawer_v2 import CNNDrawerV2
from neural_bezier.models.patch_gan_discriminator import PatchGanDiscriminator

MODELS: Dict[str, nn.Module] = {
    'CNNDrawerV2': lambda config: CNNDrawerV2(),
    'CNNDrawerV1': lambda config: CNNDrawerV1(),
    'PatchGanDiscriminator': lambda config: PatchGanDiscriminator(
        in_channels1=config.experiment.discriminator.in_channels1,
        in_channels2=config.experiment.discriminator.in_channels2
    )
}


def get_model(name: str, config: DictConfig) -> nn.Module:
    factory = MODELS.get(name)
    if factory is None:
        raise NotImplementedError(f"Model {name} is not found. May be you mean one of them: {GENERATORS.keys()}")

    return factory(config)
