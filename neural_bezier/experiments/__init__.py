from typing import Dict

import pytorch_lightning as pl
from omegaconf import DictConfig

from neural_bezier.experiments.basic_experiment import BasicExperiment
from neural_bezier.experiments.gan_experiment import GanExperiment

EXPERIMENTS: Dict[str, pl.LightningModule] = {
    'basic': BasicExperiment,
    'gan': GanExperiment
}


def get_experiment(config: DictConfig) -> pl.LightningModule:
    key = config.experiment.name
    factory = EXPERIMENTS.get(key)
    if factory is None:
        raise NotImplementedError(f"Experiment {key} is not found. May be you mean one of them: {EXPERIMENTS.keys()}")

    return factory(config)
