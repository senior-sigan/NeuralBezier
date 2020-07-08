import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from neural_bezier.experiments import get_experiment


def train(config: DictConfig):
    pl.seed_everything(config.seed)
    lm = get_experiment(config)
    trainer = Trainer(
        fast_dev_run=False,
        max_epochs=config.training.epochs,
        checkpoint_callback=ModelCheckpoint(
            filepath='checkpoints_{epoch:02d}-{val_loss:.2f}',
            save_last=True,
            monitor='val_loss',
            mode='min'
        ),
        logger=TensorBoardLogger(
            save_dir='logs'
        ),
        gpus=config.training.gpus,
        distributed_backend='ddp'
    )
    trainer.fit(lm)
