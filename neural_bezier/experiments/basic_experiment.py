from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from neural_bezier.dataset import BezierDataset, BezierDatasetStatic
from neural_bezier.models import get_model


class BasicExperiment(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.hparams = OmegaConf.to_container(config, resolve=True)
        self.n_demo_images = config.training.n_demo_images
        self.n_batches_per_epoch = config.dataset.n_batches_per_epoch
        self.n_batches_val = config.dataset.n_batches_val
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.image_size = config.dataset.image_size

        self.lr = config.optimizer.learning_rate

        self.model = get_model(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_loss = F.mse_loss(logits, y)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(self.model.parameters(), self.lr)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = BezierDataset(
            length=self.batch_size * self.n_batches_per_epoch,
            image_transform=transform,
            size=self.image_size
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.val_dataset = BezierDatasetStatic(
            length=self.batch_size * self.n_batches_val,
            image_transform=transform,
            size=self.image_size
        )
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def on_epoch_end(self) -> None:
        params = []
        canvas = []

        for i in range(self.n_demo_images):
            params_, canvas_ = self.val_dataset[i]
            params.append(params_)
            canvas.append(canvas_)

        params = np.stack(params, axis=0)
        canvas = torch.stack(canvas)

        params = torch.from_numpy(params).to(device=self.device)
        gen_canvas = self.forward(params).detach().cpu().numpy()
        for i in range(self.n_demo_images):
            grid = np.concatenate([gen_canvas[i].squeeze(), canvas[i].squeeze()], axis=1)
            self.logger.experiment.add_image(f"canvas_{i}", grid, self.current_epoch, dataformats='HW')
