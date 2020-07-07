import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from neural_bezier.cnn_model import CNNDrawer
from neural_bezier.dataset import BezierDataset, random_canvas
from omegaconf import DictConfig


class CNNDrawerLM(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        # self.hparams = hparams
        self.n_demo_images = hparams.n_demo_images
        self.n_batches_per_epoch = hparams.n_batches_per_epoch
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.image_size = hparams.image_size
        self.dataset_seed = hparams.dataset_seed

        self.lr = hparams.learning_rate

        self.model = CNNDrawer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        return {'loss': loss, 'log': {
            'train_loss': loss
        }}

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.model.parameters(), self.lr)
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = BezierDataset(
            length=self.batch_size * self.n_batches_per_epoch,
            image_transform=transform,
            size=self.image_size, seed=self.dataset_seed
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def on_epoch_end(self) -> None:
        params = []
        canvas = []
        for i in range(self.n_demo_images):
            params_, canvas_ = random_canvas(size=self.image_size)
            params.append(params_)
            canvas.append(canvas_)
        params = np.stack(params, axis=0)
        canvas = np.stack(canvas)

        params = torch.from_numpy(params).to(device=self.device)
        gen_canvas = self.forward(params).detach().cpu().numpy()

        for i in range(self.n_demo_images):
            grid = np.concatenate([gen_canvas[i].squeeze(), canvas[i]], axis=1)
            self.logger.experiment.add_image(f"canvas_{i}", grid, self.current_epoch, dataformats='HW')


def train(config: DictConfig):
    model = CNNDrawerLM(config)
    trainer = Trainer(
        max_epochs=config.epochs,
        checkpoint_callback=ModelCheckpoint(
            filepath='checkpoints',
            save_last=True
        ),
        logger=TensorBoardLogger(
            save_dir='logs'
        ),
        gpus=config.gpus,
        distributed_backend='ddp'
    )
    trainer.fit(model)
