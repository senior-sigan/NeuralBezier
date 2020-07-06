import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from neural_bezier.cnn_model import CNNDrawer
from neural_bezier.dataset import BezierDataset, random_canvas


class CNNDrawerLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n_demo_images = 16
        self.batch_size = 8
        self.num_workers = 4
        self.image_size = 128

        self.lr = 0.001

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
        return Adam(self.model.parameters(), self.lr)

    def train_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = BezierDataset(
            length=self.batch_size * 8,
            image_transform=transform,
            size=self.image_size, seed=42
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

        params = torch.from_numpy(params)
        gen_canvas = self.forward(params)

        for i in range(self.n_demo_images):
            self.logger.experiment.add_image(f"canvas_{i}/generated", gen_canvas[i], self.current_epoch,
                                             dataformats='CHW')
            self.logger.experiment.add_image(f"canvas_{i}/origin", canvas[i], self.current_epoch,
                                             dataformats='HW')


def main():
    model = CNNDrawerLM()
    trainer = Trainer(
        val_check_interval=2,
        checkpoint_callback=ModelCheckpoint(
            filepath='output/checkpoints'
        ),
        logger=TensorBoardLogger(
            save_dir='output/logs',

        )
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
