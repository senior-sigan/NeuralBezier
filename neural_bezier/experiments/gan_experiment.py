import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import AttributeDict
from torch.utils.data import DataLoader
from torchvision import transforms

from neural_bezier.dataset import BezierDataset, BezierDatasetStatic
from neural_bezier.models import get_model
from neural_bezier.utils import weights_init_normal


class GanExperiment(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.n_demo_images = config.training.n_demo_images
        self.n_batches_per_epoch = config.dataset.n_batches_per_epoch
        self.n_batches_val = config.dataset.n_batches_val
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.image_size = config.dataset.image_size

        self.lambda_pixel = config.experiment.loss.lambda_pixel

        self.generator = get_model(config.experiment.generator.name, config)
        self.discriminator = get_model(config.experiment.discriminator.name, config)

        weights_init_normal(self.generator)
        weights_init_normal(self.discriminator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator.forward(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:  # generator
            return self.training_step_generator(batch)

        if optimizer_idx == 1:  # discriminator
            return self.training_step_discriminator(batch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        gen_out = self.generator(x)
        desc_fake_out = self.discriminator(x1=gen_out, x2=y)
        desc_real_out = self.discriminator(x1=y, x2=y)

        g_valid = torch.ones_like(desc_fake_out)
        adv_loss = F.binary_cross_entropy(desc_fake_out, g_valid)
        l1_loss = F.l1_loss(gen_out, y)
        g_loss = adv_loss + self.lambda_pixel * l1_loss

        d_fake = torch.zeros_like(desc_fake_out)
        d_real = torch.ones_like(desc_real_out)
        fake_loss = F.binary_cross_entropy(desc_fake_out, d_fake)
        real_loss = F.binary_cross_entropy(desc_real_out, d_real)
        d_loss = (fake_loss + real_loss) * 0.5

        l2_loss = F.mse_loss(gen_out, y)

        return {
            'val_l1_loss': l1_loss,
            'val_l2_loss': l2_loss,
            'val_d_loss': d_loss,
            'val_g_loss': g_loss,
        }

    def validation_epoch_end(self, outputs):
        val_d_loss = torch.stack([x['val_d_loss'] for x in outputs]).mean()
        val_g_loss = torch.stack([x['val_g_loss'] for x in outputs]).mean()
        val_l1_loss = torch.stack([x['val_l1_loss'] for x in outputs]).mean()
        val_l2_loss = torch.stack([x['val_l2_loss'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_d_loss': val_d_loss,
            'val_g_loss': val_g_loss,
            'val_l1_loss': val_l1_loss,
            'val_l2_loss': val_l2_loss,
        }
        return {
            **tensorboard_logs,
            'val_loss': val_g_loss,
            'log': tensorboard_logs,
        }

    def training_step_generator(self, batch):
        x, y = batch

        gen_out = self.generator(x)
        desc_out = self.discriminator(x1=gen_out, x2=y)

        valid = torch.ones_like(desc_out)

        adv_loss = F.binary_cross_entropy(desc_out, valid)
        l1_loss = F.l1_loss(gen_out, y)
        g_loss = adv_loss + self.lambda_pixel * l1_loss
        log = {
            'g_loss': g_loss,
            'l1_loss': l1_loss,
        }
        return {
            'loss': g_loss,
            'log': log,
            'progress_bar': log,
        }

    def training_step_discriminator(self, batch):
        x, y = batch

        desc_out = self.discriminator(x1=y, x2=y)
        valid = torch.ones_like(desc_out)
        real_loss = F.binary_cross_entropy(desc_out, valid)

        # generate fake image and detach to not optimize
        gen_out = self.generator(x).detach()
        fake_out = self.discriminator(x1=gen_out, x2=y)
        fake = torch.zeros_like(desc_out)
        fake_loss = F.binary_cross_entropy(fake_out, fake)

        d_loss = (real_loss + fake_loss) * 0.5
        log = {
            'd_loss': d_loss
        }
        return {
            'loss': d_loss,
            'log': log,
            'progress_bar': log
        }

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

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.config.experiment.generator.optimizer.learning_rate,
            betas=self.config.experiment.generator.optimizer.betas,
        )
        opt_d = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.config.experiment.discriminator.optimizer.learning_rate,
            betas=self.config.experiment.generator.optimizer.betas,
        )
        return [opt_g, opt_d], []

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

    @property
    def hparams(self) -> AttributeDict:
        return OmegaConf.to_container(self.config, resolve=True)
