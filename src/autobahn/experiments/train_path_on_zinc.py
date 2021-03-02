"""
Trains a path model.
TODO: Integrate with tensorboard, better file structure, etc.
"""

import os
import dataclasses
import random

import omegaconf
import torch
import torch.optim
import pytorch_lightning
import torch

from typing import List, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autobahn.pathnet import PathNetWithMP
from autobahn.transform import Pathifier
from autobahn.experiments.data import ZincDatasetConfiguration, ZincDataModule

from autobahn.experiments import utils

# from tqdm import tqdm


@dataclasses.dataclass
class PathModelConfiguration:
    num_channels: int = 64
    num_layers: int = 2
    path_lengths: List[int] = dataclasses.field(default_factory=lambda: [3, 4, 5, 6])
    path_depth: int = 2
    dropout: float = 0.0


@dataclasses.dataclass
class OptimConfiguration:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    min_learning_rate: float = 5e-5
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 20
    gradient_clip_norm: Optional[float] = None


@dataclasses.dataclass
class PathTrainingConfiguration:
    model: PathModelConfiguration = dataclasses.field(default_factory=PathModelConfiguration)
    optim: OptimConfiguration = dataclasses.field(default_factory=OptimConfiguration)
    data: ZincDatasetConfiguration = dataclasses.field(default_factory=ZincDatasetConfiguration)
    max_epochs: int = 1000
    batch_size: int = 64
    output_folder: Optional[str] = None
    seed: Optional[int] = None
    mixed_precision: bool = False
    num_gpus: int = 1


class PathModel(pytorch_lightning.LightningModule):
    hparams: PathTrainingConfiguration

    def __init__(self, config: PathTrainingConfiguration):
        super().__init__()

        self.save_hyperparameters(config)
        self.model = PathNetWithMP(
            config.model.num_channels, 1, config.model.num_layers,
            path_lengths=config.model.path_lengths,
            path_depth=config.model.path_depth,
            dropout=config.model.dropout)

    def forward(self, data):
        return self.model.forward(data)

    def _compute_loss(self, batch):
        y = batch.y
        out = self.forward(batch)
        loss = torch.nn.functional.l1_loss(out.squeeze(), y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr_ratio = self.hparams.batch_size / 64

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hparams.optim.learning_rate * lr_ratio,
            weight_decay=self.hparams.optim.weight_decay)

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=self.hparams.optim.lr_decay_factor,
            patience=self.hparams.optim.lr_decay_patience,
            min_lr=self.hparams.optim.min_learning_rate * lr_ratio)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'reduce_on_plateau': True,
                'monitor': 'val_loss',
            }}


def train_with_conf(config: PathTrainingConfiguration):
    trainer = utils.make_trainer(config)

    torch.manual_seed(config.seed)

    dataset = ZincDataModule(config.data, transform=Pathifier(), batch_size=config.batch_size)

    dataset.prepare_data()
    dataset.setup()

    fixture = PathModel(config)

    trainer.fit(fixture, dataset)


def main():
    conf = omegaconf.OmegaConf.structured(PathTrainingConfiguration)
    cli_conf = omegaconf.OmegaConf.from_cli()
    conf = omegaconf.OmegaConf.merge(conf, cli_conf)
    train_with_conf(conf)


if __name__ == '__main__':
    main()
