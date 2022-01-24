import dataclasses
from typing import List, Optional, Sequence

import ogb.graphproppred
import pytorch_lightning
import torch
import torch.nn.functional
import torch.optim
import torch_geometric
import torch_geometric.data
import torchmetrics

from autobahn.pathnet import PathAndCycleNet
from autobahn.experiments.data import OGBDatasetConfiguration
from autobahn.experiments.data import ZincDatasetConfiguration
from . import utils


@dataclasses.dataclass
class ModelConfiguration:
    """Configuration for model architecture.

    Attributes
    ----------
    atom_feature_cardinality : List[int]
        A list of the cardinality of each (categorical) atom feature,
        used to create the atom feature embedding layers.
    num_channels : int
        Number of channels to use throughout the ntework.
    num_layers : int
        Number of layer groups to use throughout the network.
    path_lengths : List[int]
        The lengths of paths to take into account in the model.
    path_depth : int
        Number of layers in each path convolution block.
    cycle_lengths : List[int]
        The lengths of the cycles to take into account in the model.
    cyclle_depth : int
        Number of layers in each cycle convolution block.
    dropout : float
        If non-zero, the dropout proportion to apply during training
    bottleneck : bool
        If True, indicates that the residual blocks should use a bottleneck
        architecture.
    """
    atom_feature_cardinality: Optional[List[int]] = None
    num_channels: int = 128
    num_layers: int = 5
    path_lengths: List[int] = dataclasses.field(default_factory=lambda: [3, 4, 5])
    cycle_lengths: List[int] = dataclasses.field(default_factory=lambda: [5, 6])
    path_depth: int = 2
    cycle_depth: int = 2
    dropout: float = 0.0
    bottleneck: bool = True


@dataclasses.dataclass
class OptimConfiguration:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    min_learning_rate: float = 5e-5
    lr_warmup_epochs: float = 10
    lr_decay_factor: float = 0.5
    lr_decay_milestones: List[float] = dataclasses.field(default_factory=lambda: [150, 300])
    gradient_clip_norm: Optional[float] = None


@dataclasses.dataclass
class OGBTrainingConfiguration(utils.TrainerConfiguration):
    model: ModelConfiguration = dataclasses.field(default_factory=ModelConfiguration)
    optim: OptimConfiguration = dataclasses.field(default_factory=OptimConfiguration)
    data: OGBDatasetConfiguration = dataclasses.field(default_factory=OGBDatasetConfiguration)


@dataclasses.dataclass
class ZincTrainingConfiguration(utils.TrainerConfiguration):
    model: ModelConfiguration = dataclasses.field(default_factory=ModelConfiguration)
    optim: OptimConfiguration = dataclasses.field(default_factory=OptimConfiguration)
    data: ZincDatasetConfiguration = dataclasses.field(default_factory=ZincDatasetConfiguration)


class OGBPathAndCycleModel(pytorch_lightning.LightningModule):
    hparams: OGBTrainingConfiguration

    def __init__(self, config: OGBTrainingConfiguration):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = PathAndCycleNet(
            config.model.num_channels, config.data.num_tasks, config.model.num_layers,
            atom_feature_cardinality=config.model.atom_feature_cardinality,
            path_lengths=config.model.path_lengths,
            cycle_lengths=config.model.cycle_lengths,
            path_depth=config.model.path_depth,
            cycle_depth=config.model.cycle_depth,
            dropout=config.model.dropout,
            bottleneck_blocks=config.model.bottleneck)
        self.metric = OGBMetric(config.data.data_name, compute_on_step=False)

    def forward(self, data):
        return self.model.forward(data)

    def _compute_loss(self, batch):
        y = batch.y
        raw_out = self.forward(batch)
        raw_y = y.to(torch.float)

        # Only train on elements with extant data.
        mask = ~torch.isnan(raw_y)
        y = raw_y[mask]
        out = raw_out[mask]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        return {'loss': loss, 'preds': raw_out, 'targets': raw_y}

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)['loss']
        self.log('train_loss', loss, prog_bar=False, batch_size=self.hparams.batch_size)
        return self._compute_loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.metric(loss['preds'], loss['targets'])
        self.log('val_loss', loss['loss'], prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_epoch_end(self, outputs):
        self.log(self.metric.eval_metric, self.metric.compute())

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.metric(loss['preds'], loss['targets'])
        self.log('test_loss', loss['loss'], sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log(self.metric.eval_metric, self.metric.compute())

    def configure_optimizers(self):
        lr_ratio = self.hparams.batch_size / 64

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hparams.optim.learning_rate * lr_ratio,
            weight_decay=self.hparams.optim.weight_decay)

        scheduler = utils.WarmupStepScheduler(
            optimizer, milestones=self.hparams.optim.lr_decay_milestones,
            warmup_epochs=self.hparams.optim.lr_warmup_epochs,
            gamma=self.hparams.optim.lr_decay_factor
        )
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if isinstance(batch, torch_geometric.data.Batch):
            return batch.to(device)
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @property
    def eval_metric(self) -> str:
        """Name of the metric to track for evaluation / checkpointing."""
        return self.metric.eval_metric


class OGBMetric(torchmetrics.Metric):
    preds: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(self, data_name: str, compute_on_step: bool=True):
        super().__init__(compute_on_step=compute_on_step)

        self.evaluator = ogb.graphproppred.Evaluator(data_name)

        self.add_state('preds', default=[], dist_reduce_fx=None)
        self.add_state('target', default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        try:
            # some metrics error if there are no positive labels
            # we simply return 0.0 if that happens
            return self.evaluator.eval({
                'y_pred': preds,
                'y_true': target
            })[self.evaluator.eval_metric]
        except RuntimeError:
            return 0.0

    @property
    def eval_metric(self):
        return self.evaluator.eval_metric


class ZincPathAndCycleModel(pytorch_lightning.LightningModule):
    hparams: ZincTrainingConfiguration

    def __init__(self, config: ZincTrainingConfiguration):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = PathAndCycleNet(
            config.model.num_channels, 1, config.model.num_layers,
            atom_feature_cardinality=config.model.atom_feature_cardinality,
            path_lengths=config.model.path_lengths,
            cycle_lengths=config.model.cycle_lengths,
            path_depth=config.model.path_depth,
            cycle_depth=config.model.cycle_depth,
            dropout=config.model.dropout,
            bottleneck_blocks=config.model.bottleneck)

    def forward(self, data):
        return self.model.forward(data)

    def _compute_loss(self, batch):
        y = batch.y
        out = self.forward(batch)
        loss = torch.nn.functional.l1_loss(out.squeeze(), y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('test_loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        lr_ratio = self.hparams.batch_size / 64

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hparams.optim.learning_rate * lr_ratio,
            weight_decay=self.hparams.optim.weight_decay)

        scheduler = utils.WarmupStepScheduler(
            optimizer, milestones=self.hparams.optim.lr_decay_milestones,
            warmup_epochs=self.hparams.optim.lr_warmup_epochs,
            gamma=self.hparams.optim.lr_decay_factor
        )

        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if isinstance(batch, torch_geometric.data.Batch):
            return batch.to(device)
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
