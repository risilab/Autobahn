"""Module for training path + cycle model on OGB datasets."""

import torch
import torch.optim
import torch

import hydra
from hydra.core.config_store import ConfigStore

from torch_geometric.transforms import Compose
from autobahn.transform import Pathifier, Cyclifier, OGBTransform
from autobahn.experiments.data import OGBDataModule
from autobahn.experiments import combo_models, utils


cs = ConfigStore()
cs.store(name='config_ogb', node=combo_models.OGBTrainingConfiguration)

@hydra.main(config_name='config_ogb', config_path='conf')
def train_with_conf(config: combo_models.OGBTrainingConfiguration):
    utils.ensure_config_defaults(config)

    torch.manual_seed(config.seed)
    transform = Compose([OGBTransform(), Pathifier(list(config.model.path_lengths)), Cyclifier(list(config.model.cycle_lengths))])
    batch_split = max(config.num_gpus, 1)
    dataset = OGBDataModule(config.data, transform=transform, batch_size=config.batch_size // batch_split)

    dataset.prepare_data()
    dataset.setup()

    # Get the number of tasks from the dataset.
    if config.data.num_tasks == -1:
        try:
            config.data.num_tasks = dataset._train_dataset.num_tasks
        except Exception as error:
            raise ValueError("Number of tasks was not assigned, and could not be found in the dataset."
                             + "Original error: " + repr(error))

    config.model.atom_feature_cardinality = dataset.atom_feature_cardinality
    fixture = combo_models.OGBPathAndCycleModel(config)

    trainer = utils.make_trainer(config, monitor_loss=fixture.eval_metric, monitor_mode='max')
    trainer.fit(fixture, dataset)


if __name__ == '__main__':
    train_with_conf()
