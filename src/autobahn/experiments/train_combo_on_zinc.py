"""Module for training path + cycle model on Zinc dataset."""

import hydra
import torch
import torch.optim
import torch


from torch_geometric.transforms import Compose
import autobahn.transform
from autobahn.transform import Pathifier, Cyclifier
from autobahn.experiments.data import ZincDataModule
from autobahn.experiments import combo_models, utils


def _expand_to_default(path_lengths, cycle_lengths):
    if set(path_lengths).issubset(autobahn.transform.KEPT_PATHS):
        path_lengths = autobahn.transform.KEPT_PATHS

    if set(cycle_lengths).issubset(autobahn.transform.KEPT_CYCLES):
        cycle_lengths = autobahn.transform.KEPT_CYCLES

    return list(path_lengths), list(cycle_lengths)


@hydra.main(config_name='config_zinc', config_path='conf')
def train_with_conf(config: combo_models.ZincTrainingConfiguration):
    trainer = utils.make_trainer(config)

    torch.manual_seed(config.seed)

    path_lengths, cycle_lengths = _expand_to_default(config.model.path_lengths, config.model.cycle_lengths)

    transform = Compose([Pathifier(path_lengths), Cyclifier(cycle_lengths)])
    dataset = ZincDataModule(config.data, transform=transform, batch_size=config.batch_size // config.num_gpus)

    dataset.prepare_data()
    dataset.setup()

    config.model.atom_feature_cardinality = dataset.atom_feature_cardinality
    fixture = combo_models.ZincPathAndCycleModel(config)

    trainer.fit(fixture, dataset)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='base_config_zinc', node=combo_models.ZincTrainingConfiguration)
    train_with_conf()
