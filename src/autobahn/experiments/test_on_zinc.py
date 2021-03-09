"""Module for testing performance on zinc datasets."""

import dataclasses
import pickle

from typing import Dict, Optional

import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning
import pytorch_lightning.callbacks
import tqdm


from torch_geometric.transforms import Compose
from autobahn.transform import Pathifier, Cyclifier
from autobahn.experiments.data import ZincDataModule
from autobahn.experiments import combo_models, utils, train_combo_on_zinc
from .pretrained_checkpoints import download_pretrained_checkpoint


@dataclasses.dataclass
class ZincTestingConfiguration:
    """Configuration for testing Zinc-based models.

    Attributes
    ----------
    folder : str
        Path to pytorch-lightning folder containing trained models.
    batch_size : int
        Batch size to use for evaluation
    data : ZincDatasetConfiguration
        Dataset configuration to use if you wish to override original configuration.
        Only used if `override_data_config` is set to True.
    override_data_config : bool
        If True, indicates that given data config should be used. Otherwise, uses data config
        from checkpoint
    mixed_precision : bool
        Use mixed-precision evaluation
    """
    checkpoint: Optional[str] = None
    batch_size: int = 1024
    data: combo_models.ZincDatasetConfiguration = dataclasses.field(default_factory=combo_models.ZincDatasetConfiguration)
    override_data_config: bool = False
    mixed_precision: bool = False


cs = ConfigStore.instance()
cs.store(name='config', node=ZincTestingConfiguration)

def test_with_checkpoint(config: ZincTestingConfiguration, checkpoint_path: str, dataset_cache: Dict[str, ZincDataModule], trainer: pytorch_lightning.Trainer):
    model: combo_models.ZincPathAndCycleModel = combo_models.ZincPathAndCycleModel.load_from_checkpoint(
        checkpoint_path, map_location='cpu')

    saved_config = model.hparams

    if config.override_data_config:
        data_config = config.data
    else:
        data_config = saved_config.data

    data_config.data_folder = hydra.utils.to_absolute_path(data_config.data_folder)

    path_lengths, cycle_lengths = train_combo_on_zinc._expand_to_default(saved_config.model.path_lengths, saved_config.model.cycle_lengths)
    dataset_key = tuple(path_lengths), tuple(cycle_lengths)

    if dataset_key not in dataset_cache:
        transform = Compose([Pathifier(list(path_lengths)), Cyclifier(list(cycle_lengths))])
        dataset = ZincDataModule(data_config, transform=transform, batch_size=config.batch_size)
        dataset.prepare_data()
        dataset.setup()
    else:
        dataset = dataset_cache[dataset_key]

    valid_result = trainer.test(model, test_dataloaders=[dataset.val_dataloader()], verbose=False)[0]
    test_result = trainer.test(model, test_dataloaders=[dataset.test_dataloader()], verbose=False)[0]

    return {
        'valid': valid_result,
        'test': test_result
    }


@hydra.main(config_name='config')
def train_with_conf(config: ZincTestingConfiguration):
    kwargs = {}

    if config.mixed_precision:
        kwargs['precision'] = 16

    trainer = pytorch_lightning.Trainer(gpus=1, **kwargs)

    if config.checkpoint is None:
        config.checkpoint = download_pretrained_checkpoint(config.data, hydra.utils.to_absolute_path('./checkpoints'))

    checkpoints_by_version = utils.list_checkpoints(hydra.utils.to_absolute_path(config.checkpoint))

    dataset_cache = {}

    results = {}

    for version, ckpt_path in tqdm.tqdm(checkpoints_by_version.items()):
        print(f'Testing checkpoint {ckpt_path}')
        results[version] = test_with_checkpoint(config, ckpt_path, dataset_cache, trainer)
        print(results[version])

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    train_with_conf()

