"""Module for testing performance on OGB datasets."""

import dataclasses
import pickle

from typing import Dict, Optional

import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning
import pytorch_lightning.callbacks
import tqdm


from torch_geometric.transforms import Compose
from autobahn.transform import Pathifier, Cyclifier, OGBTransform
from autobahn.experiments.data import OGBDataModule
from autobahn.experiments import combo_models, utils
from .pretrained_checkpoints import download_pretrained_checkpoint


@dataclasses.dataclass
class OGBTestingConfiguration:
    checkpoint: Optional[str] = None
    batch_size: int = 256
    data: combo_models.OGBDatasetConfiguration = dataclasses.field(default_factory=combo_models.OGBDatasetConfiguration)
    override_data_config: bool = False
    mixed_precision: bool = False


cs = ConfigStore.instance()
cs.store(name='config', node=OGBTestingConfiguration)

def test_with_checkpoint(config: OGBTestingConfiguration, checkpoint_path: str, dataset_cache: Dict[str, OGBDataModule], trainer: pytorch_lightning.Trainer):
    model: combo_models.OGBPathAndCycleModel = combo_models.OGBPathAndCycleModel.load_from_checkpoint(
        checkpoint_path, map_location='cpu')

    saved_config = model.hparams

    if config.override_data_config:
        data_config = config.data
    else:
        data_config = saved_config.data

    data_config.data_folder = hydra.utils.to_absolute_path(data_config.data_folder)

    dataset_key = tuple(saved_config.model.path_lengths), saved_config.model.path_depth

    if dataset_key not in dataset_cache:
        transform = Compose([OGBTransform(), Pathifier(list(saved_config.model.path_lengths)), Cyclifier(list(saved_config.model.cycle_lengths))])
        dataset = OGBDataModule(data_config, transform=transform, batch_size=config.batch_size)
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
def train_with_conf(config: OGBTestingConfiguration):
    kwargs = {}

    if config.mixed_precision:
        kwargs['precision'] = 16

    trainer = pytorch_lightning.Trainer(gpus=1, **kwargs)

    if config.checkpoint is None:
        config.checkpoint = download_pretrained_checkpoint(config.data, './checkpoints')
    checkpoints_by_version = utils.list_checkpoints(config.checkpoint)

    dataset_cache = {}

    results = {}

    progress = tqdm.tqdm(checkpoints_by_version.items())

    for version, ckpt_path in progress:
        progress.write(f'Testing checkpoint {ckpt_path}')
        results[version] = test_with_checkpoint(config, ckpt_path, dataset_cache, trainer)
        progress.write(str(results[version]))

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    train_with_conf()

