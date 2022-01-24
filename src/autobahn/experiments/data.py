import dataclasses
import functools

from typing import Tuple

import torch
import torch_geometric.loader
import pytorch_lightning
from autobahn.datasets import ZINC, PygGraphPropPredDataset


#  ~~~~~~~~~~~~~~~~~~~~ Basec Classes ~~~~~~~~~~~~~~~~~~~~  #

@dataclasses.dataclass
class DatasetConfiguration:
    data_folder: str = './data/'
    num_workers: int = 4


class DataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config: DatasetConfiguration, transform=None, batch_size: int=32):  # noqa E252
        super().__init__()

        self.data_folder = config.data_folder
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = config.num_workers

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage=None):
        raise NotImplementedError

    def _make_dataloader(self, dataset, shuffle):
        return torch_geometric.loader.DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self._test_dataset, shuffle=False)


#  ~~~~~~~~~~~~~~~~~~~~ Zinc ~~~~~~~~~~~~~~~~~~~~  #
@dataclasses.dataclass
class ZincDatasetConfiguration(DatasetConfiguration):
    data_folder: str = './data/zinc'
    use_subset: bool = True


class ZincDataModule(DataModule):
    """Data module encapsulating the zinc dataset.
    """
    def __init__(self, config: ZincDatasetConfiguration, transform=None, batch_size: int=32):  # noqa E252
        super().__init__(config, transform, batch_size)
        self.subset = config.use_subset

    def setup(self, stage=None):
        self._train_dataset = ZINC(self.data_folder, subset=self.subset, split='train', pre_transform=self.transform)
        self._val_dataset = ZINC(self.data_folder, subset=self.subset, split='val', pre_transform=self.transform)
        self._test_dataset = ZINC(self.data_folder, subset=self.subset, split='test', pre_transform=self.transform)

    @property
    def atom_feature_cardinality(self):
        return self._train_dataset.atom_features_cardinality


#  ~~~~~~~~~~~~~~~~~~~~ OGB ~~~~~~~~~~~~~~~~~~~~ #
@dataclasses.dataclass
class OGBDatasetConfiguration(DatasetConfiguration):
    num_tasks: int = -1
    data_folder: str = './data/ogb'
    data_name: str = 'ogbg-molhiv'


class OGBDataModule(DataModule):
    """Data module encapsulating an OGB Dataset
    """
    def __init__(self, config: OGBDatasetConfiguration, transform=None, batch_size: int=32):  # noqa E252
        super().__init__(config, transform, batch_size)
        self.data_name = config.data_name
        self.data_folder = config.data_folder
        self.num_tasks = None

    def setup(self, stage=None):
        dataset = PygGraphPropPredDataset(self.data_name, self.data_folder, pre_transform=self.transform)
        split_idx = dataset.get_idx_split()
        self._train_dataset = dataset[split_idx['train']]
        self._val_dataset = dataset[split_idx['valid']]
        self._test_dataset = dataset[split_idx['test']]
        self.num_tasks = dataset.num_tasks

    @functools.cached_property
    def atom_feature_cardinality(self) -> Tuple[int, ...]:
        """Cardinality (number of distinct values) of each atom feature.""" 
        # Note: because feature values are not contiguous, we compute
        # maximum so that it integrates with nn.Embedding
        atom_features = self._train_dataset.data.x
        return tuple(torch.max(atom_features, dim=0).values.add_(1).tolist())
