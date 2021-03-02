"""Utilities to compute statistics on the datasets."""

import collections
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import torch.utils.data
import torch_geometric.data
import tqdm

from autobahn import decompositions, utils


def _update_append_dictionary(accumulator: dict, update: dict):
    for k, v in update.items():
        accumulator.setdefault(k, []).append(v.clone())


def _make_values_to_array(d: dict):
    return {k: torch.cat(v) for k, v in d.items()}



class _TransformShuffleDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, transform=None, limit: int=None, generator=None):
        length = min(limit, len(base_dataset)) if limit is not None else len(base_dataset)

        self._base_dataset = base_dataset
        self._permutation = torch.randperm(len(base_dataset), generator=generator).narrow(0, 0, length)
        self._transform = transform

    def __getitem__(self, idx):
        base_idx = int(self._permutation[idx])
        value = self._base_dataset[base_idx]
        if self._transform is not None:
            value = self._transform(value)
        return value

    def __len__(self):
        return len(self._permutation)


def compute_dataset_statistics(dataset: Mapping[int, torch_geometric.data.Data],
                               max_sample: int=10000, progress: bool=False,
                               statistics_fn=None):
    """Computes statistics on the given dataset.

    Parameters
    ----------
    dataset : Mapping[int, Data]
        A dataset of graphs to analyze
    max_sample : int
        Subsample the dataset to this number of observations to compute statistics.
    progress : bool
        If True, indicates that a progress bar should be printed.
    """
    if statistics_fn is None:
        statistics_fn = path_and_cycle_statistics

    stats = {
        'num_nodes': [],
        'num_edges': []
    }

    path_stats = {}
    cycle_stats = {}

    dataset = _TransformShuffleDataset(dataset, statistics_fn, max_sample)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=20, shuffle=False,
        num_workers=0 if max_sample < 2000 else 8)

    if progress:
        dataloader = tqdm.tqdm(dataloader)

    for s, p, c in dataloader:
        _update_append_dictionary(stats, s)
        _update_append_dictionary(path_stats, p)
        _update_append_dictionary(cycle_stats, c)

    stats = _make_values_to_array(stats)
    path_stats = _make_values_to_array(path_stats)
    cycle_stats = _make_values_to_array(cycle_stats)

    return stats, path_stats, cycle_stats


def global_statistics(data: torch_geometric.data.Data) -> Dict[str, int]:
    return {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
    }


def path_statistics(data: torch_geometric.data.Data, max_length: int=6) -> Dict[int, int]:
    num_paths = {}

    for i in range(3, max_length + 1):
        paths = decompositions.get_path_list(data, i)
        num_paths[i] = len(paths)

    return collections.Counter(num_paths)


def cycle_statistics(data: torch_geometric.data.Data, max_length: int=6) -> Dict[int, int]:
    _, cycles, _, _ = decompositions.cycle_decomposition(data, max_length)

    cycle_length, cycle_count = np.unique([len(c) for c in cycles], return_counts=True)

    return collections.Counter(dict(zip(cycle_length, cycle_count)))


def path_and_cycle_statistics(data: torch_geometric.data.Data, max_path_length=6, max_cycle_length=6) -> Tuple[Dict[str, int], Dict[int, int], Dict[int, int]]:
    """Computes global, cycle and path statistics for the given data sample."""
    global_stat = global_statistics(data)
    path_stat = path_statistics(data, max_path_length)
    cycle_stat = cycle_statistics(data, max_cycle_length)
    return global_stat, path_stat, cycle_stat
