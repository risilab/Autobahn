"""Script which computes the statistics for the various datasets."""

import functools
import pickle
import os

import torch
import tqdm
import autobahn.statistics
import autobahn.datasets
import autobahn.transform


def compute_dataset_mean_statistics(dataset, progress=False):
    max_samples = 20000

    global_stats, path_stats, cycle_stats = autobahn.statistics.compute_dataset_statistics(
        dataset, max_sample=max_samples,
        progress=progress,
        statistics_fn=functools.partial(
            autobahn.statistics.path_and_cycle_statistics,
            max_path_length=8,
            max_cycle_length=8))

    total_number = min(len(dataset), max_samples)

    return {
        'global': {k: torch.sum(v).item() / total_number for k, v in global_stats.items()},
        'path': {k: torch.sum(v).item() / total_number for k, v in path_stats.items()},
        'cycle': {k: torch.sum(v).item() / total_number for k, v in cycle_stats.items()}
    }


def _ogb_factory_fn(root, name):
    return autobahn.datasets.PygGraphPropPredDataset(name, root=root, transform=autobahn.transform.OGBTransform())


DATASETS = {
    'zinc': (functools.partial(autobahn.datasets.ZINC, subset=False), 'zinc'),
    'zinc_subset': (functools.partial(autobahn.datasets.ZINC, subset=True), 'zinc'),
    'molpcba': (functools.partial(_ogb_factory_fn, name='ogbg-molpcba'), 'ogb'),
    'muv': (functools.partial(_ogb_factory_fn, name='ogbg-molmuv'), 'ogb'),
    'hiv': (functools.partial(_ogb_factory_fn, name='ogbg-molhiv'), 'ogb'),
}


def format_results_latex(results):
    def _format_number(num, digits):
        return '\\num{' + '{{:.{}f}}'.format(digits).format(num) + '}'

    def _format_line(v):
        values = []
        values.append(_format_number(v['global']['num_nodes'], 1))
        values.append(_format_number(v['global']['num_edges'], 1))

        for i in range(3, 8 + 1):
            values.append(_format_number(v['path'].get(i, 0) / 2, 1))

        for i in [5, 6]:
            values.append(_format_number(v['cycle'].get(i, 0), 2))

        return '&'.join(values)

    lines = []

    for k, v in results.items():
        lines.append(k + '&' + _format_line(v) + '\n')

    return ''.join(lines)


def main():
    dataset_path = '/mnt/ceph/users/wzhou/projects/autobahn/data'

    results = {}

    for k, (dataset_factory, path_suffix) in tqdm.tqdm(DATASETS.items()):
        ds = dataset_factory(os.path.join(dataset_path, path_suffix))
        result = compute_dataset_mean_statistics(ds, progress=True)
        results[k] = result

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
