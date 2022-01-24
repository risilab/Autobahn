"""Subclasses of torch geometric dataset with better behaviour. """

import functools
import hashlib
import itertools
import os
import pickle
import shutil

from typing import Any, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.multiprocessing
import torch_geometric.data
import torch_geometric.datasets
import torch_geometric.loader
import tqdm

from autobahn.transform import PathData


def _hash(objs: Sequence[Any]) -> bytes:
    m = hashlib.sha256()

    for x in objs:
        m.update(pickle.dumps(x))

    return m.hexdigest()


def _dataset_to_dict(dataset) -> Dict[str, np.ndarray]:
    """Converts a tuple of data and slices from a `InMemoryDataset` to a dictionary of numpy arrays."""
    data, slices = dataset
    data_class = type(data)
    data = data.to_dict()

    merged = {}

    for k, v in data.items():
        merged['data_' + k] = v.numpy()

    for k, v in slices.items():
        merged['slices_' + k] = v.numpy()

    merged['_data_class'] = np.frombuffer(pickle.dumps(data_class), dtype=np.byte)

    return merged


def _load_dataset(path, mmap_mode=None, as_torch_tensors=True) -> Tuple[torch_geometric.data.Data, Dict[str, torch.Tensor]]:
    data  = np.load(path, mmap_mode=mmap_mode)
    data_class = pickle.loads(data['_data_class'])

    data_dict = {}
    slices_dict = {}

    if as_torch_tensors:
        def convert_tensor(x):
            return torch.from_numpy(x)
    else:
        def convert_tensor(x):
            return x

    for k, v in data.items():
        if k == '_data_class':
            continue

        group, key = k.split('_', 1)

        if group == 'data':
            data_dict[key] = convert_tensor(v)
        elif group == 'slices':
            slices_dict[key] = convert_tensor(v)
        else:
            raise ValueError('Unknown key prefix {} for key {}'.format(group, k))

    return data_class.from_dict(data_dict), slices_dict


class TensorSliceDataset(torch.utils.data.Dataset):
    """Thin wrapper over a pair of data and slices viewed as a dataset."""
    def __init__(self, data, slices: Dict[str, torch.Tensor]):
        self.data = data
        self.slices = slices

    def __getitem__(self, idx):
        data = self.data.__class__()

        if hasattr(data, '__num_nodes__'):
            data.num_nodes = data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = int(slices[idx]), int(slices[idx + 1])

            if torch.is_tensor(item):
                s = [slice(None) for _ in range(len(item.shape))]
                s[data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        return data


    def __len__(self):
        return len(next(iter(self.slices.values()))) - 1


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.base[idx])

    def __len__(self):
        return len(self.base)
    

def _process_dataset_pretransform_torch(process_fn, data, num_workers=None):
    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))

    dataloader = torch_geometric.loader.DataLoader(
        dataset=TransformDataset(data, process_fn),
        batch_size=16, shuffle=False,
        num_workers=num_workers, prefetch_factor=1)

    result = tqdm.tqdm(dataloader)
    return list(itertools.chain.from_iterable(x.clone().to_data_list() for x in result))


class ZINC(torch_geometric.data.InMemoryDataset):
    """ZINC dataset (see `torch_geometric.datasets.ZINC`).

    This dataset improves on the original dataset with better handling of multiple pre-transforms
    by splitting the cache location and using multiprocessing to accelerate pre_transform.
    """
    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    data: PathData

    def __init__(self, root, subset=False, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.subset = subset
        super(ZINC, self).__init__(root, transform, pre_transform, pre_filter)
        if split not in ('train', 'val', 'test'):
            raise ValueError('Invalid split {}'.format(split))
        path = os.path.join(self.processed_dir, f'{split}.npz')
        self.data, self.slices = _load_dataset(path)


    @functools.cached_property
    def atom_features_cardinality(self) -> Tuple[int, ...]:
        return tuple(len(torch.unique(self.data.x[:, i])) for i in range(self.data.x.shape[1]))

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        basename = os.path.join(self.root, name, 'processed')
        param_hash = _hash((self.pre_transform, self.pre_filter))
        return os.path.join(basename, param_hash[:8])

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_file_names(self):
        return ['train.npz', 'val.npz', 'test.npz']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = torch_geometric.data.download_url(self.url, self.root)
        torch_geometric.data.extract_zip(path, self.root)
        os.rename(os.path.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            torch_geometric.data.download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            print('Processing split {}'.format(split))
            with open(os.path.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                print('Loading raw data for split {}'.format(split))
                mols = pickle.load(f)

            indices = range(len(mols))

            if self.subset:
                with open(os.path.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                data_list.append(data)

            tg_dataset = self.collate(data_list)

            if self.pre_transform is not None:
                dataset = TensorSliceDataset(*tg_dataset)

                print('Processing data for split {}'.format(split))
                data_list = _process_dataset_pretransform_torch(self.pre_transform, dataset)
                tg_dataset = self.collate(data_list)

            ds_dict = _dataset_to_dict(tg_dataset)
            np.savez(os.path.join(self.processed_dir, f'{split}.npz'), **ds_dict)


class PygGraphPropPredDataset(torch_geometric.data.InMemoryDataset):
    """Re-implementation of `ogb.graphproppred.PygGraphPropPredDataset` which supports faster
    loading and multiprocessing for pre-transform computation.
    """
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 

        self.name = name ## original name, e.g., ogbg-molhiv
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if os.path.exists(os.path.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = os.path.join(root, self.dir_name)
            
            from ogb.graphproppred import dataset_pyg
            master = pd.read_csv(os.path.join(os.path.dirname(dataset_pyg.__file__), 'master.csv'), index_col = 0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict
        
        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if os.path.isdir(self.root) and (not os.path.exists(os.path.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = _load_dataset(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = os.path.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(os.path.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(os.path.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(os.path.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def processed_dir(self):
        basename = os.path.join(self.root, 'processed')
        param_hash = _hash((self.pre_transform, self.pre_filter))
        return os.path.join(basename, param_hash[:8])

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.npz'

    def download(self):
        from ogb.utils.url import decide_download, download_url, extract_zip
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(os.path.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        from ogb.io.read_graph_pyg import read_graph_pyg

        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(
            self.raw_dir, add_inverse_edge=add_inverse_edge,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files,
            binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(os.path.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(os.path.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(os.path.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)


        if self.pre_transform is not None:
            dataset = self.collate(data_list)

            data_list = _process_dataset_pretransform_torch(
                self.pre_transform, TensorSliceDataset(*dataset))

        dataset = self.collate(data_list)
        np.savez(self.processed_paths[0], **_dataset_to_dict(dataset))
