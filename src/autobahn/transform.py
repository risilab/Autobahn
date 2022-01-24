import dataclasses
from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import tree_decomposition, to_dense_adj, to_dense_batch
from autobahn.decompositions import cycle_decomposition, path_decomposition
from autobahn.utils import mol_from_data


class JunctionTreeData(Data):
    def __inc__(self, key, item, *args, **kwargs):
        if key == 'tree_edge_index':
            return self.x_clique.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(JunctionTreeData, self).__inc__(key, item, *args, **kwargs)


class JunctionTree(object):
    def __call__(self, data):
        mol = mol_from_data(data)
        out = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = out

        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique

        return data


class OGBTransform(object):
    """This class converts zero-based indices to one-based indices for atom and bond types
    found in the OGB datasets.

    Taken from the HIMP library
    """
    def __call__(self, data):
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data


KEPT_CYCLES = [4, 5, 6, 7, 8]
KEPT_PATHS = [3, 4, 5, 6, 7, 8]
A2C_KEYS = tuple(["atom2clique_%d_cycle" % k for k in KEPT_CYCLES])
A2C_PATH_KEYS = tuple(["atom2clique_%d_path" % k for k in KEPT_PATHS])


class PathData(Data):
    def __inc__(self, key, item, *args, **kwargs):
        if key in A2C_PATH_KEYS:
            n_atoms = self.x.size(0)
            total_path_positions = item.size(1)
            return torch.tensor([[n_atoms], [total_path_positions]])
        elif key in A2C_KEYS:
            n_atoms = self.x.size(0)
            total_cycle_positions = item.size(1)
            return torch.tensor([[n_atoms], [total_cycle_positions]])
        else:
            return super(PathData, self).__inc__(key, item, *args, **kwargs)

    def __cat_dim__(self, key, item, *args, **kwargs):
        # print(key, key in A2C_KEYS, A2C_KEYS)
        if key in A2C_PATH_KEYS:
            return -1
        elif key in A2C_KEYS:
            return -1
        else:
            return super(PathData, self).__cat_dim__(key, item, *args, **kwargs)


@dataclasses.dataclass(frozen=True)
class Cyclifier(object):
    kept_cycles: Tuple[int, ...] = dataclasses.field(default_factory=lambda: tuple(KEPT_CYCLES))

    def __call__(self, data):
        # mol = mol_from_data(data)
        a2c_indexes, cycles, n_cycles, x_list = cycle_decomposition(data)

        data = PathData(**{k: v for k, v in data})
        # data.smiles = Chem.MolToSmiles

        for k in KEPT_CYCLES:
            a2c_k = a2c_indexes[k]
            x_k = x_list[k]
            # print(x_k)
            data.__setitem__("atom2clique_%d_cycle" % k, a2c_k)
            data.__setitem__("x_clique_%d_cycle" % k, x_k)
        data.n_cycles = n_cycles
        return data


@dataclasses.dataclass(frozen=True)
class Pathifier:
    kept_paths: Tuple[int, ...] = dataclasses.field(default_factory=lambda: tuple(KEPT_PATHS))

    def __call__(self, data):
        a2c_indexes, paths_by_length, x_list = path_decomposition(data, min(*self.kept_paths), max(*self.kept_paths))
        n_paths = sum([len(i) for i in paths_by_length])

        # Repurposing cycle data since the
        data = PathData(**{k: v for k, v in data})
        # data.smiles = Chem.MolToSmiles

        for k in self.kept_paths:
            a2c_k = a2c_indexes[k]
            x_k = x_list[k]
            data.__setitem__("atom2clique_%d_path" % k, a2c_k)
            data.__setitem__("x_clique_%d_path" % k, x_k)
        data.n_paths = n_paths
        return data


@dataclasses.dataclass(frozen=True)
class CycleAndPathifier:
    """
    Evaluates the cycles by connecting paths.
    """
    kept_paths: Tuple[int, ...] = dataclasses.field(default_factory=lambda: tuple(KEPT_PATHS))

    def __call__(self, data):
        a2path_indexes, paths, n_paths, x_list = path_decomposition(data, self.kept_paths[-1])
        a2cycle_indexes, cycles, n_cycles, x_list = cycles_from_paths(data, paths)

        # Repurposing cycle data since the
        data = PathData(**{k: v for k, v in data})

        for k in self.kept_paths:
            data.__setitem__("atom2clique_%d_path" % k, a2path_indexes[k])
            data.__setitem__("x_clique_%d_path" % k, x_list[k])
        data.n_paths = n_paths
        return data


class Densify(object):
    def __init__(self, max_size=38, num_atom_classes=21):
        self.max_size = max_size
        self.num_atom_classes = num_atom_classes

    def __call__(self, data):
        x = data.x if data.x.dim() == 1 else data.x[:, 0]

        adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr,
                           max_num_nodes=self.max_size)
        x_node, x_node_sizes = to_dense_batch(x, data.batch, max_num_nodes=self.max_size)
        data.num_atoms = x_node_sizes

        data.dense_adj = _one_hotify_adj(adj).float()
        try:
            data.dense_x = torch.nn.functional.one_hot(x_node, self.num_atom_classes).float()
        except:  # noqa: E722  # Ignore bare except since the error is raised anyway.
            print(self.num_atom_classes, ", but saw: ", torch.max(x_node))
            raise
        return data


def _one_hotify_adj(adj, max_num=3):
    B, N, __ = adj.shape
    onehotified_adj = torch.zeros((B, N, N, max_num)).float()
    nonzero_locs = torch.nonzero(adj, as_tuple=True)
    nonzero_vals = adj[nonzero_locs]
    new_indices = nonzero_locs + (nonzero_vals.long()-1,)
    onehotified_adj[new_indices] = 1.
    return onehotified_adj
