"""
Cycle decomposition code, adapted from pytorch geometric.
"""
import torch
import torch_geometric.data
import torch_geometric.utils
from itertools import chain
from copy import deepcopy
from autobahn.utils import mol_from_data, bonds, _get_edge_loc

from typing import List


try:
    import rdkit.Chem as Chem
    from rdkit.Chem.rdchem import BondType
except ImportError:
    Chem = None
    BondType = None


class _PathSet:
    """Class which implements a system to check whether path super-set is contained in """
    _paths : List[str]

    def __init__(self):
        self._paths = []

    def try_update(self, path) -> bool:
        path_str = ':'.join(str(x) for x in path)

        for p in self._paths:
            if path_str in p:
                return False

        self._paths.append(path_str)
        return True


def remove_non_maximal_paths(paths: List[List[int]]):
    maximal_path_set = _PathSet()
    maximal_paths = [[] for _ in paths]

    for i in reversed(range(len(paths))):
        for path in paths[i]:
            if maximal_path_set.try_update(path):
                maximal_paths[i].append(path)

    return maximal_paths


def get_path_list(data: torch_geometric.data.Data, n: int) -> List[int]:
    """Computes all paths of length n in the given graph."""
    G = torch_geometric.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)
    N = G.number_of_nodes()
    all_paths = []
    for i in range(N):
        all_paths.extend(_find_paths_from_i(G, i, n-1))
    return all_paths


def get_paths_by_length(data: torch_geometric.data.Data, max_length: int) -> List[List[int]]:
    paths_by_length = [[], [], []]
    num_paths = 0

    for i in range(3, max_length + 1):
        pl = get_path_list(data, i)
        paths_by_length.append(pl)
        num_paths += len(pl)

    return paths_by_length


def _find_paths_from_i(G, u: int, n: int) -> List[int]:
    """
    Code for finding paths, taken from https://stackoverflow.com/a/28103735.
    """
    if n == 0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in _find_paths_from_i(G, neighbor, n-1):
            if u not in path:
                paths.append([u]+path)
    return paths


def get_paths_by_length(data: torch_geometric.data.Data, min_length: int, max_length: int) -> List[List[int]]:
    paths_by_length = [[] for _ in range(min_length)]

    for i in range(min_length, max_length + 1):
        pl = get_path_list(data, i)
        paths_by_length.append(pl)

    return paths_by_length


def path_decomposition(data, min_length: int=3, max_length: int=4, remove_non_maximal=False, use_mol_path_features=True):
    """Path Decomposition code

    Parameters
    ----------
    data
        The molecule from which to extract paths
    max_length : int
        The maximal length of the paths to be extracted
    remove_non_maximal : bool, optional
        If True, indicates that paths that are not maximal (i.e. contained in another path)
        are not included in the returned paths.
    """
    paths_by_length, feats_by_length = get_paths_by_lengths_with_features(data, min_length, max_length, remove_non_maximal, use_mol_path_features)
    a2c_indexes, x_cycles = _create_index_sets(paths_by_length, feats_by_length)
    # For all k, get the indexes converting the atom list to list of k-cycles
    return a2c_indexes, paths_by_length, x_cycles


def get_paths_by_lengths_with_features(data, min_length: int=3, max_length: int=4, remove_non_maximal=False, use_mol_path_features=True):  # noqa E252
    """
    Extracts all paths of length between min_length and max_length, ordered by their length.
    """
    # Extract all the cycles
    paths_by_length = get_paths_by_length(data, min_length, max_length)

    if remove_non_maximal:
        paths_by_length = remove_non_maximal_paths(paths_by_length)

    if use_mol_path_features:
        feats_by_length = [get_mol_path_features(data, path_list) for path_list in paths_by_length]
    else:
        feats_by_length = [get_basic_path_features(data, path_list) for path_list in paths_by_length]
    return paths_by_length, feats_by_length


def get_path_list(data: torch_geometric.data.Data, n: int) -> List[int]:
    """Computes all paths of length n in the given graph."""
    G = torch_geometric.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)
    N = G.number_of_nodes()
    all_paths = []
    for i in range(N):
        all_paths.extend(_find_paths_from_i(G, i, n-1))
    return all_paths


def _find_paths_from_i(G, u: int, n: int) -> List[int]:
    """
    Code for finding paths, taken from https://stackoverflow.com/a/28103735.
    """
    if n == 0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in _find_paths_from_i(G, neighbor, n-1):
            if u not in path:
                paths.append([u]+path)
    return paths


def get_mol_path_features(data, paths):
    mol = mol_from_data(data)

    path_features = []
    for path in paths:
        previous_node = path[0]
        edge_features = []
        for node in path[1:]:
            potential_bond = mol.GetBondBetweenAtoms(node, previous_node)
            bond_type = bonds.index(potential_bond.GetBondType())
            edge_features.append(bond_type + 1)
            previous_node = node
        edge_features.append(0)
        path_features.append(edge_features)
    return path_features


def get_basic_path_features(data, paths):
    path_features = []
    for path in paths:
        edge_features = [1 for node in path[1:]]
        edge_features.append(0)
        path_features.append(edge_features)
    return path_features


def get_nx_bond_features(data, paths):
    path_features = []
    for path in paths:
        edge_features = [1 for node in path[1:]]
        edge_features.append(0)
        path_features.append(edge_features)
    return path_features


def get_basic_cycle_features(data, paths):
    path_features = []
    for path in paths:
        edge_features = [1 for node in path]
        path_features.append(edge_features)
    return path_features


def cycle_decomposition(data, max_length=20):
    r"""Cycle Decomposition code

    Args:
        mol (rdkit.Chem.Mol): A :obj:`rdkit` molecule.
        return_vocab (bool, optional): If set to :obj:`True`, will return an
            identifier for each clique (ring, bond, bridged compounds, single).
            (default: :obj:`False`)
    """
    cycles = []
    cycle_features = []
    if Chem is None:
        raise ImportError('Package `rdkit` could not be found.')
    mol = mol_from_data(data)
    # Extract all the cycles
    for x in Chem.GetSymmSSSR(mol):
        c, f = _canonicalize_cycle(x, mol)
        cycles.append(c)
        cycle_features.append(f)
    return _format_cycles(cycles, cycle_features, max_length)


def _format_cycles(cycles, cycle_features, max_length):
    """
    Format a list of cycles into appropriate_indices
    """
    # Sort the cycles by their length: element i contains cycles of length i
    cycles_by_length = [[] for i in range(max_length+1)]
    feats_by_length = [[] for i in range(max_length+1)]
    for c, a in zip(cycles, cycle_features):
        if len(c) > max_length:
            continue
        cycles_by_length[len(c)].append(c)
        feats_by_length[len(c)].append(a)
    n_cycles = torch.tensor([len(c) for c in cycles_by_length])

    a2c_indexes, x_cycles = _create_index_sets(cycles_by_length, feats_by_length)
    return a2c_indexes, cycles, n_cycles, x_cycles


def _create_index_sets(structures_by_length, feats_by_length):
    a2c_indexes = []
    for c_list in structures_by_length:
        row = torch.tensor(list(chain.from_iterable(c_list)))
        col = torch.arange(len(row))
        a2c_idx = torch.stack([row, col], dim=0).to(torch.long)
        a2c_indexes.append(a2c_idx)

    # Get list of features
    x_cycles = []
    for x_list in feats_by_length:
        row = torch.tensor(list(chain.from_iterable(x_list)))
        x_cycles.append(row.to(torch.long))
    return a2c_indexes, x_cycles


def cycles_from_paths(data, paths, path_features=None, max_length=20):
    """
    Extracts cycles from a list of paths
    """
    edge_indices = data.edge_index.transpose(0, 1).float()  # easier accessing.
    cycles = []
    cycle_features = []
    for n, paths_n in enumerate(paths):  # Loop over cycle lengths
        cycles_n = []
        cycles_n_feats = []
        for i, path_i in enumerate(paths_n):  # loop over cycles of length n
            # Cycle is a path whose ends are connected.
            edge_candidate = torch.tensor([path_i[-1], path_i[0]]).float()
            edge_loc = _get_edge_loc(edge_candidate, edge_indices)

            if edge_loc > -1:  # Tripped if we find an edge.
                cycles_n.append(path_i)
                if path_features is not None:
                    path_feats = path_features[n][i]
                    link_feat = path_features[edge_loc]
                    cycles_n_feats.append(path_feats[:-1] + [link_feat])
                else:
                    cycles_n_feats.append([1.] * n)

        cycles.append(cycles_n)
        cycles.append(cycles_n_feats)
    cycles = _format_cycles(cycles, cycle_features, max_length=20)
    return cycles, cycle_features


def _get_atom2clique_index(cliques, num_atoms):
    """
    Takes a list of cliques and converts them into input and output indices
    for use in pytorch_scatter.
    """
    atom2clique = [[] for i in range(num_atoms)]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)
    return atom2clique


def _canonicalize_cycle(cycle_indices, mol):
    """
    Takes the indices in a cycle and canonicalizes them, ensuring that
    subsequent indices in the list are connected.
    """
    if BondType is None:
        raise ImportError('Package `rdkit` could not be found.')

    bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    cycle_indices = deepcopy(list(cycle_indices))  # Be safe!

    # Add first atom
    corrected_cycle = [cycle_indices[0]]
    cycle_features = []
    current_index = cycle_indices[0]
    cycle_indices.remove(current_index)

    while len(cycle_indices) > 0:
        successfully_added_atom = False
        # Find a connected atom and append it to the list.
        for i in cycle_indices:
            potential_bond = mol.GetBondBetweenAtoms(current_index, i)
            if potential_bond is not None:
                corrected_cycle.append(i)
                cycle_indices.remove(i)
                current_index = i
                successfully_added_atom = True
                # Bond type, arbitrarily appended to first atom.  This
                # slightly breaks permutation equivariance, but not more so than
                # kekulization.
                bond_type = bonds.index(potential_bond.GetBondType())
                cycle_features.append(bond_type)
                break

        if not successfully_added_atom:
            raise RuntimeError("No contigious cycle found from atom %d to atoms" % current_index, cycle_indices)

    # Add bond type between start and end.
    last_bond = mol.GetBondBetweenAtoms(corrected_cycle[0], corrected_cycle[-1])
    if last_bond is None:
        raise Exception
    else:
        cycle_features.append(bonds.index(last_bond.GetBondType()))

    return corrected_cycle, cycle_features


def _get_all_aromatic(c, mol):
    """
    Checks that every atom in the cycle described is aromatic.
    """
    atom_is_aromatic = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in c]
    all_are_aromatic = all(atom_is_aromatic)
    return [all_are_aromatic] * len(c)
