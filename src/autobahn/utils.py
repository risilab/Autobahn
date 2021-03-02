from torch_geometric.utils import to_networkx
from itertools import combinations
from networkx.algorithms.components import is_connected
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def mol_from_data(data):
    mol = Chem.RWMol()

    x = data.x if data.x.dim() == 1 else data.x[:, 0]
    for z in x.tolist():
        mol.AddAtom(Chem.Atom(z))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr
    bond_type = bond_type if bond_type.dim() == 1 else bond_type[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        assert bond >= 1 and bond <= 4
        mol.AddBond(i, j, bonds[bond - 1])

    return mol.GetMol()


def get_subgraph_list(data, n):
    """
    Gets all subgroups of size n.

    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    k_paths : list of k-paths
    """
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    N = G.number_of_nodes()
    indices = list(range(N))

    subgraphs = []
    for node_set in combinations(indices, n):
        G_sub = G.subgraph(node_set)
        if is_connected(G_sub):
            subgraphs.append(node_set)
    return subgraphs


def get_path_list(data, n):
    """
    Get path list of 
    """
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    N = G.number_of_nodes()
    all_paths = []
    for i in range(N):
        all_paths += _find_paths_from_i(G, i, n-1)
    return all_paths


def _find_paths_from_i(G, u, n):
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


def _get_edge_loc(candidate_edge, edge_index):
    edge_index = edge_index.float()
    delta = torch.linalg.norm(edge_index - candidate_edge, dim=1, ord=1)
    loc = torch.nonzero(delta == 0)
    if len(loc) > 0:
        return int(loc)
    else:
        return -1
