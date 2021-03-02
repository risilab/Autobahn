"""
Model for a gnn that operates on paths
"""

from typing import Dict, Sequence, Tuple, Optional

import torch
import torch.autograd.profiler
import torch.nn
import torch.nn.functional as F
from torch.nn import ModuleList, ModuleDict
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import scatter
from torch_geometric.nn import GINEConv
from autobahn import blocks


PathActivationCollection = Dict[str, torch.Tensor]
AtomToPathMapping = Dict[str, torch.Tensor]


class PathEmbeddingLayer(torch.nn.Module):
    """Convenience layer which regroups a number of embeddings.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, path_lengths: Sequence[int]):
        super().__init__()

        self.embeddings = torch.nn.ModuleDict({
            str(k): torch.nn.Embedding(num_embeddings, embedding_dim)
            for k in path_lengths
        })

    def forward(self, x_paths: PathActivationCollection) -> PathActivationCollection:
        return {
            k: self.embeddings[k](v) for k, v in x_paths.items()
        }

    def __call__(self, x_paths: PathActivationCollection) -> PathActivationCollection:
        return super(PathEmbeddingLayer, self).__call__(x_paths)


class PathConvolutionLayer(torch.nn.Module):
    """This module implements a layer which convolves over paths in the network."""
    def __init__(self, num_channels: int, path_depth: int, path_lengths: Sequence[int],
                 dropout: float=0.0, bottleneck_blocks: bool=False, cyclic: bool=False):
        """Creates a new PathConvolutionLayer with the given parameters.

        Parameters
        ----------
        num_channels : int
            Number of channels to use.
        path_depth : int
            Depth of path transformation.
        dropout : float
            Use dropout between layers.
        bottleneck_blocks : bool
            Whether to use bottleneck residual blocks.
        cyclic : bool
            If True, indicates that the path is cyclic and should use cyclic padding and symmetric convolution.
        """
        super().__init__()

        self.path_lengths = tuple(path_lengths)
        self.path_keys = [str(k) for k in path_lengths]

        self.path_blocks = ModuleDict()
        self.atom_to_path_linear = ModuleDict()
        self.path_to_atom_linear = ModuleDict()

        if cyclic:
            padding_mode = 'circular'
            use_symmetric_convs = True
        else:
            padding_mode = 'zeros'
            use_symmetric_convs = False

        # Construct layers that convolve over paths and move to/from atom activations
        for k, key in zip(self.path_lengths, self.path_keys):
            self.path_blocks[key] = blocks.PathBlock(
                num_channels, k, num_resid_blocks=path_depth,
                padding_mode=padding_mode, use_symmetric_convs=use_symmetric_convs,
                dropout=dropout, bottleneck_blocks=bottleneck_blocks)
            self.atom_to_path_linear[key] = Linear(num_channels, num_channels)
            self.path_to_atom_linear[key] = Linear(num_channels, num_channels)

    def _mix_into_paths(self, x: torch.Tensor, x_paths: Dict[str, torch.Tensor], atom_to_path_map) -> Dict[str, torch.Tensor]:
        """Transfer atom activations into path activations.

        Parameters
        ----------
        x : torch.Tensor
            Tensor representing the atom activations.
        x_paths : Dict[str, torch.Tensor]
            Tensors representing the path activations, indexed by path length.
        atom_to_path_map : Dict[str, torch.Tensor]
            Tensors representing the mapping between atoms and paths.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tensors representing mixed-in path activations, indexed by path length.
        """
        x_out = {}
        # for key in self.path_keys:
        for key in x_paths.keys():
            # Gather data for that path length
            row, col = atom_to_path_map[key]
            lin_k = self.atom_to_path_linear[key]
            x_path_k = x_paths[key]
            x_sc_k = scatter(x[row], col, dim=0, dim_size=x_path_k.size(0), reduce='mean')
            x_out[key] = x_path_k + F.relu(lin_k(x_sc_k))
        return x_out

    def _mix_into_atoms(self, x: torch.Tensor, x_paths: PathActivationCollection, atom_to_path_map) -> torch.Tensor:
        x_out = x
        for key in self.path_keys:
            # Gather data for that path length
            row, col = atom_to_path_map[key]
            lin_k = self.path_to_atom_linear[key]
            x_path_k = x_paths[key]
            x_sc_k = scatter(x_path_k[col], row, dim=0, dim_size=x.size(0), reduce='mean')
            x_out = x_out + F.relu(lin_k(x_sc_k))
        return x_out

    def _apply_pathblocks(self, x_paths: PathActivationCollection) -> PathActivationCollection:
        result = {}

        for k, v in x_paths.items():
            result[k] = self.path_blocks[k](v)

        return result

    def forward(self, x: torch.Tensor, x_paths: PathActivationCollection,
                atom_to_path_map: AtomToPathMapping) -> Tuple[torch.Tensor, PathActivationCollection]:
        """Computes the path convolution layer.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing the atom-level activations.
        x_paths : PathActivationCollection
            A dictionary of tensors representing the path-level activations, indexed by path length.
        atom_to_path_map : AtomToPathMapping
            A dictionary of tensors representing the atom to path mappings.

        Returns
        -------
        torch.Tensor
            The new activations on the atoms
        PathActivationCollection
            The new activations on the paths
        """
        with torch.autograd.profiler.record_function('mix_in'):
            x_paths = self._mix_into_paths(x, x_paths, atom_to_path_map)

        with torch.autograd.profiler.record_function('conv'):
            x_paths = self._apply_pathblocks(x_paths)

        with torch.autograd.profiler.record_function('mix_out'):
            x = self._mix_into_atoms(x, x_paths, atom_to_path_map)

        return x, x_paths

    def __call__(self, x, x_paths, atom_to_path_map) -> Tuple[torch.Tensor, PathActivationCollection]:
        return super(PathConvolutionLayer, self).__call__(x, x_paths, atom_to_path_map)


class PathNetWithMP(torch.nn.Module):
    path_layers : Sequence[PathConvolutionLayer]

    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.0,
                 path_lengths=None, path_depth=1, inter_message_passing=True):
        super(PathNetWithMP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing

        if path_lengths is None:
            path_lengths = [3, 4]
        self.path_lengths = path_lengths
        self.path_keys = [str(k) for k in path_lengths]

        self._build_base_mp_layers(hidden_channels, num_layers)

        self.lin = Linear(hidden_channels, out_channels)

        self.path_layers = ModuleList(
            PathConvolutionLayer(hidden_channels, path_depth, path_lengths, dropout)
            for _ in range(num_layers))

        self.path_embedding = PathEmbeddingLayer(6, hidden_channels, path_lengths)


    def _build_base_mp_layers(self, hidden_channels, num_layers):
        self.atom_encoder = blocks.AtomEncoder(hidden_channels)
        self.bond_encoders = ModuleList()
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        for _ in range(num_layers):
            self.bond_encoders.append(blocks.BondEncoder(hidden_channels))
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))
        self.atom_lin = Linear(hidden_channels, hidden_channels)

    def forward(self, data):
        x = self.atom_encoder(data.x.squeeze())
        x_paths = {k: data['x_clique_%s_path' % k] for k in self.path_keys}
        atom_to_path_mapping = {k: data['atom2clique_%s_path' % k] for k in self.path_keys}

        # Apply initial embedding of path features.
        x_paths = self.path_embedding(x_paths)

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, edge_attr)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.inter_message_passing:
                # Blocks used in this layer
                x, x_paths = self.path_layers[i](x, x_paths, atom_to_path_mapping)

        # Aggregate output and run an MLP on top.
        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_lin(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return x


class PathAndCycleNet(torch.nn.Module):
    """
    Network that operates on paths only (no message passing)
    """
    path_layers : Sequence[PathConvolutionLayer]
    cycle_layers : Sequence[PathConvolutionLayer]

    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float=0.0,
                 atom_feature_cardinality: Optional[Sequence[int]]=None,
                 path_lengths: Optional[Sequence[int]]=None, cycle_lengths: Optional[Sequence[int]]=None,
                 path_depth: int=1, cycle_depth: int=1, bottleneck_blocks: bool=False):
        """Creates a new path-and-cycle model.

        Parameters
        ----------
        hidden_channels : int
            Number of channels to use in the inner layers.
        out_channels : int
            Number of output channels
        num_layers : int
            Number of layers to use.
        dropout : float
            Dropout parameter to use.
        atom_feature_cardinality : Sequence[int]
            Cardinality (number of values) of each atom feature.
        path_lengths : Sequence[int]
            Sequence of lengths of paths included in the model.
        cycle_lengths : Sequence[int]
            Sequence of lengths of cycles included in the model.
        path_depth : int
            Depth of each path convolution block.
        cycle_depth : int
            Depth of each cycle convolution block.
        bottleneck_blocks : bool
            Whether to use bottleneck blocks in the path convolutions.
        """

        super(PathAndCycleNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        if path_lengths is None:
            path_lengths = [3, 4, 5]
        if cycle_lengths is None:
            cycle_lengths = [5, 6]

        self.path_lengths = tuple(path_lengths)
        self.path_keys = [str(k) for k in path_lengths]
        self.cycle_lengths = tuple(cycle_lengths)
        self.cycle_keys = [str(k) for k in cycle_lengths]

        self.path_embedding = PathEmbeddingLayer(6, hidden_channels, path_lengths)
        self.cycle_embedding = PathEmbeddingLayer(4, hidden_channels, cycle_lengths)

        self.path_layers = torch.nn.ModuleList(
            PathConvolutionLayer(hidden_channels, path_depth, path_lengths, dropout, bottleneck_blocks)
            for _ in range(num_layers))
        self.cycle_layers = torch.nn.ModuleList(
            PathConvolutionLayer(hidden_channels, cycle_depth, cycle_lengths, dropout, bottleneck_blocks, cyclic=True)
            for _ in range(num_layers))

        # Bulid layers for input and output
        self.atom_encoder = blocks.AtomEncoder(hidden_channels, atom_feature_cardinality)
        self.atom_lin = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)


    def forward(self, data):
        x = self.atom_encoder(data.x.squeeze())
        x_paths = {k: data['x_clique_%s_path' % k] for k in self.path_keys}
        x_cycles = {k: data['x_clique_%s_cycle' % k] for k in self.cycle_keys}
        a2ps = {k: data['atom2clique_%s_path' % k] for k in self.path_keys}
        a2cs = {k: data['atom2clique_%s_cycle' % k] for k in self.cycle_keys}

        # Apply initial embedding of path features.
        x_paths = self.path_embedding(x_paths)
        x_cycles = self.cycle_embedding(x_cycles)

        for i in range(self.num_layers):
            # Send atom activations into their path/cycle activations
            with torch.autograd.profiler.record_function(f'path_{i}'):
                x, x_paths = self.path_layers[i](x, x_paths, a2ps)
            with torch.autograd.profiler.record_function(f'cycle_{i}'):
                x, x_cycles = self.cycle_layers[i](x, x_cycles, a2cs)

        with torch.autograd.profiler.record_function('readout'):
            # Aggregate output and run an MLP on top.
            x = scatter(x, data.batch, dim=0, reduce='mean')
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.atom_lin(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.lin(x)
        return x
