"""

"""
import torch
import torch.nn.functional as F
from torch.nn import Embedding, ModuleList, ModuleDict
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import scatter
from torch_geometric.nn import GINEConv
from autobahn import blocks


class CycleNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.0,
                 used_cycles=None, cycle_depth=1, inter_message_passing=True):
        super(CycleNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing

        if used_cycles is None:
            used_cycles = [5, 6]
        self.used_cycles = used_cycles
        self._cycle_keys = [str(k) for k in used_cycles]

        self._build_base_mp_layers(hidden_channels, num_layers)
        self._build_cycle_layers(hidden_channels, num_layers, cycle_depth)

        self.atom2clique_lins = ModuleList()
        self.clique2atom_lins = ModuleList()
        for _ in range(num_layers):
            self.atom2clique_lins.append(
                Linear(hidden_channels, hidden_channels))
            self.clique2atom_lins.append(
                Linear(hidden_channels, hidden_channels))

        self.clique_lin = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

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

    def _build_cycle_layers(self, hidden_channels, num_layers, cycle_depth):
        """
        Build layers that convolve over the cyclic activations.
        """
        # Construct embedding layers for input
        self.cycle_embedders = ModuleDict()
        for k, key in zip(self.used_cycles, self._cycle_keys):
            self.cycle_embedders[key] = Embedding(4, hidden_channels)

        self.cycle_blocks = ModuleList()
        self.a2c_blocks = ModuleList()
        self.c2a_blocks = ModuleList()

        # Construct layers that convolve over cycles and move to/from atom activations
        for _ in range(num_layers):
            cb_k_dict = ModuleDict()  # Dict of intracycle blocks
            a2c_k_dict = ModuleDict()  # Dict of linears for conversions
            c2a_k_dict = ModuleDict()  # Dict of linears for conversions
            for k, key in zip(self.used_cycles, self._cycle_keys):
                cb_k_dict[key] = blocks.PathBlock(hidden_channels, k,
                                                  num_resid_blocks=cycle_depth,
                                                  dropout=self.dropout)
                a2c_k_dict[key] = Linear(hidden_channels, hidden_channels)
                c2a_k_dict[key] = Linear(hidden_channels, hidden_channels)
            self.cycle_blocks.append(cb_k_dict)
            self.a2c_blocks.append(a2c_k_dict)
            self.c2a_blocks.append(c2a_k_dict)

    def forward(self, data):
        x = self.atom_encoder(data.x.squeeze())
        x_cycles = {k: data['x_clique_%s_cycle' % k] for k in self._cycle_keys}
        a2c_cycles = {k: data['atom2clique_%s_cycle' % k] for k in self._cycle_keys}

        # Apply initial embedding of cycle features.
        x_cycles = self._apply_cycle_layers(x_cycles, self.cycle_embedders)

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, edge_attr)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.inter_message_passing:
                # Blocks used in this layer
                mix_in_cycle_blocks = self.a2c_blocks[i]
                mix_in_atom_blocks = self.c2a_blocks[i]

                x_cycles = self._mix_into_cycles(x, x_cycles, a2c_cycles,
                                                 mix_in_cycle_blocks)
                x_cycles = self._apply_cycle_layers(x_cycles, self.cycle_blocks[i])
                x = self._mix_into_atoms(x, x_cycles, a2c_cycles,
                                         mix_in_atom_blocks)

        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_lin(x)

        # if self.inter_message_passing:
        #     tree_batch = torch.repeat_interleave(data.num_cliques)
        #     x_clique = scatter(x_clique, tree_batch, dim=0, dim_size=x.size(0),
        #                        reduce='mean')
        #     x_clique = F.dropout(x_clique, self.dropout,
        #                          training=self.training)
        #     x_clique = self.clique_lin(x_clique)
        #     x = x + x_clique

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return x

    def _apply_cycle_layers(self, x_dict, layer_dict):
        """
        Applies the layers in the dict to a dict of activations.
        """
        x_out = {}
        for key in self._cycle_keys:
            x_k = x_dict[key]
            x_out[key] = layer_dict[key](x_k)
        return x_out

    def _mix_into_cycles(self, x, x_cycles, a2c, mix_in_layers):
        """
        Transfer x (atom activation) to cycle activation and average it in.
        """
        x_out = {}
        for key in self._cycle_keys:
            # Gather data for that cycle length
            row, col = a2c[key]
            lin_k = mix_in_layers[key]
            x_cycle_k = x_cycles[key]

            x_sc_k = scatter(x[row], col, dim=0, dim_size=x_cycle_k.size(0), reduce='mean')
            x_out[key] = x_cycle_k + F.relu(lin_k(x_sc_k))
        return x_out

    def _mix_into_atoms(self, x, x_cycles, a2c, mix_in_layers):
        """
        Transfer x (atom activation) to cycle activation and average it in.
        """
        x_out = x
        for key in self._cycle_keys:
            # Gather data for that cycle length
            row, col = a2c[key]
            lin_k = mix_in_layers[key]
            x_cycle_k = x_cycles[key]
            x_sc_k = scatter(x_cycle_k[col], row, dim=0, dim_size=x.size(0), reduce='mean')
            x_out = x_out + F.relu(lin_k(x_sc_k))
        return x_out
