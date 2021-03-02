from typing import Optional, Sequence

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single


class SymmetricConv1D(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(SymmetricConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        flipped_weight = torch.flip(self.weight, dims=(2,))
        sym_weight = (self.weight + flipped_weight) / 2.
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            sym_weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, sym_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, stride: int=1, kernel_size: int=3, bias: bool=False, padding_mode="circular",
                 use_symmetric_convs: bool=True, dropout: float=0.0, bottleneck: bool=False):
        """Residual convolution block.

        Parameters
        ----------
        num_channels : int
            Number of input / output channels
        stride : int
            Stride of the convolutions
        kernel_size : int
            Kernel size of the convolutions
        bias : bool
            Whether the convolutional layers should include bias.
        padding_mode : str
            Padding mode for the convolutions.
        use_symmetric_convs : bool
            Whether to enforce symmetric convolutions.
        dropout : float
            Dropout rate to apply
        bottleneck : bool
            If True, indicates to use a bottleneck architecture.
        """

        super(ResidualBlock, self).__init__()
        if use_symmetric_convs:
            conv_layer = SymmetricConv1D
        else:
            conv_layer = nn.Conv1d

        inner_channels = num_channels // 2 if bottleneck else num_channels

        self.conv1 = conv_layer(num_channels, inner_channels, kernel_size, stride=stride,
                                padding=(kernel_size-1)//2, padding_mode=padding_mode,
                                bias=bias)
        self.bn1 = nn.BatchNorm1d(inner_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(inner_channels, num_channels, kernel_size, stride=stride,
                                padding=(kernel_size-1)//2, padding_mode=padding_mode,
                                bias=bias)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class PathBlock(torch.nn.Module):
    def __init__(self, hidden_channels, cycle_size, num_resid_blocks=1, dropout=0.0,
                 use_symmetric_convs=True, padding_mode="circular",
                 bottleneck_blocks=False):
        super(PathBlock, self).__init__()
        self.cycle_size = cycle_size
        self.num_resid_blocks = num_resid_blocks

        self.conv_blocks = nn.ModuleList(
            ResidualBlock(
                hidden_channels, use_symmetric_convs=use_symmetric_convs, dropout=dropout,
                padding_mode=padding_mode, bottleneck=bottleneck_blocks)
            for _ in range(num_resid_blocks))

    def forward(self, x):
        x = self._reshape_to_cycle_format(x)
        x = self._run_convs(x)

        x = self._reshape_to_vector_format(x)
        return x

    def _run_convs(self, x):
        for i in range(self.num_resid_blocks):
            x = self.conv_blocks[i](x)
        return x

    def _reshape_to_cycle_format(self, x):
        x_shape = x.shape
        x = x.reshape(-1, self.cycle_size, x_shape[1]).transpose(1, 2)  # NCL
        return x

    def _reshape_to_vector_format(self, x):
        x = x.transpose(1, 2)  # NLC
        x_shape = x.shape
        return x.reshape(x_shape[0] * x_shape[1], x_shape[2])


class AtomEncoder(torch.nn.Module):
    embeddings: Sequence[torch.nn.Embedding]

    def __init__(self, hidden_channels, embedding_dimensions: Optional[Sequence[int]]=None):
        """Creates a new set of encoders
        """
        super(AtomEncoder, self).__init__()

        if embedding_dimensions is None:
            embedding_dimensions = [64] * 4

        self.embeddings = torch.nn.ModuleList(
            torch.nn.Embedding(dim, hidden_channels) for dim in embedding_dimensions)

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])

        return out


class BondEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(3):
            self.embeddings.append(nn.Embedding(6, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out
