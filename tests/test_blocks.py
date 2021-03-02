import pytest
import torch
from autobahn import blocks

shift_matrix_5 = torch.tensor(
    [[0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0]]
).float()


shift_matrix_6 = torch.tensor(
    [[0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0]]
).float()

shift_matrices = {5: shift_matrix_5, 6: shift_matrix_6}


class TestSymmetricConv1D():
    @pytest.mark.parametrize('num_channels', [1, 5, 10])
    @pytest.mark.parametrize('signal_size', [4, 6, 10])
    def test_symmetry(self, num_channels, signal_size):
        conv = blocks.SymmetricConv1D(num_channels, num_channels, 3, padding=1, padding_mode='circular')
        x = torch.randn(5, num_channels, signal_size)  # NCL
        x_flip = torch.flip(x, dims=(2,))

        x_out = conv(x)
        x_out_flip = torch.flip(conv(x_flip), dims=(2,))
        assert(torch.allclose(x_out, x_out_flip, atol=1e-6, rtol=1e-4))

    @pytest.mark.parametrize('k', [5, 6])
    def test_equivariance(self, k):
        shift_mat = shift_matrices[k]
        num_cycles = 4
        num_channels = 3

        x = torch.randn(num_cycles, num_channels, k)
        x_shift = torch.matmul(x, shift_mat)

        conv = blocks.SymmetricConv1D(num_channels, num_channels, 3, padding=1, padding_mode='circular')
        x_out = conv(x)
        x_out_shift = torch.matmul(x_out, shift_mat)
        x_shift_out = conv(x_shift)
        assert(torch.allclose(x_out_shift, x_shift_out, atol=1e-6, rtol=1e-4))


class TestCycleBlock():
    def test_vec_to_cycle(self):
        k = 6  # Cycle Length
        num_cycles = 4
        hidden_channels = 3
        cb = blocks.CycleBlock(hidden_channels, k)

        x = torch.stack([torch.arange(k * num_cycles)] * hidden_channels, dim=1)
        x_formatted = cb._reshape_to_cycle_format(x).transpose(1, 2)

        for i in range(num_cycles):
            true_values = torch.arange(i * k, (i+1) * k)
            x_i_true = torch.stack([true_values] * hidden_channels, dim=1)
            assert(torch.allclose(x_formatted[i], x_i_true, atol=1e-6, rtol=1e-4))

    def test_cycle_to_vec(self):
        k = 6  # Cycle Length
        num_cycles = 4
        hidden_channels = 3
        cb = blocks.CycleBlock(hidden_channels, k)

        x = []
        for i in range(num_cycles):
            true_values = torch.arange(i * k, (i+1) * k)
            x_i = torch.stack([true_values] * hidden_channels, dim=1)
            x.append(x_i)
        x = torch.stack(x, dim=0)
        x_formatted = cb._reshape_to_vector_format(x.transpose(1, 2))

        x_true = torch.stack([torch.arange(k * num_cycles)] * hidden_channels, dim=1)
        assert(torch.allclose(x_formatted, x_true, atol=1e-6, rtol=1e-4))

    @pytest.mark.parametrize('k', [5, 6])
    def test_run_conv_equivariance(self, k):
        shift_mat = shift_matrices[k]
        num_cycles = 4
        hidden_channels = 3
        cb = blocks.CycleBlock(hidden_channels, k)

        x = torch.randn(num_cycles, hidden_channels, k)
        x_shift = torch.matmul(x, shift_mat)

        x_out = cb._run_convs(x)
        x_shift_out = cb._run_convs(x_shift)

        x_out_shift = torch.matmul(x_out, shift_mat)
        assert(torch.allclose(x_shift_out, x_out_shift, atol=1e-6, rtol=1e-4))

    @pytest.mark.parametrize('k', [5, 6])
    def test_equivariance(self, k):
        shift_mat = shift_matrices[k]
        num_cycles = 4
        hidden_channels = 3
        cb = blocks.CycleBlock(hidden_channels, k)

        x = torch.randn(num_cycles, hidden_channels, k)
        x_shift = torch.matmul(x, shift_mat)

        # Convert to single long vector
        x_shift = x_shift.transpose(1, 2).reshape(-1, hidden_channels)
        x = x.transpose(1, 2).reshape(-1, hidden_channels)

        # Run Network
        x_out = cb.forward(x)
        x_out = cb._reshape_to_cycle_format(x_out)
        x_out = torch.matmul(x_out, shift_mat)
        x_out = cb._reshape_to_vector_format(x_out)
        x_out_shift = cb.forward(x_shift)
        assert(torch.allclose(x_out, x_out_shift, atol=1e-6, rtol=1e-4))


class TestResidualBlock():
    @pytest.mark.parametrize('k', [5, 6])
    def test_equivariance(self, k):
        shift_mat = shift_matrices[k]
        num_cycles = 4
        hidden_channels = 3

        rb = blocks.ResidualBlock(hidden_channels)

        # Construct sample activation
        x = torch.randn(num_cycles, hidden_channels, k)
        x_out = rb(x)

        shift_mat_i = shift_mat
        for i in range(1, k):
            shift_mat_i = torch.matmul(shift_mat_i, shift_mat)
            x_shift = torch.matmul(x, shift_mat_i)
            x_shift_out = rb(x_shift)
            x_out_shift = torch.matmul(x_out, shift_mat_i)
            assert(torch.allclose(x_shift_out, x_out_shift, atol=1e-6, rtol=1e-4))
