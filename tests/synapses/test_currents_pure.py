import pytest
import torch

from neurosush.synapses.currents import (
    avg_pool_current,
    conv2d_current,
    conv_output_size,
    dense_current,
    lateral_current,
    local2d_current,
    one_to_one_current,
    sparse_current,
)


def test_dense_current_sums_weights_of_spiking_sources():
    w = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    spikes = torch.tensor([False, True])
    assert dense_current(spikes, w).tolist() == [10.0, 20.0, 30.0]


def test_one_to_one_current():
    out = one_to_one_current(torch.tensor([True, False, True]), torch.tensor([0.5, 2.0, 3.0]))
    assert out.tolist() == [0.5, 0.0, 3.0]


def test_sparse_current():
    # synapses: 0->1 (w 2), 1->1 (w 3), 1->0 (w 5)
    src_idx, dst_idx = torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])
    values = torch.tensor([2.0, 3.0, 5.0])
    out = sparse_current(torch.tensor([True, True]), values, src_idx, dst_idx, n_dst=3)
    assert out.tolist() == [5.0, 5.0, 0.0]
    out = sparse_current(torch.tensor([True, False]), values, src_idx, dst_idx, n_dst=3)
    assert out.tolist() == [0.0, 2.0, 0.0]


def test_conv_output_size():
    assert conv_output_size(5, kernel=3, stride=1, padding=0) == 3
    assert conv_output_size(5, kernel=3, stride=2, padding=1) == 3
    assert conv_output_size(4, kernel=2, stride=2, padding=0) == 2


def test_conv2d_current():
    # source (1, 3, 3) with spikes on the diagonal; one 2x2 all-ones kernel
    spikes = torch.eye(3, dtype=torch.bool).flatten()
    w = torch.ones(1, 1, 2, 2)
    out = conv2d_current(spikes, w, src_shape=(1, 3, 3), stride=1, padding=0)
    assert out.tolist() == [2.0, 1.0, 1.0, 2.0]


def test_conv2d_current_two_output_channels():
    spikes = torch.ones(4, dtype=torch.bool)
    w = torch.stack([torch.ones(1, 2, 2), 2 * torch.ones(1, 2, 2)])
    out = conv2d_current(spikes, w, src_shape=(1, 2, 2), stride=1, padding=0)
    assert out.tolist() == [4.0, 8.0]


def test_local2d_current_matches_conv_when_weights_are_shared():
    spikes = torch.rand(2 * 4 * 4, generator=torch.Generator().manual_seed(0)) > 0.5
    kernel = torch.rand(3, 2, 2, 2, generator=torch.Generator().manual_seed(1))
    out_positions = 3 * 3
    local = kernel.flatten(1).unsqueeze(1).expand(3, out_positions, 8).contiguous()
    a = local2d_current(spikes, local, src_shape=(2, 4, 4), kernel_size=(2, 2), stride=1, padding=0)
    b = conv2d_current(spikes, kernel, src_shape=(2, 4, 4), stride=1, padding=0)
    assert torch.allclose(a, b)


def test_local2d_current_uses_position_specific_weights():
    spikes = torch.ones(4, dtype=torch.bool)  # (1, 2, 2) source
    # one output channel, 2 positions (1x2 output with a 2x1 kernel), 2 inputs each
    w = torch.tensor([[[1.0, 2.0], [10.0, 20.0]]])
    out = local2d_current(spikes, w, src_shape=(1, 2, 2), kernel_size=(2, 1), stride=1, padding=0)
    assert out.tolist() == [3.0, 30.0]


def test_lateral_current_keeps_the_shape():
    spikes = torch.zeros(9, dtype=torch.bool)
    spikes[4] = True  # center of (1, 3, 3)
    w = torch.arange(9, dtype=torch.float32).view(1, 1, 1, 3, 3)
    out = lateral_current(spikes, w, shape=(1, 3, 3))
    # cross-correlation: output (i, j) sees the center through kernel entry (2 - i, 2 - j)
    assert out.tolist() == [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]


def test_avg_pool_current():
    spikes = torch.tensor([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool)
    out = avg_pool_current(spikes, src_shape=(1, 4, 4), out_size=(2, 2))
    assert out.tolist() == [0.75, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("fn", [dense_current, one_to_one_current])
def test_currents_do_not_modify_inputs(fn):
    w = torch.ones(2, 2) if fn is dense_current else torch.ones(2)
    spikes = torch.tensor([True, False])
    fn(spikes, w)
    assert spikes.tolist() == [True, False]
    assert w.sum().item() == w.numel()
