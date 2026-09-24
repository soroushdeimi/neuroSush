import pytest
import torch

from neurosush.synapses.bounds import BOUNDS, hard_bound, no_bound, soft_bound
from neurosush.synapses.plasticity import (
    istdp_dense,
    stdp_conv2d,
    stdp_dense,
    stdp_local2d,
    stdp_one_to_one,
    stdp_sparse,
)

B = torch.tensor


class TestBounds:
    def test_soft_bound_is_directional(self):
        ltp, ltd = soft_bound(B([0.0, 0.25, 1.0]), 0.0, 1.0)
        assert ltp.tolist() == [1.0, 0.75, 0.0]
        assert ltd.tolist() == [0.0, 0.25, 1.0]

    def test_soft_bound_never_negative_outside_range(self):
        ltp, ltd = soft_bound(B([-0.5, 1.5]), 0.0, 1.0)
        assert ltp.tolist() == [1.5, 0.0]
        assert ltd.tolist() == [0.0, 1.5]

    def test_hard_bound_lets_weights_recover(self):
        ltp, ltd = hard_bound(B([1.2, 0.5, -0.1]), 0.0, 1.0)
        # above w_max: no more potentiation, but depression still works
        assert ltp.tolist() == [0.0, 1.0, 1.0]
        assert ltd.tolist() == [1.0, 1.0, 0.0]

    def test_no_bound(self):
        ltp, ltd = no_bound(B([5.0, -5.0]), 0.0, 1.0)
        assert ltp.tolist() == [1.0, 1.0]
        assert ltd.tolist() == [1.0, 1.0]

    def test_registry(self):
        assert {"soft": soft_bound, "hard": hard_bound, "none": no_bound} == BOUNDS


def test_stdp_dense_pairs():
    # pre neuron 0 spiked now with post trace 0.5 on post 1 -> depression at (0, 1)
    # post neuron 0 spiked now with pre trace 2.0 on pre 1 -> potentiation at (1, 0)
    dw = stdp_dense(
        pre_spike=B([True, False]),
        pre_trace=B([0.0, 2.0]),
        post_spike=B([True, False]),
        post_trace=B([0.0, 0.5]),
        a_plus=0.1,
        a_minus=0.2,
    )
    torch.testing.assert_close(dw, torch.tensor([[0.0, -0.1], [0.2, 0.0]]))


def test_stdp_dense_gates():
    dw = stdp_dense(
        pre_spike=B([True]),
        pre_trace=B([1.0]),
        post_spike=B([True]),
        post_trace=B([1.0]),
        a_plus=1.0,
        a_minus=0.5,
        ltp_gate=B([[0.5]]),
        ltd_gate=B([[2.0]]),
    )
    assert dw.tolist() == [[0.5 - 1.0]]


def test_stdp_one_to_one():
    dw = stdp_one_to_one(
        pre_spike=B([True, False]),
        pre_trace=B([0.0, 3.0]),
        post_spike=B([False, True]),
        post_trace=B([1.0, 0.0]),
        a_plus=1.0,
        a_minus=1.0,
    )
    assert dw.tolist() == [-1.0, 3.0]


def test_stdp_sparse_matches_dense_on_the_connections():
    src_idx, dst_idx = B([0, 1, 1]), B([1, 0, 1])
    args = {
        "pre_spike": B([True, True]),
        "pre_trace": B([0.3, 0.7]),
        "post_spike": B([True, False]),
        "post_trace": B([0.2, 0.9]),
        "a_plus": 0.5,
        "a_minus": 0.25,
    }
    dense = stdp_dense(**args)
    sparse = stdp_sparse(**args, src_idx=src_idx, dst_idx=dst_idx)
    assert sparse.tolist() == pytest.approx(dense[src_idx, dst_idx].tolist())


def test_stdp_conv2d_averages_over_positions():
    # source (1, 2, 2), destination (1, 1, 2), kernel 2x1 -> two positions
    pre_trace = B([1.0, 2.0, 3.0, 4.0])
    post_spike = B([True, True])
    dw = stdp_conv2d(
        pre_spike=torch.zeros(4, dtype=torch.bool),
        pre_trace=pre_trace,
        post_spike=post_spike,
        post_trace=torch.zeros(2),
        a_plus=1.0,
        a_minus=1.0,
        src_shape=(1, 2, 2),
        dst_shape=(1, 1, 2),
        kernel_size=(2, 1),
        stride=(1, 1),
        padding=(0, 0),
    )
    # position 0 sees traces (1, 3), position 1 sees (2, 4); mean = (1.5, 3.5)
    assert dw.shape == (1, 1, 2, 1)
    assert dw.flatten().tolist() == pytest.approx([1.5, 3.5])


def test_stdp_local2d_potentiation_needs_a_post_spike():
    common = {
        "pre_trace": B([2.0, 2.0, 2.0, 2.0]),
        "post_trace": torch.zeros(2),
        "a_plus": 1.0,
        "a_minus": 0.0,
        "src_shape": (1, 2, 2),
        "dst_shape": (1, 1, 2),
        "kernel_size": (2, 1),
        "stride": (1, 1),
        "padding": (0, 0),
    }
    silent = stdp_local2d(
        pre_spike=torch.ones(4, dtype=torch.bool), post_spike=B([False, False]), **common
    )
    assert silent.abs().sum().item() == 0.0
    one = stdp_local2d(
        pre_spike=torch.zeros(4, dtype=torch.bool), post_spike=B([False, True]), **common
    )
    assert one.shape == (1, 2, 2)
    assert one.tolist() == [[[0.0, 0.0], [2.0, 2.0]]]


def test_stdp_local2d_depression():
    dw = stdp_local2d(
        pre_spike=B([True, False, False, True]),
        pre_trace=torch.zeros(4),
        post_spike=B([False, False]),
        post_trace=B([0.5, 1.0]),
        a_plus=1.0,
        a_minus=2.0,
        src_shape=(1, 2, 2),
        dst_shape=(1, 1, 2),
        kernel_size=(2, 1),
        stride=(1, 1),
        padding=(0, 0),
    )
    # position 0 sees pre spikes (1, 0) -> -2 * 0.5; position 1 sees (0, 1) -> -2 * 1.0
    assert dw.tolist() == [[[-1.0, 0.0], [0.0, -2.0]]]


def test_istdp_dense():
    # Vogels et al. 2011: pre spike -> lr * (post_trace - alpha); post spike -> lr * pre_trace
    dw = istdp_dense(
        pre_spike=B([True, False]),
        pre_trace=B([0.0, 0.4]),
        post_spike=B([False, True]),
        post_trace=B([0.3, 0.1]),
        lr=0.5,
        alpha=0.2,
    )
    torch.testing.assert_close(dw, torch.tensor([[0.05, -0.05], [0.0, 0.2]]))
