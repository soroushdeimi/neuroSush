"""Structured connectivity against the dense synapse-by-synapse definition.

A conv2d synapse group is a dense weight matrix whose entries repeat one kernel at every
position; a local2d group has a separate kernel per position; a lateral group is a 3-D
convolution that keeps the shape; adaptive average pooling weights every source in a
region by one over its size. Currents must therefore equal ``spikes @ W_dense``, and STDP
must change every synapse by the dense pair rule
``a_plus pre_trace[i] post_spike[j] - a_minus pre_spike[i] post_trace[j]`` (averaged over
the positions that share a conv kernel entry). Here every synapse is enumerated with plain
loops, independently of the unfold/conv code under test.
"""

import itertools
import math

import pytest
import torch

from neurosush.synapses.currents import (
    avg_pool_current,
    conv2d_current,
    conv_output_size,
    lateral_current,
    local2d_current,
)
from neurosush.synapses.plasticity import stdp_conv2d, stdp_dense, stdp_local2d

f64 = torch.float64
GEOMETRIES = [
    # (in_channels, height, width), out_channels, kernel, stride, padding
    ((1, 5, 5), 1, (3, 3), (1, 1), (0, 0)),
    ((2, 6, 7), 3, (3, 2), (2, 1), (1, 0)),
    ((3, 7, 5), 2, (2, 3), (1, 2), (1, 1)),
]


def flat(c, y, x, shape):
    return (c * shape[1] + y) * shape[2] + x


def conv_synapses(src, out_channels, kernel, stride, padding):
    """Every synapse ``(kernel entry, source index, destination index, position)``."""
    height = conv_output_size(src[1], kernel=kernel[0], stride=stride[0], padding=padding[0])
    width = conv_output_size(src[2], kernel=kernel[1], stride=stride[1], padding=padding[1])
    dst = (out_channels, height, width)
    for o, y, x in itertools.product(range(out_channels), range(height), range(width)):
        for c, i, j in itertools.product(range(src[0]), range(kernel[0]), range(kernel[1])):
            sy, sx = y * stride[0] + i - padding[0], x * stride[1] + j - padding[1]
            if 0 <= sy < src[1] and 0 <= sx < src[2]:  # zero padding: no synapse
                entry = (o, c, i, j)
                yield entry, flat(c, sy, sx, src), flat(o, y, x, dst), y * width + x
    return dst


def geometry_dst(src, out_channels, kernel, stride, padding):
    height = conv_output_size(src[1], kernel=kernel[0], stride=stride[0], padding=padding[0])
    width = conv_output_size(src[2], kernel=kernel[1], stride=stride[1], padding=padding[1])
    return (out_channels, height, width)


def random_state(n, g, batch=()):
    return (torch.rand(*batch, n, generator=g, dtype=f64) < 0.3), torch.rand(
        *batch, n, generator=g, dtype=f64
    )


@pytest.mark.parametrize(("src", "out", "kernel", "stride", "padding"), GEOMETRIES)
class TestConv2d:
    def test_current_is_the_dense_product(self, src, out, kernel, stride, padding):
        g = torch.Generator().manual_seed(0)
        dst = geometry_dst(src, out, kernel, stride, padding)
        weights = torch.randn(out, src[0], *kernel, generator=g, dtype=f64)
        dense = torch.zeros(math.prod(src), math.prod(dst), dtype=f64)
        for entry, i, j, _ in conv_synapses(src, out, kernel, stride, padding):
            dense[i, j] += weights[entry]
        spikes = torch.rand(4, math.prod(src), generator=g) < 0.4
        got = conv2d_current(spikes, weights, src_shape=src, stride=stride, padding=padding)
        torch.testing.assert_close(got, spikes.to(f64) @ dense)

    def test_stdp_averages_the_pair_rule_over_positions(self, src, out, kernel, stride, padding):
        g = torch.Generator().manual_seed(1)
        dst = geometry_dst(src, out, kernel, stride, padding)
        pre_spike, pre_trace = random_state(math.prod(src), g, batch=(3,))
        post_spike, post_trace = random_state(math.prod(dst), g, batch=(3,))
        a_plus, a_minus = 0.7, 0.4
        expected = torch.zeros(out, src[0], *kernel, dtype=f64)
        for entry, i, j, _ in conv_synapses(src, out, kernel, stride, padding):
            pair = (
                a_plus * pre_trace[:, i] * post_spike[:, j]
                - a_minus * pre_spike[:, i] * post_trace[:, j]
            )
            expected[entry] += pair.mean()  # batch mean
        expected /= dst[1] * dst[2]  # shared entry: mean over positions
        got = stdp_conv2d(
            pre_spike=pre_spike,
            pre_trace=pre_trace,
            post_spike=post_spike,
            post_trace=post_trace,
            a_plus=a_plus,
            a_minus=a_minus,
            src_shape=src,
            dst_shape=dst,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        torch.testing.assert_close(got, expected)


@pytest.mark.parametrize(("src", "out", "kernel", "stride", "padding"), GEOMETRIES)
class TestLocal2d:
    def local_index(self, entry, position):
        o, c, i, j = entry
        return o, position, (c * self.kernel[0] + i) * self.kernel[1] + j  # unfold order

    def test_current_is_the_dense_product(self, src, out, kernel, stride, padding):
        self.kernel = kernel
        g = torch.Generator().manual_seed(2)
        dst = geometry_dst(src, out, kernel, stride, padding)
        positions = dst[1] * dst[2]
        weights = torch.randn(
            out, positions, src[0] * kernel[0] * kernel[1], generator=g, dtype=f64
        )
        dense = torch.zeros(math.prod(src), math.prod(dst), dtype=f64)
        for entry, i, j, position in conv_synapses(src, out, kernel, stride, padding):
            dense[i, j] += weights[self.local_index(entry, position)]
        spikes = torch.rand(4, math.prod(src), generator=g) < 0.4
        got = local2d_current(
            spikes, weights, src_shape=src, kernel_size=kernel, stride=stride, padding=padding
        )
        torch.testing.assert_close(got, spikes.to(f64) @ dense)

    def test_stdp_is_the_pair_rule_per_synapse(self, src, out, kernel, stride, padding):
        self.kernel = kernel
        g = torch.Generator().manual_seed(3)
        dst = geometry_dst(src, out, kernel, stride, padding)
        pre_spike, pre_trace = random_state(math.prod(src), g, batch=(3,))
        post_spike, post_trace = random_state(math.prod(dst), g, batch=(3,))
        expected = torch.zeros(out, dst[1] * dst[2], src[0] * kernel[0] * kernel[1], dtype=f64)
        for entry, i, j, position in conv_synapses(src, out, kernel, stride, padding):
            pair = (
                0.7 * pre_trace[:, i] * post_spike[:, j] - 0.4 * pre_spike[:, i] * post_trace[:, j]
            )
            expected[self.local_index(entry, position)] = pair.mean()
        got = stdp_local2d(
            pre_spike=pre_spike,
            pre_trace=pre_trace,
            post_spike=post_spike,
            post_trace=post_trace,
            a_plus=0.7,
            a_minus=0.4,
            src_shape=src,
            dst_shape=dst,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("kernel", [(1, 3, 3), (3, 1, 5), (3, 3, 3)])
def test_lateral_current_is_a_same_padded_3d_correlation(kernel):
    shape = (3, 4, 5)
    g = torch.Generator().manual_seed(4)
    weights = torch.randn(1, 1, *kernel, generator=g, dtype=f64)
    pad = [(k - 1) // 2 for k in kernel]
    dense = torch.zeros(math.prod(shape), math.prod(shape), dtype=f64)
    for d, y, x in itertools.product(*(range(s) for s in shape)):
        for a, b, c in itertools.product(*(range(k) for k in kernel)):
            sd, sy, sx = d + a - pad[0], y + b - pad[1], x + c - pad[2]
            if 0 <= sd < shape[0] and 0 <= sy < shape[1] and 0 <= sx < shape[2]:
                dense[flat(sd, sy, sx, shape), flat(d, y, x, shape)] += weights[0, 0, a, b, c]
    spikes = torch.rand(2, math.prod(shape), generator=g) < 0.4
    torch.testing.assert_close(
        lateral_current(spikes, weights, shape=shape), spikes.to(f64) @ dense
    )


@pytest.mark.parametrize(
    ("src", "out_size"), [((2, 6, 6), (3, 3)), ((1, 7, 5), (3, 2)), ((2, 5, 5), (5, 5))]
)
def test_adaptive_pooling_averages_its_regions(src, out_size):
    # region i covers rows floor(i H / h) to ceil((i + 1) H / h) - 1
    g = torch.Generator().manual_seed(5)
    dst = (src[0], *out_size)
    dense = torch.zeros(math.prod(src), math.prod(dst), dtype=f64)
    for c, i, j in itertools.product(range(src[0]), range(out_size[0]), range(out_size[1])):
        rows = range(i * src[1] // out_size[0], -(-(i + 1) * src[1] // out_size[0]))
        cols = range(j * src[2] // out_size[1], -(-(j + 1) * src[2] // out_size[1]))
        for y, x in itertools.product(rows, cols):
            dense[flat(c, y, x, src), flat(c, i, j, dst)] = 1 / (len(rows) * len(cols))
    spikes = torch.rand(3, math.prod(src), generator=g) < 0.5
    got = avg_pool_current(spikes, src_shape=src, out_size=out_size, dtype=f64)
    torch.testing.assert_close(got, spikes.to(f64) @ dense)


def test_stdp_keeps_the_precision_of_the_traces():
    # boolean spikes carry no dtype; the update must be computed in the traces' dtype (the
    # network's), not in torch's default float32
    third, seventh = torch.tensor([1 / 3], dtype=f64), torch.tensor([1 / 7], dtype=f64)
    spike = torch.tensor([True])
    dense = stdp_dense(
        pre_spike=spike,
        pre_trace=third,
        post_spike=spike,
        post_trace=seventh,
        a_plus=1.0,
        a_minus=1.0,
    )
    assert dense.dtype == f64
    assert dense.item() == 1 / 3 - 1 / 7  # exact in float64, off by ~1e-8 in float32
    geometry = {"src_shape": (1, 2, 2), "dst_shape": (1, 1, 1), "kernel_size": (2, 2)}
    for rule in (stdp_conv2d, stdp_local2d):
        change = rule(
            pre_spike=torch.ones(4, dtype=torch.bool),
            pre_trace=third.expand(4),
            post_spike=spike,
            post_trace=seventh,
            a_plus=1.0,
            a_minus=1.0,
            stride=(1, 1),
            padding=(0, 0),
            **geometry,
        )
        assert change.dtype == f64
        torch.testing.assert_close(change, torch.full_like(change, 1 / 3 - 1 / 7), rtol=0, atol=0)
