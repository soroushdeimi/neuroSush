"""Compute syn.I, one current per destination, from syn.pre_spike and syn.weights.

Each input behavior sets syn.connectivity and syn.input (itself) so learning and
normalization can choose matching kernels. Inhibitory sources produce negative
currents; average pooling needs no weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order
from neurosush.synapses.traces import SpikeGather


def conv_output_size(size: int, *, kernel: int, stride: int, padding: int) -> int:
    """Return (size + 2 * padding - kernel) // stride + 1.

    Args:
        size: Input extent along one axis.
        kernel: Kernel extent along that axis.
        stride: Kernel step along that axis.
        padding: Zero padding on each side.
    """
    return (size + 2 * padding - kernel) // stride + 1


def dense_current(spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Sum the weights of spiking sources for each destination.

    Args:
        spikes: Flat source spike vector.
        weights: Weights shaped (n_src, n_dst).
    """
    return spikes.to(weights.dtype) @ weights


def one_to_one_current(spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Multiply each source spike by its corresponding weight.

    Args:
        spikes: Flat source spike vector.
        weights: One weight per source and destination pair.
    """
    return spikes.to(weights.dtype) * weights


def sparse_current(
    spikes: torch.Tensor,
    values: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    n_dst: int,
) -> torch.Tensor:
    """Accumulate the weights of active sparse edges at their destinations.

    Args:
        spikes: Flat source spike vector.
        values: One weight per edge.
        src_idx: Source index of each edge.
        dst_idx: Destination index of each edge.
        n_dst: Number of destination neurons.
    """
    contributions = spikes[..., src_idx] * values
    return values.new_zeros(*spikes.shape[:-1], n_dst).index_add_(-1, dst_idx, contributions)


def conv2d_current(
    spikes: torch.Tensor,
    weights: torch.Tensor,
    *,
    src_shape: tuple[int, int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> torch.Tensor:
    """Cross-correlate source spikes with shared spatial kernels.

    Args:
        spikes: Flat source spike vector.
        weights: Kernels shaped (out_channels, in_channels, kh, kw).
        src_shape: Source depth, height and width.
        stride: Spatial kernel step.
        padding: Spatial zero padding.
    """
    image = spikes.to(weights.dtype).reshape(-1, *src_shape)
    out = F.conv2d(image, weights, stride=stride, padding=padding)
    return out.reshape(*spikes.shape[:-1], -1)


def local2d_current(
    spikes: torch.Tensor,
    weights: torch.Tensor,
    *,
    src_shape: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> torch.Tensor:
    """Sum weighted spike patches using a separate kernel at each position.

    Args:
        spikes: Flat source spike vector.
        weights: Weights shaped (out_channels, positions, in_channels * kh * kw).
        src_shape: Source depth, height and width.
        kernel_size: Kernel height and width.
        stride: Spatial kernel step.
        padding: Spatial zero padding.
    """
    image = spikes.to(weights.dtype).reshape(-1, *src_shape)
    patches = F.unfold(image, kernel_size, stride=stride, padding=padding)  # (N, K, L)
    out = torch.einsum("olk,nkl->nol", weights, patches)
    return out.reshape(*spikes.shape[:-1], -1)


def lateral_current(
    spikes: torch.Tensor, weights: torch.Tensor, *, shape: tuple[int, int, int]
) -> torch.Tensor:
    """Cross-correlate spikes across three axes while preserving the group shape.

    Args:
        spikes: Flat source spike vector.
        weights: Odd kernels shaped (1, 1, kd, kh, kw).
        shape: Group depth, height and width.
    """
    image = spikes.to(weights.dtype).reshape(-1, 1, *shape)
    padding = tuple((size - 1) // 2 for size in weights.shape[2:])
    return F.conv3d(image, weights, padding=padding).reshape(*spikes.shape[:-1], -1)


def avg_pool_current(
    spikes: torch.Tensor,
    *,
    src_shape: tuple[int, int, int],
    out_size: tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Average source spikes over adaptive spatial pooling regions.

    Args:
        spikes: Flat source spike vector.
        src_shape: Source depth, height and width.
        out_size: Destination height and width.
        dtype: Floating point dtype for pooling and the result.
    """
    image = spikes.to(dtype).reshape(-1, *src_shape)
    return F.adaptive_avg_pool2d(image, out_size).reshape(*spikes.shape[:-1], -1)


def _pair(value: int | tuple[int, int], name: str, minimum: int) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else value
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v < minimum for v in pair)
    ):
        raise ValueError(f"{name} must be an int or pair of ints >= {minimum}, got {value!r}")
    return pair


def _check_shape(syn: SynapseGroup, expected: tuple[int, ...]) -> None:
    assert syn.weights is not None  # Weighted inputs are checked before validate().
    if syn.weights.shape != expected:
        raise ValueError(
            f"weights of {syn.name} must have shape {expected}, got {tuple(syn.weights.shape)}"
        )


def _check_grid(
    syn: SynapseGroup,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    for axis, kernel, step, pad in zip(
        ("height", "width"), kernel_size, stride, padding, strict=True
    ):
        expected = conv_output_size(getattr(syn.src, axis), kernel=kernel, stride=step, padding=pad)
        actual = getattr(syn.dst, axis)
        if actual != expected:
            raise ValueError(f"destination {axis} of {syn.name} must be {expected}, got {actual}")


class _SynapticInput(Behavior, ABC):
    """Scale connectivity-specific currents by source sign and coefficient."""

    order = Order.SYNAPTIC_INPUT
    connectivity: str
    needs_weights = True

    def __init__(self, *, coef: float = 1.0) -> None:
        self.coef = coef

    def initialize(self, syn: SynapseGroup) -> None:
        """Validate connectivity and allocate destination current and source spikes.

        Args:
            syn: Synapse group receiving the input state.
        """
        if self.needs_weights and syn.weights is None:
            raise RuntimeError(f"{type(self).__name__} on {syn.name} needs weights (WeightInit)")
        if not any(isinstance(behavior, SpikeGather) for behavior in syn.behaviors):
            # without it pre_spike would stay silent forever
            raise RuntimeError(
                f"{type(self).__name__} on {syn.name} needs SpikeGather to receive spikes"
            )
        self.validate(syn)
        syn.connectivity = self.connectivity
        syn.input = self
        syn.I = syn.dst.state()
        if not hasattr(syn, "pre_spike"):
            syn.pre_spike = syn.src.state(False, dtype=torch.bool)
        self.sign = -1.0 if syn.src.inhibitory else 1.0

    def validate(self, syn: SynapseGroup) -> None:
        """Check connectivity-specific geometry before allocating state.

        Args:
            syn: Synapse group whose geometry is checked.
        """

    @abstractmethod
    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Return currents before applying the coefficient and source sign.

        Args:
            syn: Synapse group providing spikes, weights and geometry.
        """

    def forward(self, syn: SynapseGroup) -> None:
        """Write currents scaled by the coefficient and source sign.

        Args:
            syn: Synapse group receiving the destination current vector.
        """
        syn.I = self.sign * self.coef * self.current(syn)


class DenseInput(_SynapticInput):
    """Sum dense weights from spiking sources into destination currents.

    Args:
        coef: Multiplier applied to the current.
    """

    connectivity = "dense"

    def validate(self, syn: SynapseGroup) -> None:
        """Require one weight for every source and destination pair.

        Args:
            syn: Synapse group whose weight shape is checked.
        """
        _check_shape(syn, (syn.src.size, syn.dst.size))

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Sum active rows of the dense weight matrix.

        Args:
            syn: Synapse group providing spikes and weights.
        """
        assert syn.weights is not None  # initialize() requires weights.
        return dense_current(syn.pre_spike, syn.weights)


class OneToOneInput(_SynapticInput):
    """Multiply each source spike by its paired destination weight.

    Args:
        coef: Multiplier applied to the current.
    """

    connectivity = "one_to_one"

    def validate(self, syn: SynapseGroup) -> None:
        """Require equal group sizes and a vector of paired weights.

        Args:
            syn: Synapse group whose sizes and weights are checked.
        """
        if syn.src.size != syn.dst.size:
            raise ValueError(
                f"group size for {syn.name} must match, got src={syn.src.size}, dst={syn.dst.size}"
            )
        _check_shape(syn, (syn.src.size,))

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Return the weights gated by their paired source spikes.

        Args:
            syn: Synapse group providing spikes and weights.
        """
        assert syn.weights is not None  # initialize() requires weights.
        return one_to_one_current(syn.pre_spike, syn.weights)


class SparseInput(_SynapticInput):
    """Accumulate active sparse edge weights into destination currents.

    Args:
        coef: Multiplier applied to the current.
    """

    connectivity = "sparse"

    def validate(self, syn: SynapseGroup) -> None:
        """Require sparse source indices and a vector of edge weights.

        Args:
            syn: Synapse group whose sparse storage is checked.
        """
        assert syn.weights is not None  # initialize() requires weights.
        if not hasattr(syn, "src_idx") or syn.weights.ndim != 1:
            raise RuntimeError(f"SparseInput on {syn.name} needs sparse weights (sparse=True)")

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Sum active sparse edges at each destination index.

        Args:
            syn: Synapse group providing spikes, weights and edge indices.
        """
        assert syn.weights is not None  # initialize() requires weights.
        return sparse_current(syn.pre_spike, syn.weights, syn.src_idx, syn.dst_idx, syn.dst.size)


class Conv2dInput(_SynapticInput):
    """Cross-correlate source spikes with shared spatial weight kernels.

    Args:
        coef: Multiplier applied to the current.
        stride: Spatial kernel step, as an int or height-width pair.
        padding: Spatial zero padding, as an int or height-width pair.
    """

    connectivity = "conv2d"

    def __init__(
        self,
        *,
        coef: float = 1.0,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__(coef=coef)
        self.stride = _pair(stride, "stride", 1)
        self.padding = _pair(padding, "padding", 0)

    def validate(self, syn: SynapseGroup) -> None:
        """Require matching kernel channels and destination spatial geometry.

        Args:
            syn: Synapse group whose convolution geometry is checked.
        """
        assert syn.weights is not None  # initialize() requires weights.
        shape = tuple(syn.weights.shape)
        if syn.weights.ndim != 4:
            raise ValueError(f"weights of {syn.name} must have a 4-D shape, got {shape}")
        for label, actual, expected in (
            ("in_channels", shape[1], syn.src.depth),
            ("out_channels", shape[0], syn.dst.depth),
        ):
            if actual != expected:
                raise ValueError(f"{label} of {syn.name} must be {expected}, got {actual}")
        _check_grid(syn, (shape[2], shape[3]), self.stride, self.padding)

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Return flattened spatial cross-correlations of the source spikes.

        Args:
            syn: Synapse group providing spikes, weights and source shape.
        """
        assert syn.weights is not None  # initialize() requires weights.
        return conv2d_current(
            syn.pre_spike,
            syn.weights,
            src_shape=syn.src.shape,
            stride=self.stride,
            padding=self.padding,
        )


class Local2dInput(Conv2dInput):
    """Weight source spike patches separately at each destination position.

    Args:
        kernel_size: Spatial kernel height and width.
        coef: Multiplier applied to the current.
        stride: Spatial kernel step, as an int or height-width pair.
        padding: Spatial zero padding, as an int or height-width pair.
    """

    connectivity = "local2d"

    def __init__(
        self,
        *,
        kernel_size: int | tuple[int, int],
        coef: float = 1.0,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__(coef=coef, stride=stride, padding=padding)
        self.kernel_size = _pair(kernel_size, "kernel_size", 1)

    def validate(self, syn: SynapseGroup) -> None:
        """Require a matching output grid and one weight kernel per position.

        Args:
            syn: Synapse group whose local geometry and weights are checked.
        """
        _check_grid(syn, self.kernel_size, self.stride, self.padding)
        kh, kw = self.kernel_size
        _check_shape(syn, (syn.dst.depth, syn.dst.height * syn.dst.width, syn.src.depth * kh * kw))

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Return currents from the position-specific weighted source patches.

        Args:
            syn: Synapse group providing spikes, weights and source shape.
        """
        assert syn.weights is not None  # initialize() requires weights.
        return local2d_current(
            syn.pre_spike,
            syn.weights,
            src_shape=syn.src.shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class LateralInput(_SynapticInput):
    """Cross-correlate spikes within one group using an odd three-axis kernel.

    Args:
        coef: Multiplier applied to the current.
    """

    connectivity = "lateral"

    def validate(self, syn: SynapseGroup) -> None:
        """Require the same group and a (1, 1, kd, kh, kw) kernel with odd extents.

        Args:
            syn: Synapse group whose recurrence and kernel are checked.
        """
        if syn.src is not syn.dst:
            raise ValueError(
                f"{syn.name} requires the same group, got src={syn.src.name}, dst={syn.dst.name}"
            )
        assert syn.weights is not None  # initialize() requires weights.
        shape = tuple(syn.weights.shape)
        if len(shape) != 5 or shape[:2] != (1, 1) or any(k % 2 != 1 for k in shape[2:]):
            raise ValueError(
                f"weights of {syn.name} must have shape (1, 1, kd, kh, kw) "
                f"with odd kernel sizes, got {shape}"
            )

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Return lateral currents with the group's original shape flattened.

        Args:
            syn: Synapse group providing spikes, weights and group shape.
        """
        assert syn.weights is not None  # initialize() requires weights.
        return lateral_current(syn.pre_spike, syn.weights, shape=syn.src.shape)


class AvgPool2dInput(_SynapticInput):
    """Average source spikes over adaptive spatial pooling regions per depth plane.

    Args:
        coef: Multiplier applied to the current.
    """

    connectivity = "avg_pool"
    needs_weights = False

    def validate(self, syn: SynapseGroup) -> None:
        """Require equal source and destination depths.

        Args:
            syn: Synapse group whose depth is checked.
        """
        if syn.src.depth != syn.dst.depth:
            raise ValueError(
                f"group depth for {syn.name} must match, "
                f"got src={syn.src.depth}, dst={syn.dst.depth}"
            )

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """Return adaptive average pooled spikes in the network dtype.

        Args:
            syn: Synapse group providing spikes and pooling geometry.
        """
        return avg_pool_current(
            syn.pre_spike,
            src_shape=syn.src.shape,
            out_size=(syn.dst.height, syn.dst.width),
            dtype=syn.net.dtype,
        )
