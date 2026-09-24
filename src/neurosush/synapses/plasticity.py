"""Spike-timing dependent plasticity: pure kernels per connectivity and the learning rules.

Kernels return the weight change in the layout of the weights. Potentiation pairs the
presynaptic trace with a postsynaptic spike; depression pairs a presynaptic spike with the
postsynaptic trace. Updates are per spike pair and are not scaled by ``dt``.

Activity may carry leading batch dimensions; the weights are shared, so a batch contributes
the mean of the per-sample changes.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
from typing_extensions import NotRequired, Unpack

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order
from neurosush.synapses.bounds import BOUNDS
from neurosush.synapses.currents import Conv2dInput, Local2dInput
from neurosush.synapses.traces import Traces

Gate = float | torch.Tensor
Pair = tuple[int, int]


class _STDPOptions(TypedDict):
    a_plus: float
    a_minus: float
    w_min: NotRequired[float]
    w_max: NotRequired[float]
    bound: NotRequired[Literal["none", "soft", "hard"]]


class _SpikeArgs(TypedDict):
    pre_spike: torch.Tensor
    pre_trace: torch.Tensor
    post_spike: torch.Tensor
    post_trace: torch.Tensor


class _STDPArgs(_SpikeArgs):
    a_plus: float
    a_minus: float
    ltp_gate: Gate
    ltd_gate: Gate


class _ISTDPArgs(_SpikeArgs):
    lr: float
    alpha: float


class _Geometry(TypedDict):
    src_shape: tuple[int, int, int]
    dst_shape: tuple[int, int, int]


def _f(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.get_default_dtype()) if not x.is_floating_point() else x


def _pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch mean of the outer products ``a[n] x b[n]``, shape ``(len_a, len_b)``."""
    a2 = _f(a).reshape(-1, a.shape[-1])
    b2 = _f(b).to(a2.dtype).reshape(-1, b.shape[-1])
    return a2.T @ b2 / a2.shape[0]


def _batch_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean over leading batch dimensions, keeping the last one."""
    return x.reshape(-1, x.shape[-1]).mean(0)


def stdp_dense(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float,
    a_minus: float,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change ``(n_src, n_dst)`` of all-to-all synapses."""
    ltp = _pairs(pre_trace, post_spike)
    ltd = _pairs(pre_spike, post_trace)
    return a_plus * ltp * ltp_gate - a_minus * ltd * ltd_gate


def apply_stdp_dense_(
    weights: torch.Tensor,
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float,
    a_minus: float,
    bound: str = "none",
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> None:
    """Apply :func:`stdp_dense` to ``weights`` in place, touching only active rows and columns.

    Potentiation changes only the columns of spiking postsynaptic neurons and depression only
    the rows of spiking presynaptic neurons, so a step costs O(active * size) instead of
    O(n_src * n_dst). Both changes are computed from the weights before the update, which
    gives exactly the result of ``weights + stdp_dense(...)``.
    """
    post_idx = post_spike.nonzero().squeeze(1)
    pre_idx = pre_spike.nonzero().squeeze(1)
    ltp = ltd = None
    if post_idx.numel() and a_plus:
        columns = weights.index_select(1, post_idx)
        gate = BOUNDS[bound](columns, w_min, w_max)[0]
        ltp = gate.mul_(pre_trace.unsqueeze(1)).mul_(a_plus)
    if pre_idx.numel() and a_minus:
        rows = weights.index_select(0, pre_idx)
        gate = BOUNDS[bound](rows, w_min, w_max)[1]
        ltd = gate.mul_(post_trace.unsqueeze(0)).mul_(-a_minus)
    if ltp is not None:
        weights.index_add_(1, post_idx, ltp)
    if ltd is not None:
        weights.index_add_(0, pre_idx, ltd)


def stdp_one_to_one(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float,
    a_minus: float,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change ``(size,)`` of one-to-one synapses."""
    ltp = _batch_mean(pre_trace * post_spike.to(pre_trace.dtype))
    ltd = _batch_mean(pre_spike.to(post_trace.dtype) * post_trace)
    return a_plus * ltp * ltp_gate - a_minus * ltd * ltd_gate


def stdp_sparse(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float,
    a_minus: float,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change of a connection list ``src_idx[k] -> dst_idx[k]``."""
    return stdp_one_to_one(
        pre_spike=pre_spike[..., src_idx],
        pre_trace=pre_trace[..., src_idx],
        post_spike=post_spike[..., dst_idx],
        post_trace=post_trace[..., dst_idx],
        a_plus=a_plus,
        a_minus=a_minus,
        ltp_gate=ltp_gate,
        ltd_gate=ltd_gate,
    )


def _patches(
    values: torch.Tensor,
    shape: tuple[int, int, int],
    kernel_size: Pair,
    stride: Pair,
    padding: Pair,
) -> torch.Tensor:
    """Unfolded source patches, shape ``(samples, in_channels * kh * kw, positions)``."""
    image = values.to(torch.get_default_dtype()).reshape(-1, *shape)
    return F.unfold(image, kernel_size=kernel_size, stride=stride, padding=padding)


def stdp_conv2d(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float,
    a_minus: float,
    src_shape: tuple[int, int, int],
    dst_shape: tuple[int, int, int],
    kernel_size: Pair,
    stride: Pair,
    padding: Pair,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change ``(out, in, kh, kw)`` of a shared kernel, averaged over positions."""
    geometry = (src_shape, kernel_size, stride, padding)
    out_channels, positions = dst_shape[0], dst_shape[1] * dst_shape[2]
    weight_shape = (out_channels, src_shape[0], *kernel_size)
    post_s = post_spike.to(torch.get_default_dtype()).reshape(-1, out_channels, positions)
    post_t = post_trace.to(torch.get_default_dtype()).reshape(-1, out_channels, positions)
    samples = post_s.shape[0]
    pre_t, pre_s = _patches(pre_trace, *geometry), _patches(pre_spike, *geometry)
    ltp = torch.einsum("nol,nkl->ok", post_s, pre_t).reshape(weight_shape) / samples
    ltd = torch.einsum("nol,nkl->ok", post_t, pre_s).reshape(weight_shape) / samples
    return (a_plus * ltp * ltp_gate - a_minus * ltd * ltd_gate) / positions


def stdp_local2d(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float,
    a_minus: float,
    src_shape: tuple[int, int, int],
    dst_shape: tuple[int, int, int],
    kernel_size: Pair,
    stride: Pair,
    padding: Pair,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change ``(out, positions, in * kh * kw)`` of unshared local kernels."""
    geometry = (src_shape, kernel_size, stride, padding)
    out_channels, positions = dst_shape[0], dst_shape[1] * dst_shape[2]
    post_s = post_spike.to(torch.get_default_dtype()).reshape(-1, out_channels, positions)
    post_t = post_trace.to(torch.get_default_dtype()).reshape(-1, out_channels, positions)
    samples = post_s.shape[0]
    pre_t, pre_s = _patches(pre_trace, *geometry), _patches(pre_spike, *geometry)
    ltp = torch.einsum("nol,nkl->olk", post_s, pre_t) / samples
    ltd = torch.einsum("nol,nkl->olk", post_t, pre_s) / samples
    return a_plus * ltp * ltp_gate - a_minus * ltd * ltd_gate


def istdp_dense(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    lr: float,
    alpha: float,
) -> torch.Tensor:
    """Symmetric inhibitory STDP (Vogels et al. 2011) for all-to-all synapses.

    A presynaptic spike adds ``lr * (post_trace - alpha)``; a postsynaptic spike adds
    ``lr * pre_trace``.
    """
    return lr * (_pairs(pre_spike, post_trace - alpha) + _pairs(pre_trace, post_spike))


def istdp_one_to_one(
    *,
    pre_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_spike: torch.Tensor,
    post_trace: torch.Tensor,
    lr: float,
    alpha: float,
) -> torch.Tensor:
    """Symmetric inhibitory STDP for one-to-one synapses."""
    on_pre = pre_spike.to(post_trace.dtype) * (post_trace - alpha)
    on_post = pre_trace * post_spike.to(pre_trace.dtype)
    return lr * _batch_mean(on_pre + on_post)


class STDP(Behavior):
    """Pair-based STDP for dense, one-to-one, sparse, conv2d and local2d synapses.

    Needs :class:`~neurosush.synapses.traces.Traces` on the synapse.

    Args:
        a_plus: Potentiation rate.
        a_minus: Depression rate.
        w_min: Lower weight used by the bound.
        w_max: Upper weight used by the bound.
        bound: ``"none"``, ``"soft"`` or ``"hard"`` (see :mod:`neurosush.synapses.bounds`).
    """

    order = Order.PLASTICITY
    supported: tuple[str, ...] = ("dense", "one_to_one", "sparse", "conv2d", "local2d")

    def __init__(
        self,
        *,
        a_plus: float,
        a_minus: float,
        w_min: float = 0.0,
        w_max: float = 1.0,
        bound: Literal["none", "soft", "hard"] = "none",
    ) -> None:
        for name, value in (("a_plus", a_plus), ("a_minus", a_minus)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if w_min >= w_max:
            raise ValueError(f"w_min ({w_min}) must be below w_max ({w_max})")
        if bound not in BOUNDS:
            raise ValueError(f"bound must be one of {sorted(BOUNDS)}, got {bound!r}")
        self.a_plus, self.a_minus, self.w_min, self.w_max = a_plus, a_minus, w_min, w_max
        self.bound = bound

    def initialize(self, syn: SynapseGroup) -> None:
        """Check for traces and a supported connectivity."""
        if not hasattr(syn, "pre_trace"):
            raise RuntimeError(f"{type(self).__name__} on {syn.name} needs Traces")
        kind = getattr(syn, "connectivity", None)
        if kind not in self.supported:
            raise ValueError(
                f"{type(self).__name__} does not support connectivity {kind!r} ({syn.name})"
            )

    def compute(self, syn: SynapseGroup) -> torch.Tensor:
        """Weight change of this step."""
        assert syn.weights is not None  # Supported inputs require weights at initialization.
        ltp_gate, ltd_gate = BOUNDS[self.bound](syn.weights, self.w_min, self.w_max)
        args: _STDPArgs = {
            "pre_spike": syn.pre_spike,
            "pre_trace": syn.pre_trace,
            "post_spike": syn.post_spike,
            "post_trace": syn.post_trace,
            "a_plus": self.a_plus,
            "a_minus": self.a_minus,
            "ltp_gate": ltp_gate,
            "ltd_gate": ltd_gate,
        }
        kind = syn.connectivity
        if kind == "dense":
            return stdp_dense(**args)
        if kind == "one_to_one":
            return stdp_one_to_one(**args)
        if kind == "sparse":
            return stdp_sparse(**args, src_idx=syn.src_idx, dst_idx=syn.dst_idx)
        geometry: _Geometry = {"src_shape": syn.src.shape, "dst_shape": syn.dst.shape}
        if kind == "conv2d":
            assert isinstance(syn.input, Conv2dInput)  # Sets the conv2d connectivity.
            kernel = (syn.weights.shape[2], syn.weights.shape[3])
            return stdp_conv2d(
                **args,
                **geometry,
                kernel_size=kernel,
                stride=syn.input.stride,
                padding=syn.input.padding,
            )
        assert isinstance(syn.input, Local2dInput)  # The remaining supported connectivity.
        return stdp_local2d(
            **args,
            **geometry,
            kernel_size=syn.input.kernel_size,
            stride=syn.input.stride,
            padding=syn.input.padding,
        )

    def forward(self, syn: SynapseGroup) -> None:
        """Apply this step's weight change (in place and event-driven for dense synapses)."""
        assert syn.weights is not None  # Supported inputs require weights at initialization.
        if syn.connectivity == "dense" and syn.pre_spike.dim() == 1:
            apply_stdp_dense_(
                syn.weights,
                pre_spike=syn.pre_spike,
                pre_trace=syn.pre_trace,
                post_spike=syn.post_spike,
                post_trace=syn.post_trace,
                a_plus=self.a_plus,
                a_minus=self.a_minus,
                bound=self.bound,
                w_min=self.w_min,
                w_max=self.w_max,
            )
        else:
            syn.weights = syn.weights + self.compute(syn)


class RSTDP(STDP):
    """Reward-modulated STDP through an eligibility trace ``syn.eligibility``.

    ``c = c * (1 - dt / tau_c) + dw_stdp`` and ``w += dt * dopamine * c``. Needs
    :class:`~neurosush.modulation.Dopamine` on the network.

    Args:
        tau_c: Eligibility time constant; must exceed ``dt``.
        **kwargs: :class:`STDP` arguments.
    """

    def __init__(self, *, tau_c: float, **kwargs: Unpack[_STDPOptions]) -> None:
        super().__init__(**kwargs)
        if tau_c <= 0:
            raise ValueError(f"tau_c must be positive, got {tau_c}")
        self.tau_c = tau_c

    def initialize(self, syn: SynapseGroup) -> None:
        """Check the time constant and dopamine; allocate the eligibility trace."""
        super().initialize(syn)
        if self.tau_c <= syn.net.dt:
            raise ValueError(f"tau_c ({self.tau_c}) must exceed dt ({syn.net.dt})")
        if not hasattr(syn.net, "dopamine"):
            raise RuntimeError(f"RSTDP on {syn.name} needs Dopamine on the network")
        assert syn.weights is not None  # Supported inputs require weights at initialization.
        syn.eligibility = torch.zeros_like(syn.weights)

    def forward(self, syn: SynapseGroup) -> None:
        """Update the eligibility trace and the weights."""
        assert syn.weights is not None  # initialize() allocates eligibility from weights.
        dt = syn.net.dt
        syn.eligibility = syn.eligibility * (1 - dt / self.tau_c) + self.compute(syn)
        syn.weights = syn.weights + dt * syn.net.dopamine * syn.eligibility


class ISTDP(Behavior):
    """Symmetric inhibitory STDP (Vogels et al. 2011) that drives postsynaptic rates to ``rho``.

    ``alpha = 2 * rho * tau * scale`` from the synapse's symmetric
    :class:`~neurosush.synapses.traces.Traces`, unless ``alpha`` is given. Supports dense,
    one-to-one and sparse synapses.

    Args:
        lr: Learning rate.
        rho: Target postsynaptic rate, in spikes per time unit of ``dt``.
        alpha: Depression offset, instead of ``rho``.
    """

    order = Order.PLASTICITY
    supported = ("dense", "one_to_one", "sparse")

    def __init__(self, *, lr: float, rho: float | None = None, alpha: float | None = None) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if (rho is None) == (alpha is None):
            raise ValueError("give exactly one of rho and alpha")
        self.lr, self.rho, self.alpha = lr, rho, alpha

    def initialize(self, syn: SynapseGroup) -> None:
        """Derive ``alpha`` from the traces and check the connectivity."""
        traces = next((b for b in syn.behaviors if isinstance(b, Traces)), None)
        if traces is None:
            raise RuntimeError(f"ISTDP on {syn.name} needs Traces")
        if traces.tau_pre != traces.tau_post:
            raise ValueError(f"ISTDP on {syn.name} needs symmetric traces (tau_pre == tau_post)")
        if getattr(syn, "connectivity", None) not in self.supported:
            raise ValueError(
                f"ISTDP does not support connectivity {getattr(syn, 'connectivity', None)!r} "
                f"({syn.name})"
            )
        if self.alpha is None:
            assert self.rho is not None  # __init__ requires exactly one of rho and alpha.
            self.alpha = 2 * self.rho * traces.tau_pre * traces.scale

    def forward(self, syn: SynapseGroup) -> None:
        """Apply this step's weight change."""
        assert syn.weights is not None  # Supported inputs require weights at initialization.
        assert self.alpha is not None  # initialize() derives alpha when omitted.
        args: _ISTDPArgs = {
            "pre_spike": syn.pre_spike,
            "pre_trace": syn.pre_trace,
            "post_spike": syn.post_spike,
            "post_trace": syn.post_trace,
            "lr": self.lr,
            "alpha": self.alpha,
        }
        if syn.connectivity == "dense":
            dw = istdp_dense(**args)
        elif syn.connectivity == "one_to_one":
            dw = istdp_one_to_one(**args)
        else:
            dw = istdp_one_to_one(
                pre_spike=syn.pre_spike[..., syn.src_idx],
                pre_trace=syn.pre_trace[..., syn.src_idx],
                post_spike=syn.post_spike[..., syn.dst_idx],
                post_trace=syn.post_trace[..., syn.dst_idx],
                lr=self.lr,
                alpha=self.alpha,
            )
        syn.weights = syn.weights + dw
