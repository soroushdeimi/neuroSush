"""Spike-timing dependent plasticity: pure kernels per connectivity and the learning rules.

Kernels return the weight change in the layout of the weights. Potentiation pairs the
presynaptic trace with a postsynaptic spike; depression pairs a presynaptic spike with the
postsynaptic trace. Updates are per spike pair and are not scaled by ``dt``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order
from neurosush.synapses.bounds import BOUNDS
from neurosush.synapses.traces import Traces

Gate = float | torch.Tensor
Pair = tuple[int, int]


def _f(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.get_default_dtype()) if not x.is_floating_point() else x


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
    ltp = torch.outer(pre_trace, _f(post_spike).to(pre_trace.dtype))
    ltd = torch.outer(_f(pre_spike).to(post_trace.dtype), post_trace)
    return a_plus * ltp * ltp_gate - a_minus * ltd * ltd_gate


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
    ltp = pre_trace * post_spike.to(pre_trace.dtype)
    ltd = pre_spike.to(post_trace.dtype) * post_trace
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
        pre_spike=pre_spike[src_idx],
        pre_trace=pre_trace[src_idx],
        post_spike=post_spike[dst_idx],
        post_trace=post_trace[dst_idx],
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
    """Unfolded source patches, shape ``(in_channels * kh * kw, positions)``."""
    image = values.to(torch.get_default_dtype()).view(1, *shape)
    return F.unfold(image, kernel_size=kernel_size, stride=stride, padding=padding)[0]


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
    post_s = post_spike.to(torch.get_default_dtype()).view(out_channels, positions)
    post_t = post_trace.to(torch.get_default_dtype()).view(out_channels, positions)
    ltp = (post_s @ _patches(pre_trace, *geometry).T).view(weight_shape)
    ltd = (post_t @ _patches(pre_spike, *geometry).T).view(weight_shape)
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
    post_s = post_spike.to(torch.get_default_dtype()).view(out_channels, positions, 1)
    post_t = post_trace.to(torch.get_default_dtype()).view(out_channels, positions, 1)
    ltp = post_s * _patches(pre_trace, *geometry).T.unsqueeze(0)
    ltd = post_t * _patches(pre_spike, *geometry).T.unsqueeze(0)
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
    on_pre = torch.outer(pre_spike.to(post_trace.dtype), post_trace - alpha)
    on_post = torch.outer(pre_trace, post_spike.to(pre_trace.dtype))
    return lr * (on_pre + on_post)


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
    return lr * (on_pre + on_post)


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
        ltp_gate, ltd_gate = BOUNDS[self.bound](syn.weights, self.w_min, self.w_max)
        args = {
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
        geometry = {"src_shape": syn.src.shape, "dst_shape": syn.dst.shape}
        if kind == "conv2d":
            kernel = tuple(syn.weights.shape[2:])
            return stdp_conv2d(
                **args,
                **geometry,
                kernel_size=kernel,
                stride=syn.input.stride,
                padding=syn.input.padding,
            )
        return stdp_local2d(
            **args,
            **geometry,
            kernel_size=syn.input.kernel_size,
            stride=syn.input.stride,
            padding=syn.input.padding,
        )

    def forward(self, syn: SynapseGroup) -> None:
        """Apply this step's weight change."""
        syn.weights = syn.weights + self.compute(syn)


class RSTDP(STDP):
    """Reward-modulated STDP through an eligibility trace ``syn.eligibility``.

    ``c = c * (1 - dt / tau_c) + dw_stdp`` and ``w += dt * dopamine * c``. Needs
    :class:`~neurosush.modulation.Dopamine` on the network.

    Args:
        tau_c: Eligibility time constant; must exceed ``dt``.
        **kwargs: :class:`STDP` arguments.
    """

    def __init__(self, *, tau_c: float, **kwargs) -> None:
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
        syn.eligibility = torch.zeros_like(syn.weights)

    def forward(self, syn: SynapseGroup) -> None:
        """Update the eligibility trace and the weights."""
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

    def __init__(self, *, lr: float, rho: float | None = None, alpha: float | None = None):
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
            self.alpha = 2 * self.rho * traces.tau_pre * traces.scale

    def forward(self, syn: SynapseGroup) -> None:
        """Apply this step's weight change."""
        args = {
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
            idx = {
                "pre_spike": syn.src_idx,
                "pre_trace": syn.src_idx,
                "post_spike": syn.dst_idx,
                "post_trace": syn.dst_idx,
            }
            dw = istdp_one_to_one(**{k: (v[idx[k]] if k in idx else v) for k, v in args.items()})
        syn.weights = syn.weights + dw
