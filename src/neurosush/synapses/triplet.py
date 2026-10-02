"""Triplet STDP (Pfister and Gerstner 2006): pair terms plus spike-triplet terms.

Four traces are kept by the behavior itself: presynaptic ``r1`` (``tau_plus``) and ``r2``
(``tau_x``), postsynaptic ``o1`` (``tau_minus``) and ``o2`` (``tau_y``). Pure functions
compute the trace updates and the weight change; :class:`TripletSTDP` is a thin wrapper.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order
from neurosush.synapses.bounds import BOUNDS
from neurosush.synapses.plasticity import Gate, _batch_mean, _pairs

Interaction = Literal["all", "nearest"]
_INTERACTIONS = ("all", "nearest")


def trace_decay(trace: torch.Tensor, *, tau: float, dt: float) -> torch.Tensor:
    """Euler decay ``trace * (1 - dt / tau)`` as a new tensor."""
    return trace * (1 - dt / tau)


def trace_increment(
    decayed: torch.Tensor, spikes: torch.Tensor, *, interaction: Interaction
) -> torch.Tensor:
    """Register spikes on an already decayed trace, as a new tensor.

    ``"all"`` adds one per spike (all-to-all); ``"nearest"`` sets the trace to one at a
    spike (nearest-spike, as in the Diehl and Cook 2015 code).
    """
    if interaction == "all":
        return decayed + spikes.to(decayed.dtype)
    return decayed.masked_fill(spikes.bool(), 1.0)


def triplet_dense(
    *,
    pre_spike: torch.Tensor,
    post_spike: torch.Tensor,
    r1: torch.Tensor,
    o1: torch.Tensor,
    r2_before: torch.Tensor,
    o2_before: torch.Tensor,
    a2_plus: float,
    a3_plus: float,
    a2_minus: float,
    a3_minus: float,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change ``(n_src, n_dst)`` of all-to-all synapses (batch mean of the samples)."""
    post_gain = post_spike.to(r1.dtype) * (a2_plus + a3_plus * o2_before)
    pre_gain = pre_spike.to(o1.dtype) * (a2_minus + a3_minus * r2_before)
    return _pairs(r1, post_gain) * ltp_gate - _pairs(pre_gain, o1) * ltd_gate


def triplet_one_to_one(
    *,
    pre_spike: torch.Tensor,
    post_spike: torch.Tensor,
    r1: torch.Tensor,
    o1: torch.Tensor,
    r2_before: torch.Tensor,
    o2_before: torch.Tensor,
    a2_plus: float,
    a3_plus: float,
    a2_minus: float,
    a3_minus: float,
    ltp_gate: Gate = 1.0,
    ltd_gate: Gate = 1.0,
) -> torch.Tensor:
    """Weight change ``(size,)`` of one-to-one synapses."""
    post_gain = post_spike.to(r1.dtype) * (a2_plus + a3_plus * o2_before)
    pre_gain = pre_spike.to(o1.dtype) * (a2_minus + a3_minus * r2_before)
    return _batch_mean(r1 * post_gain) * ltp_gate - _batch_mean(pre_gain * o1) * ltd_gate


class TripletSTDP(Behavior):
    """Triplet STDP (Pfister and Gerstner 2006, all-to-all minimal and general rule).

    Step semantics (the pair STDP convention: traces include the current step's spike).
    Each step the four traces first decay by ``1 - dt / tau`` (Euler, like
    :func:`~neurosush.synapses.traces.trace_step`). ``r1`` and ``o1`` then register this
    step's spikes (add 1 for ``interaction="all"``, set to 1 for ``"nearest"``) and are used
    including the current spike, exactly as :class:`~neurosush.synapses.plasticity.STDP`
    uses its pair traces after ``Traces`` ran. The triplet traces ``r2`` and ``o2`` enter
    the update as their decayed values from *before* this step's increment (the
    ``t - epsilon`` of the paper), and are incremented afterwards. With ``s_pre`` and
    ``s_post`` the spikes of the step::

        dw  = r1 * s_post * (a2_plus  + a3_plus  * o2_before)
        dw -= o1 * s_pre  * (a2_minus + a3_minus * r2_before)

    A simultaneous pre/post spike therefore contributes both terms, each pairing the spike
    with the other neuron's trace that already includes the simultaneous spike (so the pair
    parts equal ``a2_plus - a2_minus``, as in ``STDP``) while the triplet factors see only
    earlier spikes. A previous spike ``k`` steps before gives a factor ``(1 - dt / tau)^k``.

    Diehl and Cook (2015) is the special case ``a2_plus=0, a3_plus=nu_post, a2_minus=nu_pre,
    a3_minus=0``, ``interaction="nearest"``, ``tau_plus=20, tau_minus=20, tau_y=40``
    (``tau_x`` is then unused).

    Supports dense and one-to-one synapses; needs ``SpikeGather`` and an Axon on the
    destination. Activity may be batched: the weight change is the batch mean of the
    per-sample changes (traces are per sample). Weight-dependent gates from ``bound`` scale
    the potentiation and depression parts.

    Args:
        a2_plus: Pair potentiation amplitude.
        a3_plus: Triplet potentiation amplitude (post-pre-post).
        a2_minus: Pair depression amplitude.
        a3_minus: Triplet depression amplitude (pre-post-pre).
        tau_plus: Presynaptic fast trace ``r1`` time constant.
        tau_minus: Postsynaptic fast trace ``o1`` time constant.
        tau_x: Presynaptic slow trace ``r2`` time constant.
        tau_y: Postsynaptic slow trace ``o2`` time constant.
        interaction: ``"all"`` (all-to-all) or ``"nearest"`` (nearest-spike).
        w_min: Lower weight used by the bound.
        w_max: Upper weight used by the bound.
        bound: ``"none"``, ``"soft"`` or ``"hard"`` (see :mod:`neurosush.synapses.bounds`).
    """

    order = Order.PLASTICITY
    graph_safe = True
    supported: tuple[str, ...] = ("dense", "one_to_one")

    def __init__(
        self,
        *,
        a2_plus: float,
        a3_plus: float,
        a2_minus: float,
        a3_minus: float,
        tau_plus: float,
        tau_minus: float,
        tau_x: float,
        tau_y: float,
        interaction: Interaction = "all",
        w_min: float = 0.0,
        w_max: float = 1.0,
        bound: Literal["none", "soft", "hard"] = "none",
    ) -> None:
        amplitudes = (
            ("a2_plus", a2_plus),
            ("a3_plus", a3_plus),
            ("a2_minus", a2_minus),
            ("a3_minus", a3_minus),
        )
        for name, value in amplitudes:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        taus = (
            ("tau_plus", tau_plus),
            ("tau_minus", tau_minus),
            ("tau_x", tau_x),
            ("tau_y", tau_y),
        )
        for name, value in taus:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if interaction not in _INTERACTIONS:
            raise ValueError(f"interaction must be one of {_INTERACTIONS}, got {interaction!r}")
        if w_min >= w_max:
            raise ValueError(f"w_min ({w_min}) must be below w_max ({w_max})")
        if bound not in BOUNDS:
            raise ValueError(f"bound must be one of {sorted(BOUNDS)}, got {bound!r}")
        self.a2_plus, self.a3_plus = a2_plus, a3_plus
        self.a2_minus, self.a3_minus = a2_minus, a3_minus
        self.tau_plus, self.tau_minus, self.tau_x, self.tau_y = tau_plus, tau_minus, tau_x, tau_y
        self.interaction: Interaction = interaction
        self.w_min, self.w_max, self.bound = w_min, w_max, bound

    def initialize(self, syn: SynapseGroup) -> None:
        """Check collaborators, time constants and connectivity; allocate the traces."""
        name = type(self).__name__
        if not hasattr(syn, "post_spike"):
            raise RuntimeError(
                f"{name} on {syn.name} needs SpikeGather and an Axon on {syn.dst.name}"
            )
        kind = getattr(syn, "connectivity", None)
        if kind not in self.supported:
            raise ValueError(f"{name} does not support connectivity {kind!r} ({syn.name})")
        dt = syn.net.dt
        for label, tau in (
            ("tau_plus", self.tau_plus),
            ("tau_minus", self.tau_minus),
            ("tau_x", self.tau_x),
            ("tau_y", self.tau_y),
        ):
            if tau < dt:
                raise ValueError(f"{label} ({tau}) must be at least dt ({dt})")
        self.r1 = syn.src.state()
        self.r2 = syn.src.state()
        self.o1 = syn.dst.state()
        self.o2 = syn.dst.state()

    def compute(self, syn: SynapseGroup) -> torch.Tensor:
        """Weight change of this step; advances the four traces."""
        assert syn.weights is not None  # Supported inputs require weights at initialization.
        dt = syn.net.dt
        pre, post = syn.pre_spike, syn.post_spike
        r2_before = trace_decay(self.r2, tau=self.tau_x, dt=dt)
        o2_before = trace_decay(self.o2, tau=self.tau_y, dt=dt)
        self.r1 = trace_increment(
            trace_decay(self.r1, tau=self.tau_plus, dt=dt), pre, interaction=self.interaction
        )
        self.o1 = trace_increment(
            trace_decay(self.o1, tau=self.tau_minus, dt=dt), post, interaction=self.interaction
        )
        self.r2 = trace_increment(r2_before, pre, interaction=self.interaction)
        self.o2 = trace_increment(o2_before, post, interaction=self.interaction)
        ltp_gate, ltd_gate = BOUNDS[self.bound](syn.weights, self.w_min, self.w_max)
        kernel = triplet_dense if syn.connectivity == "dense" else triplet_one_to_one
        return kernel(
            pre_spike=pre,
            post_spike=post,
            r1=self.r1,
            o1=self.o1,
            r2_before=r2_before,
            o2_before=o2_before,
            a2_plus=self.a2_plus,
            a3_plus=self.a3_plus,
            a2_minus=self.a2_minus,
            a3_minus=self.a3_minus,
            ltp_gate=ltp_gate,
            ltd_gate=ltd_gate,
        )

    def forward(self, syn: SynapseGroup) -> None:
        """Apply this step's weight change."""
        assert syn.weights is not None  # Supported inputs require weights at initialization.
        syn.weights = syn.weights + self.compute(syn)

    def reset_state(self, syn: SynapseGroup) -> None:
        """Zero the four traces in place."""
        for trace in (self.r1, self.r2, self.o1, self.o2):
            trace.zero_()

    def state_dict(self) -> dict[str, Any]:
        """The four traces."""
        return {k: getattr(self, k).clone() for k in ("r1", "r2", "o1", "o2")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the traces saved by :meth:`state_dict`."""
        if set(state) != {"r1", "r2", "o1", "o2"}:
            raise KeyError(f"expected keys r1, r2, o1, o2, got {sorted(state)}")
        for key, value in state.items():
            current: torch.Tensor = getattr(self, key)
            value = torch.as_tensor(value)
            if value.shape != current.shape:
                raise ValueError(
                    f"{key} shape must be {tuple(current.shape)}, got {tuple(value.shape)}"
                )
            setattr(self, key, value.to(current))
