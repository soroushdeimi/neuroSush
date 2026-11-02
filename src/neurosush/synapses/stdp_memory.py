"""Trace-based STDP memory update for spiking networks."""

from __future__ import annotations

import torch

from neurosush.synapses.plasticity import stdp_dense
from neurosush.synapses.traces import trace_step


def stdp_memory_update(
    memory: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    *,
    a_plus: float,
    a_minus: float,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    tau_pre: float,
    tau_post: float,
    dt: float,
    clip: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace-based STDP on a (pre, post) fast-memory matrix M, clamped to [-clip, clip].

    Sign convention: a_plus > 0 for potentiation, a_minus >= 0 for depression subtracted
    (-a_minus * pre_spike * post_trace). Note that the paper's Eq. 9 and Table 1
    disagree on the sign of A-.
    """
    if a_minus < 0:
        raise ValueError(f"a_minus must be non-negative, got {a_minus}")

    trace_pre_new = trace_step(trace_pre, pre, tau=tau_pre, dt=dt)
    trace_post_new = trace_step(trace_post, post, tau=tau_post, dt=dt)

    dw = stdp_dense(
        pre_spike=pre,
        pre_trace=trace_pre_new,
        post_spike=post,
        post_trace=trace_post_new,
        a_plus=a_plus,
        a_minus=a_minus,
    )
    memory_new = (memory + dw).clamp(-clip, clip)
    return memory_new, trace_pre_new, trace_post_new
