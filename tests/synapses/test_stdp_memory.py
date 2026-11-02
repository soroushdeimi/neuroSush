"""Tests for trace-based STDP memory update."""

from __future__ import annotations

import pytest
import torch

from neurosush.synapses.stdp_memory import stdp_memory_update


def test_stdp_memory_update():
    memory = torch.zeros(2, 2)
    pre = torch.tensor([1.0, 0.0])
    post = torch.tensor([0.0, 1.0])
    trace_pre = torch.zeros(2)
    trace_post = torch.zeros(2)

    with pytest.raises(ValueError, match="a_minus"):
        stdp_memory_update(
            memory,
            pre,
            post,
            a_plus=1.0,
            a_minus=-0.1,
            trace_pre=trace_pre,
            trace_post=trace_post,
            tau_pre=10.0,
            tau_post=10.0,
            dt=1.0,
        )

    memory2 = torch.zeros(2, 2)
    pre2 = torch.tensor([1.0, 0.0])
    post2 = torch.tensor([0.0, 1.0])
    tp2 = torch.tensor([1.0, 0.0])
    tq2 = torch.tensor([0.0, 1.0])
    memory_new, tp_new, tq_new = stdp_memory_update(
        memory2,
        pre2,
        post2,
        a_plus=0.5,
        a_minus=0.2,
        trace_pre=tp2,
        trace_post=tq2,
        tau_pre=10.0,
        tau_post=10.0,
        dt=1.0,
        clip=2.0,
    )
    assert memory_new.shape == (2, 2)
    assert tp_new.shape == (2,)
    assert tq_new.shape == (2,)

    memory_clamp = torch.zeros(5, 5)
    pre_c = torch.ones(5)
    post_c = torch.ones(5)
    tp_c = torch.ones(5)
    tq_c = torch.ones(5)
    for _ in range(100):
        memory_clamp, tp_c, tq_c = stdp_memory_update(
            memory_clamp,
            pre_c,
            post_c,
            a_plus=1.0,
            a_minus=0.0,
            trace_pre=tp_c,
            trace_post=tq_c,
            tau_pre=10.0,
            tau_post=10.0,
            dt=1.0,
            clip=2.0,
        )
    assert (memory_clamp <= 2.0).all()
    assert (memory_clamp >= -2.0).all()
