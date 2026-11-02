"""Surrogate gradient spike functions and LIF dynamics with surrogate gradients."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import Tensor

from neurosush.neurons.dynamics import euler_step, lif_derivative


class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient function for spike generation."""

    @staticmethod
    def forward(
        ctx: Any,
        v: Tensor,
        threshold: float,
        width: float,
        kind: str,
    ) -> Tensor:
        """Compute forward surrogate step: 1 if v > threshold else 0."""
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.width = width
        ctx.kind = kind
        return (v - threshold > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        """Compute backward surrogate gradient."""
        (v,) = ctx.saved_tensors
        threshold = ctx.threshold
        width = ctx.width
        kind = ctx.kind
        x = v - threshold
        if kind == "arctan":
            d_v = width / (1.0 + (math.pi * width * x / 2.0).square()) / 2.0
        elif kind == "triangle":
            d_v = (1.0 - x.abs() / width).clamp(min=0.0)
        else:
            raise ValueError(f"Unknown surrogate kind: {kind}")
        return grad_output * d_v, None, None, None


def spike(v: Tensor, threshold: float = 1.0, width: float = 2.0, kind: str = "arctan") -> Tensor:
    """Return 0/1 spike tensor with surrogate gradient."""
    if width <= 0 or not math.isfinite(width):
        raise ValueError(f"width must be positive and finite, got {width}")
    if kind not in ("arctan", "triangle"):
        raise ValueError(f"kind must be 'arctan' or 'triangle', got {kind}")
    return cast(Tensor, SurrogateSpike.apply(v, threshold, width, kind))


def lif_step(
    v: Tensor,
    current: Tensor | float,
    *,
    tau: float,
    dt: float,
    v_rest: float,
    threshold: float,
    reset: float,
    resistance: float = 1.0,
    width: float = 2.0,
    kind: str = "arctan",
) -> tuple[Tensor, Tensor]:
    """One Euler step of Leaky Integrate-and-Fire with surrogate spiking and hard reset."""
    dv_dt = lif_derivative(v, current, v_rest=v_rest, resistance=resistance)
    v_trial = euler_step(v, dv_dt, tau=tau, dt=dt)
    spikes = spike(v_trial, threshold=threshold, width=width, kind=kind)
    reset_tensor = torch.as_tensor(reset, dtype=v_trial.dtype, device=v_trial.device)
    v_next = torch.where(spikes > 0.5, reset_tensor, v_trial)
    return v_next, spikes
