"""Read-only gradient and latent saturation diagnostics for TEM training."""

from __future__ import annotations

from torch import Tensor, nn

from .model import TEMOutput


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return each parameter's Euclidean gradient norm before global clipping."""
    return {
        f"grad/{name}": float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0
        for name, parameter in model.named_parameters()
    }


def saturation(output: TEMOutput) -> dict[str, Tensor]:
    """Return fractions at g's two bounds and the generative attractor's upper cap."""
    return {
        "g_prior_cap": (output.g_prior.detach().abs() >= 1 - 1e-6).float().mean(),
        "g_inf_cap": (output.g_inf.detach().abs() >= 1 - 1e-6).float().mean(),
        "p_gen_cap": (output.p_gen.detach() >= 1 - 1e-6).float().mean(),
        "g_inf_rms": output.g_inf.detach().square().mean().sqrt(),
        "p_gen_rms": output.p_gen.detach().square().mean().sqrt(),
    }
