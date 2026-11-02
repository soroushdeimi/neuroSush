"""Pure TEM objective terms with an optional seen-target mask."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TEMConfig
from .model import TEMOutput


def _mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    if mask.shape != values.shape or mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean with shape {values.shape}, got {mask}")
    return (values * mask).sum() / mask.sum().clamp_min(1)


def sensory_loss(logits: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean categorical cross entropy over selected observations."""
    return _mean(F.cross_entropy(logits, target, reduction="none"), mask)


def squared_loss(inferred: Tensor, generated: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared latent discrepancy, averaged over units then examples."""
    return _mean((inferred - generated).square().mean(-1), mask)


def activity_loss(g: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared entorhinal activity, before the configured weight."""
    return _mean(g.square().mean(-1), mask)


def sparsity_loss(p: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean absolute hippocampal activity, before the configured weight."""
    return _mean(p.abs().mean(-1), mask)


def total_loss(
    output: TEMOutput,
    target: Tensor,
    config: TEMConfig,
    *,
    iteration: int = 0,
    mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the weighted sum and raw terms; latent weights ramp by iteration.

    Targets are the observation associated with this output, predicted from the
    preceding state and action. Mask unseen targets when fitting slow weights.
    """
    terms = {
        "L_x_gen": sensory_loss(output.logits_gen, target, mask),
        "L_x_g": sensory_loss(output.logits_g, target, mask),
        "L_x_p": sensory_loss(output.logits_p, target, mask),
        "L_p": squared_loss(output.p_inf, output.p_gen, mask),
        "L_g": squared_loss(output.g_inf, output.g_prior, mask),
        "reg_g": activity_loss(output.g_inf, mask),
        "reg_p": sparsity_loss(output.p_inf, mask),
    }
    if config.calibrated_reductions:
        terms["L_p"] = _mean(0.5 * (output.p_inf - output.p_gen).square().sum(-1), mask)
        terms["L_g"] = _mean(0.5 * (output.g_inf - output.g_prior).square().sum(-1), mask)
        terms["reg_g"] = _mean(output.g_inf.square().sum(-1), mask)
        terms["reg_p"] = _mean(output.p_inf.abs().sum(-1), mask)
    weights = loss_weights(config, iteration)
    if config.inference_memory_loss:
        if output.p_memory is None:
            raise ValueError("inference_memory_loss requires p_memory")
        terms["L_p_x"] = _mean(0.5 * (output.p_inf - output.p_memory).square().sum(-1), mask)
    loss = torch.stack([term * weights[name] for name, term in terms.items()]).sum()
    return loss, terms


def loss_weights(config: TEMConfig, iteration: int) -> dict[str, float]:
    """Return objective coefficients at an optimizer update."""
    latent = config.schedules(iteration)[3]
    return {
        "L_x_gen": config.weight_x_gen,
        "L_x_g": config.weight_x_g,
        "L_x_p": config.weight_x_p,
        "L_p": config.weight_p * latent,
        "L_g": config.weight_g * latent,
        "reg_g": config.reg_g
        * (max(0.0, 1 - iteration / config.reg_g_horizon) if config.calibrated_reductions else 1),
        "reg_p": config.reg_p
        * (max(0.0, 1 - iteration / config.reg_p_horizon) if config.calibrated_reductions else 1),
        "L_p_x": config.weight_p * latent * min(1.0, iteration / config.correction_ramp),
    }
