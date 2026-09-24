"""Active dendrites: context-gated neurons for multi-task learning.

Follows Iyer et al. (2022), "Avoiding catastrophe: active dendrites enable multi-task
learning in dynamic environments", Front. Neurorobotics 16:846219. A neuron computes a
feedforward response ``y = w . x + b``; each of its dendritic segments ``u_j`` compares
itself with a context vector ``c``, and the strongest segment ``j* = argmax_j u_j . c``
gates the output: ``y * sigmoid(u_j* . c)``. A k-winners-take-all step then keeps the
``k`` largest outputs of the layer. Gradients reach only the selected segment, so
different contexts train different segments and route through different neurons.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` largest values of the last dimension and set the rest to zero."""
    if not 1 <= k <= x.shape[-1]:
        raise ValueError(f"k must be in [1, {x.shape[-1]}], got {k}")
    winners = x.topk(k, dim=-1).indices
    mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, winners, True)
    return x * mask


class ActiveDendrites(nn.Module):
    """Linear layer whose units are gated by dendritic segments reading a context vector.

    Args:
        in_features: Size of the feedforward input.
        out_features: Number of units.
        context_features: Size of the context vector.
        segments: Dendritic segments per unit.
        absolute: Select the segment with the largest ``|u_j . c|`` instead of the
            largest ``u_j . c`` so that a segment can also silence its unit
            ("absolute max gating").
        k: Keep only the ``k`` largest outputs (k-winners-take-all); ``None`` keeps all.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        segments: int,
        *,
        absolute: bool = False,
        k: int | None = None,
    ) -> None:
        super().__init__()
        if segments < 1:
            raise ValueError(f"segments must be positive, got {segments}")
        if k is not None and not 1 <= k <= out_features:
            raise ValueError(f"k must be in [1, {out_features}], got {k}")
        self.linear = nn.Linear(in_features, out_features)
        bound = 1 / math.sqrt(context_features)
        self.segments = nn.Parameter(
            torch.empty(out_features, segments, context_features).uniform_(-bound, bound)
        )
        self.absolute, self.k = absolute, k

    def dendritic_activation(self, context: torch.Tensor) -> torch.Tensor:
        """``u_j* . c`` of every unit, shape ``(..., out_features)``."""
        responses = torch.einsum("osc,...c->...os", self.segments, context)
        score = responses.abs() if self.absolute else responses
        selected = score.argmax(-1, keepdim=True)
        return responses.gather(-1, selected).squeeze(-1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Gated, optionally k-WTA-sparsified output of shape ``(..., out_features)``."""
        out = self.linear(x) * torch.sigmoid(self.dendritic_activation(context))
        return out if self.k is None else kwta(out, self.k)
