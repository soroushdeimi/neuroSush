"""Hierarchical predictive coding: inference and learning by minimizing free energy.

Follows Rao and Ballard (1999), "Predictive coding in the visual cortex", Nat. Neurosci.
2:79, in the notation of Bogacz (2017), "A tutorial on the free-energy framework for
modelling perception and learning", J. Math. Psychol. 76:198.

Layer 0 holds the observation and layer ``L`` the most abstract causes. Every layer
``l < L`` is predicted from the one above as ``W_l f(mu_{l+1})``; the top layer has a
Gaussian prior with mean ``prior_mean``. With per-unit variances ``sigma_l`` the free
energy of one sample is

    F = sum_l 1/2 sum_i (e_l,i^2 / sigma_l,i + log sigma_l,i),
    e_l = mu_l - W_l f(mu_{l+1})  (l < L),   e_L = mu_L - prior_mean.

With prediction errors ``eps_l = e_l / sigma_l``, gradient descent on ``F`` gives the
local rules of the model: inference ``dmu_l = -eps_l + f'(mu_l) * (eps_{l-1} W_{l-1})``,
learning ``dW_l = eps_l^T f(mu_{l+1})``, ``dprior = eps_L`` and
``dsigma_l = (eps_l^2 - 1 / sigma_l) / 2``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise

import torch

_Activation = Callable[[torch.Tensor], torch.Tensor]
_ACTIVATIONS: dict[str, tuple[_Activation, _Activation]] = {
    "linear": (lambda x: x, torch.ones_like),
    "tanh": (torch.tanh, lambda x: 1 - torch.tanh(x) ** 2),
}


class PredictiveCodingNetwork:
    """A hierarchy of layers that predict the layer below.

    Args:
        sizes: Units per layer, from the observation (``sizes[0]``) to the top.
        activation: ``"linear"`` or ``"tanh"``: the ``f`` applied before predicting.
        variances: Initial variance of every layer (one value per layer); default 1.
        dtype: Floating point type of all tensors.
        seed: Seed of the random initial weights.
    """

    def __init__(
        self,
        sizes: list[int],
        *,
        activation: str = "tanh",
        variances: list[float] | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        if len(sizes) < 1 or any(n < 1 for n in sizes):
            raise ValueError(f"sizes must be positive, got {sizes}")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )
        variances = variances if variances is not None else [1.0] * len(sizes)
        if len(variances) != len(sizes) or any(v <= 0 for v in variances):
            raise ValueError(f"need one positive variance per layer, got {variances}")
        self.sizes, self.dtype = list(sizes), dtype
        self.f, self.df = _ACTIVATIONS[activation]
        generator = torch.Generator().manual_seed(seed)
        self.weights = [
            torch.randn(lower, upper, generator=generator, dtype=dtype) / math.sqrt(upper)
            for lower, upper in pairwise(sizes)
        ]
        self.variances = [
            torch.full((n,), v, dtype=dtype) for n, v in zip(sizes, variances, strict=True)
        ]
        self.prior_mean = torch.zeros(sizes[-1], dtype=dtype)

    @property
    def depth(self) -> int:
        """Index ``L`` of the top layer."""
        return len(self.sizes) - 1

    def predict(self, top: torch.Tensor) -> list[torch.Tensor]:
        """Top-down sweep: every layer set to the prediction from the layer above."""
        mu = [top.to(self.dtype)]
        for weight in reversed(self.weights):
            mu.insert(0, self.f(mu[0]) @ weight.T)
        return mu

    def errors(self, mu: list[torch.Tensor]) -> list[torch.Tensor]:
        """Raw prediction errors ``e_l`` of every layer."""
        below = [m - self.f(up) @ w.T for m, up, w in zip(mu, mu[1:], self.weights, strict=False)]
        return [*below, mu[-1] - self.prior_mean]

    def free_energy(self, mu: list[torch.Tensor]) -> torch.Tensor:
        """``F`` of every sample, shape ``(batch,)`` (or a scalar for one sample)."""
        energy = sum(
            0.5 * (e**2 / v + v.log()).sum(-1)
            for e, v in zip(self.errors(mu), self.variances, strict=True)
        )
        assert isinstance(energy, torch.Tensor)  # The hierarchy has at least one layer.
        return energy

    def _eps(self, mu: list[torch.Tensor]) -> list[torch.Tensor]:
        return [e / v for e, v in zip(self.errors(mu), self.variances, strict=True)]

    def infer(
        self,
        observation: torch.Tensor,
        *,
        top: torch.Tensor | None = None,
        steps: int = 100,
        rate: float = 0.1,
        mu: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Relax the free layers by gradient descent on ``F``.

        Args:
            observation: Clamped layer 0, shape ``(sizes[0],)`` or ``(batch, sizes[0])``.
            top: Clamp the top layer as well (supervised use); otherwise it is free.
            steps: Gradient steps.
            rate: Step size.
            mu: Starting state; defaults to the top-down prediction from ``top``, or
                from the prior mean.
        """
        observation = observation.to(self.dtype)
        if mu is None:
            start = top if top is not None else self.prior_mean.expand(*observation.shape[:-1], -1)
            mu = self.predict(start)
        mu = [m.clone() for m in mu]
        mu[0] = observation.clone()
        last = self.depth if top is not None else self.depth + 1
        for _ in range(steps if last > 1 else 0):
            eps = self._eps(mu)
            for layer in range(1, last):
                feedback = eps[layer - 1] @ self.weights[layer - 1]
                mu[layer] = mu[layer] - rate * (eps[layer] - self.df(mu[layer]) * feedback)
        return mu

    def learn(
        self,
        mu: list[torch.Tensor],
        rate: float,
        *,
        prior_rate: float = 0.0,
        variance_rate: float = 0.0,
        min_variance: float = 1e-4,
    ) -> None:
        """One gradient step on ``F`` for the parameters, averaged over the batch.

        Args:
            mu: Relaxed state from :meth:`infer`.
            rate: Step size of the weights.
            prior_rate: Step size of the top layer's prior mean.
            variance_rate: Step size of the variances.
            min_variance: Floor that keeps variances positive.
        """
        eps = [e.reshape(-1, e.shape[-1]) for e in self._eps(mu)]
        rows = eps[0].shape[0]
        for layer, weight in enumerate(self.weights):
            upper = self.f(mu[layer + 1]).reshape(rows, -1)
            weight += rate * eps[layer].T @ upper / rows
        self.prior_mean += prior_rate * eps[-1].mean(0)
        if variance_rate:
            for layer, variance in enumerate(self.variances):
                change = 0.5 * (eps[layer] ** 2 - 1 / variance).mean(0)
                variance.copy_((variance + variance_rate * change).clamp(min=min_variance))
