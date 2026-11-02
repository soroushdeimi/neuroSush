"""Deterministic TEM inference, prediction, and differentiable fast memory.

Equations follow Whittington et al. (2020), PDF pp. 21-23. Dynamic state is
explicit: reset for every walk and detach only at BPTT boundaries. Predictions
for observation t are computed before incorporating observation t.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurosush.htm.sdr import random_sdr
from neurosush.memory import attractor, hebbian_update

from .config import TEMConfig
from .rotation import Rotation


def precision_mean(
    prior: Tensor, memory: Tensor, sigma_prior: Tensor, sigma_memory: Tensor
) -> Tensor:
    """Combine Gaussian means using inverse variances.

    Shapes: inputs broadcast to the returned mean shape. Units: sigmas are
    positive standard deviations in the same units as their means.
    """
    for name, sigma in (("sigma_prior", sigma_prior), ("sigma_memory", sigma_memory)):
        if not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
            raise ValueError(f"{name} must be finite and positive, got {sigma.tolist()}")
    return _precision_mean(prior, memory, sigma_prior, sigma_memory)


def _precision_mean(
    prior: Tensor, memory: Tensor, sigma_prior: Tensor, sigma_memory: Tensor
) -> Tensor:
    prior_precision = sigma_prior.reciprocal().square()
    memory_precision = sigma_memory.reciprocal().square()
    return (prior * prior_precision + memory * memory_precision) / (
        prior_precision + memory_precision
    )


def bind(g: Tensor, x: Tensor) -> Tensor:
    """Flatten the outer product, with sensory coordinates varying fastest.

    Shapes: (..., G), (..., X) -> (..., G*X). Units: dimensionless activations.
    """
    return F.leaky_relu((g.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2), 0.01).clamp(max=1)


@dataclass(frozen=True)
class TEMState:
    """Per-walk state; memories are tensors rather than optimizer parameters.

    Shapes per module: g (B, G), x (B, X), gen/inf (B, G*X, G*X).
    Units: dimensionless activations and associative weights.
    """

    g: tuple[Tensor, ...]
    x: tuple[Tensor, ...]
    gen: tuple[Tensor, ...]
    inf: tuple[Tensor, ...]

    def detach(self) -> TEMState:
        """Cut the graph at a BPTT boundary without changing values."""
        return TEMState(
            *(tuple(t.detach() for t in group) for group in (self.g, self.x, self.gen, self.inf))
        )


@dataclass(frozen=True)
class TEMOutput:
    """Prediction before observation, reconstructions, and aligned latent variables.

    Shapes: logits (B, S), g (B, sum(G)), p (B, sum(G*X)); all align
    with the current observation. Units: logits and dimensionless activations.
    """

    logits_gen: Tensor
    logits_g: Tensor
    logits_p: Tensor
    g_prior: Tensor
    g_inf: Tensor
    p_gen: Tensor
    p_inf: Tensor
    p_memory: Tensor | None = None


class Dense(nn.Module):
    """Linear map initialized exclusively from the supplied generator."""

    def __init__(
        self, inputs: int, outputs: int, generator: torch.Generator, *, zero: bool = False
    ) -> None:
        super().__init__()
        weight = torch.empty(outputs, inputs, device=generator.device)
        if zero:
            weight.zero_()
        else:
            weight.uniform_(-1 / sqrt(inputs), 1 / sqrt(inputs), generator=generator)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(outputs, device=generator.device))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the affine map."""
        return F.linear(x, self.weight, self.bias)


class MLP(nn.Module):
    """Two affine layers with a tanh hidden activation."""

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        generator: torch.Generator,
        *,
        zero: bool = False,
    ) -> None:
        super().__init__()
        self.first = Dense(inputs, hidden, generator)
        self.last = Dense(hidden, outputs, generator, zero=zero)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the network."""
        return self.last.forward(torch.tanh(self.first.forward(x)))


class TEM(nn.Module):
    """Independent TEM streams with separate generative and inference memories.

    Actions 0-3 are graph directions; action 4 is stay. Each action MLP produces
    a transition matrix, initially zero, applied as a residual to g. No node IDs
    or coordinates enter inference. ``prior_override`` is reserved for oracle tests.
    """

    codes: Tensor

    def __init__(self, config: TEMConfig, *, generator: torch.Generator) -> None:
        super().__init__()
        self.config = config
        codes: list[Tensor] = []
        while len(codes) < config.sensory:
            code = random_sdr(config.compressed, 2, generator=generator, device=generator.device)
            if not any(torch.equal(code, old) for old in codes):
                codes.append(code)
        self.register_buffer("codes", torch.stack(codes).float())
        self.g_init = nn.ParameterList(
            nn.Parameter(
                torch.randn(n, generator=generator, device=generator.device) * config.initial_std
            )
            for n in config.modules
        )
        self.transitions = nn.ModuleList(
            Rotation(n, generator)
            if config.transition == "rotation"
            else MLP(5, config.hidden, n * n, generator, zero=True)
            for n in config.modules
        )
        self.memory_means = nn.ModuleList(
            MLP(n, config.hidden, n, generator) for n in config.modules
        )
        self.sigmas = nn.ModuleList(
            MLP(2 * n, config.hidden, 2 * n, generator) for n in config.modules
        )
        self.alpha = nn.Parameter(
            torch.linspace(2.0, -2.0, len(config.modules), device=generator.device)
        )
        self.decoder = Dense(config.compressed, config.sensory, generator)
        if config.reliable_fusion:
            self.path_sigmas = nn.ModuleList(
                MLP(n, config.hidden, n, generator) for n in config.modules
            )
            self.memory_sigmas = nn.ModuleList(
                Dense(2, n, generator, zero=True) for n in config.modules
            )
            for module in self.memory_sigmas:
                network = cast(Dense, module)
                with torch.no_grad():
                    network.bias.fill_(10.0)
                    network.weight[:, 1].fill_(1.0)

    def reset(self, batch: int, *, generator: torch.Generator | None = None) -> TEMState:
        """Start a walk from the learned anchor with zero filters and memories.

        The optional generator is retained for compatibility; reset draws no noise.
        """
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        device = self.alpha.device
        gs = tuple(
            (F.normalize(g, dim=-1) if self.config.transition == "rotation" else g.clamp(-1, 1))
            .unsqueeze(0)
            .expand(batch, -1)
            for g in self.g_init
        )
        xs = tuple(torch.zeros(batch, self.config.compressed, device=device) for _ in gs)
        memories = tuple(
            torch.zeros(
                batch, n * self.config.compressed, n * self.config.compressed, device=device
            )
            for n in self.config.modules
        )
        return TEMState(gs, xs, memories, tuple(m.clone() for m in memories))

    def transition(self, g: Tensor, action: Tensor, module: int) -> Tensor:
        """Apply the configured rotation or bounded residual matrix transition."""
        network = self.transitions[module]
        if isinstance(network, Rotation):
            return network.forward(g, action)
        if bool(((action < 0) | (action > 4)).any()):
            raise ValueError(f"action must be in [0, 4], got {action}")
        n = self.config.modules[module]
        weights = (
            self.transitions[module].forward(F.one_hot(action, 5).to(g.dtype)).reshape(-1, n, n)
        )
        delta = cast(Tensor, (weights @ g.unsqueeze(-1)).squeeze(-1))
        return (g + delta).clamp(-1, 1)

    def _transition(self, g: Tensor, features: Tensor, module: int) -> Tensor:
        network = self.transitions[module]
        n = self.config.modules[module]
        if isinstance(network, Rotation):
            angle = features @ network.angles
            x, y = g.reshape(-1, n // 2, 2).unbind(-1)
            cosine, sine = angle.cos(), angle.sin()
            return torch.stack((cosine * x - sine * y, sine * x + cosine * y), -1).flatten(1)
        weights = network.forward(features).reshape(-1, n, n)
        delta = cast(Tensor, (weights @ g.unsqueeze(-1)).squeeze(-1))
        return (g + delta).clamp(-1, 1)

    def _validate_ranges(self, observation: Tensor | None, action: Tensor | None) -> None:
        if observation is not None:
            valid = (observation >= 0) & (observation < self.config.sensory)
            if not torch.equal(valid, torch.ones_like(valid)):
                raise ValueError(
                    f"observation must be in [0, {self.config.sensory}), got {observation}"
                )
        if action is not None:
            valid = (action >= 0) & (action <= 4)
            if not torch.equal(valid, torch.ones_like(valid)):
                raise ValueError(f"action must be in [0, 4], got {action}")

    def _retrieve(self, cue: Tensor, memory: Tensor) -> Tensor:
        if not self.config.bounded_attractor_activation:
            return attractor(cue, memory, self.config.iterations, retention=self.config.retention)
        h = F.leaky_relu(cue.clamp(-1, 1), 0.01)
        for _ in range(self.config.iterations):
            drive = self.config.retention * h + (memory @ h.unsqueeze(-1)).squeeze(-1)
            h = F.leaky_relu(drive.clamp(-1, 1), 0.01)
        return h

    def _memory_sigma(self, mean: Tensor, error: Tensor, module: int) -> Tensor:
        quality = torch.stack((mean.square().sum(-1), error), -1).detach()
        return F.softplus(self.memory_sigmas[module].forward(quality)) + self.config.sigma_floor

    def _decode(self, patterns: list[Tensor]) -> Tensor:
        compressed = self.config.compressed
        pooled = torch.stack([p.reshape(p.shape[0], -1, compressed).sum(1) for p in patterns]).mean(
            0
        )
        return self.decoder.forward(pooled)

    def forward(
        self,
        observation: Tensor,
        action: Tensor,
        state: TEMState,
        *,
        decay: float | None = None,
        rate: float | None = None,
        prior_override: tuple[Tensor, ...] | None = None,
    ) -> tuple[TEMOutput, TEMState]:
        """Predict observation, infer its latents, then write both fast memories.

        ``action`` leads from the previous observation to this observation. At a
        walk's initial observation use stay. ``logits_gen`` never uses its target.
        Memory remains differentiable through the chunk, including earlier writes.
        """
        c = self.config
        if observation.shape != (state.g[0].shape[0],):
            raise ValueError(
                f"observation must have shape {(state.g[0].shape[0],)}, got {observation.shape}"
            )
        self._validate_ranges(observation, None)
        if action.shape != observation.shape:
            raise ValueError(f"action shape must match {observation.shape}, got {action.shape}")
        if prior_override is None:
            if c.transition == "rotation":
                for f, n in enumerate(c.modules):
                    g = state.g[f]
                    if g.ndim != 2 or g.shape[-1] != n:
                        raise ValueError(f"g must have shape (batch, {n}), got {g.shape}")
                    if action.shape != g.shape[:1]:
                        raise ValueError(f"action shape must be {g.shape[:1]}, got {action.shape}")
                    if action.dtype != torch.long:
                        raise ValueError(f"action dtype must be torch.int64, got {action.dtype}")
                    if f == 0:
                        self._validate_ranges(None, action)
            else:
                self._validate_ranges(None, action)
        return self._forward(
            observation, action, state, decay=decay, rate=rate, prior_override=prior_override
        )

    def _forward(
        self,
        observation: Tensor,
        action: Tensor,
        state: TEMState,
        *,
        decay: float | None = None,
        rate: float | None = None,
        prior_override: tuple[Tensor, ...] | None = None,
    ) -> tuple[TEMOutput, TEMState]:
        c = self.config
        codes = self.codes[observation]
        features = None
        encoded: dict[torch.dtype, Tensor] = {}
        if prior_override is None:
            if c.transition == "rotation":
                rotation = cast(Rotation, self.transitions[0])
                features = rotation.directions[action]
            else:
                # Explicit classes avoid one_hot's data-dependent range checks.
                if action.dtype != torch.long:
                    raise RuntimeError(
                        "one_hot is only applicable to index tensor of type LongTensor."
                    )
                features = (action.unsqueeze(-1) == torch.arange(5, device=action.device)).long()
        xs, gs, priors, ps, predictions, generated, gens, infs, retrieved = ([] for _ in range(9))
        for f, n in enumerate(c.modules):
            if features is not None and state.g[f].dtype not in encoded:
                encoded[state.g[f].dtype] = features.to(state.g[f].dtype)
            prior = (
                self._transition(state.g[f], encoded[state.g[f].dtype], f)
                if prior_override is None
                else prior_override[f]
            )
            location_cue = prior.repeat_interleave(c.compressed, -1)
            predictions.append(self._retrieve(location_cue, state.gen[f]))
            alpha = self.alpha[f].sigmoid()
            x = (1 - alpha) * state.x[f] + alpha * codes
            sensory = F.normalize(F.relu(x - x.mean(-1, keepdim=True)), dim=-1)
            sensory_cue = sensory.repeat(1, n)
            p_memory = self._retrieve(sensory_cue, state.inf[f])
            pooled = p_memory.reshape(-1, n, c.compressed).sum(-1)
            raw_mean = self.memory_means[f].forward(pooled)
            memory_mean = raw_mean.clamp(-1, 1)
            if c.reliable_fusion:
                error = F.cross_entropy(self._decode([p_memory]), observation, reduction="none")
                sigma_prior = F.softplus(self.path_sigmas[f].forward(state.g[f])) + c.sigma_floor
                sigma_memory = self._memory_sigma(raw_mean, error, f)
            else:
                sigma = (
                    F.softplus(self.sigmas[f].forward(torch.cat((prior, memory_mean), -1))) + 1e-3
                )
                sigma_prior, sigma_memory = sigma.chunk(2, -1)
            g = (
                _precision_mean(prior, memory_mean, sigma_prior, sigma_memory)
                if prior_override is None
                else prior
            )
            if c.reliable_fusion and prior_override is None:
                empty = state.inf[f].detach().abs().sum((-2, -1)) == 0
                g = torch.where(empty.unsqueeze(-1), prior, g)
            p = bind(g, sensory)
            p_gen = self._retrieve(g.repeat_interleave(c.compressed, -1), state.gen[f])
            gen = hebbian_update(
                state.gen[f],
                p,
                p_gen,
                decay=c.memory_decay if decay is None else decay,
                rate=c.memory_rate if rate is None else rate,
            )
            inf = hebbian_update(
                state.inf[f],
                p,
                p_memory,
                decay=c.memory_decay if decay is None else decay,
                rate=c.memory_rate if rate is None else rate,
            )
            retrieved.append(p_memory)
            xs.append(x)
            gs.append(g)
            priors.append(prior)
            ps.append(p)
            generated.append(p_gen)
            gens.append(gen)
            infs.append(inf)
        output = TEMOutput(
            self._decode(predictions),
            self._decode(generated),
            self._decode(ps),
            torch.cat(priors, -1),
            torch.cat(gs, -1),
            torch.cat(generated, -1),
            torch.cat(ps, -1),
            torch.cat(retrieved, -1),
        )
        return output, TEMState(tuple(gs), tuple(xs), tuple(gens), tuple(infs))
