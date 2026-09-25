"""The HTM spatial pooler: learns a stable, sparse code of its input.

Follows Cui, Ahmad and Hawkins (2017), "The HTM Spatial Pooler - a neocortical algorithm
for online sparse distributed coding", Front. Comput. Neurosci. 11:111. Each column has a
potential pool of inputs near its center; synapses with permanence at least ``connected``
count toward the column's overlap with the input; inhibition keeps a fixed fraction of the
columns active; active columns learn Hebbian permanence updates; boosting and weak-column
bumping keep every column in use.
"""

from __future__ import annotations

import math
from itertools import product

import torch
import torch.nn.functional as F

from neurosush.htm._state import checked

# what learning changes, plus the random pools and tie-breaks it started from
_STATE = ("potential", "permanences", "tie_break", "boost", "active_duty", "overlap_duty")


def _as_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if not 1 <= len(shape) <= 2 or any(d < 1 for d in shape):
        raise ValueError(f"shapes must have one or two positive dimensions, got {shape}")
    return shape


class SpatialPooler:
    """Spatial pooler with topology, global or local inhibition, boosting and bumping.

    Args:
        input_shape: Shape of the input (1-D or 2-D).
        column_shape: Shape of the column grid, same number of dimensions as the input.
        potential_radius: Half-width, in inputs, of the square around a column's center
            from which its potential pool is drawn.
        potential_pct: Fraction of that square that is in the pool.
        density: Target fraction of active columns (``local_area_density``).
        global_inhibition: Keep the top columns of the whole grid; otherwise of every
            neighborhood of radius ``inhibition_radius``.
        inhibition_radius: Half-width, in columns, of the local inhibition neighborhood.
        stimulus_threshold: Overlaps below this count as zero.
        connected: Permanence at which a synapse is connected.
        active_inc: Permanence increase of synapses from active inputs.
        inactive_dec: Permanence decrease of synapses from inactive inputs.
        boost_strength: ``beta`` of ``b = exp(-beta * (duty - target))``; 0 disables it.
        duty_cycle_period: Averaging window of the duty cycles.
        min_overlap_duty: Columns whose overlap duty falls below this fraction of their
            neighborhood's maximum have all potential permanences raised.
        seed: Seed of the random potential pools and permanences.
    """

    def __init__(
        self,
        input_shape: int | tuple[int, ...],
        column_shape: int | tuple[int, ...],
        *,
        potential_radius: int = 16,
        potential_pct: float = 0.5,
        density: float = 0.02,
        global_inhibition: bool = True,
        inhibition_radius: int = 5,
        stimulus_threshold: int = 0,
        connected: float = 0.1,
        active_inc: float = 0.05,
        inactive_dec: float = 0.008,
        boost_strength: float = 0.0,
        duty_cycle_period: int = 1000,
        min_overlap_duty: float = 0.001,
        seed: int = 0,
    ) -> None:
        self.input_shape, self.column_shape = _as_shape(input_shape), _as_shape(column_shape)
        if len(self.input_shape) != len(self.column_shape):
            raise ValueError("input_shape and column_shape need the same number of dimensions")
        if not 0 < density <= 1:
            raise ValueError(f"density must be in (0, 1], got {density}")
        if not 0 < potential_pct <= 1:
            raise ValueError(f"potential_pct must be in (0, 1], got {potential_pct}")
        for name, value in (
            ("potential_radius", potential_radius),
            ("inhibition_radius", inhibition_radius),
            ("boost_strength", boost_strength),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if duty_cycle_period < 1:
            raise ValueError(f"duty_cycle_period must be positive, got {duty_cycle_period}")
        self.n_inputs = math.prod(self.input_shape)
        self.n_columns = math.prod(self.column_shape)
        self.density, self.global_inhibition = density, global_inhibition
        self.inhibition_radius, self.stimulus_threshold = inhibition_radius, stimulus_threshold
        self.connected, self.active_inc, self.inactive_dec = connected, active_inc, inactive_dec
        self.boost_strength, self.duty_cycle_period = boost_strength, duty_cycle_period
        self.min_overlap_duty = min_overlap_duty
        generator = torch.Generator().manual_seed(seed)
        self.potential = self._potential_pools(potential_radius, potential_pct, generator)
        jitter = (torch.rand(self.n_columns, self.n_inputs, generator=generator) - 0.5) * 0.2
        self.permanences = ((connected + jitter).clamp(0, 1)) * self.potential
        # a tiny fixed per-column offset breaks ties in inhibition deterministically
        self.tie_break = torch.rand(self.n_columns, generator=generator) * 1e-6
        self.boost = torch.ones(self.n_columns)
        self.active_duty = torch.zeros(self.n_columns)
        self.overlap_duty = torch.zeros(self.n_columns)
        self.iteration = 0
        # connected synapses as floats for the overlap product; kept in step with the rows
        # that learning changes, and rebuilt when permanences are replaced from outside
        self._connected = torch.empty(0)
        self._connected_key: tuple[int, int] | None = None

    def _center(self, column: tuple[int, ...]) -> tuple[int, ...]:
        """Input coordinates at the center of a column's receptive field."""
        return tuple(
            int((c + 0.5) * i / k)
            for c, i, k in zip(column, self.input_shape, self.column_shape, strict=True)
        )

    def _potential_pools(self, radius: int, pct: float, generator: torch.Generator) -> torch.Tensor:
        """Bool ``(columns, inputs)``: ``pct`` of the inputs within ``radius`` of each center."""
        grid = torch.stack(
            torch.meshgrid(*(torch.arange(d) for d in self.input_shape), indexing="ij"), -1
        ).reshape(-1, len(self.input_shape))
        pools = torch.zeros(self.n_columns, self.n_inputs, dtype=torch.bool)
        for index, column in enumerate(product(*(range(d) for d in self.column_shape))):
            center = torch.tensor(self._center(column))
            near = ((grid - center).abs() <= radius).all(-1).nonzero().flatten()
            count = max(1, round(pct * len(near)))
            chosen = near[torch.randperm(len(near), generator=generator)[:count]]
            pools[index, chosen] = True
        return pools

    def overlap(self, x: torch.Tensor) -> torch.Tensor:
        """Connected synapses with active inputs per column, zero below the threshold.

        ``x`` has shape ``(*batch, *input_shape)``; the result ``(*batch, columns)``.
        """
        return self._overlap(x.reshape(*x.shape[: x.dim() - len(self.input_shape)], -1))

    def _connected_synapses(self) -> torch.Tensor:
        """``permanences >= connected`` as floats, rebuilt only when permanences changed."""
        key = (id(self.permanences), self.permanences._version)
        if self._connected_key != key:
            self._connected = (self.permanences >= self.connected).to(torch.float32)
            self._connected_key = key
        return self._connected

    def _set_rows(self, rows: torch.Tensor, values: torch.Tensor) -> None:
        """Write new permanences for ``rows`` and update their connected synapses."""
        connected = self._connected_synapses()
        self.permanences[rows] = values
        connected[rows] = (values >= self.connected).to(torch.float32)
        self._connected_key = (id(self.permanences), self.permanences._version)

    def _overlap(self, flat: torch.Tensor) -> torch.Tensor:
        """:meth:`overlap` of inputs already flattened to ``(..., n_inputs)``."""
        scores = flat.float() @ self._connected_synapses().T
        return torch.where(scores >= self.stimulus_threshold, scores, torch.zeros_like(scores))

    def inhibit(self, scores: torch.Tensor) -> torch.Tensor:
        """Active columns (bool) from boosted overlap scores of shape ``(..., columns)``."""
        candidate = scores > 0
        ranked = torch.where(candidate, scores + self.tie_break, torch.full_like(scores, -1.0))
        if self.global_inhibition:
            k = max(1, round(self.density * self.n_columns))
            winners = ranked.topk(k, dim=-1).indices
            active = torch.zeros_like(candidate).scatter_(-1, winners, True)
            return active & candidate
        return candidate & (self._stronger_neighbors(ranked) < self._local_quota())

    def _neighborhoods(self, values: torch.Tensor, fill: float) -> torch.Tensor:
        """Every column's neighborhood values, shape ``(..., columns, neighborhood)``."""
        r, dims = self.inhibition_radius, len(self.column_shape)
        lead = values.shape[:-1]
        grid = values.reshape(-1, 1, *self.column_shape)
        if dims == 1:
            padded = F.pad(grid, (r, r), value=fill)
            patches = padded.unfold(-1, 2 * r + 1, 1)
        else:
            padded = F.pad(grid, (r, r, r, r), value=fill)
            patches = padded.unfold(-2, 2 * r + 1, 1).unfold(-2, 2 * r + 1, 1).flatten(-2)
        return patches.reshape(*lead, self.n_columns, -1)

    def _stronger_neighbors(self, ranked: torch.Tensor) -> torch.Tensor:
        return (self._neighborhoods(ranked, -2.0) > ranked.unsqueeze(-1)).sum(-1)

    def _local_quota(self) -> torch.Tensor:
        """Winners allowed in each neighborhood: ``density * its size`` (at least one)."""
        size = self._neighborhoods(torch.ones(self.n_columns), 0.0).sum(-1)
        return (self.density * size).round().clamp(min=1)

    def compute(self, x: torch.Tensor, *, learn: bool = True) -> torch.Tensor:
        """Active columns for input ``x`` of shape ``(*batch, *input_shape)``.

        With ``learn``, every sample of a batch updates the pooler in turn.
        """
        batch = x.shape[: x.dim() - len(self.input_shape)]
        flat = x.reshape(-1, self.n_inputs).bool()
        if not learn:
            active = self.inhibit(self._overlap(flat) * self.boost)
            return active.reshape(*batch, *self.column_shape)
        outputs = [self._learn_one(sample) for sample in flat]
        return torch.stack(outputs).reshape(*batch, *self.column_shape)

    def _learn_one(self, x: torch.Tensor) -> torch.Tensor:
        overlaps = self._overlap(x)
        active = self.inhibit(overlaps * self.boost)
        # Hebbian update of active columns' potential synapses (Cui et al. 2017, eq. 4); the
        # other rows would only be clamped and masked again, which leaves them unchanged
        rows = active.nonzero().flatten()
        delta = torch.where(x, self.active_inc, -self.inactive_dec)
        pool = self.potential[rows]
        self._set_rows(rows, (self.permanences[rows] + delta * pool).clamp(0, 1) * pool)
        self.iteration += 1
        period = min(self.iteration, self.duty_cycle_period)
        self.active_duty += (active.float() - self.active_duty) / period
        self.overlap_duty += ((overlaps > 0).float() - self.overlap_duty) / period
        self._update_boost()
        self._bump_weak_columns()
        return active

    def _neighborhood_mean(self, values: torch.Tensor) -> torch.Tensor:
        if self.global_inhibition:
            return values.mean().expand_as(values)
        total = self._neighborhoods(values, 0.0).sum(-1)
        return total / self._neighborhoods(torch.ones_like(values), 0.0).sum(-1)

    def _update_boost(self) -> None:
        """``b = exp(-beta * (active_duty - target))`` (Cui et al. 2017, eq. 6)."""
        if self.boost_strength > 0:
            target = self._neighborhood_mean(self.active_duty)
            self.boost = torch.exp(-self.boost_strength * (self.active_duty - target))

    def _bump_weak_columns(self) -> None:
        """Raise all potential permanences of columns that rarely overlap their input."""
        if self.global_inhibition:
            reference = self.overlap_duty.max().expand_as(self.overlap_duty)
        else:
            reference = self._neighborhoods(self.overlap_duty, 0.0).max(-1).values
        weak = (self.overlap_duty < self.min_overlap_duty * reference).nonzero().flatten()
        if len(weak):
            bump = (0.1 * self.connected) * self.potential[weak]
            self._set_rows(weak, (self.permanences[weak] + bump).clamp(0, 1))

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """A copy of the learned state; with it a pooler of the same shape continues exactly."""
        state: dict[str, torch.Tensor | int] = {
            name: getattr(self, name).clone() for name in _STATE
        }
        state["iteration"] = self.iteration
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor | int]) -> None:
        """Restore a :meth:`state_dict` saved from a pooler with the same shapes."""
        for name in _STATE:
            setattr(self, name, checked(name, state[name], getattr(self, name)))
        self.iteration = int(state["iteration"])
