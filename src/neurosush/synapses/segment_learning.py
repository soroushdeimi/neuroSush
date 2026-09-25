"""Learning on active dendritic segments: the temporal memory rules in spike time.

This is the learning of Hawkins and Ahmad (2016), driven by spikes instead of discrete
steps. The "previous element" of the temporal memory becomes the presynaptic cells that
fired, or won, within ``context`` time units before now; the current element's own spikes
are more recent than ``context[0]`` and do not count.

When cells of a minicolumn fire:

- if some fired cells have a segment whose plateau started within the context window
  (they were predicted), each such segment is reinforced: ``+increment`` for synapses from
  context cells, ``-decrement`` for the others, and it grows synapses to context winners
  up to ``max_new_synapses`` matched ones;
- otherwise the minicolumn burst: its best matching segment (most synapses from context
  cells, at least ``min_threshold``) learns the same way, or else its least used cell gets
  a new segment (replacing its least recently used one when full) grown to the context
  winners. That cell is the minicolumn's winner.

A plateau that ends without its cell having fired loses ``predicted_decrement`` on the
synapses that started it (a wrong prediction). Winners are tracked per cell, so the
synapse group must connect a layer to itself.

Spikes, wins and plateau starts keep their last two times: a cell of the previous element
that fires again in the current one (its minicolumn is active twice in a row) still counts
as context, and a plateau restarted by the current element still counts as a prediction.
Minicolumn inhibition allows one spike per cell per element, so two times are enough.
"""

from __future__ import annotations

import math

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order
from neurosush.synapses.segments import ActiveSegments

_NEVER = -(10**9)


def _record(times: torch.Tensor, where: torch.Tensor, t: int) -> None:
    """Make ``t`` the latest of the two times ``(..., 2)`` where ``where`` holds."""
    # torch.where rather than times[where, 1]: older torch versions index a mask followed
    # by an integer differently
    latest, previous = times[..., 0], times[..., 1]
    shifted = torch.stack([torch.where(where, t, latest), torch.where(where, latest, previous)], -1)
    times.copy_(shifted)


def _mask(like: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """A bool mask shaped like ``like`` that is true at ``index``."""
    mask = torch.zeros_like(like, dtype=torch.bool)
    mask[index] = True
    return mask


def _within(times: torch.Tensor, t: int, earliest: int, latest: int) -> torch.Tensor:
    """Whether either of the two times ``(..., 2)`` is ``earliest`` to ``latest`` steps old."""
    age = t - times
    return ((age >= earliest) & (age <= latest)).any(-1)


class SegmentLearning(Behavior):
    """Temporal memory learning for an :class:`ActiveSegments` layer-to-itself synapse group.

    Args:
        cells_per_column: Cells in every minicolumn (consecutive cells).
        context: ``(earliest, latest)`` age, in the unit of ``dt``, of the spikes that form the
            previous element; for elements shown every ``P`` steps, about ``(P / 2, 3 P / 2)``.
        min_threshold: Context synapses that make a segment match in a bursting minicolumn.
        initial_permanence: Permanence of new synapses.
        increment: Permanence increase of synapses from context cells.
        decrement: Permanence decrease of the other synapses of a learning segment.
        predicted_decrement: Decrease of the synapses that started a wrong prediction.
        max_new_synapses: Context synapses a learning segment grows up to.
    """

    order = Order.PLASTICITY

    def __init__(
        self,
        *,
        cells_per_column: int,
        context: tuple[float, float],
        min_threshold: int,
        initial_permanence: float = 0.21,
        increment: float = 0.1,
        decrement: float = 0.1,
        predicted_decrement: float = 0.0,
        max_new_synapses: int = 20,
    ) -> None:
        if cells_per_column < 1:
            raise ValueError(f"cells_per_column must be positive, got {cells_per_column}")
        if not 0 < context[0] < context[1]:
            raise ValueError(
                f"context must be (earliest, latest) with 0 < earliest < latest, got {context}"
            )
        if min_threshold < 1 or max_new_synapses < 1:
            raise ValueError(
                "min_threshold and max_new_synapses must be positive, "
                f"got {min_threshold}, {max_new_synapses}"
            )
        self.cells_per_column, self.context = cells_per_column, context
        self.min_threshold, self.max_new_synapses = min_threshold, max_new_synapses
        self.initial_permanence, self.increment, self.decrement = (
            initial_permanence,
            increment,
            decrement,
        )
        self.predicted_decrement = predicted_decrement

    def initialize(self, syn: SynapseGroup) -> None:
        """Check the synapse group and allocate the learning state."""
        segments = getattr(syn, "input", None)
        if not isinstance(segments, ActiveSegments):
            raise RuntimeError(f"SegmentLearning on {syn.name} needs ActiveSegments")
        if syn.src is not syn.dst:
            raise ValueError(f"SegmentLearning on {syn.name} needs a layer connected to itself")
        if syn.net.batch_size is not None:
            raise ValueError(
                "SegmentLearning does not support batches: samples would share segments"
            )
        if not hasattr(syn, "post_spike"):
            raise RuntimeError(f"SegmentLearning on {syn.name} needs an Axon on {syn.dst.name}")
        if syn.dst.size % self.cells_per_column:
            raise ValueError(
                f"{syn.dst.name} has {syn.dst.size} cells, not a multiple of "
                f"cells_per_column={self.cells_per_column}"
            )
        self.segments = segments
        n, s = syn.presynaptic.shape[:2]
        device = syn.net.device
        syn.segment_start = torch.full((n, s, 2), _NEVER, dtype=torch.long, device=device)
        syn.segment_used = torch.full((n, s), _NEVER, dtype=torch.long, device=device)
        syn.activation_synapses = torch.zeros_like(syn.presynaptic, dtype=torch.bool)
        syn.last_spike = torch.full((n, 2), _NEVER, dtype=torch.long, device=device)
        syn.last_win = torch.full((n, 2), _NEVER, dtype=torch.long, device=device)
        dt = syn.net.dt
        self.earliest = math.ceil(self.context[0] / dt - 1e-9)
        self.latest = math.floor(self.context[1] / dt + 1e-9)

    # --- one step ---------------------------------------------------------------------

    def forward(self, syn: SynapseGroup) -> None:
        """Record dendritic spikes, punish ended wrong predictions, learn where cells fired."""
        t = syn.net.iteration
        used = syn.presynaptic >= 0
        started = syn.active_segments
        _record(syn.segment_start, started, t)
        syn.segment_used[started] = t
        recent = syn.pre_recent[syn.presynaptic.clamp(min=0)] > 0
        syn.activation_synapses[started] = (recent & used)[started]
        self._punish(syn, t)
        fired = syn.post_spike
        if bool(fired.any()):
            context = _within(syn.last_spike, t, self.earliest, self.latest)
            winners = _within(syn.last_win, t, self.earliest, self.latest)
            for column in (
                fired.view(-1, self.cells_per_column).any(-1).nonzero().flatten().tolist()
            ):
                self._learn_column(syn, t, column, fired, context, winners)
            _record(syn.last_spike, fired, t)

    def _punish(self, syn: SynapseGroup, t: int) -> None:
        """Plateaus that end now without their cell having fired weaken what started them."""
        if not self.predicted_decrement:
            return
        start = syn.segment_start[..., 0]
        ended = (start == t - self.segments.duration) & (syn.plateau_steps == 0)
        silent = syn.last_spike[:, :1] < start
        for cell, segment in (ended & silent).nonzero().tolist():
            mask = syn.activation_synapses[cell, segment]
            syn.permanence[cell, segment, mask] = (
                syn.permanence[cell, segment, mask] - self.predicted_decrement
            ).clamp(min=0.0)

    def _learn_column(
        self,
        syn: SynapseGroup,
        t: int,
        column: int,
        fired: torch.Tensor,
        context: torch.Tensor,
        winners: torch.Tensor,
    ) -> None:
        k = self.cells_per_column
        cells = torch.arange(column * k, (column + 1) * k, device=fired.device)
        by_context = _within(syn.segment_start[cells], t, self.earliest, self.latest)
        predicted = (syn.plateau_steps[cells] > 0) & by_context
        predicted &= fired[cells].unsqueeze(-1)
        if bool(predicted.any()):
            for i, segment in predicted.nonzero().tolist():
                self._reinforce(syn, t, int(cells[i]), segment, context, winners)
            _record(syn.last_win, _mask(fired, cells[predicted.any(-1)]), t)
            return
        potential = self._potential(syn, cells, context)
        if int(potential.max()) >= self.min_threshold:
            i, segment = divmod(int(potential.argmax()), potential.shape[1])
            winner = int(cells[i])
            self._reinforce(syn, t, winner, segment, context, winners)
        else:
            winner = self._least_used(syn, cells)
            if bool(winners.any()):
                segment = self._free_segment(syn, winner)
                syn.segment_used[winner, segment] = t
                self._grow(syn, winner, segment, winners, self.max_new_synapses)
        _record(syn.last_win, _mask(fired, torch.tensor([winner], device=fired.device)), t)

    # --- learning primitives ----------------------------------------------------------

    def _potential(
        self, syn: SynapseGroup, cells: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Synapses from context cells on every segment of ``cells``."""
        pre = syn.presynaptic[cells]
        return ((pre >= 0) & context[pre.clamp(min=0)]).sum(-1)

    def _reinforce(
        self,
        syn: SynapseGroup,
        t: int,
        cell: int,
        segment: int,
        context: torch.Tensor,
        winners: torch.Tensor,
    ) -> None:
        pre = syn.presynaptic[cell, segment]
        used = pre >= 0
        from_context = used & context[pre.clamp(min=0)]
        change = torch.where(from_context, self.increment, -self.decrement) * used
        syn.permanence[cell, segment] = (syn.permanence[cell, segment] + change).clamp(0.0, 1.0)
        syn.segment_used[cell, segment] = t
        self._grow(syn, cell, segment, winners, self.max_new_synapses - int(from_context.sum()))

    def _grow(
        self, syn: SynapseGroup, cell: int, segment: int, winners: torch.Tensor, count: int
    ) -> None:
        """Add up to ``count`` synapses to winners not yet on the segment."""
        pre = syn.presynaptic[cell, segment]
        on_segment = torch.zeros_like(winners)
        on_segment[pre[pre >= 0]] = True
        candidates = (winners & ~on_segment).nonzero().flatten()
        count = min(count, len(candidates))
        if count <= 0:
            return
        order = torch.randperm(
            len(candidates), generator=syn.net.generator, device=candidates.device
        )
        chosen = candidates[order[:count]]
        free = (pre < 0).nonzero().flatten()
        if len(free) < count:  # replace the weakest synapses (Hawkins and Ahmad 2016)
            weakest = syn.permanence[cell, segment].masked_fill(pre < 0, 2.0).argsort()
            pre[weakest[: count - len(free)]] = -1
            free = (pre < 0).nonzero().flatten()
        slots = free[:count]
        syn.presynaptic[cell, segment, slots] = chosen
        syn.permanence[cell, segment, slots] = self.initial_permanence

    def _least_used(self, syn: SynapseGroup, cells: torch.Tensor) -> int:
        """The cell with the fewest segments; ties broken at random."""
        counts = (syn.presynaptic[cells] >= 0).any(-1).sum(-1)
        fewest = (counts == counts.min()).nonzero().flatten()
        pick = torch.randint(len(fewest), (1,), generator=syn.net.generator, device=fewest.device)
        return int(cells[fewest[pick]])

    def _free_segment(self, syn: SynapseGroup, cell: int) -> int:
        """An empty segment of ``cell``, or its least recently used one emptied."""
        empty = ~(syn.presynaptic[cell] >= 0).any(-1)
        if bool(empty.any()):
            return int(empty.nonzero()[0])
        segment = int(syn.segment_used[cell].argmin())
        syn.presynaptic[cell, segment] = -1
        syn.permanence[cell, segment] = 0.0
        syn.plateau_steps[cell, segment] = 0
        return segment
