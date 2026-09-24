"""HTM temporal memory: learns sequences as transitions between sparse cell codes.

Follows Hawkins and Ahmad (2016), "Why neurons have thousands of synapses, a theory of
sequence memory in neocortex", Front. Neural Circuits 10:23, and the pseudocode of
Numenta's "Biological and Machine Intelligence" temporal memory chapter.

Columns contain cells; a cell's distal segments hold synapses to other cells. A segment
whose connected synapses see at least ``activation_threshold`` active cells makes its cell
predictive. When a column becomes active, its predicted cells fire; if none were predicted
the whole column bursts. Learning reinforces the segments that predicted correctly, grows a
segment on one cell of every bursting column, and punishes segments whose prediction failed.

Segment activity is computed for all segments at once from tensors; only the few segments
that learn in a step (about one per active column) are handled one by one.
"""

from __future__ import annotations

import torch


class TemporalMemory:
    """Sequence memory over a sheet of columns with ``cells_per_column`` cells each.

    Args:
        columns: Number of columns (the size of the input SDR).
        cells_per_column: Cells in every column.
        activation_threshold: Active connected synapses that make a segment active.
        min_threshold: Active potential synapses that make a segment matching.
        initial_permanence: Permanence of new synapses.
        connected: Permanence at which a synapse is connected.
        increment: Permanence increase of synapses from previously active cells.
        decrement: Permanence decrease of the other synapses of a learning segment.
        predicted_decrement: Decrease of synapses on segments that predicted wrongly.
        max_new_synapses: Synapses a learning segment grows toward previous winner cells.
        max_synapses_per_segment: Capacity of a segment.
        max_segments_per_cell: Segments a cell may hold; the least recently used goes first.
        seed: Seed for tie-breaking and synapse sampling.
    """

    def __init__(
        self,
        columns: int,
        cells_per_column: int = 32,
        *,
        activation_threshold: int = 13,
        min_threshold: int = 10,
        initial_permanence: float = 0.21,
        connected: float = 0.5,
        increment: float = 0.1,
        decrement: float = 0.1,
        predicted_decrement: float = 0.0,
        max_new_synapses: int = 20,
        max_synapses_per_segment: int = 32,
        max_segments_per_cell: int = 128,
        seed: int = 0,
    ) -> None:
        if columns < 1 or cells_per_column < 1:
            raise ValueError(
                f"columns and cells_per_column must be positive, got {columns}, {cells_per_column}"
            )
        if not 0 < min_threshold <= activation_threshold:
            raise ValueError(
                f"need 0 < min_threshold <= activation_threshold, got {min_threshold}, "
                f"{activation_threshold}"
            )
        if max_new_synapses > max_synapses_per_segment:
            raise ValueError("max_new_synapses cannot exceed max_synapses_per_segment")
        self.columns, self.cells_per_column = columns, cells_per_column
        self.n_cells = columns * cells_per_column
        self.activation_threshold, self.min_threshold = activation_threshold, min_threshold
        self.initial_permanence, self.connected = initial_permanence, connected
        self.increment, self.decrement = increment, decrement
        self.predicted_decrement, self.max_new_synapses = predicted_decrement, max_new_synapses
        self.max_synapses, self.max_segments_per_cell = (
            max_synapses_per_segment,
            max_segments_per_cell,
        )
        self.generator = torch.Generator().manual_seed(seed)
        capacity = 64
        self.segment_cell = torch.full((capacity,), -1, dtype=torch.long)
        self.presynaptic = torch.full((capacity, self.max_synapses), -1, dtype=torch.long)
        self.permanence = torch.zeros(capacity, self.max_synapses)
        self.last_used = torch.zeros(capacity, dtype=torch.long)
        self.n_segments = 0
        self.iteration = 0
        self.reset()

    def reset(self) -> None:
        """Forget the current sequence context (not what was learned)."""
        self.active_cells = torch.zeros(self.n_cells, dtype=torch.bool)
        self.winner_cells = torch.zeros(self.n_cells, dtype=torch.bool)
        self.predictive_cells = torch.zeros(self.n_cells, dtype=torch.bool)
        self.active_segments = torch.zeros(self.n_segments, dtype=torch.bool)
        self.matching_segments = torch.zeros(self.n_segments, dtype=torch.bool)
        self.potential_counts = torch.zeros(self.n_segments, dtype=torch.long)
        self.anomaly = 0.0

    # --- dendrite activity -------------------------------------------------------------

    def _segment_activity(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Active connected and active potential synapse counts of every used segment."""
        pre = self.presynaptic[: self.n_segments]
        valid = pre >= 0
        active = valid & cells[pre.clamp(min=0)]
        connected = active & (self.permanence[: self.n_segments] >= self.connected)
        return connected.sum(-1), active.sum(-1)

    def _activate_dendrites(self) -> None:
        connected, potential = self._segment_activity(self.active_cells)
        self.active_segments = connected >= self.activation_threshold
        self.matching_segments = potential >= self.min_threshold
        self.potential_counts = potential
        self.predictive_cells = torch.zeros(self.n_cells, dtype=torch.bool)
        self.predictive_cells[self.segment_cell[: self.n_segments][self.active_segments]] = True
        self.last_used[: self.n_segments][self.active_segments] = self.iteration

    # --- one step ---------------------------------------------------------------------

    def compute(self, active_columns: torch.Tensor, *, learn: bool = True) -> torch.Tensor:
        """Process one input; returns the active cells.

        Args:
            active_columns: Bool ``(columns,)`` input, for example spatial pooler output.
            learn: Update synapses.
        """
        active_columns = torch.as_tensor(active_columns, dtype=torch.bool)
        if active_columns.shape != (self.columns,):
            raise ValueError(
                f"expected ({self.columns},) columns, got {tuple(active_columns.shape)}"
            )
        self.iteration += 1
        prev_active, prev_winners = self.active_cells, self.winner_cells
        segment_columns = self.segment_cell[: self.n_segments] // self.cells_per_column
        active_segments, matching = self.active_segments, self.matching_segments
        potential = self.potential_counts
        self.active_cells = torch.zeros(self.n_cells, dtype=torch.bool)
        self.winner_cells = torch.zeros(self.n_cells, dtype=torch.bool)
        predicted_columns = torch.zeros(self.columns, dtype=torch.bool)
        predicted_columns[segment_columns[active_segments]] = True
        bursting = 0
        for column in active_columns.nonzero().flatten().tolist():
            in_column = segment_columns == column
            if bool(predicted_columns[column]):
                self._activate_predicted_column(
                    in_column & active_segments, potential, prev_active, prev_winners, learn
                )
            else:
                bursting += 1
                self._burst_column(
                    column, in_column & matching, potential, prev_active, prev_winners, learn
                )
        if learn and self.predicted_decrement > 0:
            wrong = matching & ~active_columns[segment_columns]
            for segment in wrong.nonzero().flatten().tolist():
                self._adapt(segment, prev_active, -self.predicted_decrement, 0.0)
        total = int(active_columns.sum())
        self.anomaly = bursting / total if total else 0.0
        self._activate_dendrites()
        return self.active_cells

    def _activate_predicted_column(
        self,
        segments: torch.Tensor,
        potential: torch.Tensor,
        prev_active: torch.Tensor,
        prev_winners: torch.Tensor,
        learn: bool,
    ) -> None:
        for segment in segments.nonzero().flatten().tolist():
            cell = int(self.segment_cell[segment])
            self.active_cells[cell] = self.winner_cells[cell] = True
            if learn:
                self._adapt(segment, prev_active, self.increment, self.decrement)
                self._grow(segment, prev_winners, self.max_new_synapses - int(potential[segment]))

    def _burst_column(
        self,
        column: int,
        matching: torch.Tensor,
        potential: torch.Tensor,
        prev_active: torch.Tensor,
        prev_winners: torch.Tensor,
        learn: bool,
    ) -> None:
        start = column * self.cells_per_column
        self.active_cells[start : start + self.cells_per_column] = True
        candidates = matching.nonzero().flatten()
        if len(candidates):
            best = int(candidates[potential[candidates].argmax()])
            self.winner_cells[int(self.segment_cell[best])] = True
            if learn:
                self._adapt(best, prev_active, self.increment, self.decrement)
                self._grow(best, prev_winners, self.max_new_synapses - int(potential[best]))
            return
        winner = self._least_used_cell(start)
        self.winner_cells[winner] = True
        if learn and bool(prev_winners.any()):
            segment = self._create_segment(winner)
            self._grow(segment, prev_winners, self.max_new_synapses)

    # --- learning primitives ----------------------------------------------------------

    def _least_used_cell(self, start: int) -> int:
        """Cell of the column with the fewest segments; ties broken at random."""
        cells = self.segment_cell[: self.n_segments]
        counts = torch.bincount(
            cells[(cells >= start) & (cells < start + self.cells_per_column)] - start,
            minlength=self.cells_per_column,
        )
        fewest = (counts == counts.min()).nonzero().flatten()
        pick = torch.randint(len(fewest), (1,), generator=self.generator)
        return start + int(fewest[pick])

    def _adapt(self, segment: int, prev_active: torch.Tensor, inc: float, dec: float) -> None:
        """``+inc`` for synapses from previously active cells, ``-dec`` for the rest."""
        pre = self.presynaptic[segment]
        valid = pre >= 0
        from_active = valid & prev_active[pre.clamp(min=0)]
        change = torch.where(from_active, inc, -dec) * valid
        self.permanence[segment] = (self.permanence[segment] + change).clamp(0, 1)
        self.last_used[segment] = self.iteration

    def _grow(self, segment: int, prev_winners: torch.Tensor, count: int) -> None:
        """Add up to ``count`` synapses to previous winner cells not yet on the segment."""
        pre = self.presynaptic[segment]
        existing = torch.zeros(self.n_cells, dtype=torch.bool)
        existing[pre[pre >= 0]] = True
        candidates = (prev_winners & ~existing).nonzero().flatten()
        count = min(count, len(candidates))
        if count <= 0:
            return
        chosen = candidates[torch.randperm(len(candidates), generator=self.generator)[:count]]
        free = (pre < 0).nonzero().flatten()
        if len(free) < count:
            # make room by removing the weakest synapses (Hawkins and Ahmad 2016)
            used = (pre >= 0).nonzero().flatten()
            weakest = used[self.permanence[segment, used].argsort()[: count - len(free)]]
            self.presynaptic[segment, weakest] = -1
            free = (self.presynaptic[segment] < 0).nonzero().flatten()
        slots = free[:count]
        self.presynaptic[segment, slots] = chosen
        self.permanence[segment, slots] = self.initial_permanence

    def _create_segment(self, cell: int) -> int:
        """A new empty segment on ``cell``, evicting its least recently used one if full."""
        on_cell = (self.segment_cell[: self.n_segments] == cell).nonzero().flatten()
        if len(on_cell) >= self.max_segments_per_cell:
            segment = int(on_cell[self.last_used[on_cell].argmin()])
        else:
            if self.n_segments == len(self.segment_cell):
                self._grow_storage()
            segment = self.n_segments
            self.n_segments += 1
        self.segment_cell[segment] = cell
        self.presynaptic[segment] = -1
        self.permanence[segment] = 0.0
        self.last_used[segment] = self.iteration
        return segment

    def _grow_storage(self) -> None:
        extra = len(self.segment_cell)
        self.segment_cell = torch.cat(
            [self.segment_cell, torch.full((extra,), -1, dtype=torch.long)]
        )
        self.presynaptic = torch.cat(
            [self.presynaptic, torch.full((extra, self.max_synapses), -1, dtype=torch.long)]
        )
        self.permanence = torch.cat([self.permanence, torch.zeros(extra, self.max_synapses)])
        self.last_used = torch.cat([self.last_used, torch.zeros(extra, dtype=torch.long)])

    # --- read-outs --------------------------------------------------------------------

    def predicted_columns(self) -> torch.Tensor:
        """Columns containing at least one predictive cell: the prediction for the next input."""
        return self.predictive_cells.view(self.columns, self.cells_per_column).any(-1)
