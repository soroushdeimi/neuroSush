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

from neurosush.htm._state import checked

# per-segment storage, grown by doubling; only the first n_segments rows are in use
_STORAGE = ("segment_cell", "presynaptic", "permanence", "last_used")
# the current context: cells and the dendrite activity computed from them
_CELLS = ("active_cells", "winner_cells", "predictive_cells")
_SEGMENTS = ("active_segments", "matching_segments", "potential_counts")


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
        empty = self._empty_storage(64)
        self.segment_cell, self.presynaptic = empty["segment_cell"], empty["presynaptic"]
        self.permanence, self.last_used = empty["permanence"], empty["last_used"]
        self.n_segments = 0
        self.cell_segments = torch.zeros(self.n_cells, dtype=torch.long)  # segments per cell
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
        # the few active and matching segments, grouped by column once
        by_column = _by_column(active_segments, segment_columns)
        matching_by_column = _by_column(matching, segment_columns)
        winners = prev_winners.nonzero().flatten()
        bursting = 0
        for column in active_columns.nonzero().flatten().tolist():
            if column in by_column:
                self._activate_predicted_column(
                    by_column[column], potential, prev_active, winners, learn
                )
            else:
                bursting += 1
                self._burst_column(
                    column,
                    matching_by_column.get(column, []),
                    potential,
                    prev_active,
                    winners,
                    learn,
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
        segments: list[int],
        potential: torch.Tensor,
        prev_active: torch.Tensor,
        prev_winners: torch.Tensor,
        learn: bool,
    ) -> None:
        for segment in segments:
            cell = int(self.segment_cell[segment])
            self.active_cells[cell] = self.winner_cells[cell] = True
            if learn:
                self._adapt(segment, prev_active, self.increment, self.decrement)
                self._grow(segment, prev_winners, self.max_new_synapses - int(potential[segment]))

    def _burst_column(
        self,
        column: int,
        matching: list[int],
        potential: torch.Tensor,
        prev_active: torch.Tensor,
        prev_winners: torch.Tensor,
        learn: bool,
    ) -> None:
        start = column * self.cells_per_column
        self.active_cells[start : start + self.cells_per_column] = True
        if matching:
            candidates = torch.tensor(matching)
            best = int(candidates[potential[candidates].argmax()])
            self.winner_cells[int(self.segment_cell[best])] = True
            if learn:
                self._adapt(best, prev_active, self.increment, self.decrement)
                self._grow(best, prev_winners, self.max_new_synapses - int(potential[best]))
            return
        winner = self._least_used_cell(start)
        self.winner_cells[winner] = True
        if learn and len(prev_winners):
            segment = self._create_segment(winner)
            self._grow(segment, prev_winners, self.max_new_synapses)

    # --- learning primitives ----------------------------------------------------------

    def _least_used_cell(self, start: int) -> int:
        """Cell of the column with the fewest segments; ties broken at random."""
        counts = self.cell_segments[start : start + self.cells_per_column]
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
        """Add up to ``count`` synapses to previous winner cells not yet on the segment.

        ``prev_winners`` holds the winner cells' indices in increasing order.
        """
        pre = self.presynaptic[segment]
        candidates = prev_winners[~torch.isin(prev_winners, pre[pre >= 0])]
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
        if int(self.cell_segments[cell]) >= self.max_segments_per_cell:
            on_cell = (self.segment_cell[: self.n_segments] == cell).nonzero().flatten()
            segment = int(on_cell[self.last_used[on_cell].argmin()])
        else:
            if self.n_segments == len(self.segment_cell):
                self._grow_storage()
            segment = self.n_segments
            self.n_segments += 1
            self.cell_segments[cell] += 1
        self.segment_cell[segment] = cell
        self.presynaptic[segment] = -1
        self.permanence[segment] = 0.0
        self.last_used[segment] = self.iteration
        return segment

    def _empty_storage(self, rows: int) -> dict[str, torch.Tensor]:
        """Unused segment rows: no cell, no synapses (-1), zero permanence and use time."""
        return {
            "segment_cell": torch.full((rows,), -1, dtype=torch.long),
            "presynaptic": torch.full((rows, self.max_synapses), -1, dtype=torch.long),
            "permanence": torch.zeros(rows, self.max_synapses),
            "last_used": torch.zeros(rows, dtype=torch.long),
        }

    def _grow_storage(self) -> None:
        """Double the segment capacity."""
        self._set_storage({name: getattr(self, name) for name in _STORAGE}, len(self.segment_cell))

    def _set_storage(self, used: dict[str, torch.Tensor], extra: int) -> None:
        """Make ``used`` the segment storage, followed by ``extra`` unused rows."""
        empty = self._empty_storage(extra)
        self.segment_cell = torch.cat([used["segment_cell"], empty["segment_cell"]])
        self.presynaptic = torch.cat([used["presynaptic"], empty["presynaptic"]])
        self.permanence = torch.cat([used["permanence"], empty["permanence"]])
        self.last_used = torch.cat([used["last_used"], empty["last_used"]])

    # --- checkpoints ------------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        """A copy of the segments, the current context and the random generator.

        A memory with the same arguments continues exactly after :meth:`load_state_dict`.
        """
        n = self.n_segments
        state: dict[str, torch.Tensor | int | float] = {
            name: getattr(self, name)[:n].clone() for name in _STORAGE
        }
        for name in (*_CELLS, *_SEGMENTS):
            state[name] = getattr(self, name).clone()
        state.update(
            n_segments=n,
            iteration=self.iteration,
            anomaly=self.anomaly,
            generator=self.generator.get_state(),
        )
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float]) -> None:
        """Restore a :meth:`state_dict` saved from a memory with the same arguments."""
        n = int(state["n_segments"])
        storage = {
            name: checked(name, state[name], getattr(self, name), rows=n) for name in _STORAGE
        }
        cells, synapses = storage["segment_cell"], storage["presynaptic"]
        if bool(((cells < 0) | (cells >= self.n_cells)).any()):
            raise ValueError(f"segment_cell must index {self.n_cells} cells")
        if bool(((synapses < -1) | (synapses >= self.n_cells)).any()):
            raise ValueError(f"presynaptic must index {self.n_cells} cells or be -1")
        self._set_storage(storage, max(64 - n, 0))
        self.n_segments = n
        self.cell_segments = torch.bincount(cells, minlength=self.n_cells)
        for name in _CELLS:
            setattr(self, name, checked(name, state[name], getattr(self, name)))
        for name in _SEGMENTS:
            setattr(self, name, checked(name, state[name], getattr(self, name)[:0], rows=n))
        self.iteration = int(state["iteration"])
        self.anomaly = float(state["anomaly"])
        generator = state["generator"]
        if not isinstance(generator, torch.Tensor):
            raise ValueError("generator must be a generator state tensor")
        self.generator.set_state(generator.cpu())

    # --- read-outs --------------------------------------------------------------------

    def predicted_columns(self) -> torch.Tensor:
        """Columns containing at least one predictive cell: the prediction for the next input."""
        return self.predictive_cells.view(self.columns, self.cells_per_column).any(-1)


def _by_column(segments: torch.Tensor, columns: torch.Tensor) -> dict[int, list[int]]:
    """The segments where ``segments`` holds, by column, each list in increasing order."""
    index = segments.nonzero().flatten()
    groups: dict[int, list[int]] = {}
    for segment, column in zip(index.tolist(), columns[index].tolist(), strict=True):
        groups.setdefault(column, []).append(segment)
    return groups
