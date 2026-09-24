"""A spiking layer learns sequences the way the temporal memory does.

The layer (``sequence_memory``) and a ``TemporalMemory`` with the same parameters see the
same two sequences, ABCDE and XBCDY, which share their middle: only the first element tells
the endings apart. The number of bursting minicolumns of every element, repetition by
repetition, is the learning curve. The two differ only in their random choices (winner
cells, sampled synapses), so the curves must agree while learning follows from the
structure of the input and must end in the same state.
"""

import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.htm.sdr import random_sdr
from neurosush.htm.temporal_memory import TemporalMemory
from neurosush.neurons.axon import Axon
from neurosush.structure.sequence import sequence_memory

from .common import ScriptedSpikes

COLUMNS, CELLS, PERIOD, WINDOW, REPETITIONS = 64, 4, 60, 25, 10


def sequences():
    first, second = (
        random_sdr(COLUMNS, 8, batch=(5,), generator=torch.Generator().manual_seed(s))
        for s in (1, 2)
    )
    second[1:4] = first[1:4]  # ABCDE and XBCDY
    return [first, second]


def spiking_bursts(order):
    """Bursting minicolumns of every presented element."""
    schedule, starts, t = {}, [], 1
    for sequence in order:
        for x in sequence:
            for step in range(t, t + WINDOW):
                schedule.setdefault(step, []).extend(x.nonzero().flatten().tolist())
            starts.append((t, x))
            t += PERIOD
        t += 2 * PERIOD  # silence: the temporal memory's reset
    net = Network(dtype=torch.float64, seed=0)
    columns = NeuronGroup(net, COLUMNS, [ScriptedSpikes(schedule), Axon()])
    layer, _ = sequence_memory(net, columns, cells_per_column=CELLS, period=PERIOD, window=WINDOW)
    fired = torch.zeros(len(starts), layer.size, dtype=torch.bool)
    element = -1
    for _ in range(t):
        net.step()
        while element + 1 < len(starts) and starts[element + 1][0] <= net.iteration:
            element += 1
        if element >= 0 and net.iteration < starts[element][0] + PERIOD:
            fired[element] |= layer.spikes
    return [
        int(fired[i].view(COLUMNS, CELLS)[x.nonzero().flatten()].all(-1).sum())
        for i, (_, x) in enumerate(starts)
    ]


def temporal_memory_bursts(order):
    tm = TemporalMemory(
        COLUMNS,
        CELLS,
        activation_threshold=6,
        min_threshold=4,
        max_new_synapses=8,
        max_synapses_per_segment=10,
        initial_permanence=0.21,
        predicted_decrement=0.05,
        seed=0,
    )
    bursts = []
    for sequence in order:
        tm.reset()
        for x in sequence:
            tm.compute(x)
            active = tm.active_cells.view(COLUMNS, CELLS)[x.nonzero().flatten()]
            bursts.append(int(active.all(-1).sum()))
    return bursts


def test_learning_curves_match_the_temporal_memory():
    order = [s for _ in range(REPETITIONS) for s in sequences()]
    spiking, reference = spiking_bursts(order), temporal_memory_bursts(order)
    per_repetition = 2 * 5
    rows = [
        (spiking[i : i + per_repetition], reference[i : i + per_repetition])
        for i in range(0, len(order) * 5, per_repetition)
    ]
    # learning follows the input: identical while every column still bursts or is learned
    for spiking_row, reference_row in rows[:4]:
        assert spiking_row == reference_row
    # and both end in the same state: only the first elements (A, X) and the D after XBC
    # burst; E follows ABCD and Y follows XBCD
    final = [8, 0, 0, 0, 0, 8, 0, 0, 8, 0]
    for spiking_row, reference_row in rows[-2:]:
        assert spiking_row == reference_row == final
