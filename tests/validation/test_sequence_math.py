"""A spiking layer computes exactly what the temporal memory computes.

Minicolumns of LIF cells receive their column's input on the proximal compartment and
their learned segments on the distal one. A cell with a segment in its plateau is primed
towards (but below) threshold, so when its column is driven it reaches threshold after
``a`` steps; an unprimed cell needs ``b > a`` steps. Minicolumn inhibition therefore lets a
predicted cell fire alone, and makes an unpredicted column fire all at once (a burst).

The equivalence holds under timing conditions that are checked here: ``a < b <= W`` (both
fire inside the input window, and the inhibition lasts the window), and a plateau started
by one element covers the next element's firing but ends before the one after.
"""

import math

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.htm.sdr import random_sdr
from neurosush.htm.temporal_memory import TemporalMemory
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import MinicolumnInhibition
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.segments import ActiveSegments
from neurosush.synapses.traces import SpikeGather

from .common import ScriptedSpikes
from .test_segment_math import tm_segments

COLUMNS, CELLS = 64, 4
PERIOD, WINDOW, PLATEAU = 100, 25, 120  # steps per element, input window, plateau
DRIVE, GAIN, AMPLITUDE = 12.0, 0.8, 0.5
TAU, V_REST, THETA = 10.0, -65.0, -55.0


def steps_to_threshold(v0, primed):
    """Exact Euler steps from ``v0`` to threshold under the drive (dt = 1)."""
    k, limit = math.tanh(AMPLITUDE), V_REST + GAIN * (THETA - V_REST)
    v, n = v0, 0
    while v < THETA:
        n += 1
        prime = k * max(limit - v, 0.0) if primed else 0.0
        v = v + ((V_REST - v) + DRIVE) / TAU + prime
    return n


def primed_rest():
    """Where a plateau alone holds the membrane: the fixed point of the primed map."""
    k, limit = math.tanh(AMPLITUDE), V_REST + GAIN * (THETA - V_REST)
    return (V_REST / TAU + k * limit) / (1 / TAU + k)


def trained_memory():
    tm = TemporalMemory(
        COLUMNS,
        CELLS,
        activation_threshold=6,
        min_threshold=4,
        max_new_synapses=8,
        max_synapses_per_segment=10,
        initial_permanence=0.51,
        seed=0,
    )
    gen = [torch.Generator().manual_seed(s) for s in (1, 2, 3)]
    sequences = [random_sdr(COLUMNS, 8, batch=(5,), generator=g) for g in gen]
    sequences[1][1:4] = sequences[0][1:4]  # shared middle: only the first element tells them apart
    for _ in range(6):
        for sequence in sequences:
            tm.reset()
            for x in sequence:
                tm.compute(x)
    return tm, sequences


def spiking_layer(tm, sequences, coincidence=None):
    """The layer, fed every sequence after a silence long enough to end all plateaus."""
    schedule, starts, t = {}, [], 1
    for sequence in sequences:
        for x in sequence:
            for step in range(t, t + WINDOW):
                schedule.setdefault(step, []).extend(x.nonzero().flatten().tolist())
            starts.append(t)
            t += PERIOD
        t += 3 * PERIOD
    net = Network(dtype=torch.float64)
    columns = NeuronGroup(net, COLUMNS, [ScriptedSpikes(schedule), Axon()])
    layer = NeuronGroup(
        net,
        COLUMNS * CELLS,
        [
            DendriteStructure(),
            DendriteIntegration(distal_gain=GAIN),
            LIF(tau=TAU, threshold=THETA, v_reset=-70.0, v_rest=V_REST),
            MinicolumnInhibition(cells_per_column=CELLS, duration=WINDOW),
            Fire(),
            Axon(),
        ],
    )
    block = torch.zeros(COLUMNS, COLUMNS * CELLS)
    for c in range(COLUMNS):
        block[c, c * CELLS : (c + 1) * CELLS] = 1.0
    SynapseGroup(
        net, columns, layer, [WeightInit(weights=block), DenseInput(coef=DRIVE), SpikeGather()]
    )
    per_cell = int(torch.bincount(tm.segment_cell[: tm.n_segments]).max())
    presynaptic, permanence, _ = tm_segments(tm, per_cell)
    SynapseGroup(
        net,
        layer,
        layer,
        [
            ActiveSegments(
                segments=per_cell,
                synapses=tm.max_synapses,
                activation_threshold=tm.activation_threshold,
                connected=tm.connected,
                plateau=PLATEAU,
                coincidence=coincidence,
                amplitude=AMPLITUDE,
                presynaptic=presynaptic,
                permanence=permanence,
            ),
            SpikeGather(),
        ],
        compartment="distal",
    )
    return net, layer, starts, t


def test_timing_conditions_of_the_equivalence():
    a, b = steps_to_threshold(primed_rest(), True), steps_to_threshold(V_REST, False)
    # predicted cells win the race, and every column fires inside the input window
    assert a < b <= WINDOW
    # an element fires at offset a or b and its plateaus start right after: one started by
    # a burst (offset b) still primes the next element's cells when they cross (offset a)
    assert b + PLATEAU >= PERIOD + a
    # one started at offset a has ended before the element after next is driven
    assert a + PLATEAU <= 2 * PERIOD


@pytest.mark.parametrize("coincidence", [None, 11.0])
def test_spiking_layer_equals_the_temporal_memory(coincidence):
    # with a coincidence window of b - a + 1 = 11 steps a segment also sums an element's
    # predicted (offset a) and bursting (offset b) cells; the previous element's spikes,
    # 90 or more steps old, stay outside it
    tm, sequences = trained_memory()
    net, layer, starts, end = spiking_layer(tm, sequences, coincidence)
    fired = torch.zeros(len(starts), layer.size, dtype=torch.bool)
    offsets = set()
    for _ in range(end):
        net.step()
        element = max((i for i, s in enumerate(starts) if s <= net.iteration), default=None)
        if element is not None and net.iteration < starts[element] + PERIOD and layer.spikes.any():
            fired[element] |= layer.spikes
            offsets.add(net.iteration - starts[element])
    element = 0
    for sequence in sequences:
        tm.reset()
        for x in sequence:
            tm.compute(x, learn=False)
            assert torch.equal(fired[element], tm.active_cells), f"element {element}"
            element += 1
    # predicted cells fire after a steps, bursting columns after b steps
    assert offsets == {steps_to_threshold(primed_rest(), True), steps_to_threshold(V_REST, False)}
