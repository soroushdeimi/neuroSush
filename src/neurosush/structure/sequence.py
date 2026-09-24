"""A spiking sequence memory layer whose timing is derived and checked, not tuned.

Minicolumns of LIF cells get their column's input on the proximal compartment and learned
dendritic segments (:class:`~neurosush.synapses.segments.ActiveSegments`, with
:class:`~neurosush.synapses.segment_learning.SegmentLearning`) on the distal one. Every
element of a sequence drives its columns for ``window`` time units, one element every
``period``. A cell primed by a plateau reaches threshold after ``a`` steps of drive, an
unprimed one after ``b``; :func:`sequence_timing` computes both from the exact Euler map
of the neuron and derives the plateau, the coincidence window and the learning context,
refusing parameters under which the layer would not compute the temporal memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import MinicolumnInhibition
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.segment_learning import SegmentLearning
from neurosush.synapses.segments import ActiveSegments
from neurosush.synapses.traces import SpikeGather


@dataclass(frozen=True)
class Neuron:
    """LIF parameters of the layer and the strength of its inputs.

    Args:
        tau: Membrane time constant.
        v_rest: Resting voltage.
        v_reset: Voltage after a spike (and of inhibited cells).
        threshold: Spike threshold.
        drive: ``R I`` of an active column's input.
        gain: Distal priming limit, ``v_rest + gain (threshold - v_rest)``; below 1.
        amplitude: Plateau current; the priming rate is ``tanh(amplitude)``.
    """

    tau: float = 10.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    threshold: float = -55.0
    drive: float = 12.0
    gain: float = 0.8
    amplitude: float = 0.5


@dataclass(frozen=True)
class SequenceTiming:
    """Race times and the derived time constants, all in steps.

    Args:
        primed: Steps of drive a primed cell needs to fire (``a``).
        unprimed: Steps an unprimed cell needs (``b``).
        plateau: Plateau length.
        coincidence: Coincidence window of the segments.
        context: Ages of the spikes that form the previous element.
    """

    primed: int
    unprimed: int
    plateau: int
    coincidence: int
    context: tuple[int, int]


def _steps_to_threshold(neuron: Neuron, dt: float, *, primed: bool) -> int:
    """Euler steps of drive from rest (or from the primed rest) to threshold."""
    k = math.tanh(neuron.amplitude) if primed else 0.0
    limit = neuron.v_rest + neuron.gain * (neuron.threshold - neuron.v_rest)
    # the plateau alone holds the membrane at the fixed point of the primed map
    v = (neuron.v_rest / neuron.tau + k * limit) / (1 / neuron.tau + k) if primed else neuron.v_rest
    for n in range(1, 100_000):
        v = (
            v
            + dt * ((neuron.v_rest - v) + neuron.drive) / neuron.tau
            + dt * k * max(limit - v, 0.0)
        )
        if v >= neuron.threshold:
            return n
    raise ValueError(f"the drive {neuron.drive} never brings a cell to threshold")


def sequence_timing(period: int, window: int, neuron: Neuron, dt: float = 1.0) -> SequenceTiming:
    """Derive the layer's time constants and check that it computes the temporal memory.

    Args:
        period: Steps from one element to the next.
        window: Steps an element drives its columns (and a fired minicolumn stays inhibited).
        neuron: Neuron and input parameters.
        dt: Time step.

    Raises:
        ValueError: When primed cells would not win the race inside the window, or when no
            plateau covers exactly the next element.
    """
    if not 0 < neuron.gain < 1:
        raise ValueError(
            f"gain must be in (0, 1) so that priming stays below threshold, got {neuron.gain}"
        )
    a = _steps_to_threshold(neuron, dt, primed=True)
    b = _steps_to_threshold(neuron, dt, primed=False)
    if not a < b <= window:
        raise ValueError(
            f"primed cells must fire first and every column inside the window: need "
            f"primed {a} < unprimed {b} <= window {window}"
        )
    # an element fires at offset a or b and its plateaus start right after: they must still
    # prime the next element's crossing (b + plateau >= period + a) and end before the element
    # after it is driven (a + plateau <= 2 period); the midpoint leaves room on both sides
    low, high = period + a - b, 2 * period - a
    if low > high:
        raise ValueError(f"period {period} is too short for firing offsets {a} and {b}")
    plateau = (low + high) // 2
    spread = b - a
    if spread >= period // 2:
        raise ValueError(f"period {period} must exceed twice the firing spread {spread}")
    context = (
        period - spread - (period // 2 - spread) // 2,
        period + spread + (period // 2 - spread) // 2,
    )
    return SequenceTiming(a, b, plateau, spread + 1, context)


def sequence_memory(
    net: Network,
    columns: NeuronGroup,
    *,
    cells_per_column: int,
    period: int,
    window: int,
    neuron: Neuron | None = None,
    segments: int = 8,
    synapses: int = 10,
    activation_threshold: int = 6,
    min_threshold: int = 4,
    max_new_synapses: int = 8,
    initial_permanence: float = 0.21,
    predicted_decrement: float = 0.05,
    learn: bool = True,
    name: str = "sequence",
) -> tuple[NeuronGroup, SynapseGroup]:
    """Add a sequence memory layer driven by ``columns`` (which needs an Axon).

    Every column neuron drives its ``cells_per_column`` cells; to present an element, make
    its columns fire on every step of the window.

    Returns:
        The layer and its distal (segment) synapse group.
    """
    neuron = neuron or Neuron()
    timing = sequence_timing(period, window, neuron, net.dt)
    size = columns.size * cells_per_column
    layer = NeuronGroup(
        net,
        size,
        [
            DendriteStructure(),
            DendriteIntegration(distal_gain=neuron.gain),
            LIF(
                tau=neuron.tau,
                threshold=neuron.threshold,
                v_reset=neuron.v_reset,
                v_rest=neuron.v_rest,
            ),
            MinicolumnInhibition(cells_per_column=cells_per_column, duration=window * net.dt),
            Fire(),
            Axon(),
        ],
        name=name,
    )
    feed = torch.repeat_interleave(torch.eye(columns.size), cells_per_column, dim=1)
    SynapseGroup(
        net,
        columns,
        layer,
        [WeightInit(weights=feed), DenseInput(coef=neuron.drive), SpikeGather()],
    )
    behaviors = [
        ActiveSegments(
            segments=segments,
            synapses=synapses,
            activation_threshold=activation_threshold,
            plateau=timing.plateau * net.dt,
            coincidence=timing.coincidence * net.dt,
            amplitude=neuron.amplitude,
        ),
        SpikeGather(),
    ]
    if learn:
        behaviors.append(
            SegmentLearning(
                cells_per_column=cells_per_column,
                context=(timing.context[0] * net.dt, timing.context[1] * net.dt),
                min_threshold=min_threshold,
                initial_permanence=initial_permanence,
                max_new_synapses=max_new_synapses,
                predicted_decrement=predicted_decrement,
            )
        )
    distal = SynapseGroup(net, layer, layer, behaviors, compartment="distal", name=f"{name}.distal")
    return layer, distal
