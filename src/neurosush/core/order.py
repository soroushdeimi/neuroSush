"""Execution order for the simulation."""

from __future__ import annotations

from enum import IntEnum


class Order(IntEnum):
    """Lower values run first, in initialization and in every step."""

    INITIALIZATION = 0
    PAYOFF = 100
    NEUROMODULATOR = 120
    SYNAPTIC_INPUT = 180
    CURRENT_NORMALIZATION = 200
    DENDRITE_STRUCTURE = 220
    DENDRITE_INTEGRATION = 240
    NEURON_DYNAMICS = 260
    NOISE = 280
    COMPETITION = 300
    VOLTAGE_HOMEOSTASIS = 310
    FIRE = 340
    ACTIVITY_HOMEOSTASIS = 350
    AXON = 380
    SPIKE_GATHER = 420
    TRACE = 460
    PLASTICITY = 500
    WEIGHT_NORMALIZATION = 520
    WEIGHT_CLIP = 540
    RECORD = 1000
