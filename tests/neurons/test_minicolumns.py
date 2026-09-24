import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.competition import MinicolumnInhibition, minicolumn_inhibition
from neurosush.neurons.models import LIF


def step(v, inhibition, duration=3):
    return minicolumn_inhibition(
        torch.tensor(v),
        -55.0,
        -70.0,
        torch.tensor(inhibition),
        cells_per_column=2,
        duration=duration,
    )


def test_crossing_cells_fire_together_and_start_the_inhibition():
    v, inhibition = step([-54.0, -54.5, -60.0, -56.0], [0, 0])
    assert v.tolist() == [-54.0, -54.5, -60.0, -56.0]  # nobody is held down yet
    assert inhibition.tolist() == [3, 0]


def test_inhibited_columns_are_held_at_reset_and_count_down():
    v, inhibition = step([-50.0, -58.0, -54.0, -60.0], [2, 0])
    assert v.tolist() == [-70.0, -70.0, -54.0, -60.0]
    assert inhibition.tolist() == [1, 3]


def test_batches_are_independent():
    _, inhibition = step([[-54.0, -60.0], [-60.0, -60.0]], [[0], [0]])
    assert inhibition.tolist() == [[3], [0]]


def test_behavior_checks_the_group():
    net = Network()
    NeuronGroup(
        net,
        5,
        [
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            MinicolumnInhibition(cells_per_column=2, duration=3.0),
        ],
        name="g",
    )
    with pytest.raises(ValueError, match="not a multiple of cells_per_column=2"):
        net.initialize()
    net = Network()
    NeuronGroup(net, 4, [MinicolumnInhibition(cells_per_column=2, duration=3.0)])
    with pytest.raises(RuntimeError, match="neuron model"):
        net.initialize()


def test_duration_is_rounded_up_to_steps():
    net = Network(dt=0.5)
    group = NeuronGroup(
        net,
        4,
        [
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            inhibition := MinicolumnInhibition(cells_per_column=2, duration=1.2),
        ],
    )
    net.initialize()
    assert inhibition.steps == 3
    assert group.column_inhibition.tolist() == [0, 0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cells_per_column": 0, "duration": 1.0}, "cells_per_column"),
        ({"cells_per_column": 2, "duration": 0.0}, "duration"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MinicolumnInhibition(**kwargs)
