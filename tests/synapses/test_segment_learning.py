"""The temporal memory learning rules on spiking segments, one hand-computed case each.

A layer of two minicolumns with two cells each (cells 0-1 and 2-3). Context is the spikes
5 to 15 steps old; the tests set the state directly and run one learning step at t = 20.
"""

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.synapses.segment_learning import SegmentLearning
from neurosush.synapses.segments import ActiveSegments
from neurosush.synapses.traces import SpikeGather

T = 20


def layer(**kwargs):
    options = {
        "cells_per_column": 2,
        "context": (5.0, 15.0),
        "min_threshold": 2,
        "max_new_synapses": 3,
        "predicted_decrement": 0.05,
    }
    net = Network(seed=0)
    group = NeuronGroup(net, 4, [Axon()])
    learning = SegmentLearning(**{**options, **kwargs})
    syn = SynapseGroup(
        net,
        group,
        group,
        [
            ActiveSegments(segments=2, synapses=4, activation_threshold=2, plateau=10.0),
            SpikeGather(),
            learning,
        ],
        compartment="distal",
    )
    net.initialize()
    net.iteration = T
    syn.post_spike = torch.zeros(4, dtype=torch.bool)
    return net, syn, learning


def spiked(syn, cells, age, *, won=True):
    """Record a (latest) spike ``age`` steps ago."""
    syn.last_spike[cells, 0] = T - age
    if won:
        syn.last_win[cells, 0] = T - age


def segment(syn, cell, index, presynaptic, permanence):
    syn.presynaptic[cell, index] = torch.tensor(presynaptic)
    syn.permanence[cell, index] = torch.tensor(permanence, dtype=syn.permanence.dtype)


class TestPredictedCells:
    def test_reinforces_and_grows_the_predicting_segment(self):
        _, syn, learning = layer()
        spiked(syn, [0, 1], age=10)  # the previous element: context and winners
        segment(syn, 3, 0, [0, 2, -1, -1], [0.3, 0.3, 0.0, 0.0])
        syn.plateau_steps[3, 0], syn.segment_start[3, 0, 0] = 5, T - 8  # started by the context
        syn.post_spike[3] = True
        learning.forward(syn)
        # +0.1 for the context synapse, -0.1 for the other; 3 - 1 context synapses to grow,
        # but only winner 1 is not on the segment yet
        assert syn.presynaptic[3, 0].tolist() == [0, 2, 1, -1]
        assert syn.permanence[3, 0].tolist() == pytest.approx([0.4, 0.2, 0.21, 0.0])
        assert syn.last_win[3, 0].item() == T
        assert syn.last_win[2, 0].item() != T  # only the predicted cell wins

    def test_a_plateau_started_by_the_current_element_is_not_a_prediction(self):
        _, syn, learning = layer()
        spiked(syn, [0, 1], age=10)
        segment(syn, 3, 0, [0, 1, -1, -1], [0.6, 0.6, 0.0, 0.0])
        syn.plateau_steps[3, 0], syn.segment_start[3, 0, 0] = 9, T - 1  # 1 step old
        syn.post_spike[2:] = True  # the column bursts
        learning.forward(syn)
        # treated as a burst: the segment only matches (2 context synapses) and learns
        assert syn.permanence[3, 0].tolist() == pytest.approx([0.7, 0.7, 0.0, 0.0])
        assert syn.last_win[3, 0].item() == T


class TestRepeatedMinicolumns:
    def test_a_cell_that_fires_again_still_counts_as_context(self):
        # cell 0 won in the previous element (10 steps ago) and fired again 1 step ago
        # because its minicolumn is active twice in a row: it is still context
        _, syn, learning = layer()
        spiked(syn, [0, 1], age=10)
        syn.last_spike[0] = torch.tensor([T - 1, T - 10])
        segment(syn, 2, 0, [0, 1, -1, -1], [0.3, 0.3, 0.0, 0.0])
        syn.post_spike[2:] = True
        learning.forward(syn)
        assert syn.permanence[2, 0].tolist() == pytest.approx([0.4, 0.4, 0.0, 0.0])

    def test_times_keep_the_previous_one(self):
        _, syn, learning = layer()
        syn.post_spike[1] = True
        learning.forward(syn)
        assert syn.last_spike[1].tolist() == [T, -(10**9)]
        syn.net.iteration = T + 7
        learning.forward(syn)
        assert syn.last_spike[1].tolist() == [T + 7, T]


class TestBurstingColumns:
    def test_best_matching_segment_learns_and_its_cell_wins(self):
        _, syn, learning = layer()
        spiked(syn, [0, 1], age=10)
        segment(syn, 2, 0, [0, 1, -1, -1], [0.3, 0.3, 0.0, 0.0])  # 2 context synapses
        segment(syn, 3, 0, [0, -1, -1, -1], [0.3, 0.0, 0.0, 0.0])  # 1: below min_threshold
        syn.post_spike[2:] = True
        learning.forward(syn)
        assert syn.permanence[2, 0].tolist() == pytest.approx([0.4, 0.4, 0.0, 0.0])
        assert syn.permanence[3, 0].tolist() == pytest.approx([0.3, 0.0, 0.0, 0.0])
        assert syn.last_win[[2, 3], 0].tolist() == [T, -(10**9)]

    def test_without_a_match_the_least_used_cell_grows_a_segment(self):
        _, syn, learning = layer()
        spiked(syn, [2, 3], age=10)
        segment(syn, 0, 0, [3, -1, -1, -1], [0.3, 0.0, 0.0, 0.0])  # cell 0 has one segment
        syn.post_spike[:2] = True
        learning.forward(syn)
        assert sorted(syn.presynaptic[1, 0].tolist()) == [-1, -1, 2, 3]
        assert sorted(syn.permanence[1, 0].tolist()) == pytest.approx([0.0, 0.0, 0.21, 0.21])
        assert syn.last_win[1, 0].item() == T

    def test_a_full_cell_replaces_its_least_recently_used_segment(self):
        _, syn, learning = layer()
        spiked(syn, [2, 3], age=10)
        for cell in (0, 1):
            for index, used in ((0, 3), (1, 7)):
                segment(syn, cell, index, [cell, -1, -1, -1], [0.5, 0.0, 0.0, 0.0])
                syn.segment_used[cell, index] = used
        syn.post_spike[:2] = True
        learning.forward(syn)
        winner = int(syn.last_win[:2, 0].argmax())
        assert sorted(syn.presynaptic[winner, 0].tolist()) == [-1, -1, 2, 3]  # used at 3
        assert syn.presynaptic[winner, 1].tolist() == [winner, -1, -1, -1]

    def test_no_context_means_no_new_segment(self):
        _, syn, learning = layer()
        syn.post_spike[:2] = True
        learning.forward(syn)
        assert (syn.presynaptic == -1).all()
        assert (syn.last_win[:2, 0] == T).sum() == 1  # a winner is still chosen


class TestWrongPredictions:
    def test_an_unanswered_plateau_weakens_what_started_it(self):
        _, syn, learning = layer()
        duration = syn.input.duration
        segment(syn, 3, 0, [0, 1, 2, -1], [0.5, 0.5, 0.5, 0.0])
        syn.segment_start[3, 0, 0] = T - duration
        syn.activation_synapses[3, 0] = torch.tensor([True, True, False, False])
        learning.forward(syn)
        assert syn.permanence[3, 0].tolist() == pytest.approx([0.45, 0.45, 0.5, 0.0])

    def test_a_fired_cell_is_not_punished(self):
        _, syn, learning = layer()
        duration = syn.input.duration
        segment(syn, 3, 0, [0, 1, -1, -1], [0.5, 0.5, 0.0, 0.0])
        syn.segment_start[3, 0, 0] = T - duration
        syn.activation_synapses[3, 0] = torch.tensor([True, True, False, False])
        syn.last_spike[3, 0] = T - 3  # it fired during the plateau
        learning.forward(syn)
        assert syn.permanence[3, 0].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])

    def test_a_dendritic_spike_records_its_synapses(self):
        _, syn, learning = layer()
        segment(syn, 3, 0, [0, 1, 2, -1], [0.6, 0.6, 0.6, 0.0])
        syn.pre_recent = torch.tensor([2, 0, 1, 0])
        syn.active_segments[3, 0] = True
        learning.forward(syn)
        assert syn.segment_start[3, 0, 0].item() == T
        assert syn.activation_synapses[3, 0].tolist() == [True, False, True, False]


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"cells_per_column": 0}, "cells_per_column"),
            ({"context": (10.0, 5.0)}, "context"),
            ({"min_threshold": 0}, "min_threshold"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        options = {"cells_per_column": 2, "context": (5.0, 15.0), "min_threshold": 2}
        with pytest.raises(ValueError, match=match):
            SegmentLearning(**{**options, **kwargs})

    def test_needs_segments_on_a_layer_connected_to_itself(self):
        net = Network()
        a, b = NeuronGroup(net, 4, [Axon()]), NeuronGroup(net, 4, [Axon()])
        SynapseGroup(
            net,
            a,
            b,
            [
                ActiveSegments(segments=1, synapses=2, activation_threshold=1, plateau=5.0),
                SpikeGather(),
                SegmentLearning(cells_per_column=2, context=(5.0, 15.0), min_threshold=1),
            ],
        )
        with pytest.raises(ValueError, match="connected to itself"):
            net.initialize()

    def test_batches_are_rejected(self):
        net = Network(batch_size=2)
        group = NeuronGroup(net, 4, [Axon()])
        SynapseGroup(
            net,
            group,
            group,
            [
                ActiveSegments(segments=1, synapses=2, activation_threshold=1, plateau=5.0),
                SpikeGather(),
                SegmentLearning(cells_per_column=2, context=(5.0, 15.0), min_threshold=1),
            ],
        )
        with pytest.raises(ValueError, match="batches"):
            net.initialize()
