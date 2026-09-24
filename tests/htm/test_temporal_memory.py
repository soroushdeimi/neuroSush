"""Temporal memory: learning rules and the sequence properties of Hawkins and Ahmad (2016)."""

import pytest
import torch

from neurosush.htm.sdr import random_sdr
from neurosush.htm.temporal_memory import TemporalMemory

COLUMNS, W = 256, 12


def memory(**kwargs):
    options = {
        "cells_per_column": 8,
        "activation_threshold": 8,
        "min_threshold": 6,
        "max_new_synapses": 12,
        "max_synapses_per_segment": 16,
        "initial_permanence": 0.51,
        "seed": 0,
    }
    return TemporalMemory(COLUMNS, **{**options, **kwargs})


def patterns(count, seed=0):
    return random_sdr(COLUMNS, W, batch=(count,), generator=torch.Generator().manual_seed(seed))


def train(tm, sequences, repeats):
    for _ in range(repeats):
        for sequence in sequences:
            tm.reset()
            for step in sequence:
                tm.compute(step)


def predictions_along(tm, sequence):
    """Predicted columns after each element (learning off)."""
    tm.reset()
    out = []
    for step in sequence:
        tm.compute(step, learn=False)
        out.append(tm.predicted_columns().clone())
    return out


class TestActivation:
    def test_unpredicted_columns_burst(self):
        tm = memory()
        x = patterns(1)[0]
        active = tm.compute(x)
        assert active.view(COLUMNS, 8)[x].all()
        assert not active.view(COLUMNS, 8)[~x].any()
        assert tm.anomaly == 1.0
        assert tm.winner_cells.view(COLUMNS, 8).sum(-1)[x].tolist() == [1] * W

    def test_input_shape_is_checked(self):
        with pytest.raises(ValueError, match="columns"):
            memory().compute(torch.zeros(3, dtype=torch.bool))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"min_threshold": 9}, "min_threshold"), ({"max_new_synapses": 40}, "max_new_synapses")],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            memory(**kwargs)


class TestLearningRules:
    def test_a_bursting_column_grows_a_segment_to_previous_winners(self):
        tm = memory()
        a, b = patterns(2)
        tm.compute(a)
        prev_winners = tm.winner_cells.clone()
        tm.compute(b)
        assert tm.n_segments == W  # one new segment per bursting column of b
        synapses = tm.presynaptic[: tm.n_segments]
        assert ((synapses >= 0).sum(-1) == W).all()  # min(max_new, |winners|) = 12
        assert prev_winners[synapses[synapses >= 0]].all()
        grown = tm.permanence[: tm.n_segments][synapses >= 0]
        torch.testing.assert_close(grown, torch.full_like(grown, 0.51))

    def test_adapt_rule(self):
        tm = memory(increment=0.1, decrement=0.05)
        segment = tm._create_segment(0)
        tm.presynaptic[segment, :3] = torch.tensor([10, 11, 12])
        tm.permanence[segment, :3] = torch.tensor([0.5, 0.5, 0.02])
        prev = torch.zeros(tm.n_cells, dtype=torch.bool)
        prev[10] = True
        tm._adapt(segment, prev, 0.1, 0.05)
        torch.testing.assert_close(tm.permanence[segment, :3], torch.tensor([0.6, 0.45, 0.0]))
        assert (tm.permanence[segment, 3:] == 0).all()  # empty slots stay empty

    def test_growth_replaces_the_weakest_synapses_when_full(self):
        tm = memory(max_synapses_per_segment=12)
        segment = tm._create_segment(0)
        tm.presynaptic[segment] = torch.arange(100, 112)
        tm.permanence[segment] = torch.linspace(0.1, 0.9, 12)
        winners = torch.zeros(tm.n_cells, dtype=torch.bool)
        winners[[200, 201]] = True
        tm._grow(segment, winners, 2)
        kept = set(tm.presynaptic[segment].tolist())
        assert {200, 201} <= kept
        assert 100 not in kept  # the two weakest were removed
        assert 101 not in kept

    def test_least_recently_used_segment_is_recycled(self):
        tm = memory(max_segments_per_cell=2)
        first = tm._create_segment(5)
        tm.iteration = 3
        second = tm._create_segment(5)
        tm.iteration = 4
        third = tm._create_segment(5)
        assert third == first != second
        assert tm.n_segments == 2

    def test_wrong_predictions_are_punished(self):
        tm = memory(predicted_decrement=0.05)
        a, b, c = patterns(3)
        train(tm, [[a, b]], repeats=3)
        before = tm.permanence[: tm.n_segments].clone()
        tm.reset()
        tm.compute(a)  # now b's cells are predicted
        tm.compute(c)  # ...but c arrives: b's matching segments lose 0.05 per active synapse
        changed = tm.permanence[: before.shape[0]] - before
        assert changed.min().item() == pytest.approx(-0.05)


class TestSequenceMemory:
    def test_learns_a_first_order_sequence(self):
        tm = memory()
        seq = list(patterns(5, seed=1))
        train(tm, [seq], repeats=5)
        predicted = predictions_along(tm, seq)
        for step in range(4):
            assert torch.equal(predicted[step], seq[step + 1])
        tm.reset()
        anomalies = []
        for x in seq:
            tm.compute(x, learn=False)
            anomalies.append(tm.anomaly)
        assert anomalies == [1.0, 0.0, 0.0, 0.0, 0.0]

    def test_high_order_context_disambiguates(self):
        # ABCD and XBCY share B and C: after C only the first-step context tells D from Y
        a, b, c, d, x, y = patterns(6, seed=2)
        tm = memory(predicted_decrement=0.05)
        train(tm, [[a, b, c, d], [x, b, c, y]], repeats=10)
        after_abc = predictions_along(tm, [a, b, c])[-1]
        after_xbc = predictions_along(tm, [x, b, c])[-1]
        assert torch.equal(after_abc, d)
        assert torch.equal(after_xbc, y)

    def test_without_punishment_a_stale_prediction_persists(self):
        # the first XBCY pass lets the A-context C cells learn to predict Y; only the
        # predicted_decrement punishment removes that link when ABC is followed by D
        a, b, c, d, x, y = patterns(6, seed=2)
        tm = memory(predicted_decrement=0.0)
        train(tm, [[a, b, c, d], [x, b, c, y]], repeats=10)
        assert torch.equal(predictions_along(tm, [a, b, c])[-1], d | y)

    def test_branching_predicts_the_union(self):
        a, b, c = patterns(3, seed=3)
        tm = memory()
        train(tm, [[a, b], [a, c]], repeats=6)
        predicted = predictions_along(tm, [a])[0]
        assert torch.equal(predicted, b | c)

    def test_reset_forgets_context_but_not_learning(self):
        seq = list(patterns(3, seed=4))
        tm = memory()
        train(tm, [seq], repeats=5)
        tm.reset()
        assert not tm.predictive_cells.any()
        assert torch.equal(predictions_along(tm, seq)[0], seq[1])
