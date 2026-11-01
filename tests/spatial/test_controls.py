"""Tests for online behavioral and sensory control baselines."""

from __future__ import annotations

import pytest
import torch

from neurosush.spatial.controls import seen_targets, sensory_count_predictions
from neurosush.spatial.graph_world import Walk


class TestControls:
    @pytest.fixture(autouse=True)
    def _single_thread(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(threads)

    def test_seen_targets_hand_computed(self):
        walk = Walk(
            nodes=torch.tensor([0, 1, 0, 2]),
            observations=torch.tensor([0, 1, 0, 2]),
            actions=torch.tensor([0, 0, 0]),
            novel_edge_to_seen=torch.tensor([False, True, False]),
            fraction_seen=torch.tensor([0.0, 0.5, 0.5]),
        )
        mask = seen_targets(walk)
        torch.testing.assert_close(mask, torch.tensor([False, True, False]))

    def test_online_sensory_counts_do_not_see_future_targets(self):
        def build_walk(labels):
            return Walk(
                nodes=torch.arange(4),
                observations=torch.tensor(labels),
                actions=torch.tensor([0, 1, 0]),
                novel_edge_to_seen=torch.zeros(3, dtype=torch.bool),
                fraction_seen=torch.zeros(3),
            )

        a = sensory_count_predictions(build_walk([2, 1, 2, 0]), 3)
        b = sensory_count_predictions(build_walk([2, 1, 2, 1]), 3)
        for name in a:
            torch.testing.assert_close(a[name], b[name])
        assert int(a["frequency"][0]) == 2

    def test_predictions_do_not_use_same_step_target(self):
        w1 = Walk(
            nodes=torch.arange(4),
            observations=torch.tensor([1, 2, 0, 2]),
            actions=torch.tensor([0, 1, 0]),
            novel_edge_to_seen=torch.zeros(3, dtype=torch.bool),
            fraction_seen=torch.zeros(3),
        )
        w2 = Walk(
            nodes=torch.arange(4),
            observations=torch.tensor([1, 2, 0, 1]),
            actions=torch.tensor([0, 1, 0]),
            novel_edge_to_seen=torch.zeros(3, dtype=torch.bool),
            fraction_seen=torch.zeros(3),
        )
        p1 = sensory_count_predictions(w1, 3)
        p2 = sensory_count_predictions(w2, 3)
        torch.testing.assert_close(p1["frequency"][:2], p2["frequency"][:2])
        torch.testing.assert_close(p1["conditional"][:2], p2["conditional"][:2])

    def test_controls_validation(self):
        walk = Walk(
            nodes=torch.zeros(2, dtype=torch.long),
            observations=torch.zeros(2, dtype=torch.long),
            actions=torch.zeros(1, dtype=torch.long),
            novel_edge_to_seen=torch.zeros(1, dtype=torch.bool),
            fraction_seen=torch.zeros(1),
        )
        with pytest.raises(ValueError, match="sensory"):
            sensory_count_predictions(walk, 0)
