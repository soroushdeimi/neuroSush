"""Discrete TEM world checks."""

from __future__ import annotations

import pytest
import torch

from neurosush.spatial import GraphWorld


class TestGraphWorld:
    def test_square_walk_seed_boundaries_and_labels(self):
        for width in (5, 6, 7, 8):
            world = GraphWorld(width, seed=4)
            walk = world.random_walk(200, start=0)
            other = GraphWorld(width, seed=4).random_walk(200, start=0)
            torch.testing.assert_close(walk.nodes, other.nodes)
            torch.testing.assert_close(walk.observations, world.labels[walk.nodes])
            assert walk.nodes.shape == (201,)
            assert walk.actions.shape == (200,)
            assert world.step(0, 1) == 0
            assert world.step(0, 3) == 0
            for source, action, destination in zip(
                walk.nodes[:-1], walk.actions, walk.nodes[1:], strict=True
            ):
                assert world.step(int(source), int(action)) == int(destination)
            assert not walk.novel_edge_to_seen[0]
            assert float(walk.fraction_seen[0]) == pytest.approx(1 / world.size)
            assert (walk.fraction_seen[1:] >= walk.fraction_seen[:-1]).all()

    def test_track_and_shuttle(self):
        world = GraphWorld(40, track=True, sensory=10, seed=2)
        walk = world.random_walk(100, stay=False)
        assert (walk.actions < 2).all()
        assert ((walk.nodes >= 0) & (walk.nodes < 40)).all()
        shuttle = world.shuttle(3)
        assert len(shuttle) == 118
        assert shuttle[[0, 39, 78]].tolist() == [0, 39, 0]
        assert (shuttle.diff().abs() == 1).all()

    def test_novel_edge_evaluation_has_no_future_leakage(self):
        walk = GraphWorld(5, seed=7).random_walk(500)
        seen, edges = {int(walk.nodes[0])}, set()
        for t, (source, target) in enumerate(
            zip(walk.nodes[:-1].tolist(), walk.nodes[1:].tolist(), strict=True)
        ):
            expected = source != target and target in seen and (source, target) not in edges
            assert bool(walk.novel_edge_to_seen[t]) == expected
            assert float(walk.fraction_seen[t]) == pytest.approx(len(seen) / 25)
            seen.add(target)
            edges.add((source, target))
        assert walk.novel_edge_to_seen.any()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 4},
            {"width": 8, "track": True},
            {"sensory": 0},
        ],
    )
    def test_invalid_world(self, kwargs):
        with pytest.raises(ValueError, match=r"invalid|sensory"):
            GraphWorld(**kwargs)

    def test_small_world_legal_actions_match_supplied_displacements(self):
        world = GraphWorld(3, diagnostic_small=True, seed=3)
        walk = world.random_walk(200, stay=False, valid_only=True)
        assert (walk.nodes.diff() != 0).all()
        for source, action, target in zip(
            walk.nodes[:-1], walk.actions, walk.nodes[1:], strict=True
        ):
            assert world.step(int(source), int(action)) == int(target)
        other = GraphWorld(3, diagnostic_small=True, seed=3).random_walk(
            200, stay=False, valid_only=True
        )
        torch.testing.assert_close(walk.actions, other.actions)
