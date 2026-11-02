"""Tests for batched walks module (neurosush.spatial.batch)."""

from __future__ import annotations

import pytest
import torch

from neurosush.spatial import GraphWorld, batched_random_walks, make_worlds


class TestBatch:
    def test_make_worlds_and_shapes(self):
        worlds = make_worlds(5, seed=42)
        assert len(worlds) == 5
        # Widths cycle: 5, 6, 7, 8, 5
        assert worlds[0].width == 5
        assert worlds[1].width == 6
        assert worlds[2].width == 7
        assert worlds[3].width == 8
        assert worlds[4].width == 5

        bw = batched_random_walks(worlds, 50, stay=False, valid_only=False)
        assert bw.batch_size == 5
        assert bw.steps == 50
        assert bw.nodes.shape == (5, 51)
        assert bw.nodes.dtype == torch.long
        assert bw.observations.shape == (5, 51)
        assert bw.observations.dtype == torch.long
        assert bw.actions.shape == (5, 50)
        assert bw.actions.dtype == torch.long
        assert bw.executed_actions.shape == (5, 50)
        assert bw.executed_actions.dtype == torch.long
        assert bw.novel_edge_to_seen.shape == (5, 50)
        assert bw.novel_edge_to_seen.dtype == torch.bool
        assert bw.fraction_seen.shape == (5, 50)
        assert bw.fraction_seen.dtype == torch.float32

    def test_determinism_and_difference(self):
        worlds_a = make_worlds(3, seed=10)
        worlds_b = make_worlds(3, seed=10)
        worlds_c = make_worlds(3, seed=11)

        bw_a = batched_random_walks(worlds_a, 20)
        bw_b = batched_random_walks(worlds_b, 20)
        bw_c = batched_random_walks(worlds_c, 20)

        torch.testing.assert_close(bw_a.nodes, bw_b.nodes)
        torch.testing.assert_close(bw_a.actions, bw_b.actions)
        assert not torch.equal(bw_a.nodes, bw_c.nodes)

    def test_row_matches_single_world(self):
        batch = 3
        seed = 123
        worlds = make_worlds(batch, seed=seed)
        bw = batched_random_walks(worlds, 30, stay=False, valid_only=True)

        for i in range(batch):
            w_single = GraphWorld(
                width=5 if i % 4 == 0 else (6 if i % 4 == 1 else (7 if i % 4 == 2 else 8)),
                seed=seed * 100003 + i,
            ).random_walk(30, stay=False, valid_only=True)
            torch.testing.assert_close(bw.nodes[i], w_single.nodes)
            torch.testing.assert_close(bw.observations[i], w_single.observations)
            torch.testing.assert_close(bw.actions[i], w_single.actions)
            if w_single.executed_actions is not None:
                torch.testing.assert_close(bw.executed_actions[i], w_single.executed_actions)
            torch.testing.assert_close(bw.novel_edge_to_seen[i], w_single.novel_edge_to_seen)
            torch.testing.assert_close(bw.fraction_seen[i], w_single.fraction_seen)

    def test_track_mode(self):
        worlds = make_worlds(2, track=True, seed=5)
        for w in worlds:
            assert w.track
            assert w.width == 40
        bw = batched_random_walks(worlds, 40, stay=False, valid_only=True)
        assert bw.nodes.shape == (2, 41)
        assert ((bw.nodes >= 0) & (bw.nodes < 40)).all()

    def test_novel_edge_to_seen_correctness(self):
        worlds = make_worlds(3, seed=99)
        bw = batched_random_walks(worlds, 100)

        for b in range(bw.batch_size):
            nodes = bw.nodes[b].tolist()
            seen = {nodes[0]}
            edges = set()
            for t in range(bw.steps):
                source = nodes[t]
                target = nodes[t + 1]
                expected = source != target and target in seen and (source, target) not in edges
                assert bool(bw.novel_edge_to_seen[b, t]) == expected
                seen.add(target)
                edges.add((source, target))

    def test_error_cases(self):
        with pytest.raises(ValueError, match=r"batch"):
            make_worlds(0)
        with pytest.raises(ValueError, match=r"worlds"):
            batched_random_walks([], 10)
        with pytest.raises(ValueError, match=r"steps"):
            batched_random_walks(make_worlds(1), 0)
