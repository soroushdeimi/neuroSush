"""Tests for spatial coordinate decoding probes and normalization."""

from __future__ import annotations

import pytest
import torch

from neurosush.spatial.probe import coordinates, position_probe


class TestProbe:
    @pytest.fixture(autouse=True)
    def _single_thread(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(threads)

    def test_coordinates_hand_computed_3x3(self):
        nodes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
        coords = coordinates(nodes, 3)
        expected = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [0.5, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(coords, expected)

    def test_known_coordinate_code_beats_shift_null(self):
        nodes = torch.randint(25, (400,), generator=torch.Generator().manual_seed(7))
        code = torch.stack((nodes % 5, nodes // 5), dim=1).float()
        result = position_probe(code, nodes, 5)
        assert result["r2"] > 0.99
        assert result["shift_null_p95"] < 0.1

    def test_random_code_no_position_info(self):
        nodes = torch.randint(25, (400,), generator=torch.Generator().manual_seed(7))
        gen = torch.Generator().manual_seed(42)
        code = torch.randn(400, 10, generator=gen)
        result = position_probe(code, nodes, 5, shifts=16)
        assert result["r2"] < result["shift_null_p95"] + 0.05

    def test_probe_and_coordinate_validation(self):
        nodes = torch.zeros(10, dtype=torch.long)
        code = torch.zeros(10, 5)
        with pytest.raises(ValueError, match="width"):
            coordinates(nodes, 1)
        with pytest.raises(ValueError, match="width"):
            position_probe(code, nodes, 1)
        with pytest.raises(ValueError, match="shifts"):
            position_probe(code, nodes, 5, shifts=0)
        with pytest.raises(ValueError, match="train_fraction"):
            position_probe(code, nodes, 5, train_fraction=1.5)
        with pytest.raises(ValueError, match="gap_fraction"):
            position_probe(code, nodes, 5, gap_fraction=-0.1)
        with pytest.raises(ValueError, match="train_fraction \\+ gap_fraction"):
            position_probe(code, nodes, 5, train_fraction=0.6, gap_fraction=0.5)
        with pytest.raises(ValueError, match="length"):
            position_probe(code, torch.zeros(5, dtype=torch.long), 5)
