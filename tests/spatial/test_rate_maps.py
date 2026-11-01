"""Tests for rate maps, spatial information, and position-shuffling control."""

from __future__ import annotations

import pytest
import torch

from neurosush.spatial import rate_maps, shuffled_position_control, spatial_information


class TestRateMaps:
    @pytest.fixture(autouse=True)
    def _single_thread(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(threads)

    def test_rate_map_and_information_hand_values(self):
        activity = torch.tensor([[0.0, 2.0], [2.0, 2.0], [4.0, 0.0]])
        maps, counts = rate_maps(activity, torch.tensor([0, 0, 1]), (2, 2))
        expected = torch.tensor([[1.0, 4.0], [2.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(maps[:, 0, :], expected)
        assert maps[:, 1].isnan().all()
        assert counts.tolist() == [[2, 1], [0, 0]]
        info = spatial_information(
            torch.tensor([[0.0, 2.0], [1.0, 1.0], [0.0, 0.0]]), torch.ones(2)
        )
        torch.testing.assert_close(info, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_uniform_and_sample_position_shuffle_information(self):
        nodes = torch.arange(64).repeat(80)
        activity = torch.stack(((nodes == 20).double(), torch.ones(len(nodes))), -1)
        maps, occupancy = rate_maps(activity, nodes, (8, 8))
        info = spatial_information(maps, occupancy)
        assert float(info[0]) == pytest.approx(6.0)
        assert float(info[1]) == pytest.approx(0.0)
        null = shuffled_position_control(activity, nodes, (8, 8), repeats=8, seed=7)
        again = shuffled_position_control(activity, nodes, (8, 8), repeats=8, seed=7)
        torch.testing.assert_close(null.information, again.information)
        assert null.square.shape == (8, 2)
        assert float(null.information[:, 0].mean()) < 0.8
        assert (null.information[:, 1].abs() < 1e-12).all()

    def test_metric_validation(self):
        with pytest.raises(ValueError, match="nonnegative"):
            spatial_information(torch.tensor([[-1.0, 1.0]]), torch.ones(2))
