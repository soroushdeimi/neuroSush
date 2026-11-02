"""Tests for spatial autocorrelogram and gridness metrics."""

from __future__ import annotations

import math

import pytest
import torch

from neurosush.htm.grid_cells import hexagonal_rate
from neurosush.spatial import gridness, spatial_autocorrelogram


class TestGridness:
    @pytest.fixture(autouse=True)
    def _single_thread(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(threads)

    def test_fixed_hexagonal_reference_scores_high(self):
        axis = torch.arange(49, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        for orientation in (0.0, 0.3):
            rate = hexagonal_rate(torch.stack((x, y), -1), scale=10, orientation=orientation)
            score = gridness(rate, annulus=(5, 15))
            assert score.hexagonal > 1.0
            assert score.hexagonal > score.square

    def test_square_cosine_is_not_hexagonal(self):
        axis = torch.arange(49, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        for orientation in (0.0, math.pi / 4):
            u = x * math.cos(orientation) - y * math.sin(orientation)
            v = x * math.sin(orientation) + y * math.cos(orientation)
            rate = 2 + torch.cos(2 * math.pi * u / 10) + torch.cos(2 * math.pi * v / 10)
            score = gridness(rate, annulus=(5, 15))
            assert score.square > 0.8
            assert score.hexagonal < 0

    def test_autocorrelation_matches_overlap_pearson_with_missing_bin(self):
        rate = torch.tensor(
            [[1.0, 4.0, 2.0], [3.0, float("nan"), 7.0], [2.0, 9.0, 5.0]], dtype=torch.float64
        )
        sac = spatial_autocorrelogram(rate)
        assert float(sac[2, 2]) == pytest.approx(1.0)
        a, b = rate[:, 1:].flatten(), rate[:, :-1].flatten()
        valid = a.isfinite() & b.isfinite()
        expected = torch.corrcoef(torch.stack((a[valid], b[valid])))[0, 1]
        assert float(sac[2, 3]) == pytest.approx(float(expected))
        torch.testing.assert_close(sac, sac.flip((0, 1)), equal_nan=True)

    def test_constant_map_has_undefined_gridness(self):
        result = gridness(torch.ones(8, 8))
        assert math.isnan(result.hexagonal)
        assert math.isnan(result.square)

    def test_metric_validation(self):
        with pytest.raises(ValueError, match="annulus"):
            gridness(torch.ones(8, 8), annulus=(5, 4))

    def test_demo_resolution_references_are_not_resolvable(self):
        y, x = torch.meshgrid(*(torch.arange(8, dtype=torch.float64),) * 2, indexing="ij")
        for scale in (2.5, 3.0, 3.5):
            references = {
                "hex": hexagonal_rate(torch.stack((x, y), -1), scale=scale),
                "square": 2
                + torch.cos(2 * math.pi * x / scale)
                + torch.cos(2 * math.pi * y / scale),
            }
            for name, rate in references.items():
                score = gridness(rate)
                assert not score.resolvable
                if name == "hex":
                    assert score.hexagonal > score.square
                else:
                    assert score.square > 0.8
                    assert score.hexagonal < 0

    def test_demo_resolution_noise_and_single_field_nulls(self):
        scores = []
        for seed in range(256):
            noise = torch.randn(8, 8, generator=torch.Generator().manual_seed(seed))
            score = gridness(noise)
            scores.append((score.hexagonal, score.square))
        values = torch.tensor(scores)
        mean, q95 = values.mean(0), values.quantile(0.95, dim=0)
        assert (mean < 0.2).all()
        assert (q95 < 0.6).all()
        assert ((values > 0.8).float().mean(0) < 0.01).all()
        y, x = torch.meshgrid(*(torch.arange(8, dtype=torch.float64),) * 2, indexing="ij")
        for sigma in (0.6, 1.0, 1.5):
            bump = torch.exp(-((x - 3.5).square() + (y - 3.5).square()) / (2 * sigma**2))
            score = gridness(bump)
            assert score.hexagonal < 0.5
            assert score.square < 0.5

    def test_16x16_synthetic_validation_hex_square_random(self):
        axis = torch.arange(16, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((x, y), -1)

        hex_rate = hexagonal_rate(coords, scale=5, orientation=0.0)
        score_hex = gridness(hex_rate, annulus=(3, 7))
        assert score_hex.resolvable
        assert score_hex.hexagonal > score_hex.square

        u = x * math.cos(0) - y * math.sin(0)
        v = x * math.sin(0) + y * math.cos(0)
        sq_rate = 2 + torch.cos(2 * math.pi * u / 5) + torch.cos(2 * math.pi * v / 5)
        score_sq = gridness(sq_rate, annulus=(3, 7))
        assert score_sq.resolvable
        assert score_sq.square > score_sq.hexagonal

        rand_rate = torch.randn(16, 16, generator=torch.Generator().manual_seed(42))
        score_rand = gridness(rand_rate, annulus=(3, 7))
        assert score_rand.hexagonal < 0.5
        assert score_rand.square < 0.5

    def test_8x8_stays_resolvable_false(self):
        axis = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((x, y), -1)
        hex_rate = hexagonal_rate(coords, scale=4, orientation=0.0)
        score = gridness(hex_rate)
        assert not score.resolvable
