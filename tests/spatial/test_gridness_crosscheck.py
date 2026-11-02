"""Independent cross-check of gridness metric against Spiking TEM authors' algorithm."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from neurosush.htm.grid_cells import hexagonal_rate
from neurosush.spatial import gridness, spatial_autocorrelogram


def authors_style_gridness(rate_map: torch.Tensor, radius: float = 5.0) -> tuple[float, float]:
    """Authors' style gridness: autocorrelogram, annulus mask, and rotational correlations.

    Our reading of Spiking TEM plotting_functions.py:
    - SAC via spatial_autocorrelogram.
    - Annulus mask: radius < dist_from_center < radius + 4.
    - Rotation via bilinear interpolation at [30, 45, 60, 90, 120, 135, 150] degrees.
    - Hex score = min(corr[60], corr[120]) - max(corr[30], corr[90], corr[150]).
    - Square score = corr[90] - max(corr[45], corr[135]).
    """
    sac = spatial_autocorrelogram(rate_map)
    h, w = sac.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )
    dist = (x - cx).square() + (y - cy).square()
    mask = (dist > radius**2) & (dist < (radius + 4) ** 2) & torch.isfinite(sac)

    values = torch.nan_to_num(sac)[None, None]
    correlations: dict[int, float] = {}
    for angle in (30, 45, 60, 90, 120, 135, 150):
        a = math.radians(angle)
        grid = torch.stack(
            (
                ((x - cx) * math.cos(a) - (y - cy) * math.sin(a)) / cx,
                ((x - cx) * math.sin(a) + (y - cy) * math.cos(a)) / cy,
            ),
            -1,
        )[None]
        rotated = F.grid_sample(
            values, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[0, 0]
        use = mask & torch.isfinite(rotated)
        u, v = sac[use], rotated[use]
        u, v = u - u.mean(), v - v.mean()
        norm = u.norm() * v.norm()
        correlations[angle] = (
            float((u * v).sum() / norm) if use.sum() >= 3 and norm > 1e-12 else math.nan
        )

    c = correlations
    hex_score = min(c[60], c[120]) - max(c[30], c[90], c[150])
    square_score = c[90] - max(c[45], c[135])
    return hex_score, square_score


class TestGridnessCrosscheck:
    @pytest.fixture(autouse=True)
    def _single_thread(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(threads)

    def test_crosscheck_49x49_maps(self):
        axis = torch.arange(49, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")

        hex_map = hexagonal_rate(torch.stack((x, y), -1), scale=10, orientation=0.0)
        sq_map = 2 + torch.cos(2 * math.pi * x / 10) + torch.cos(2 * math.pi * y / 10)
        noise_map = torch.randn(49, 49, generator=torch.Generator().manual_seed(42))
        bump_map = torch.exp(-((x - 24).square() + (y - 24).square()) / (2 * 5**2))

        maps = [hex_map, sq_map, noise_map, bump_map]

        lib_hex_scores = []
        lib_sq_scores = []
        auth_hex_scores = []
        auth_sq_scores = []

        for m in maps:
            g_lib = gridness(m, annulus=(5, 15))
            h_auth, s_auth = authors_style_gridness(m, radius=5.0)
            lib_hex_scores.append(g_lib.hexagonal)
            lib_sq_scores.append(g_lib.square)
            auth_hex_scores.append(h_auth)
            auth_sq_scores.append(s_auth)

        print("Lib Hex:", lib_hex_scores)
        print("Auth Hex:", auth_hex_scores)
        print("Lib Sq:", lib_sq_scores)
        print("Auth Sq:", auth_sq_scores)

        assert lib_hex_scores[0] > 0.8
        assert auth_hex_scores[0] > 0.8
        for i in range(1, 4):
            assert lib_hex_scores[i] < 0.8
            assert auth_hex_scores[i] < 0.8

        assert lib_sq_scores[1] > 0.8
        assert auth_sq_scores[1] > 0.8
        assert lib_sq_scores[0] < 0.8
        assert auth_sq_scores[0] < 0.8

        lib_hex_rank = sorted(
            range(len(lib_hex_scores)), key=lambda i: lib_hex_scores[i], reverse=True
        )
        auth_hex_rank = sorted(
            range(len(auth_hex_scores)), key=lambda i: auth_hex_scores[i], reverse=True
        )
        assert lib_hex_rank == auth_hex_rank

        lib_sq_rank = sorted(
            range(len(lib_sq_scores)), key=lambda i: lib_sq_scores[i], reverse=True
        )
        auth_sq_rank = sorted(
            range(len(auth_sq_scores)), key=lambda i: auth_sq_scores[i], reverse=True
        )
        assert lib_sq_rank == auth_sq_rank

    def test_8x8_noise_maps_below_threshold(self):
        for seed in range(50):
            noise = torch.randn(8, 8, generator=torch.Generator().manual_seed(seed))
            g_lib = gridness(noise)
            h_auth, _s_auth = authors_style_gridness(noise, radius=1.0)
            if math.isfinite(g_lib.hexagonal):
                assert g_lib.hexagonal < 0.8
            if math.isfinite(h_auth):
                assert h_auth < 0.8
