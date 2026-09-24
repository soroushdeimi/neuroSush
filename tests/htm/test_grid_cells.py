"""Grid cells: lattice geometry, path integration and the population location code."""

import math
from itertools import pairwise

import pytest
import torch

from neurosush.htm.grid_cells import GridCellModule, GridCellModules, hexagonal_rate


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return v @ torch.tensor([[c, s], [-s, c]], dtype=v.dtype)


class TestHexagonalRate:
    def test_peaks_on_the_lattice(self):
        lattice = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3)], [-3.0, math.sqrt(3)]], dtype=torch.float64
        )
        rates = hexagonal_rate(lattice, scale=2.0)
        torch.testing.assert_close(rates, torch.ones(4, dtype=torch.float64))

    def test_range_is_zero_to_one(self):
        grid = torch.cartesian_prod(*(torch.linspace(-5, 5, 101, dtype=torch.float64),) * 2)
        rates = hexagonal_rate(grid, scale=1.7, orientation=0.3)
        assert rates.min().item() == pytest.approx(0.0, abs=1e-3)
        assert rates.max().item() <= 1.0 + 1e-12

    def test_sixfold_rotational_symmetry(self):
        points = torch.randn(
            200, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
        )
        rates = hexagonal_rate(points, scale=1.3)
        torch.testing.assert_close(hexagonal_rate(rotate(points, math.pi / 3), scale=1.3), rates)


class TestModule:
    def module(self):
        return GridCellModule(scale=1.5, orientation=0.4, cells_per_axis=8)

    def test_phase_is_periodic_on_the_lattice(self):
        m = self.module()
        x = torch.randn(50, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        for lattice_vector in m.basis.T:
            diff = m.phase(x + 3 * lattice_vector) - m.phase(x)
            torch.testing.assert_close(
                diff - diff.round(), torch.zeros_like(diff), atol=1e-9, rtol=0
            )

    def test_path_integration_is_path_independent(self):
        m = self.module()
        g = torch.Generator().manual_seed(2)
        start = torch.randn(2, dtype=torch.float64, generator=g)
        steps = torch.randn(300, 2, dtype=torch.float64, generator=g) * 0.2
        phase = m.phase(start)
        for step in steps:
            phase = m.move(phase, step)
        direct = m.phase(start + steps.sum(0))
        diff = phase - direct
        torch.testing.assert_close(
            diff - diff.round(), torch.zeros(2, dtype=torch.float64), atol=1e-9, rtol=0
        )

    def test_closed_loop_returns_to_the_same_phase(self):
        m = self.module()
        phase = m.phase(torch.tensor([0.3, -0.7], dtype=torch.float64))
        square = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]], dtype=torch.float64
        )
        end = phase
        for step in square:
            end = m.move(end, step)
        torch.testing.assert_close(end, phase, atol=1e-9, rtol=0)

    def test_firing_fields_form_the_lattice(self):
        m = self.module()
        cell = 0
        center = m.preferred[cell] @ m.basis.T  # a position with the cell's preferred phase
        for lattice_vector in (m.basis[:, 0], m.basis[:, 1], m.basis[:, 0] - m.basis[:, 1]):
            rate = m.activity(m.phase(center + lattice_vector))[cell]
            assert rate.item() == pytest.approx(1.0)
        # halfway between two fields the cell is silent
        between = m.activity(m.phase(center + m.basis[:, 0] / 2))[cell]
        assert between.item() < 1e-3

    def test_bump_is_gaussian_in_world_distance(self):
        m = GridCellModule(scale=2.0, cells_per_axis=4, bump_width=0.1)
        center = m.preferred[5] @ m.basis.T
        offset = torch.tensor([0.15, 0.0], dtype=torch.float64)
        rate = m.activity(m.phase(center + offset))[5].item()
        assert rate == pytest.approx(math.exp(-(0.15**2) / (2 * 0.2**2)))

    def test_noisy_path_integration_drifts_like_a_random_walk(self):
        # position error of integrating noisy velocity grows with variance t * sigma^2
        g = torch.Generator().manual_seed(3)
        sigma, walkers = 0.05, 4000
        noise = torch.randn(100, walkers, 2, dtype=torch.float64, generator=g) * sigma
        error = noise.cumsum(0)
        variance = error.var(1).mean(-1)
        for t in (10, 50, 100):
            assert variance[t - 1].item() == pytest.approx(t * sigma**2, rel=0.1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"scale": 0.0}, "scale"), ({"scale": 1.0, "bump_width": 0.0}, "bump_width")],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GridCellModule(**kwargs)


class TestPopulationCode:
    def test_modules_locate_far_beyond_their_scales(self):
        modules = GridCellModules([1.0, 1.4, 1.9, 2.6], [0.0, 0.3, 0.7, 1.1])
        axis = torch.arange(-10, 10.01, 0.25, dtype=torch.float64)
        candidates = torch.cartesian_prod(axis, axis)
        g = torch.Generator().manual_seed(4)
        targets = candidates[torch.randint(len(candidates), (40,), generator=g)]
        decoded = torch.stack([modules.decode(modules.phases(x), candidates) for x in targets])
        torch.testing.assert_close(decoded, targets)

    def test_more_modules_make_distant_places_less_ambiguous(self):
        # a random pair of places looks alike only if every module's phases happen to align
        g = torch.Generator().manual_seed(5)
        a, b = (torch.rand(5000, 2, dtype=torch.float64, generator=g) * 40 - 20 for _ in range(2))
        scales, orientations = [1.0, 1.4, 1.9, 2.6], [0.0, 0.5, 1.0, 1.4]

        def similarity(modules, x, y):
            ex, ey = modules.encode(modules.phases(x)), modules.encode(modules.phases(y))
            return (ex * ey).sum(-1) / (ex.norm(dim=-1) * ey.norm(dim=-1))

        aliased = []
        for k in range(1, 5):
            modules = GridCellModules(scales[:k], orientations[:k])
            aliased.append((similarity(modules, a, b) > 0.5).double().mean().item())
        assert all(later < earlier for earlier, later in pairwise(aliased))
        assert aliased[-1] < 0.01
        assert similarity(modules, a, a + 0.05).mean().item() > 0.9

    def test_moving_all_modules_agrees_with_direct_phases(self):
        modules = GridCellModules([1.0, 1.7])
        start = torch.tensor([0.4, 0.4], dtype=torch.float64)
        step = torch.tensor([3.1, -2.2], dtype=torch.float64)
        moved = modules.move(modules.phases(start), step)
        diff = moved - modules.phases(start + step)
        torch.testing.assert_close(diff - diff.round(), torch.zeros_like(diff), atol=1e-9, rtol=0)

    def test_needs_matching_orientations(self):
        with pytest.raises(ValueError, match="orientation"):
            GridCellModules([1.0, 2.0], [0.0])
