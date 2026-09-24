"""Spatial pooler: exact update rules and the properties reported by Cui et al. (2017)."""

import math
from itertools import pairwise

import pytest
import torch

from neurosush.htm.sdr import overlap, random_sdr
from neurosush.htm.spatial_pooler import SpatialPooler


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def move_bits(x, fraction, seed):
    """Move ``fraction`` of each SDR's active bits to random inactive positions."""
    g = gen(seed)
    out = x.clone()
    for row in out:
        on, off = row.nonzero().flatten(), (~row).nonzero().flatten()
        k = int(fraction * len(on))
        row[on[torch.randperm(len(on), generator=g)[:k]]] = False
        row[off[torch.randperm(len(off), generator=g)[:k]]] = True
    return out


def pooler(**kwargs):
    """A small fully connected pooler on 400 inputs and 256 columns."""
    options = {"potential_radius": 400, "potential_pct": 0.8, "density": 0.04, "seed": 1}
    return SpatialPooler(400, 256, **{**options, **kwargs})


class TestStructure:
    def test_potential_pools_respect_topology(self):
        sp = SpatialPooler(100, 20, potential_radius=5, potential_pct=0.6, seed=0)
        for column in range(20):
            inputs = sp.potential[column].nonzero().flatten()
            center = int((column + 0.5) * 100 / 20)
            assert ((inputs - center).abs() <= 5).all()
            in_range = min(center + 5, 99) - max(center - 5, 0) + 1  # clipped at the edges
            assert len(inputs) == round(0.6 * in_range)

    def test_two_dimensional_pools(self):
        sp = SpatialPooler((10, 10), (5, 5), potential_radius=1, potential_pct=1.0)
        pool = sp.potential[0].reshape(10, 10)
        assert pool[0:3, 0:3].all()
        assert pool.sum().item() == 9

    def test_permanences_live_only_in_the_pool(self):
        sp = SpatialPooler(50, 10, potential_radius=3)
        assert (sp.permanences[~sp.potential] == 0).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"density": 0.0}, "density"),
            ({"potential_pct": 1.5}, "potential_pct"),
            ({"boost_strength": -1.0}, "boost_strength"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SpatialPooler(10, 10, **kwargs)

    def test_dimensions_must_agree(self):
        with pytest.raises(ValueError, match="dimensions"):
            SpatialPooler((4, 4), 8)


class TestCompute:
    def test_exact_sparsity_with_global_inhibition(self):
        sp = SpatialPooler(200, 100, potential_radius=200, density=0.05)
        active = sp.compute(random_sdr(200, 30, batch=(20,), generator=gen()), learn=False)
        assert active.sum(-1).tolist() == [5] * 20

    def test_columns_without_overlap_never_win(self):
        sp = SpatialPooler(200, 100, potential_radius=200, density=0.5)
        assert not sp.compute(torch.zeros(200, dtype=torch.bool), learn=False).any()

    def test_overlap_counts_connected_synapses_to_active_inputs(self):
        sp = SpatialPooler(4, 2, potential_radius=4, potential_pct=1.0, connected=0.5)
        sp.permanences = torch.tensor([[0.6, 0.6, 0.4, 0.0], [0.5, 0.2, 0.9, 0.9]])
        x = torch.tensor([True, True, True, False])
        assert sp.overlap(x).tolist() == [2.0, 2.0]

    def test_stimulus_threshold(self):
        sp = SpatialPooler(
            4, 2, potential_radius=4, potential_pct=1.0, connected=0.5, stimulus_threshold=2
        )
        sp.permanences = torch.tensor([[0.6, 0.0, 0.0, 0.0], [0.6, 0.6, 0.0, 0.0]])
        assert sp.overlap(torch.ones(4, dtype=torch.bool)).tolist() == [0.0, 2.0]

    def test_local_inhibition_matches_a_direct_implementation(self):
        # a column wins if fewer than density * |neighborhood| neighbors score higher
        sp = SpatialPooler(
            300, 60, potential_radius=20, density=0.1, global_inhibition=False, inhibition_radius=5
        )
        x = random_sdr(300, 60, batch=(10,), generator=gen())
        active = sp.compute(x, learn=False)
        scores = sp.overlap(x) * sp.boost + sp.tie_break
        for row, score in zip(active, scores, strict=True):
            for c in range(60):
                lo, hi = max(0, c - 5), min(60, c + 6)
                stronger = (score[lo:hi] > score[c]).sum().item()
                quota = max(1, round(0.1 * (hi - lo)))
                expected = bool(score[c] > sp.tie_break[c]) and stronger < quota
                assert bool(row[c]) == expected

    def test_two_dimensional_local_inhibition_matches_a_direct_implementation(self):
        sp = SpatialPooler(
            (20, 20),
            (10, 10),
            potential_radius=3,
            density=0.2,
            global_inhibition=False,
            inhibition_radius=2,
        )
        x = random_sdr(400, 60, batch=(4,), generator=gen(1)).reshape(4, 20, 20)
        active = sp.compute(x, learn=False).reshape(4, 10, 10)
        scores = (sp.overlap(x) * sp.boost + sp.tie_break).reshape(4, 10, 10)
        own = sp.tie_break.reshape(10, 10)
        for row, score in zip(active, scores, strict=True):
            for i in range(10):
                for j in range(10):
                    window = score[max(0, i - 2) : i + 3, max(0, j - 2) : j + 3]
                    stronger = (window > score[i, j]).sum().item()
                    quota = max(1, round(0.2 * window.numel()))
                    expected = bool(score[i, j] > own[i, j]) and stronger < quota
                    assert bool(row[i, j]) == expected

    def test_batched_inference_matches_single_samples(self):
        sp = SpatialPooler(100, 50, potential_radius=100, density=0.1)
        x = random_sdr(100, 20, batch=(6,), generator=gen())
        batched = sp.compute(x, learn=False)
        assert torch.equal(batched, torch.stack([sp.compute(s, learn=False) for s in x]))


class TestLearningRules:
    def make(self, **kwargs):
        options = {"min_overlap_duty": 0.0, **kwargs}
        sp = SpatialPooler(
            4,
            2,
            potential_radius=4,
            potential_pct=1.0,
            density=0.5,
            connected=0.5,
            active_inc=0.1,
            inactive_dec=0.05,
            **options,
        )
        # column 1 has no connected synapse to the inputs used below
        sp.permanences = torch.tensor([[0.6, 0.6, 0.4, 0.3], [0.0, 0.0, 0.4, 0.4]])
        return sp

    def test_hebbian_update_of_the_winning_column(self):
        sp = self.make()
        active = sp.compute(torch.tensor([True, True, False, False]))
        assert active.tolist() == [True, False]
        torch.testing.assert_close(sp.permanences[0], torch.tensor([0.7, 0.7, 0.35, 0.25]))
        torch.testing.assert_close(sp.permanences[1], torch.tensor([0.0, 0.0, 0.4, 0.4]))

    def test_duty_cycles_are_running_means(self):
        sp = self.make(duty_cycle_period=2)
        sp.compute(torch.tensor([True, True, False, False]))
        sp.compute(torch.zeros(4, dtype=torch.bool))
        # the period is min(t, 2): after two steps the mean of (1, 0) is 0.5
        assert sp.active_duty.tolist() == pytest.approx([0.5, 0.0])

    def test_boost_formula(self):
        sp = self.make(boost_strength=2.0, duty_cycle_period=1)
        sp.compute(torch.tensor([True, True, False, False]))
        # duties (1, 0); the target is their mean, 0.5
        assert sp.boost.tolist() == pytest.approx([math.exp(-1.0), math.exp(1.0)])

    def test_local_boost_targets_the_neighborhood_mean(self):
        sp = SpatialPooler(
            8,
            4,
            potential_radius=8,
            potential_pct=1.0,
            density=0.25,
            global_inhibition=False,
            inhibition_radius=1,
            boost_strength=1.0,
            duty_cycle_period=1,
        )
        sp.active_duty = torch.tensor([1.0, 0.0, 0.0, 1.0])
        sp._update_boost()
        # neighborhoods {0,1}, {0,1,2}, {1,2,3}, {2,3}
        targets = torch.tensor([0.5, 1 / 3, 1 / 3, 0.5])
        torch.testing.assert_close(sp.boost, torch.exp(-(sp.active_duty - targets)))

    def test_local_bumping_compares_with_the_neighborhood_maximum(self):
        sp = SpatialPooler(
            8,
            4,
            potential_radius=8,
            potential_pct=1.0,
            global_inhibition=False,
            inhibition_radius=1,
            min_overlap_duty=0.5,
        )
        sp.overlap_duty = torch.tensor([1.0, 0.4, 0.0, 0.0])
        before = sp.permanences.clone()
        sp._bump_weak_columns()
        # neighborhood maxima (1, 1, 0.4, 0): columns 1 and 2 fall below half of theirs;
        # column 3 only sees zeros, so nothing counts as more active than it
        bumped = (sp.permanences - before).abs().sum(-1) > 0
        assert bumped.tolist() == [False, True, True, False]

    def test_weak_columns_are_bumped(self):
        sp = self.make(min_overlap_duty=0.5)
        sp.compute(torch.tensor([True, True, False, False]))
        # column 1 had no overlap: all its potential permanences rise by 0.1 * connected
        torch.testing.assert_close(sp.permanences[1], torch.tensor([0.05, 0.05, 0.45, 0.45]))


class TestPaperProperties:
    def test_learning_makes_codes_noise_robust(self):
        patterns = random_sdr(400, 40, batch=(30,), generator=gen(1))
        sp = pooler()

        def stability(noise):
            clean = sp.compute(patterns, learn=False)
            noisy = sp.compute(move_bits(patterns, noise, seed=9), learn=False)
            return (overlap(noisy, clean).float() / clean.sum(-1)).mean().item()

        before = stability(0.2)
        g = gen(2)
        for _ in range(20):
            sp.compute(patterns[torch.randperm(30, generator=g)])
        after = [stability(noise) for noise in (0.1, 0.2, 0.3, 0.4)]
        assert after[1] > 0.95 > before
        assert all(a >= b for a, b in pairwise(after))

    def test_boosting_spreads_activity_over_the_columns(self):
        def busiest_share(boost):
            sp = pooler(boost_strength=boost, duty_cycle_period=100)
            for x in random_sdr(400, 40, batch=(1500,), generator=gen(3)):
                sp.compute(x)
            return sp.active_duty.max().item()

        without, weak, strong = busiest_share(0.0), busiest_share(3.0), busiest_share(20.0)
        assert strong < weak < without
        assert strong < 2 * 0.04


class TestCheckpoint:
    def test_resuming_continues_exactly(self, tmp_path):
        data = random_sdr(400, 40, batch=(60,), generator=gen(5))
        reference = pooler(boost_strength=2.0, duty_cycle_period=20)
        reference.compute(data)
        first = pooler(boost_strength=2.0, duty_cycle_period=20)
        first.compute(data[:25])
        torch.save(first.state_dict(), tmp_path / "sp.pt")
        resumed = pooler(boost_strength=2.0, duty_cycle_period=20, seed=99)  # other pools
        resumed.load_state_dict(torch.load(tmp_path / "sp.pt", weights_only=True))
        resumed.compute(data[25:])
        for key, value in reference.state_dict().items():
            got = resumed.state_dict()[key]
            assert torch.equal(got, value) if isinstance(value, torch.Tensor) else got == value, key

    def test_shapes_must_match(self):
        with pytest.raises(ValueError, match=r"potential must have shape \(256, 400\)"):
            pooler().load_state_dict(SpatialPooler(100, 256).state_dict())
