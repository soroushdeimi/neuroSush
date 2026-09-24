"""SDR operations, and the matching statistics validated against exact identities and
Monte-Carlo simulation."""

from fractions import Fraction
from itertools import pairwise
from math import comb

import pytest
import torch

from neurosush.htm.sdr import (
    expected_union_size,
    match_probability,
    overlap,
    overlap_count,
    random_sdr,
    sparsity,
    subsample,
    union,
    union_match_probability,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


class TestOperations:
    def test_random_sdr_has_exactly_w_bits(self):
        sdrs = random_sdr(100, 7, batch=(50,), generator=gen())
        assert sdrs.shape == (50, 100)
        assert sdrs.sum(-1).tolist() == [7] * 50

    def test_random_sdr_is_uniform_over_bits(self):
        sdrs = random_sdr(20, 4, batch=(20_000,), generator=gen())
        # every bit is active with probability w / n = 0.2
        assert torch.allclose(sdrs.float().mean(0), torch.full((20,), 0.2), atol=0.01)

    def test_overlap_union_sparsity(self):
        a = torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool)
        b = torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool)
        assert overlap(a, b).item() == 2
        assert union(torch.stack([a, b])).tolist() == [True, True, True, False, True]
        assert sparsity(a).item() == pytest.approx(0.6)

    def test_overlap_broadcasts(self):
        a = random_sdr(30, 5, batch=(4,), generator=gen())
        assert overlap(a, a[0]).shape == (4,)
        assert overlap(a, a[0])[0].item() == 5

    def test_subsample_keeps_k_of_the_active_bits(self):
        a = random_sdr(50, 10, batch=(8,), generator=gen())
        s = subsample(a, 4, generator=gen(1))
        assert s.sum(-1).tolist() == [4] * 8
        assert not (s & ~a).any()
        assert torch.equal(subsample(a, 20, generator=gen()), a)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="w <= n"):
            random_sdr(5, 6)
        with pytest.raises(ValueError, match="k"):
            subsample(torch.zeros(3, dtype=torch.bool), -1)


class TestMatchingMath:
    @pytest.mark.parametrize(("n", "w_x", "w"), [(30, 5, 7), (64, 8, 8), (100, 20, 3)])
    def test_counts_partition_all_sdrs(self, n, w_x, w):
        # Vandermonde: every w-bit SDR has some overlap b with the fixed one
        assert sum(overlap_count(n, w_x, w, b) for b in range(w + 1)) == comb(n, w)

    def test_mean_overlap_is_hypergeometric(self):
        n, w_x, w = 64, 10, 12
        mean = Fraction(sum(b * overlap_count(n, w_x, w, b) for b in range(w + 1)), comb(n, w))
        assert mean == Fraction(w * w_x, n)

    def test_probability_limits_and_monotonicity(self):
        assert match_probability(100, 10, 10, 0) == 1.0
        assert match_probability(100, 10, 10, 11) == 0.0
        values = [match_probability(200, 20, 20, t) for t in range(21)]
        assert all(a >= b for a, b in pairwise(values))

    def test_exact_value_for_a_small_case(self):
        # n=4, w_x=w=2: of the 6 two-bit SDRs, 1 equals the fixed one and 4 share one bit
        assert match_probability(4, 2, 2, 2) == pytest.approx(1 / 6)
        assert match_probability(4, 2, 2, 1) == pytest.approx(5 / 6)

    def test_false_match_rate_matches_simulation(self):
        n, w, theta, trials = 64, 8, 3, 200_000
        fixed = random_sdr(n, w, generator=gen(1))
        samples = random_sdr(n, w, batch=(trials,), generator=gen(2))
        simulated = (overlap(samples, fixed) >= theta).float().mean().item()
        predicted = match_probability(n, w, w, theta)
        standard_error = (predicted * (1 - predicted) / trials) ** 0.5
        assert abs(simulated - predicted) < 5 * standard_error

    def test_sparse_codes_are_robust(self):
        # the paper's point: with n=2048, w=40 and theta=20 a false match is astronomically rare
        assert match_probability(2048, 40, 40, 20) < 1e-25

    def test_expected_union_size_matches_simulation(self):
        n, w, m = 200, 10, 15
        stacks = random_sdr(n, w, batch=(3000, m), generator=gen(3))
        simulated = stacks.any(1).sum(-1).float().mean().item()
        assert simulated == pytest.approx(expected_union_size(n, w, m), rel=0.01)

    def test_union_false_match_rate_matches_simulation(self):
        n, w, m, theta, trials = 256, 10, 12, 5, 20_000
        stored = random_sdr(n, w, batch=(trials, m), generator=gen(4)).any(1)
        probes = random_sdr(n, w, batch=(trials,), generator=gen(5))
        simulated = (overlap(probes, stored) >= theta).float().mean().item()
        predicted = union_match_probability(n, w, m, theta)
        assert simulated == pytest.approx(predicted, rel=0.15)

    def test_union_capacity_decreases_with_more_members(self):
        rates = [union_match_probability(1024, 20, m, 10) for m in (5, 20, 80)]
        assert rates[0] < rates[1] < rates[2]
