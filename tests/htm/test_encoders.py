"""Encoders: exact overlap laws and their validation."""

import pytest
import torch

from neurosush.htm.encoders import CategoryEncoder, RandomDistributedScalarEncoder, ScalarEncoder
from neurosush.htm.sdr import overlap


class TestScalarEncoder:
    def test_exactly_w_contiguous_bits(self):
        enc = ScalarEncoder(n=100, w=11, minimum=0.0, maximum=10.0)
        code = enc.encode(3.0)
        assert code.sum().item() == 11
        active = code.nonzero().flatten()
        assert (active.diff() == 1).all()

    def test_bucket_formula_and_ends(self):
        enc = ScalarEncoder(n=100, w=11, minimum=0.0, maximum=10.0)
        # (x - min) / range * (n - w) = 0.3 * 89 = 26.7 -> bucket 27
        assert enc.bucket(torch.tensor(3.0)).item() == 27
        assert enc.encode(0.0)[:11].all()
        assert enc.encode(10.0)[-11:].all()

    def test_overlap_law(self):
        enc = ScalarEncoder(n=120, w=15, minimum=0.0, maximum=1.0)
        xs = torch.linspace(0, 1, 60, dtype=torch.float64)
        codes = enc.encode(xs)
        buckets = enc.bucket(xs)
        measured = overlap(codes.unsqueeze(1), codes.unsqueeze(0))
        distance = (buckets.unsqueeze(1) - buckets.unsqueeze(0)).abs()
        assert torch.equal(measured, (15 - distance).clamp(min=0))

    def test_periodic_overlap_uses_circular_distance(self):
        enc = ScalarEncoder(n=24, w=5, minimum=0.0, maximum=24.0, periodic=True)
        assert enc.bucket(torch.tensor(25.0)).item() == 1
        # buckets 23 and 1 are 2 apart around the circle
        assert overlap(enc.encode(23.0), enc.encode(1.0)).item() == 3
        assert enc.encode(23.0).sum().item() == 5

    def test_batches(self):
        enc = ScalarEncoder(n=50, w=5, minimum=0.0, maximum=1.0)
        assert enc.encode(torch.rand(3, 4)).shape == (3, 4, 50)

    def test_range_checks(self):
        enc = ScalarEncoder(n=50, w=5, minimum=0.0, maximum=1.0)
        with pytest.raises(ValueError, match="values must lie"):
            enc.encode(1.5)
        clipped = ScalarEncoder(n=50, w=5, minimum=0.0, maximum=1.0, clip=True)
        assert torch.equal(clipped.encode(1.5), clipped.encode(1.0))
        with pytest.raises(ValueError, match="maximum"):
            ScalarEncoder(n=50, w=5, minimum=1.0, maximum=1.0)


class TestRandomDistributedScalarEncoder:
    def test_nearby_buckets_share_w_minus_d_bits(self):
        enc = RandomDistributedScalarEncoder(n=2000, w=21, resolution=0.5, seed=3)
        base = enc.encode(10.0)
        for d in range(21):
            shared = overlap(base, enc.encode(10.0 + 0.5 * d)).item()
            assert shared >= 21 - d - 1  # at most one collision at this sparsity
        assert base.sum().item() >= 20

    def test_far_values_overlap_only_by_chance(self):
        enc = RandomDistributedScalarEncoder(n=2000, w=21, resolution=1.0, seed=0)
        codes = enc.encode(torch.arange(0, 2000, 50, dtype=torch.float64))
        pairwise = overlap(codes.unsqueeze(1), codes.unsqueeze(0))
        off_diagonal = pairwise[~torch.eye(len(codes), dtype=torch.bool)].float()
        # expected chance overlap w**2 / n = 0.22
        assert off_diagonal.mean().item() < 0.6

    def test_negative_values_and_determinism(self):
        a = RandomDistributedScalarEncoder(n=500, w=11, resolution=1.0, seed=7)
        b = RandomDistributedScalarEncoder(n=500, w=11, resolution=1.0, seed=7)
        assert torch.equal(a.encode(-3.2), b.encode(-3.2))
        assert overlap(a.encode(-3.2), a.encode(-2.2)).item() >= 9

    def test_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            RandomDistributedScalarEncoder(n=10, w=3, resolution=0.0)


class TestCategoryEncoder:
    def test_codes_are_fixed_and_distinct(self):
        enc = CategoryEncoder(n=400, w=20, categories=5, seed=1)
        codes = enc.encode(torch.arange(5))
        assert codes.sum(-1).tolist() == [20] * 5
        pairwise = overlap(codes.unsqueeze(1), codes.unsqueeze(0))
        assert (pairwise[~torch.eye(5, dtype=torch.bool)] < 8).all()
        assert torch.equal(CategoryEncoder(n=400, w=20, categories=5, seed=1).encode(3), codes[3])

    def test_unknown_category(self):
        with pytest.raises(ValueError, match="categories"):
            CategoryEncoder(n=40, w=4, categories=3).encode(3)
