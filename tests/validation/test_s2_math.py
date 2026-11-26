"""S2 radial-basis response and prototype sampling against hand-computed values."""

import math

import pytest
import torch

from neurosush.hmax.s2 import s2_rbf, sample_c1_patches


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestS2Rbf:
    def test_identical_patch_gives_one(self):
        c1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out = s2_rbf(c1, c1.clone(), beta=1.0)
        assert out.shape == (1, 1, 1, 1)
        assert out.item() == pytest.approx(1.0, abs=1e-9)

    def test_zero_prototype(self):
        c1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
        p = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        out = s2_rbf(c1, p, beta=0.1)
        # ||X||^2 = 1 + 4 + 9 + 16 = 30, so r = exp(-0.1 * 30) = exp(-3)
        assert out.item() == pytest.approx(math.exp(-3.0), abs=1e-9)

    def test_sliding_window(self):
        c1 = torch.tensor([[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]], dtype=torch.float64)
        p = torch.ones(1, 1, 2, 2, dtype=torch.float64)
        out = s2_rbf(c1, p, beta=1.0)
        # windows [[1,2],[1,2]] and [[2,3],[2,3]]: d2 = 2 and 10
        assert out.shape == (1, 1, 1, 2)
        assert out[0, 0, 0, 0].item() == pytest.approx(math.exp(-2.0), abs=1e-9)
        assert out[0, 0, 0, 1].item() == pytest.approx(math.exp(-10.0), abs=1e-9)

    def test_matches_brute_force(self):
        g = gen(7)
        c1 = torch.rand(2, 2, 6, 6, generator=g, dtype=torch.float64)
        protos = torch.rand(3, 2, 3, 3, generator=g, dtype=torch.float64)
        beta = 0.7
        out = s2_rbf(c1, protos, beta=beta)
        assert out.shape == (2, 3, 4, 4)
        for b in range(2):
            for p in range(3):
                for i in range(4):
                    for j in range(4):
                        d2 = 0.0
                        for t in range(2):
                            for u in range(3):
                                for v in range(3):
                                    d = c1[b, t, i + u, j + v].item() - protos[p, t, u, v].item()
                                    d2 += d * d
                        expected = math.exp(-beta * d2)
                        assert out[b, p, i, j].item() == pytest.approx(expected, abs=1e-9)

    def test_cancellation_guard(self):
        g = gen(11)
        c1 = (1000.0 + 1e-3 * torch.rand(1, 1, 6, 6, generator=g)).float()
        p = c1[:, :, 0:3, 0:3].clone()
        out = s2_rbf(c1, p)
        assert out[0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert torch.isfinite(out).all()
        assert (out <= 1.0).all()

    def test_dtype_preserved(self):
        c1 = torch.rand(1, 1, 4, 4, dtype=torch.float32)
        p = torch.rand(1, 1, 2, 2, dtype=torch.float32)
        assert s2_rbf(c1, p).dtype == torch.float32
        assert s2_rbf(c1.double(), p.double()).dtype == torch.float64

    def test_prototype_channels_independent(self):
        c1 = torch.rand(1, 2, 5, 5, dtype=torch.float64)
        p0 = torch.rand(1, 2, 3, 3, dtype=torch.float64)
        p1 = torch.rand(1, 2, 3, 3, dtype=torch.float64)
        a = s2_rbf(c1, torch.cat([p0, p1]))
        b = s2_rbf(c1, torch.cat([p1, p0]))
        assert torch.allclose(a[:, 0], b[:, 1], atol=1e-12)
        assert torch.allclose(a[:, 1], b[:, 0], atol=1e-12)

    def test_does_not_mutate_inputs(self):
        c1 = torch.rand(1, 2, 5, 5, dtype=torch.float64)
        p = torch.rand(2, 2, 3, 3, dtype=torch.float64)
        c1_ref, p_ref = c1.clone(), p.clone()
        s2_rbf(c1, p)
        assert torch.equal(c1, c1_ref)
        assert torch.equal(p, p_ref)

    @pytest.mark.parametrize("beta", [0.0, -1.0])
    def test_beta_must_be_positive(self, beta):
        c1 = torch.rand(1, 1, 4, 4)
        p = torch.rand(1, 1, 2, 2)
        with pytest.raises(ValueError, match=str(beta)):
            s2_rbf(c1, p, beta=beta)

    def test_c1_must_be_4d(self):
        with pytest.raises(ValueError, match=r"\(1, 1, 4\)"):
            s2_rbf(torch.rand(1, 1, 4), torch.rand(1, 1, 2, 2))

    def test_prototypes_must_be_4d(self):
        with pytest.raises(ValueError, match=r"\(1, 2, 2\)"):
            s2_rbf(torch.rand(1, 1, 4, 4), torch.rand(1, 2, 2))

    def test_t_mismatch(self):
        with pytest.raises(ValueError, match="2 != 3"):
            s2_rbf(torch.rand(1, 2, 4, 4), torch.rand(1, 3, 2, 2))

    def test_prototype_must_be_square(self):
        with pytest.raises(ValueError, match="2 x 3"):
            s2_rbf(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 2, 3))

    def test_image_smaller_than_prototype(self):
        with pytest.raises(ValueError, match="smaller than prototype 5 x 5"):
            s2_rbf(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 5, 5))

    def test_rejects_integer_c1(self):
        # integer C1 must be rejected, not silently cast (int 0 vs proto 1 gives RBF 0, not exp(-1))
        c1 = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
        p = torch.ones(1, 1, 1, 1, dtype=torch.int64)
        with pytest.raises(ValueError, match="int64"):
            s2_rbf(c1, p)

    def test_rejects_integer_prototypes(self):
        with pytest.raises(ValueError, match="int64"):
            s2_rbf(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1, dtype=torch.int64))

    def test_float64_large_magnitude_limitation(self):
        # around 1e8 the float64 expansion loses the distance: C1=1e8, P=1e8+1 -> 1.0, not exp(-1)
        c1 = torch.tensor([[[[1e8]]]])
        p = torch.tensor([[[[1e8 + 1]]]])
        out = s2_rbf(c1, p)
        assert out.item() == pytest.approx(1.0, abs=1e-6)


class TestSampleC1Patches:
    def test_same_seed_same_patches(self):
        bands = [torch.rand(2, 8, 8, generator=gen(1)) for _ in range(3)]
        a = sample_c1_patches(bands, 5, 3, gen(42))
        b = sample_c1_patches(bands, 5, 3, gen(42))
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        bands = [torch.rand(2, 8, 8, generator=gen(1)) for _ in range(3)]
        a = sample_c1_patches(bands, 5, 3, gen(1))
        b = sample_c1_patches(bands, 5, 3, gen(2))
        assert not torch.equal(a, b)

    def test_single_band_patch_equals_band(self):
        band = torch.rand(2, 3, 3, generator=gen(3))
        out = sample_c1_patches([band], 4, 3, gen(4))
        assert out.shape == (4, 2, 3, 3)
        assert torch.equal(out, band.unsqueeze(0).expand(4, 2, 3, 3))

    def test_only_eligible_bands_are_used(self):
        small = torch.zeros(2, 2, 2)
        big = torch.ones(2, 6, 6)
        out = sample_c1_patches([small, big], 10, 3, gen(5))
        assert out.shape == (10, 2, 3, 3)
        assert (out == 1.0).all()

    def test_samples_every_valid_row(self):
        # band rows carry the row index; size=1 must be able to draw the last row too
        band = torch.arange(4.0).view(1, 4, 1)
        out = sample_c1_patches([band], 200, 1, gen(0))
        assert out.shape == (200, 1, 1, 1)
        assert sorted(out[:, 0, 0, 0].unique().tolist()) == [0.0, 1.0, 2.0, 3.0]

    def test_samples_every_valid_column(self):
        band = torch.arange(4.0).view(1, 1, 4)
        out = sample_c1_patches([band], 200, 1, gen(0))
        assert out.shape == (200, 1, 1, 1)
        assert sorted(out[:, 0, 0, 0].unique().tolist()) == [0.0, 1.0, 2.0, 3.0]

    def test_does_not_mutate_bands(self):
        bands = [torch.rand(2, 5, 5, generator=gen(6)) for _ in range(2)]
        refs = [b.clone() for b in bands]
        out = sample_c1_patches(bands, 3, 2, gen(7))
        out += 1.0
        for b, ref in zip(bands, refs, strict=True):
            assert torch.equal(b, ref)

    def test_n_patches_must_be_positive(self):
        with pytest.raises(ValueError, match="0"):
            sample_c1_patches([torch.rand(1, 4, 4)], 0, 2, gen(0))

    def test_size_must_be_positive(self):
        with pytest.raises(ValueError, match="0"):
            sample_c1_patches([torch.rand(1, 4, 4)], 1, 0, gen(0))

    def test_bands_must_be_3d(self):
        with pytest.raises(ValueError, match=r"\(1, 4\)"):
            sample_c1_patches([torch.rand(1, 4)], 1, 2, gen(0))

    def test_band_t_mismatch(self):
        with pytest.raises(ValueError, match="1 != 2"):
            sample_c1_patches([torch.rand(1, 4, 4), torch.rand(2, 4, 4)], 1, 2, gen(0))

    def test_no_band_large_enough(self):
        with pytest.raises(ValueError, match="3"):
            sample_c1_patches([torch.rand(1, 2, 2)], 1, 3, gen(0))

    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            sample_c1_patches([], 1, 2, gen(0))
