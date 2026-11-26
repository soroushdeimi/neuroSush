"""Serre 2007 Gabor bank against hand-computed kernels and the published band table."""

import math

import pytest
import torch

from neurosush.hmax.gabor import (
    SERRE2007_BANDS,
    GaborBandSpec,
    gabor_bank,
    serre2007_gabor_bank,
)


class TestExactKernel:
    def test_3x3_hand_computed(self):
        # sigma = 1/sqrt(2 ln 2) makes the envelope 2^-(x0^2 + y0^2); with wavelength 4 the
        # carrier vanishes on the grid except at the centre: [[0,0,0],[.5,1,.5],[0,0,0]]
        sigma = 1.0 / math.sqrt(2.0 * math.log(2.0))
        bank = gabor_bank([3], [4.0], [sigma], [0.0], gamma=1.0, dtype=torch.float64)
        expected = (
            torch.tensor([[-4.0, -4.0, -4.0], [5.0, 14.0, 5.0], [-4.0, -4.0, -4.0]])
            / math.sqrt(342.0)
        ).double()
        assert torch.allclose(bank[0][0, 0], expected, atol=1e-6)


class TestSerreBank:
    def test_zero_mean_unit_norm(self):
        for tensor in serre2007_gabor_bank():
            for orientation in tensor:
                for filter_ in orientation:
                    assert abs(filter_.sum().item()) < 1e-5
                    assert abs((filter_**2).sum().item() - 1.0) < 1e-5

    def test_shapes(self):
        bank = serre2007_gabor_bank()
        assert len(bank) == 16
        for i, tensor in enumerate(bank):
            assert tensor.shape == (4, 1, 7 + 2 * i, 7 + 2 * i)

    def test_band_table(self):
        assert len(SERRE2007_BANDS) == 8
        assert SERRE2007_BANDS[0] == GaborBandSpec(1, (7, 9), (2.8, 3.6), (3.5, 4.6), 8, 4)
        assert SERRE2007_BANDS[3] == GaborBandSpec(4, (19, 21), (8.2, 9.2), (10.3, 11.5), 14, 7)
        assert SERRE2007_BANDS[7] == GaborBandSpec(8, (35, 37), (17.0, 18.2), (21.2, 22.8), 22, 11)
        for spec in SERRE2007_BANDS:
            assert spec.c1_stride == spec.c1_grid // 2

    def test_all_bands_match_table_1(self):
        expected = (
            GaborBandSpec(1, (7, 9), (2.8, 3.6), (3.5, 4.6), 8, 4),
            GaborBandSpec(2, (11, 13), (4.5, 5.4), (5.6, 6.8), 10, 5),
            GaborBandSpec(3, (15, 17), (6.3, 7.3), (7.9, 9.1), 12, 6),
            GaborBandSpec(4, (19, 21), (8.2, 9.2), (10.3, 11.5), 14, 7),
            GaborBandSpec(5, (23, 25), (10.2, 11.3), (12.7, 14.1), 16, 8),
            GaborBandSpec(6, (27, 29), (12.3, 13.4), (15.4, 16.8), 18, 9),
            GaborBandSpec(7, (31, 33), (14.6, 15.8), (18.2, 19.7), 20, 10),
            GaborBandSpec(8, (35, 37), (17.0, 18.2), (21.2, 22.8), 22, 11),
        )
        assert expected == SERRE2007_BANDS


class TestOrientation:
    @pytest.mark.parametrize("gamma", [0.3, 1.0])
    def test_quarter_turn_is_transpose(self, gamma):
        # rotating the orientation by pi/2 transposes the kernel (x along rows, y along columns)
        bank = gabor_bank([7], [3.5], [2.8], [0.0, math.pi / 2], gamma=gamma, dtype=torch.float64)
        assert torch.allclose(bank[0][1, 0], bank[0][0, 0].T, atol=1e-6)

    def test_theta_order_preserved(self):
        bank = gabor_bank([7], [3.5], [2.8], [0.0, math.pi / 2], dtype=torch.float64)
        assert torch.allclose(bank[0][1, 0], bank[0][0, 0].T, atol=1e-6)


class TestFailures:
    def test_even_size(self):
        with pytest.raises(ValueError, match="8"):
            gabor_bank([8], [3.5], [2.8], [0.0])

    def test_zero_size(self):
        with pytest.raises(ValueError, match="0"):
            gabor_bank([0], [3.5], [2.8], [0.0])

    def test_zero_wavelength(self):
        with pytest.raises(ValueError, match="0"):
            gabor_bank([7], [0.0], [2.8], [0.0])

    def test_negative_sigma(self):
        with pytest.raises(ValueError, match="-1"):
            gabor_bank([7], [3.5], [-1.0], [0.0])

    def test_zero_gamma(self):
        with pytest.raises(ValueError, match="0"):
            gabor_bank([7], [3.5], [2.8], [0.0], gamma=0.0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            gabor_bank([7, 9], [3.5], [2.8], [0.0])

    def test_empty_thetas(self):
        with pytest.raises(ValueError, match="thetas"):
            gabor_bank([7], [3.5], [2.8], [])

    def test_zero_norm(self):
        # a single-pixel kernel is constant, so mean removal leaves the zero filter
        with pytest.raises(ValueError, match="norm"):
            gabor_bank([1], [3.5], [2.8], [0.0])

    def test_integer_dtype(self):
        with pytest.raises(ValueError, match="int64"):
            gabor_bank([7], [3.5], [2.8], [0.0], dtype=torch.int64)

    def test_integer_dtype_serre_bank(self):
        with pytest.raises(ValueError, match="int64"):
            serre2007_gabor_bank(torch.int64)

    def test_nan_wavelength(self):
        with pytest.raises(ValueError, match="nan"):
            gabor_bank([7], [float("nan")], [2.8], [0.0])

    def test_inf_wavelength(self):
        with pytest.raises(ValueError, match="inf"):
            gabor_bank([7], [float("inf")], [2.8], [0.0])

    def test_nan_sigma(self):
        with pytest.raises(ValueError, match="nan"):
            gabor_bank([7], [3.5], [float("nan")], [0.0])

    def test_inf_gamma(self):
        with pytest.raises(ValueError, match="inf"):
            gabor_bank([7], [3.5], [2.8], [0.0], gamma=float("inf"))

    def test_nan_theta(self):
        with pytest.raises(ValueError, match="nan"):
            gabor_bank([7], [3.5], [2.8], [float("nan")])


class TestDocstring:
    def test_orientation_documented(self):
        # the docstring must state which direction theta=0 prefers
        assert "horizontal" in gabor_bank.__doc__
