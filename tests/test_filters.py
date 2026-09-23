import math

import pytest
import torch

from neurosush.filters import dog_kernel, gabor_kernel


class TestDoG:
    def test_shape_and_dtype(self):
        k = dog_kernel(5, 1.0, 2.0, dtype=torch.float64)
        assert k.shape == (5, 5)
        assert k.dtype == torch.float64

    def test_center_value(self):
        k = dog_kernel(3, 1.0, 2.0, dtype=torch.float64)
        expected = (1 / 1.0 - 1 / 2.0) / math.sqrt(2 * math.pi)
        assert k[1, 1].item() == pytest.approx(expected)

    def test_off_center_value(self):
        k = dog_kernel(3, 1.0, 2.0, dtype=torch.float64)
        # point (0, 1): squared distance 1 from the center
        expected = (math.exp(-0.5) / 1.0 - math.exp(-0.5 / 4) / 2.0) / math.sqrt(2 * math.pi)
        assert k[0, 1].item() == pytest.approx(expected)

    def test_radially_symmetric(self):
        k = dog_kernel(7, 1.0, 3.0)
        assert torch.allclose(k, k.T)
        assert torch.allclose(k, k.flip(0))
        assert torch.allclose(k, k.flip(1))

    def test_even_size_is_centered_between_pixels(self):
        k = dog_kernel(4, 1.0, 2.0)
        assert k.shape == (4, 4)
        assert torch.allclose(k, k.flip(0))

    def test_spacing_scales_coordinates(self):
        coarse = dog_kernel(3, 1.0, 2.0, spacing=2.0, dtype=torch.float64)
        wide = dog_kernel(5, 1.0, 2.0, dtype=torch.float64)
        assert torch.allclose(coarse, wide[::2, ::2])

    def test_zero_mean(self):
        k = dog_kernel(7, 1.0, 2.0, zero_mean=True, dtype=torch.float64)
        assert k.sum().item() == pytest.approx(0.0, abs=1e-12)

    def test_unit_l1(self):
        k = dog_kernel(7, 1.0, 2.0, unit_l1=True, dtype=torch.float64)
        assert k.abs().sum().item() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("args", "match"),
        [((0, 1.0, 2.0), "size"), ((3, 0.0, 2.0), "sigma"), ((3, 1.0, -1.0), "sigma")],
    )
    def test_invalid_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            dog_kernel(*args)

    def test_invalid_spacing(self):
        with pytest.raises(ValueError, match="spacing"):
            dog_kernel(3, 1.0, 2.0, spacing=0.0)


class TestGabor:
    def test_center_is_one(self):
        k = gabor_kernel(5, wavelength=4.0, theta=0.3, sigma=2.0, gamma=0.5)
        assert k.shape == (5, 5)
        assert k[2, 2].item() == pytest.approx(1.0)

    def test_value_along_first_axis(self):
        k = gabor_kernel(5, wavelength=4.0, theta=0.0, sigma=2.0, gamma=0.5, dtype=torch.float64)
        # row offset 1 from the center: x' = 1, y' = 0
        expected = math.exp(-1 / 8) * math.cos(2 * math.pi / 4)
        assert k[3, 2].item() == pytest.approx(expected, abs=1e-12)
        # column offset 1: x' = 0, y' = 1, gamma scales y'
        expected = math.exp(-(0.25) / 8) * math.cos(0.0)
        assert k[2, 3].item() == pytest.approx(expected)

    def test_quarter_turn_transposes(self):
        a = gabor_kernel(7, wavelength=3.0, theta=0.0, sigma=2.0, gamma=0.7, dtype=torch.float64)
        b = gabor_kernel(
            7, wavelength=3.0, theta=math.pi / 2, sigma=2.0, gamma=0.7, dtype=torch.float64
        )
        assert torch.allclose(a, b.T, atol=1e-12)

    def test_zero_mean_and_unit_l1(self):
        k = gabor_kernel(
            9, wavelength=4.0, theta=0.5, sigma=2.0, gamma=0.5, zero_mean=True, unit_l1=True
        )
        assert k.sum().item() == pytest.approx(0.0, abs=1e-6)
        assert k.abs().sum().item() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"wavelength": 0.0}, "wavelength"),
            ({"sigma": 0.0}, "sigma"),
            ({"gamma": 0.0}, "gamma"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        params = {"wavelength": 4.0, "theta": 0.0, "sigma": 2.0, "gamma": 0.5, **kwargs}
        with pytest.raises(ValueError, match=match):
            gabor_kernel(5, **params)
