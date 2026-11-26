"""C1 pooling (Serre et al. 2007, Sec. 2): local max over a grid and two adjacent scales."""

from types import SimpleNamespace

import pytest
import torch

from neurosush.hmax.c1 import c1_bands_from_spec, c1_pool


class TestC1Pool:
    def test_single_band_max_over_two_scales(self):
        a = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        b = torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]])
        out = c1_pool([a, b], [(0, 1, 2, 2)])
        assert out[0].shape == (1, 1, 1, 1)
        assert torch.allclose(out[0], torch.tensor([[[[2.0]]]]))

    def test_grid2_stride1_on_arange(self):
        x = torch.arange(16.0).reshape(1, 1, 4, 4)
        out = c1_pool([x, x], [(0, 1, 2, 1)])
        expected = torch.tensor([[[[5.0, 6.0, 7.0], [9.0, 10.0, 11.0], [13.0, 14.0, 15.0]]]])
        assert out[0].shape == (1, 1, 3, 3)
        assert torch.allclose(out[0], expected)

    def test_orientations_are_not_mixed(self):
        # channel 0: one pixel 9, rest 0; channel 1: all 3
        x = torch.tensor([[[[9.0, 0.0], [0.0, 0.0]], [[3.0, 3.0], [3.0, 3.0]]]])
        out = c1_pool([x, x], [(0, 0, 2, 2)])
        assert out[0].shape == (1, 2, 1, 1)
        assert out[0][0, 0, 0, 0].item() == pytest.approx(9.0, abs=1e-6)
        assert out[0][0, 1, 0, 0].item() == pytest.approx(3.0, abs=1e-6)

    def test_scale_max_is_elementwise(self):
        # a is larger at pixel (0,0), b is larger at pixel (0,1)
        a = torch.tensor([[[[10.0, 1.0]]]])
        b = torch.tensor([[[[1.0, 10.0]]]])
        out = c1_pool([a, b], [(0, 1, 1, 1)])
        assert torch.allclose(out[0], torch.tensor([[[[10.0, 10.0]]]]))

    def test_output_shape_and_multiple_bands(self):
        x = torch.randn(2, 4, 20, 20)
        out = c1_pool([x, x], [(0, 1, 8, 4)])
        assert out[0].shape == (2, 4, 4, 4)
        out2 = c1_pool([x, x], [(0, 1, 8, 4), (0, 1, 4, 2)])
        assert len(out2) == 2
        assert out2[0].shape == (2, 4, 4, 4)
        assert out2[1].shape == (2, 4, 9, 9)

    def test_does_not_mutate_inputs(self):
        a = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        b = torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]])
        a_copy, b_copy = a.clone(), b.clone()
        c1_pool([a, b], [(0, 1, 2, 2)])
        assert torch.equal(a, a_copy)
        assert torch.equal(b, b_copy)


class TestC1BandsFromSpec:
    def test_pairs_adjacent_scales(self):
        specs = [SimpleNamespace(c1_grid=8, c1_stride=4), SimpleNamespace(c1_grid=10, c1_stride=5)]
        assert c1_bands_from_spec(specs) == [(0, 1, 8, 4), (2, 3, 10, 5)]


class TestC1PoolFailures:
    def test_empty_s1(self):
        with pytest.raises(ValueError, match="s1 must not be empty"):
            c1_pool([], [(0, 1, 2, 2)])

    def test_empty_bands(self):
        with pytest.raises(ValueError, match="bands must not be empty"):
            c1_pool([torch.zeros(1, 1, 2, 2)], [])

    def test_mismatched_scale_shapes(self):
        with pytest.raises(ValueError, match="shapes differ"):
            c1_pool([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 3)], [(0, 1, 2, 2)])

    def test_grid_larger_than_input(self):
        with pytest.raises(ValueError, match="grid 5 larger than"):
            c1_pool([torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)], [(0, 1, 5, 1)])

    def test_3d_input(self):
        with pytest.raises(ValueError, match="4-D"):
            c1_pool([torch.zeros(1, 2, 2)], [(0, 0, 1, 1)])

    def test_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            c1_pool([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)], [(7, 1, 2, 2)])

    def test_negative_scale_a_rejected(self):
        # a negative index must not silently wrap to the last scale
        with pytest.raises(ValueError, match="out of range"):
            c1_pool([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)], [(-1, 1, 1, 1)])

    def test_negative_scale_b_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            c1_pool([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)], [(0, -1, 1, 1)])

    def test_grid_zero(self):
        with pytest.raises(ValueError, match="grid must be at least 1"):
            c1_pool([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)], [(0, 1, 0, 1)])

    def test_stride_zero(self):
        with pytest.raises(ValueError, match="stride must be at least 1"):
            c1_pool([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)], [(0, 1, 2, 0)])
