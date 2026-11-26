"""C2 pooling (Serre et al. 2007): global max over all scales and positions of each S2 type."""

import pytest
import torch

from neurosush.hmax.c2 import c2_global_max


class TestC2GlobalMax:
    def test_max_across_bands(self):
        band0 = torch.tensor([[[[1.0, 7.0]]]])
        band1 = torch.tensor([[[[9.0, 2.0]]]])
        out = c2_global_max([band0, band1])
        assert out.shape == (1, 1)
        assert torch.allclose(out, torch.tensor([[9.0]]))

    def test_prototypes_stay_independent(self):
        # P=2: band0[:,0]=5, band0[:,1]=1; band1[:,0]=2, band1[:,1]=8
        band0 = torch.tensor([[[[5.0]], [[1.0]]]])
        band1 = torch.tensor([[[[2.0]], [[8.0]]]])
        out = c2_global_max([band0, band1])
        assert out.shape == (1, 2)
        assert torch.allclose(out, torch.tensor([[5.0, 8.0]]))

    def test_different_band_sizes_and_batch(self):
        band0 = torch.full((2, 3, 1, 4), 1.0)
        band0[0, 1, 0, 2] = 4.0
        band1 = torch.full((2, 3, 2, 2), 1.0)
        band1[1, 2, 1, 1] = 6.0
        out = c2_global_max([band0, band1])
        assert out.shape == (2, 3)
        assert torch.allclose(out, torch.tensor([[1.0, 4.0, 1.0], [1.0, 1.0, 6.0]]))

    def test_does_not_mutate_inputs(self):
        band0 = torch.tensor([[[[1.0, 7.0]]]])
        band1 = torch.tensor([[[[9.0, 2.0]]]])
        band0_copy, band1_copy = band0.clone(), band1.clone()
        c2_global_max([band0, band1])
        assert torch.equal(band0, band0_copy)
        assert torch.equal(band1, band1_copy)


class TestC2GlobalMaxFailures:
    def test_empty_list(self):
        with pytest.raises(ValueError, match="s2 must not be empty"):
            c2_global_max([])

    def test_3d_tensor(self):
        with pytest.raises(ValueError, match="4-D"):
            c2_global_max([torch.zeros(1, 1, 2)])

    def test_p_mismatch(self):
        with pytest.raises(ValueError, match="P differs"):
            c2_global_max([torch.zeros(1, 2, 1, 1), torch.zeros(1, 3, 1, 1)])

    def test_b_mismatch(self):
        with pytest.raises(ValueError, match="B differs"):
            c2_global_max([torch.zeros(1, 2, 1, 1), torch.zeros(2, 2, 1, 1)])
