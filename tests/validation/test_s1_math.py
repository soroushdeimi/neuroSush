"""S1 responses: cross-correlation convention, normalization, shapes and validation."""

import pytest
import torch

from neurosush.hmax.s1 import s1_response


def _zero_mean_kernel(size: int, t: int) -> torch.Tensor:
    """A zero-mean (T, 1, size, size) kernel: a column of ones and a column of minus ones."""
    k = torch.zeros(t, 1, size, size)
    k[:, :, :, 0] = 1.0
    k[:, :, :, 1] = -1.0
    return k / size


class TestS1Response:
    def test_correlation_convention(self):
        # a delta at the image centre returns the kernel mirrored: cross-correlation, no flip
        images = torch.zeros(1, 1, 3, 3)
        images[0, 0, 1, 1] = 1.0
        kernel = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        out = s1_response(images, [kernel], padding="same")
        expected = torch.tensor([[[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]]])
        assert torch.allclose(out[0], expected)

    def test_sign_invariance(self):
        images = torch.rand(1, 1, 5, 5, generator=torch.Generator().manual_seed(0))
        kernel = torch.tensor([[[[1.0, -2.0, 0.5], [-3.0, 4.0, -1.0], [0.25, -0.5, 2.0]]]])
        out_k = s1_response(images, [kernel], padding="same")
        out_neg = s1_response(images, [-kernel], padding="same")
        assert torch.allclose(out_k[0], out_neg[0])

    def test_normalize_divides_by_local_l2_norm(self):
        # raw conv of 2.0 over 9 pixels with ones/3 is 6.0; the local L2 norm is sqrt(9 * 4) = 6
        images = torch.full((1, 1, 3, 3), 2.0)
        kernel = torch.ones(1, 1, 3, 3) / 3
        raw = s1_response(images, [kernel], padding="valid", normalize=False)
        assert raw[0].shape == (1, 1, 1, 1)
        assert raw[0].item() == pytest.approx(6.0, abs=1e-6)
        norm = s1_response(images, [kernel], padding="valid", normalize=True)
        assert norm[0].shape == (1, 1, 1, 1)
        assert norm[0].item() == pytest.approx(1.0, abs=1e-6)

    def test_normalize_uses_eps(self):
        # raw conv 6.0, local norm sqrt(36 + 64) = 10.0, so the response is 0.6
        images = torch.full((1, 1, 3, 3), 2.0, dtype=torch.float64)
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float64) / 3
        out = s1_response(images, [kernel], padding="valid", normalize=True, eps=64.0)
        assert out[0].item() == pytest.approx(0.6, abs=1e-6)

    def test_normalize_float16_zero_image(self):
        # the default eps 1e-8 underflows float16; the normalisation must stay finite
        images = torch.zeros(1, 1, 3, 3, dtype=torch.float16)
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float16)
        out = s1_response(images, [kernel], padding="valid", normalize=True)
        assert out[0].dtype == torch.float16
        assert torch.isfinite(out[0]).all()
        assert torch.all(out[0] == 0.0)

    def test_normalize_float16_constant_image(self):
        # raw conv of 2.0 over 9 pixels with ones/3 is 6.0; the local L2 norm is 6.0
        images = torch.full((1, 1, 3, 3), 2.0, dtype=torch.float16)
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float16) / 3
        out = s1_response(images, [kernel], padding="valid", normalize=True)
        assert out[0].dtype == torch.float16
        assert out[0].item() == pytest.approx(1.0, abs=1e-2)

    def test_normalize_uses_each_batch_entry(self):
        # both constant images must normalise to 1.0 with their own local norm
        images = torch.stack([torch.full((1, 3, 3), 1.0), torch.full((1, 3, 3), 2.0)])
        kernel = torch.ones(1, 1, 3, 3) / 3
        out = s1_response(images, [kernel], padding="valid", normalize=True)
        assert out[0].shape == (2, 1, 1, 1)
        assert out[0][0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert out[0][1, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_zero_mean_kernel_on_constant_image(self):
        images = torch.full((1, 1, 6, 6), 5.0)
        kernel = torch.tensor([[[[1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]]]) / 2
        out = s1_response(images, [kernel], padding="valid")
        assert out[0].shape == (1, 1, 4, 4)
        assert torch.allclose(out[0], torch.zeros(1, 1, 4, 4))

    def test_output_shapes(self):
        images = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(1))
        bank = [_zero_mean_kernel(3, 2), _zero_mean_kernel(5, 2)]
        same = s1_response(images, bank, padding="same")
        assert [t.shape for t in same] == [(2, 2, 8, 8), (2, 2, 8, 8)]
        valid = s1_response(images, bank, padding="valid")
        assert [t.shape for t in valid] == [(2, 2, 6, 6), (2, 2, 4, 4)]

    def test_does_not_mutate_input(self):
        images = torch.rand(1, 1, 6, 6, generator=torch.Generator().manual_seed(2))
        before = images.clone()
        s1_response(images, [_zero_mean_kernel(3, 1)], padding="same", normalize=True)
        assert torch.equal(images, before)

    def test_rejects_3d_images(self):
        with pytest.raises(ValueError, match="4-D"):
            s1_response(torch.zeros(1, 3, 3), [_zero_mean_kernel(3, 1)])

    def test_rejects_two_channels(self):
        with pytest.raises(ValueError, match="1 channel"):
            s1_response(torch.zeros(1, 2, 4, 4), [_zero_mean_kernel(3, 1)])

    def test_rejects_empty_bank(self):
        with pytest.raises(ValueError, match="empty"):
            s1_response(torch.zeros(1, 1, 4, 4), [])

    def test_rejects_bad_kernel_shape(self):
        with pytest.raises(ValueError, match="4-D"):
            s1_response(torch.zeros(1, 1, 4, 4), [torch.zeros(3, 3)])

    def test_rejects_even_kernel_with_same(self):
        with pytest.raises(ValueError, match="odd"):
            s1_response(torch.zeros(1, 1, 4, 4), [torch.zeros(1, 1, 4, 4)], padding="same")

    def test_rejects_unknown_padding(self):
        with pytest.raises(ValueError, match="padding"):
            s1_response(torch.zeros(1, 1, 4, 4), [_zero_mean_kernel(3, 1)], padding="full")

    def test_rejects_nonpositive_eps(self):
        with pytest.raises(ValueError, match="eps"):
            s1_response(torch.zeros(1, 1, 4, 4), [_zero_mean_kernel(3, 1)], normalize=True, eps=0)

    def test_rejects_kernel_larger_than_image_with_valid(self):
        with pytest.raises(ValueError, match="larger"):
            s1_response(torch.zeros(1, 1, 4, 4), [_zero_mean_kernel(5, 1)], padding="valid")

    def test_rejects_integer_images(self):
        with pytest.raises(ValueError, match="int64"):
            s1_response(torch.zeros(1, 1, 4, 4, dtype=torch.int64), [_zero_mean_kernel(3, 1)])

    def test_rejects_integer_kernel(self):
        with pytest.raises(ValueError, match="int64"):
            s1_response(torch.zeros(1, 1, 4, 4), [torch.zeros(1, 1, 3, 3, dtype=torch.int64)])

    def test_rejects_non_square_kernel(self):
        with pytest.raises(ValueError, match="3, 1"):
            s1_response(torch.zeros(1, 1, 5, 5), [torch.zeros(1, 1, 3, 1)])
