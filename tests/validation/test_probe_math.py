"""Invariance probe: bars, shifts, normalized correlation and the invariance curve."""

import pytest
import torch

from neurosush.hmax.probe import bar_image, invariance_curve, normalized_correlation, shift_image


class TestNormalizedCorrelation:
    def test_known_values(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 3.0, 2.0])
        assert normalized_correlation(a, b).item() == pytest.approx(0.5, abs=1e-6)

    def test_affine_transform_is_one(self):
        a = torch.rand(4, 5, generator=torch.Generator().manual_seed(0))
        assert normalized_correlation(a, 2 * a + 3).item() == pytest.approx(1.0, abs=1e-6)

    def test_negation_is_minus_one(self):
        a = torch.rand(4, 5, generator=torch.Generator().manual_seed(1))
        assert normalized_correlation(a, -a).item() == pytest.approx(-1.0, abs=1e-6)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            normalized_correlation(torch.zeros(3), torch.zeros(4))

    def test_rejects_zero_variance(self):
        with pytest.raises(ValueError, match="variance"):
            normalized_correlation(torch.ones(4), torch.rand(4, generator=torch.Generator()))


class TestShiftImage:
    def test_shift_right(self):
        img = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(shift_image(img, 0, 1), torch.tensor([[0.0, 1.0], [0.0, 3.0]]))

    def test_shift_down(self):
        img = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(shift_image(img, 1, 0), torch.tensor([[0.0, 0.0], [1.0, 2.0]]))

    def test_shift_up(self):
        img = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(shift_image(img, -1, 0), torch.tensor([[3.0, 4.0], [0.0, 0.0]]))

    def test_zero_shift_is_a_copy(self):
        img = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = shift_image(img, 0, 0)
        assert torch.equal(out, img)
        assert out is not img
        assert out.data_ptr() != img.data_ptr()

    def test_does_not_mutate_input(self):
        img = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        before = img.clone()
        shift_image(img, 1, 1)
        assert torch.equal(img, before)

    def test_rejects_shift_of_full_size(self):
        with pytest.raises(ValueError, match="shift"):
            shift_image(torch.zeros(2, 2), 2, 0)


class TestBarImage:
    def test_horizontal_bar(self):
        img = bar_image(5, 3, 1, "h", (2, 2))
        assert img.shape == (5, 5)
        assert img.sum().item() == 3.0
        expected = torch.zeros(5, 5)
        expected[2, 1:4] = 1.0
        assert torch.equal(img, expected)

    def test_vertical_bar(self):
        img = bar_image(5, 3, 1, "v", (2, 2))
        expected = torch.zeros(5, 5)
        expected[1:4, 2] = 1.0
        assert torch.equal(img, expected)

    def test_value(self):
        img = bar_image(5, 3, 1, "h", (2, 2), value=2.0)
        assert img[2, 2].item() == 2.0
        assert img.sum().item() == 6.0

    def test_clipped_by_border(self):
        # a bar whose centre is inside but whose ends run past the border is clipped, no error
        img = bar_image(5, 5, 1, "h", (0, 2))
        assert img.sum().item() == 5.0
        assert torch.equal(img[0], torch.ones(5))

    def test_rejects_even_length(self):
        with pytest.raises(ValueError, match="odd"):
            bar_image(5, 4, 1, "h", (2, 2))

    def test_rejects_even_width(self):
        with pytest.raises(ValueError, match="odd"):
            bar_image(5, 3, 2, "h", (2, 2))

    def test_rejects_bad_orientation(self):
        with pytest.raises(ValueError, match="orientation"):
            bar_image(5, 3, 1, "d", (2, 2))

    def test_rejects_centre_outside(self):
        with pytest.raises(ValueError, match="centre"):
            bar_image(5, 3, 1, "h", (5, 2))


class TestInvarianceCurve:
    def test_identity_curve(self):
        gen = torch.Generator().manual_seed(0)
        img = torch.rand(8, 8, generator=gen)
        curve = invariance_curve(lambda x: x, img, [(0, 0), (0, 1)])
        assert curve.shape == (2,)
        assert curve[0].item() == pytest.approx(1.0, abs=1e-6)
        assert curve[1].item() < 1.0

    def test_does_not_mutate_input(self):
        img = torch.rand(4, 4, generator=torch.Generator().manual_seed(3))
        before = img.clone()
        invariance_curve(lambda x: x, img, [(0, 0)])
        assert torch.equal(img, before)
