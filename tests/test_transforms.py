import pytest
import torch

from neurosush.transforms import (
    FilterBank,
    GridCrop,
    GridErase,
    GridKeep,
    grid_boxes,
    split_polarity,
)


def image(channels=1, height=4, width=4):
    return torch.arange(1, channels * height * width + 1, dtype=torch.float32).reshape(
        channels, height, width
    )


class TestGridBoxes:
    def test_even_split_row_major(self):
        assert grid_boxes(4, 6, 2, 3) == [
            (0, 0, 2, 2),
            (0, 2, 2, 4),
            (0, 4, 2, 6),
            (2, 0, 4, 2),
            (2, 2, 4, 4),
            (2, 4, 4, 6),
        ]

    def test_uneven_split_clips_the_last_cell(self):
        assert grid_boxes(5, 5, 2, 2) == [(0, 0, 3, 3), (0, 3, 3, 5), (3, 0, 5, 3), (3, 3, 5, 5)]

    def test_gap_is_top_bottom_left_right(self):
        assert grid_boxes(4, 4, 1, 1, gap=(1, 0, 0, 2)) == [(1, 0, 4, 2)]

    def test_gap_that_empties_a_cell(self):
        with pytest.raises(ValueError, match="gap"):
            grid_boxes(4, 4, 2, 2, gap=(1, 1, 0, 0))

    @pytest.mark.parametrize(("rows", "cols"), [(0, 1), (1, 0), (5, 1)])
    def test_invalid_grid(self, rows, cols):
        with pytest.raises(ValueError, match="grid"):
            grid_boxes(4, 4, rows, cols)

    def test_negative_gap(self):
        with pytest.raises(ValueError, match="gap"):
            grid_boxes(4, 4, 1, 1, gap=(-1, 0, 0, 0))


class TestGridErase:
    def test_each_output_erases_one_cell(self):
        images, location = GridErase(2, 2)(image())
        assert images.shape == (4, 1, 4, 4)
        assert location.shape == (4, 2, 2)
        assert images[0, 0, :2, :2].eq(0).all()
        assert torch.equal(images[0, 0, 2:, :], image()[0, 2:, :])
        assert images[3, 0, 2:, 2:].eq(0).all()
        assert location[0].tolist() == [[False, True], [True, True]]
        assert location[3].tolist() == [[True, True], [True, False]]

    def test_input_is_not_modified(self):
        img = image()
        GridErase(2, 2)(img)
        assert torch.equal(img, image())


class TestGridKeep:
    def test_each_output_keeps_one_cell(self):
        images, location = GridKeep(2, 2)(image(channels=2))
        assert images.shape == (4, 2, 4, 4)
        assert torch.equal(images[1, :, :2, 2:], image(channels=2)[:, :2, 2:])
        assert images[1].sum() == images[1, :, :2, 2:].sum()
        assert location[1].tolist() == [[False, True], [False, False]]

    def test_gap_shrinks_the_kept_region(self):
        images, _ = GridKeep(1, 1, gap=(1, 1, 1, 1))(image())
        assert torch.equal(images[0, 0, 1:3, 1:3], image()[0, 1:3, 1:3])
        assert images[0, 0, 0].eq(0).all()


class TestGridCrop:
    def test_crops_every_cell(self):
        crops, location = GridCrop(2, 2)(image())
        assert crops.shape == (4, 1, 2, 2)
        assert torch.equal(crops[2, 0], image()[0, 2:, :2])
        assert location[2].tolist() == [[False, False], [True, False]]

    def test_uneven_grid_is_rejected(self):
        with pytest.raises(ValueError, match="divide"):
            GridCrop(3, 1)(image())


class TestShuffle:
    def test_shuffle_permutes_images_and_locations_together(self):
        images, location = GridKeep(2, 2, shuffle=True, generator=torch.Generator().manual_seed(0))(
            image()
        )
        plain, plain_location = GridKeep(2, 2)(image())
        order = [int(plain_location.flatten(1).float().argmax(1)[i]) for i in range(4)]
        assert order == [0, 1, 2, 3]
        for k in range(4):
            cell = int(location[k].flatten().float().argmax())
            assert torch.equal(images[k], plain[cell])


class TestSplitPolarity:
    def test_positive_then_negative_part(self):
        x = torch.tensor([[1.0, -2.0], [0.0, 3.0]])
        out = split_polarity(x)
        assert out.tolist() == [[1.0, 0.0], [0.0, 3.0], [0.0, 2.0], [0.0, 0.0]]

    def test_dim(self):
        x = torch.tensor([[1.0, -2.0]])
        assert split_polarity(x, dim=1).tolist() == [[1.0, 0.0, 0.0, 2.0]]


class TestFilterBank:
    def test_convolves_an_unbatched_image(self):
        bank = FilterBank(torch.ones(2, 1, 2, 2))
        out = bank(image())
        assert out.shape == (2, 3, 3)
        assert out[0, 0, 0].item() == 1 + 2 + 5 + 6

    def test_stride_and_padding(self):
        out = FilterBank(torch.ones(1, 1, 2, 2), stride=2, padding=1)(image())
        assert out.shape == (1, 3, 3)
        assert out[0, 0, 0].item() == 1.0

    def test_filters_must_be_4d(self):
        with pytest.raises(ValueError, match="filters"):
            FilterBank(torch.ones(2, 2, 2))
