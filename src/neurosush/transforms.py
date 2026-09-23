"""Grid image transforms, polarity splitting, and convolutional filter banks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_boxes(
    height: int,
    width: int,
    rows: int,
    cols: int,
    gap: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> list[tuple[int, int, int, int]]:
    """Return row-major grid boxes with exclusive bottom and right edges.

    Args:
        height: Image height.
        width: Image width.
        rows: Number of grid rows.
        cols: Number of grid columns.
        gap: Insets in top, bottom, left, right order.
    """
    if not (1 <= rows <= height and 1 <= cols <= width):
        raise ValueError("grid rows and cols must fit within the image dimensions")
    if len(gap) != 4 or any(value < 0 for value in gap):
        raise ValueError("gap must contain four nonnegative insets")
    gap_top, gap_bottom, gap_left, gap_right = gap
    cell_h = (height + rows - 1) // rows
    cell_w = (width + cols - 1) // cols
    boxes = []
    for i in range(rows):
        for j in range(cols):
            top = i * cell_h + gap_top
            left = j * cell_w + gap_left
            bottom = min((i + 1) * cell_h, height) - gap_bottom
            right = min((j + 1) * cell_w, width) - gap_right
            if bottom <= top or right <= left:
                raise ValueError("gap leaves an empty grid cell")
            boxes.append((top, left, bottom, right))
    return boxes


class _Grid:
    """Share grid traversal, location masks, and paired shuffling."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        gap: tuple[int, int, int, int] = (0, 0, 0, 0),
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("grid rows and cols must be positive")
        self.rows = rows
        self.cols = cols
        self.gap = gap
        self.shuffle = shuffle
        self.generator = generator

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if img.ndim != 3:
            raise ValueError("img must have shape (channels, height, width)")
        _, height, width = img.shape
        boxes = grid_boxes(height, width, self.rows, self.cols, self.gap)
        outputs = torch.stack([self._apply(img, box) for box in boxes])
        cells = len(boxes)
        location = self._location(cells)
        if self.shuffle:
            order = torch.randperm(cells, generator=self.generator)
            outputs = outputs[order]
            location = location[order]
        return outputs, location

    def _apply(self, img: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
        raise NotImplementedError

    def _location(self, cells: int) -> torch.Tensor:
        return torch.eye(cells, dtype=torch.bool).view(cells, self.rows, self.cols)


class GridErase(_Grid):
    """Erase one grid cell per output and mark the remaining cells as present."""

    def _apply(self, img: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
        top, left, bottom, right = box
        output = img.clone()
        output[..., top:bottom, left:right] = 0
        return output

    def _location(self, cells: int) -> torch.Tensor:
        return ~super()._location(cells)


class GridKeep(_Grid):
    """Keep one grid cell per output and mark that cell as present."""

    def _apply(self, img: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
        top, left, bottom, right = box
        output = torch.zeros_like(img)
        output[..., top:bottom, left:right] = img[..., top:bottom, left:right]
        return output


class GridCrop(_Grid):
    """Copy each cell of an evenly divisible grid into a separate crop."""

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return equal-sized crops and their grid locations."""
        height, width = img.shape[-2:]
        if height % self.rows or width % self.cols:
            raise ValueError("grid rows and cols must divide the image dimensions")
        return super().__call__(img)

    def _apply(self, img: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
        top, left, bottom, right = box
        return img[..., top:bottom, left:right].clone()


def split_polarity(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Concatenate positive and negative magnitudes along the given dimension."""
    return torch.cat([x.clamp_min(0), (-x).clamp_min(0)], dim=dim)


class FilterBank:
    """Convolve batched or unbatched images with a bank of filters."""

    def __init__(
        self,
        filters: torch.Tensor,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        if filters.ndim != 4:
            raise ValueError("filters must have shape (out, in, kh, kw)")
        self.filters = filters
        self.stride = stride
        self.padding = padding

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the filters with the configured stride and padding."""
        return F.conv2d(img, self.filters, stride=self.stride, padding=self.padding)
