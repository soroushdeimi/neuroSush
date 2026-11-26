"""Invariance probe: bars, integer shifts and normalized correlation (Serre et al. 2007, Sec. 2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def bar_image(
    size: int,
    length: int,
    width: int,
    orientation: str,
    center: tuple[int, int],
    value: float = 1.0,
) -> torch.Tensor:
    """A zero image with a single bar of ``value`` centred on ``center``.

    Args:
        size: Side length of the square image.
        length: Bar length in pixels (odd only).
        width: Bar width in pixels (odd only).
        orientation: "h" (length along columns) or "v" (length along rows).
        center: Centre pixel as (row, col); it must lie inside the image.
        value: Pixel value of the bar.

    Returns:
        A (size, size) float tensor.

    Raises:
        ValueError: If length or width is even or below 1, orientation is not "h" or "v",
            or the centre is outside the image.
    """
    if length < 1 or length % 2 == 0:
        raise ValueError(f"length must be odd and at least 1, got {length}")
    if width < 1 or width % 2 == 0:
        raise ValueError(f"width must be odd and at least 1, got {width}")
    if orientation not in ("h", "v"):
        raise ValueError(f"orientation must be 'h' or 'v', got {orientation!r}")
    row, col = center
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"centre {center} is outside a {size}x{size} image")

    img = torch.zeros(size, size)
    if orientation == "h":
        r0, c0 = row - width // 2, col - length // 2
        img[r0 : r0 + width, c0 : c0 + length] = value
    else:
        r0, c0 = row - length // 2, col - width // 2
        img[r0 : r0 + length, c0 : c0 + width] = value
    return img


def shift_image(img: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift the last two dims by integer pixels with zero fill (not a roll).

    Args:
        img: Tensor with at least two dims.
        dy: Row shift; positive moves content down.
        dx: Column shift; positive moves content right.

    Returns:
        A new tensor of the same shape.

    Raises:
        ValueError: If |dy| >= H or |dx| >= W.
    """
    h, w = img.shape[-2], img.shape[-1]
    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError(f"shift (dy={dy}, dx={dx}) must be smaller than the image ({h}x{w})")
    out = torch.zeros_like(img)
    if dy >= 0:
        dst_r, src_r = slice(dy, None), slice(0, h - dy)
    else:
        dst_r, src_r = slice(0, h + dy), slice(-dy, None)
    if dx >= 0:
        dst_c, src_c = slice(dx, None), slice(0, w - dx)
    else:
        dst_c, src_c = slice(0, w + dx), slice(-dx, None)
    out[..., dst_r, dst_c] = img[..., src_r, src_c]
    return out


def normalized_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of ``a`` and ``b`` over all elements.

    Args:
        a: First tensor.
        b: Second tensor, same shape as ``a``.

    Returns:
        A 0-dim tensor with the correlation in [-1, 1].

    Raises:
        ValueError: If the shapes differ or either input has zero variance.
    """
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm = a_c.norm() * b_c.norm()
    if norm == 0:
        raise ValueError("inputs must have nonzero variance")
    result: torch.Tensor = (a_c * b_c).sum() / norm
    return result


def invariance_curve(
    fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    shifts: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Normalized correlation of ``fn(image)`` with ``fn`` of each shifted image.

    Args:
        fn: Maps a tensor to a tensor (e.g. an S1 response).
        image: The image to probe.
        shifts: (dy, dx) pairs to test.

    Returns:
        A tensor of shape (len(shifts),); entry 0 for (0, 0) is 1.0.
    """
    base = fn(image)
    return torch.stack(
        [normalized_correlation(base, fn(shift_image(image, dy, dx))) for dy, dx in shifts]
    )
