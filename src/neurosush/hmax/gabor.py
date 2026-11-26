"""Serre et al. 2007 Gabor filter bank (TPAMI 29(3), Sec. 2, Table 1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaborBandSpec:
    """One row of Serre 2007 Table 1: the two scales of one orientation band.

    Attributes:
        band: Band index, 1..8.
        sizes: Kernel sizes of the two scales.
        sigmas: Gaussian envelope standard deviations of the two scales.
        wavelengths: Carrier wavelengths of the two scales.
        c1_grid: C1 grid size for this band.
        c1_stride: C1 grid stride for this band.
    """

    band: int
    sizes: tuple[int, int]
    sigmas: tuple[float, float]
    wavelengths: tuple[float, float]
    c1_grid: int
    c1_stride: int


SERRE2007_GAMMA = 0.3
SERRE2007_THETAS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)

SERRE2007_BANDS: tuple[GaborBandSpec, ...] = (
    GaborBandSpec(1, (7, 9), (2.8, 3.6), (3.5, 4.6), 8, 4),
    GaborBandSpec(2, (11, 13), (4.5, 5.4), (5.6, 6.8), 10, 5),
    GaborBandSpec(3, (15, 17), (6.3, 7.3), (7.9, 9.1), 12, 6),
    GaborBandSpec(4, (19, 21), (8.2, 9.2), (10.3, 11.5), 14, 7),
    GaborBandSpec(5, (23, 25), (10.2, 11.3), (12.7, 14.1), 16, 8),
    GaborBandSpec(6, (27, 29), (12.3, 13.4), (15.4, 16.8), 18, 9),
    GaborBandSpec(7, (31, 33), (14.6, 15.8), (18.2, 19.7), 20, 10),
    GaborBandSpec(8, (35, 37), (17.0, 18.2), (21.2, 22.8), 22, 11),
)


def _gabor_kernel(
    size: int,
    wavelength: float,
    theta: float,
    sigma: float,
    gamma: float,
) -> torch.Tensor:
    """Zero-mean, unit-L2 Serre Gabor kernel in float64.

    Args:
        size: Odd kernel size.
        wavelength: Carrier wavelength.
        theta: Orientation in radians.
        sigma: Envelope standard deviation.
        gamma: Envelope aspect ratio.

    Returns:
        The kernel of shape ``(size, size)``.

    Raises:
        ValueError: If size is even or non-positive, wavelength, sigma, gamma or theta is
            not finite, wavelength <= 0, sigma <= 0, gamma <= 0, or the kernel is zero
            after mean removal.
    """
    if size % 2 == 0 or size <= 0:
        raise ValueError(f"size must be a positive odd integer, got {size}")
    if not math.isfinite(wavelength):
        raise ValueError(f"wavelength must be finite, got {wavelength}")
    if not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite, got {sigma}")
    if not math.isfinite(gamma):
        raise ValueError(f"gamma must be finite, got {gamma}")
    if not math.isfinite(theta):
        raise ValueError(f"theta must be finite, got {theta}")
    if wavelength <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    c = torch.arange(size, dtype=torch.float64) - size // 2
    x, y = torch.meshgrid(c, c, indexing="ij")
    x0 = x * math.cos(theta) + y * math.sin(theta)
    y0 = -x * math.sin(theta) + y * math.cos(theta)
    kernel = torch.exp(-(x0**2 + gamma**2 * y0**2) / (2 * sigma**2)) * torch.cos(
        2 * math.pi * x0 / wavelength
    )

    kernel = kernel - kernel.mean()
    norm: torch.Tensor = kernel.norm()
    if norm == 0:
        raise ValueError(
            f"kernel has zero norm after mean removal (size={size}, wavelength={wavelength}, "
            f"sigma={sigma}, gamma={gamma})"
        )
    return kernel / norm


def gabor_bank(
    sizes: list[int],
    wavelengths: list[float],
    sigmas: list[float],
    thetas: list[float],
    gamma: float = 0.3,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Gabor bank: one tensor per scale, each holding one filter per orientation.

    theta=0 responds to horizontal-direction wave fronts (stripes running along columns).

    Args:
        sizes: Kernel size per scale; must be odd.
        wavelengths: Carrier wavelength per scale.
        sigmas: Envelope standard deviation per scale.
        thetas: Orientations in radians; order is preserved in the output.
        gamma: Envelope aspect ratio.
        dtype: Data type of the returned tensors.

    Returns:
        One tensor of shape ``(len(thetas), 1, size, size)`` per scale.

    Raises:
        ValueError: If dtype is not a floating dtype, the sequences have different lengths,
            thetas is empty, or any kernel parameter is invalid.
    """
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be a floating dtype, got {dtype}")
    if not thetas:
        raise ValueError("thetas must not be empty")
    if not (len(sizes) == len(wavelengths) == len(sigmas)):
        raise ValueError(
            f"sizes, wavelengths and sigmas must have equal length, got "
            f"{len(sizes)}, {len(wavelengths)}, {len(sigmas)}"
        )

    bank: list[torch.Tensor] = []
    for size, wavelength, sigma in zip(sizes, wavelengths, sigmas, strict=True):
        filters = []
        for theta in thetas:
            filters.append(_gabor_kernel(size, wavelength, theta, sigma, gamma).to(dtype))
        bank.append(torch.stack(filters).unsqueeze(1))
    return bank


def serre2007_gabor_bank(dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
    """The 64 Gabor filters of Serre et al. 2007, Table 1: 16 scales x 4 orientations.

    Args:
        dtype: Data type of the returned tensors.

    Returns:
        16 tensors; tensor ``i`` has shape ``(4, 1, 7 + 2*i, 7 + 2*i)``.

    Raises:
        ValueError: If dtype is not a floating dtype.
    """
    sizes: list[int] = []
    sigmas: list[float] = []
    wavelengths: list[float] = []
    for spec in SERRE2007_BANDS:
        sizes.extend(spec.sizes)
        sigmas.extend(spec.sigmas)
        wavelengths.extend(spec.wavelengths)
    return gabor_bank(sizes, wavelengths, sigmas, list(SERRE2007_THETAS), SERRE2007_GAMMA, dtype)
