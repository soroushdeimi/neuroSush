"""Filters module."""

from __future__ import annotations

import math

import torch


def _grid(
    size: int,
    spacing: float,
    dtype: type[torch.dtype] | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a coordinate grid.

    Args:
        size: Size of the grid.
        spacing: Spacing between pixels.
        dtype: Data type of the grid.
        device: Device of the grid.

    Returns:
        A tuple of (x, y) coordinate tensors.

    Raises:
        ValueError: If size < 1 or spacing <= 0.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    c = (torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2) * spacing
    x, y = torch.meshgrid(c, c, indexing="ij")
    return x, y


def _normalize(
    kernel: torch.Tensor,
    zero_mean: bool,
    unit_l1: bool,
) -> torch.Tensor:
    """Normalize the kernel.

    Args:
        kernel: The kernel to normalize.
        zero_mean: Whether to make the kernel sum to zero.
        unit_l1: Whether to make the L1 norm of the kernel equal to 1.

    Returns:
        The normalized kernel.
    """
    new_kernel = kernel.clone()
    if zero_mean:
        pos_sum = torch.sum(new_kernel.clamp(min=0))
        neg_sum = torch.sum(new_kernel.clamp(max=0))
        if neg_sum != 0:
            new_kernel = torch.where(new_kernel < 0, new_kernel * (-pos_sum / neg_sum), new_kernel)

    if unit_l1:
        l1_norm = torch.sum(torch.abs(new_kernel))
        if l1_norm != 0:
            new_kernel = new_kernel / l1_norm

    return new_kernel


def dog_kernel(
    size: int,
    sigma_1: float,
    sigma_2: float,
    *,
    spacing: float = 1.0,
    zero_mean: bool = False,
    unit_l1: bool = False,
    dtype: type[torch.dtype] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Difference of Gaussians kernel.

    Args:
        size: Size of the kernel.
        sigma_1: Standard deviation of the first Gaussian.
        sigma_2: Standard deviation of the second Gaussian.
        spacing: Spacing between pixels.
        zero_mean: Whether to make the kernel sum to zero.
        unit_l1: Whether to make the L1 norm of the kernel equal to 1.
        dtype: Data type of the kernel.
        device: Device of the kernel.

    Returns:
        The DoG kernel.

    Raises:
        ValueError: If size < 1, spacing <= 0, or sigma_1 or sigma_2 <= 0.
    """
    if sigma_1 <= 0 or sigma_2 <= 0:
        raise ValueError("sigma must be positive")

    x, y = _grid(size, spacing, dtype=dtype, device=device)
    q = -(x**2 + y**2) / 2
    term1 = torch.exp(q / sigma_1**2) / sigma_1
    term2 = torch.exp(q / sigma_2**2) / sigma_2
    kernel = (term1 - term2) / math.sqrt(2 * math.pi)

    return _normalize(kernel, zero_mean, unit_l1)


def gabor_kernel(
    size: int,
    wavelength: float,
    theta: float,
    sigma: float,
    gamma: float,
    *,
    spacing: float = 1.0,
    zero_mean: bool = False,
    unit_l1: bool = False,
    dtype: type[torch.dtype] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gabor kernel.

    Args:
        size: Size of the kernel.
        wavelength: Wavelength of the sinusoidal carrier.
        theta: Orientation of the filter.
        sigma: Standard deviation of the Gaussian envelope.
        gamma: Aspect ratio of the Gaussian envelope.
        spacing: Spacing between pixels.
        zero_mean: Whether to make the kernel sum to zero.
        unit_l1: Whether to make the L1 norm of the kernel equal to 1.
        dtype: Data type of the kernel.
        device: Device of the kernel.

    Returns:
        The Gabor kernel.

    Raises:
        ValueError: If size < 1, spacing <= 0, wavelength <= 0, sigma <= 0, or gamma <= 0.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    x, y = _grid(size, spacing, dtype=dtype, device=device)

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    x_prime = x * cos_theta + y * sin_theta
    y_prime = -x * sin_theta + y * cos_theta

    envelope = torch.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2))
    carrier = torch.cos(2 * math.pi * x_prime / wavelength)
    kernel = envelope * carrier

    return _normalize(kernel, zero_mean, unit_l1)
