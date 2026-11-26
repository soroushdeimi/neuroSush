"""S2 radial-basis response to stored C1 prototypes (Serre et al. 2007, Eq. 4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def s2_rbf(c1: torch.Tensor, prototypes: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Radial-basis response of every C1 patch to each stored prototype.

    ``r = exp(-beta * ||X - P||^2)`` where ``X`` is an ``n x n x T`` patch of C1
    responses across all orientations and ``P`` a stored prototype of the same
    shape. The squared distance is computed as
    ``conv2d(X^2, ones) - 2 * conv2d(X, P) + ||P||^2`` in float64 to avoid
    catastrophic cancellation, then clamped to be non-negative.

    Args:
        c1: C1 responses of shape ``(B, T, H, W)``.
        prototypes: Stored prototypes of shape ``(P, T, n, n)``.
        beta: Inverse squared width of the radial basis.

    Returns:
        Responses of shape ``(B, P, H - n + 1, W - n + 1)`` in ``c1``'s dtype.

    Raises:
        ValueError: If beta <= 0, c1 is not 4-D, prototypes is not 4-D, the
            orientation channels differ, a prototype is not square, or the image is
            smaller than the prototype.

    Notes:
        The float64 expansion loses the distance when responses are around 1e8 or
        larger (e.g. C1=1e8, P=1e8+1 returns 1.0 instead of exp(-1)); C1 values are
        expected to be of order 1. CPU only (float64 is not available on MPS).
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not c1.is_floating_point():
        raise ValueError(f"c1 must be floating-point, got dtype {c1.dtype}")
    if not prototypes.is_floating_point():
        raise ValueError(f"prototypes must be floating-point, got dtype {prototypes.dtype}")
    if c1.dim() != 4:
        raise ValueError(f"c1 must be 4-D, got shape {tuple(c1.shape)}")
    if prototypes.dim() != 4:
        raise ValueError(f"prototypes must be 4-D, got shape {tuple(prototypes.shape)}")
    _, t, h, w = c1.shape
    p, pt, n, m = prototypes.shape
    if t != pt:
        raise ValueError(f"c1 and prototypes time dimensions differ: {t} != {pt}")
    if n != m:
        raise ValueError(f"prototype must be square, got {n} x {m}")
    if h < n or w < n:
        raise ValueError(f"image {h} x {w} is smaller than prototype {n} x {n}")

    c1_64 = c1.to(torch.float64)
    p_64 = prototypes.to(torch.float64)
    ones = torch.ones(1, t, n, n, dtype=torch.float64, device=c1.device)
    p_norm = (p_64**2).sum(dim=(1, 2, 3)).view(1, p, 1, 1)
    d2 = F.conv2d(c1_64**2, ones) - 2.0 * F.conv2d(c1_64, p_64) + p_norm
    d2 = d2.clamp(min=0.0)
    return torch.exp(-beta * d2).to(c1.dtype)


def sample_c1_patches(
    c1_bands: list[torch.Tensor],
    n_patches: int,
    size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Randomly sample C1 patches of a single image across its scale bands.

    For each patch a band is drawn uniformly among the bands large enough to
    hold a ``size x size`` patch, then a top-left corner uniformly over the
    valid positions.

    Args:
        c1_bands: One tensor of shape ``(T, H, W)`` per scale band.
        n_patches: Number of patches to draw.
        size: Side length of each square patch.
        generator: Torch generator for all randomness.

    Returns:
        Patches of shape ``(n_patches, T, size, size)``; copies, not views.

    Raises:
        ValueError: If n_patches < 1, size < 1, a band is not 3-D, the
            orientation channels differ between bands, no band is large enough,
            or the list is empty.
    """
    if n_patches < 1:
        raise ValueError(f"n_patches must be at least 1, got {n_patches}")
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if not c1_bands:
        raise ValueError("c1_bands must not be empty")
    t = None
    for band in c1_bands:
        if band.dim() != 3:
            raise ValueError(f"each band must be 3-D, got shape {tuple(band.shape)}")
        if t is None:
            t = band.shape[0]
        elif band.shape[0] != t:
            raise ValueError(f"band time dimensions differ: {t} != {band.shape[0]}")
    eligible = [
        i for i, band in enumerate(c1_bands) if band.shape[1] >= size and band.shape[2] >= size
    ]
    if not eligible:
        raise ValueError(f"no band has H and W >= {size}")

    patches = []
    for _ in range(n_patches):
        i = int(torch.randint(len(eligible), (1,), generator=generator).item())
        band = c1_bands[eligible[i]]
        top = int(torch.randint(band.shape[1] - size + 1, (1,), generator=generator).item())
        left = int(torch.randint(band.shape[2] - size + 1, (1,), generator=generator).item())
        patches.append(band[:, top : top + size, left : left + size].clone())
    return torch.stack(patches)
