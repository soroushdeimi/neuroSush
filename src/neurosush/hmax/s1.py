"""S1 layer: simple cells as the absolute value of Gabor cross-correlations (Serre et al. 2007)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F


def s1_response(
    images: torch.Tensor,
    bank: Sequence[torch.Tensor],
    padding: Literal["same", "valid"] = "same",
    normalize: bool = False,
    eps: float = 1e-8,
) -> list[torch.Tensor]:
    """Cross-correlate every image with every kernel and take the absolute value.

    Args:
        images: Float tensor of shape (B, 1, H, W).
        bank: One kernel tensor per scale, each of shape (T, 1, s, s) with T orientations.
        padding: "same" (zero padding of s // 2, odd kernel sizes only) or "valid".
        normalize: Divide each response by the local L2 norm of the input patch.
        eps: Small constant added under the square root when ``normalize`` is true.

    Returns:
        One tensor per scale, shape (B, T, H, W) for "same" or (B, T, H - s + 1, W - s + 1)
        for "valid".

    Raises:
        ValueError: If images is not 4-D with one channel or not a floating tensor, the bank
            is empty, a kernel is not 4-D with one input channel, not a floating tensor or
            not square, padding is not "same" or "valid", a kernel is even-sized with
            "same", eps <= 0, or a kernel is larger than the image with "valid".
    """
    if images.dim() != 4:
        raise ValueError(f"images must be 4-D (B, 1, H, W), got shape {tuple(images.shape)}")
    if images.shape[1] != 1:
        raise ValueError(f"images must have 1 channel, got {images.shape[1]}")
    if not images.is_floating_point():
        raise ValueError(f"images must be a floating tensor, got dtype {images.dtype}")
    if len(bank) == 0:
        raise ValueError("bank must not be empty")
    if padding not in ("same", "valid"):
        raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    responses: list[torch.Tensor] = []
    for kernel in bank:
        if kernel.dim() != 4 or kernel.shape[1] != 1:
            raise ValueError(
                f"each kernel must be 4-D (T, 1, s, s), got shape {tuple(kernel.shape)}"
            )
        if not kernel.is_floating_point():
            raise ValueError(f"each kernel must be a floating tensor, got dtype {kernel.dtype}")
        if kernel.shape[2] != kernel.shape[3]:
            raise ValueError(f"each kernel must be square, got shape {tuple(kernel.shape)}")
        s = kernel.shape[2]
        if s % 2 == 0 and padding == "same":
            raise ValueError(f"kernel size must be odd with padding='same', got {s}")
        if padding == "valid" and (s > images.shape[2] or s > images.shape[3]):
            raise ValueError(
                f"kernel {s}x{s} is larger than the image "
                f"{images.shape[2]}x{images.shape[3]} with padding='valid'"
            )

        pad = s // 2 if padding == "same" else 0
        conv = F.conv2d(images, kernel, padding=pad)
        if normalize:
            # half precision underflows the default eps, so normalise in float32
            compute = images.float() if images.dtype in (torch.float16, torch.bfloat16) else images
            ones = torch.ones(1, 1, s, s, dtype=compute.dtype, device=images.device)
            local_norm = torch.sqrt(F.conv2d(compute**2, ones, padding=pad) + eps)
            conv = (conv / local_norm).to(images.dtype)
        responses.append(torch.abs(conv))
    return responses
