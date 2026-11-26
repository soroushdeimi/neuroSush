"""C2 pooling (Serre et al. 2007): global max over all scales and positions of each S2 type."""

from __future__ import annotations

import torch


def c2_global_max(s2: list[torch.Tensor]) -> torch.Tensor:
    """Take the global max of every S2 band over all positions and bands.

    Args:
        s2: One tensor ``(B, P, H_b, W_b)`` per band; H_b and W_b may differ between bands,
            but B and P must match.

    Returns:
        A ``(B, P)`` tensor: the maximum over every band and every position, per image and
        per prototype.

    Raises:
        ValueError: If s2 is empty, a tensor is not 4-D, or B or P differs between bands.
    """
    if not s2:
        raise ValueError("s2 must not be empty")

    for i, t in enumerate(s2):
        if t.dim() != 4:
            raise ValueError(f"s2[{i}] must be 4-D, got shape {tuple(t.shape)}")

    b, p = s2[0].shape[0], s2[0].shape[1]
    for i, t in enumerate(s2):
        if t.shape[0] != b:
            raise ValueError(f"s2[{i}]: B differs: {t.shape[0]} vs {b}")
        if t.shape[1] != p:
            raise ValueError(f"s2[{i}]: P differs: {t.shape[1]} vs {p}")

    return torch.stack([t.amax(dim=(-2, -1)) for t in s2]).amax(dim=0)
