"""Spatial lateral inhibition across feature maps at each pixel."""

from __future__ import annotations

import torch


def spatial_lateral_inhibition(
    spikes: torch.Tensor,
    potential: torch.Tensor,
    fired: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply spatial lateral inhibition across feature maps.

    At each pixel ``(b, y, x)``: if ``fired[b, y, x]`` is True every spike there is suppressed.
    Otherwise, among the maps that spike at this pixel only the one with the highest potential
    survives (ties broken by the lowest map index ``c``).

    Args:
        spikes: Spike mask, bool, shape (B, C, H, W).
        potential: Potential per entry, float, shape (B, C, H, W).
        fired: Pixels that already fired, bool, shape (B, H, W).

    Returns:
        A tuple ``(new_spikes, new_fired)`` where ``new_spikes`` is bool (B, C, H, W) and
        ``new_fired`` is bool (B, H, W) equal to ``fired | new_spikes.any(dim=1)``. Inputs are not
        mutated.

    Raises:
        ValueError: If ``spikes`` is not bool or not 4-D, ``potential`` shape differs from
            ``spikes``, or ``fired`` is not bool or not shaped (B, H, W).
    """
    if spikes.dtype != torch.bool:
        raise ValueError("spikes must be bool")
    if spikes.dim() != 4:
        raise ValueError("spikes must be 4-D")
    if potential.shape != spikes.shape:
        raise ValueError(
            f"shape mismatch: potential {tuple(potential.shape)} != spikes {tuple(spikes.shape)}"
        )
    b, _c, h, w = spikes.shape
    if fired.shape != (b, h, w):
        raise ValueError(f"shape mismatch: fired {tuple(fired.shape)} != (B, H, W) = {(b, h, w)}")
    if fired.dtype != torch.bool:
        raise ValueError("fired must be bool")

    # Only spiking maps can survive; argmax on a -inf sentinel cannot tell a masked-out map from
    # a real -inf potential, so compare against the best spiking potential instead.
    best = potential.masked_fill(~spikes, float("-inf")).amax(dim=1, keepdim=True)
    cand = spikes & (potential == best)
    new_spikes = cand & (cand.cumsum(dim=1) == 1)  # first candidate along c wins ties

    # Suppress every spike at pixels that already fired.
    new_spikes = new_spikes & ~fired.unsqueeze(1)

    new_fired = fired | new_spikes.any(dim=1)
    return new_spikes, new_fired
