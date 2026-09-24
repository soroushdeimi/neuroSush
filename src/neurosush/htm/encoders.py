"""Encoders that turn values into SDRs whose overlap reflects the similarity of the values."""

from __future__ import annotations

import torch


class ScalarEncoder:
    """Contiguous block of ``w`` active bits whose position encodes a value.

    Non-periodic: ``n - w + 1`` buckets over ``[minimum, maximum]``; the bucket of ``x`` is
    ``round((x - minimum) / (maximum - minimum) * (n - w))``. Periodic: ``n`` buckets and the
    block wraps around. Two values in buckets ``i`` and ``j`` share exactly
    ``max(0, w - |i - j|)`` bits (circular distance when periodic).

    Args:
        n: Number of bits.
        w: Number of active bits.
        minimum: Lowest value.
        maximum: Highest value (exclusive when periodic).
        periodic: Treat the range as a circle, for angles or times of day.
        clip: Clamp out-of-range values instead of raising ``ValueError``.
    """

    def __init__(
        self,
        n: int,
        w: int,
        minimum: float,
        maximum: float,
        *,
        periodic: bool = False,
        clip: bool = False,
    ) -> None:
        if not 0 < w <= n:
            raise ValueError(f"need 0 < w <= n, got n={n}, w={w}")
        if maximum <= minimum:
            raise ValueError(f"maximum ({maximum}) must exceed minimum ({minimum})")
        self.n, self.w, self.minimum, self.maximum = n, w, minimum, maximum
        self.periodic, self.clip = periodic, clip

    def bucket(self, x: torch.Tensor) -> torch.Tensor:
        """Bucket index of every value."""
        x = torch.as_tensor(x, dtype=torch.float64)
        low, high = self.minimum, self.maximum
        if self.periodic:
            fraction = torch.remainder(x - low, high - low) / (high - low)
            return (fraction * self.n).floor().long() % self.n
        outside = (x < low) | (x > high)
        if bool(outside.any()) and not self.clip:
            raise ValueError(f"values must lie in [{low}, {high}], got {x[outside].tolist()}")
        fraction = (x.clamp(low, high) - low) / (high - low)
        return (fraction * (self.n - self.w)).round().long()

    def encode(self, x: torch.Tensor | float) -> torch.Tensor:
        """SDRs of shape ``(*x.shape, n)``."""
        start = self.bucket(x)
        offsets = torch.arange(self.w)
        positions = start.unsqueeze(-1) + offsets
        if self.periodic:
            positions = positions % self.n
        sdr = torch.zeros((*start.shape, self.n), dtype=torch.bool)
        return sdr.scatter_(-1, positions, True)


def _mix(key: torch.Tensor, seed: int) -> torch.Tensor:
    """SplitMix64-style hash of int64 keys to non-negative int64 values.

    The constants are the SplitMix64 multipliers written as signed int64; torch wraps int64
    multiplication, which is what the hash needs.
    """
    z = key.to(torch.int64) + torch.tensor(seed + 1, dtype=torch.int64) * -7046029254386353131
    z = (z ^ (z >> 30)) * -4658895280553007687
    z = (z ^ (z >> 27)) * -7723592293110705685
    z = z ^ (z >> 31)
    return z & 0x7FFFFFFFFFFFFFFF


class RandomDistributedScalarEncoder:
    """Scalar encoder for unbounded values: each bucket owns ``w`` hashed bits.

    Bucket ``b = floor(x / resolution)`` activates bits ``hash(b + j) mod n`` for
    ``j = 0 .. w - 1``, so buckets ``d`` apart share the ``w - d`` hashes they have in common
    and unrelated buckets overlap only by chance (about ``w**2 / n`` bits). Hash collisions
    can make an SDR have slightly fewer than ``w`` bits; they are rare when ``n >> w``.

    Args:
        n: Number of bits.
        w: Hashes per value (active bits, up to collisions).
        resolution: Width of one bucket.
        seed: Seed of the hash, which fixes the code.
    """

    def __init__(self, n: int, w: int, resolution: float, *, seed: int = 0) -> None:
        if not 0 < w <= n:
            raise ValueError(f"need 0 < w <= n, got n={n}, w={w}")
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        self.n, self.w, self.resolution, self.seed = n, w, resolution, seed

    def bucket(self, x: torch.Tensor) -> torch.Tensor:
        """Bucket index of every value."""
        return torch.floor(torch.as_tensor(x, dtype=torch.float64) / self.resolution).long()

    def encode(self, x: torch.Tensor | float) -> torch.Tensor:
        """SDRs of shape ``(*x.shape, n)``."""
        bucket = self.bucket(x)
        keys = bucket.unsqueeze(-1) + torch.arange(self.w)
        positions = _mix(keys, self.seed) % self.n
        sdr = torch.zeros((*bucket.shape, self.n), dtype=torch.bool)
        return sdr.scatter_(-1, positions, True)


class CategoryEncoder:
    """A fixed random ``w``-bit SDR per category, drawn from a seeded generator.

    Args:
        n: Number of bits.
        w: Active bits per category.
        categories: Number of categories.
        seed: Seed that fixes the codes.
    """

    def __init__(self, n: int, w: int, categories: int, *, seed: int = 0) -> None:
        if not 0 < w <= n:
            raise ValueError(f"need 0 < w <= n, got n={n}, w={w}")
        if categories < 1:
            raise ValueError(f"categories must be positive, got {categories}")
        generator = torch.Generator().manual_seed(seed)
        keys = torch.rand(categories, n, generator=generator)
        self.codes = torch.zeros(categories, n, dtype=torch.bool).scatter_(
            -1, keys.topk(w, dim=-1).indices, True
        )

    def encode(self, category: torch.Tensor | int) -> torch.Tensor:
        """SDRs of shape ``(*category.shape, n)``."""
        category = torch.as_tensor(category, dtype=torch.long)
        if bool(((category < 0) | (category >= len(self.codes))).any()):
            raise ValueError(
                f"categories must be in [0, {len(self.codes)}), got {category.tolist()}"
            )
        return self.codes[category]
