"""dataset helpers: image-location-label samples and per-step spike frames."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeVar

import torch
from torch.utils.data import Dataset

T = TypeVar("T")


class LocationDataset(Dataset[tuple[torch.Tensor, torch.Tensor | None, Any]]):
    """A dataset that yields (image, location, label) tuples.

    Args:
        dataset: The underlying dataset yielding (image, label) tuples.
        pre_transform: A function that takes an image and returns (image, location).
        post_transform: A function that takes an image and returns a new image.
        location_transform: A function that takes a location and returns a new location.
        target_transform: A function that takes a label and returns a new label.
    """

    def __init__(
        self,
        dataset: Dataset[tuple[torch.Tensor, Any]],
        *,
        pre_transform: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        post_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        location_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.location_transform = location_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        image, label = self.dataset[idx]
        location: torch.Tensor | None = None

        if self.pre_transform is not None:
            image, location = self.pre_transform(image)

        if self.post_transform is not None:
            image = self.post_transform(image)

        if self.location_transform is not None:
            if location is None:
                raise ValueError("location_transform needs a pre_transform that returns a location")
            location = self.location_transform(location)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, location, label


def spike_frames(
    samples: Iterable[tuple[torch.Tensor, Any]],
    *,
    silence: int = 0,
) -> Generator[tuple[torch.Tensor, Any | None], None, None]:
    """A generator yielding (frame, label) tuples from spike trains.

    Each spike train has shape (steps, *shape). For each step, a flattened frame
    is yielded. After all steps in a sample, `silence` number of zero frames
    are yielded with label `None`.

    Args:
        samples: An iterable of (spike_train, label) tuples.
        silence: Number of silence frames to yield after each sample.

    Yields:
        A tuple of (frame, label).

    Raises:
        ValueError: If silence < 0.
    """
    if silence < 0:
        raise ValueError("silence must be non-negative")

    for spike_train, label in samples:
        flat = spike_train.reshape(spike_train.shape[0], -1).bool()
        for row in flat:
            yield row, label

        for _ in range(silence):
            yield torch.zeros_like(flat[0]), None
