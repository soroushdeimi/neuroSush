"""dataset helpers: image-location-label samples and per-step spike frames."""

from __future__ import annotations

import itertools
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
    batch_size: int | None = None,
) -> Generator[tuple[torch.Tensor, Any], None, None]:
    """Yield one ``(frame, label)`` per simulation step from spike trains.

    Each train has shape ``(steps, *shape)`` and each frame is one flattened step. After every
    sample (or batch), ``silence`` all-false frames follow with label ``None``. Samples are
    read lazily.

    With ``batch_size=B``, ``B`` consecutive samples run side by side: frames have shape
    ``(B, size)`` and the label is the list of the ``B`` labels. Trains in one batch must have
    the same length; a last incomplete batch is dropped.

    Args:
        samples: Iterable of ``(spike_train, label)`` pairs.
        silence: Silent steps after each sample or batch.
        batch_size: Number of samples per frame, or ``None`` for unbatched frames.
    """
    if silence < 0:
        raise ValueError(f"silence must be non-negative, got {silence}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be positive or None, got {batch_size}")
    groups = (
        ((train, label) for train, label in samples)
        if batch_size is None
        else _batches(samples, batch_size)
    )
    for spike_train, label in groups:
        steps = spike_train.shape[0]
        flat = spike_train.reshape(steps, *spike_train.shape[1 : 2 if batch_size else 1], -1)
        flat = flat.bool()
        for row in flat:
            yield row, label
        for _ in range(silence):
            yield torch.zeros_like(flat[0]), None


def _batches(
    samples: Iterable[tuple[torch.Tensor, Any]], batch_size: int
) -> Generator[tuple[torch.Tensor, list[Any]], None, None]:
    """Group samples into ``(steps, batch, *shape)`` trains with a list of labels."""
    iterator = iter(samples)
    while chunk := list(itertools.islice(iterator, batch_size)):
        if len(chunk) < batch_size:
            return
        lengths = {train.shape[0] for train, _ in chunk}
        if len(lengths) != 1:
            raise ValueError(f"trains in one batch must have equal length, got {sorted(lengths)}")
        yield torch.stack([train for train, _ in chunk], dim=1), [label for _, label in chunk]
