"""dataset helpers: image-location-label samples, per-step spike frames and MNIST."""

from __future__ import annotations

import gzip
import itertools
import struct
import urllib.request
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch.utils.data import Dataset

T = TypeVar("T")


class _ImageDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, object]: ...


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
        dataset: _ImageDataset,
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


MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = (
    ("train-images-idx3-ubyte", 3),
    ("train-labels-idx1-ubyte", 1),
    ("t10k-images-idx3-ubyte", 3),
    ("t10k-labels-idx1-ubyte", 1),
)
_IDX_UBYTE = 0x08


def read_idx(path: str | Path) -> torch.Tensor:
    """Read an IDX file of unsigned bytes, plain or gzip-compressed (``.gz``).

    Args:
        path: The file.

    Returns:
        A uint8 tensor with the dimensions the header declares.

    Raises:
        ValueError: If the magic number, the data type (only unsigned bytes) or the size is
            not that of a valid IDX file.
    """
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"{path} is not an IDX file: bad magic number {raw[:4].hex()}")
    if raw[2] != _IDX_UBYTE:
        raise ValueError(
            f"{path}: only unsigned byte IDX data (0x08) is supported, got {raw[2]:#04x}"
        )
    dims = raw[3]
    header = 4 + 4 * dims
    if dims < 1 or len(raw) < header:
        raise ValueError(f"{path}: truncated IDX header ({dims} dimensions)")
    shape = struct.unpack(f">{dims}I", raw[4:header])
    count = 1
    for size in shape:
        count *= size
    if len(raw) - header != count:
        raise ValueError(f"{path}: header declares {count} values, file holds {len(raw) - header}")
    if count == 0:
        return torch.zeros(shape, dtype=torch.uint8)
    return torch.frombuffer(bytearray(raw[header:]), dtype=torch.uint8).reshape(shape)


def load_mnist(
    root: str | Path, *, download: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST from its IDX files, with pure torch.

    Reads ``train-images-idx3-ubyte``, ``train-labels-idx1-ubyte``, ``t10k-images-idx3-ubyte``
    and ``t10k-labels-idx1-ubyte`` from ``root``, plain or with a ``.gz`` suffix.

    Args:
        root: Folder with the files.
        download: Fetch missing files as ``.gz`` from ``MNIST_URL`` into ``root`` (written
            atomically, so an interrupted download leaves no partial file).

    Returns:
        ``(train_images, train_labels, test_images, test_labels)``: images are uint8
        ``(60000, 28, 28)`` and ``(10000, 28, 28)``, labels int64.

    Raises:
        FileNotFoundError: If a file is missing and ``download`` is false (or the download
            fails).
        ValueError: If a file is not a valid IDX file of the expected kind.
    """
    root = Path(root)
    tensors = []
    for name, dims in _MNIST_FILES:
        path = next((p for p in (root / name, root / f"{name}.gz") if p.is_file()), None)
        if path is None:
            path = _fetch(name, root, download)
        tensor = read_idx(path)
        if tensor.dim() != dims:
            raise ValueError(f"{path} must have {dims} dimension(s), got {tensor.dim()}")
        tensors.append(tensor)
    train_images, train_labels, test_images, test_labels = tensors
    for images, labels in ((train_images, train_labels), (test_images, test_labels)):
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
    return train_images, train_labels.long(), test_images, test_labels.long()


def _fetch(name: str, root: Path, download: bool) -> Path:
    """Download ``name`` as ``root/name.gz``, or explain how to get it."""
    target = root / f"{name}.gz"
    if not download:
        raise FileNotFoundError(
            f"MNIST file {name} (or {name}.gz) not found in {root}; "
            "pass download=True or put the IDX files there"
        )
    root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".gz.tmp")
    try:
        with urllib.request.urlopen(f"{MNIST_URL}{name}.gz", timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(target)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"could not download {MNIST_URL}{name}.gz ({error}); fetch it manually into {root}"
        ) from error
    return target
