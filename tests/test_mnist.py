import gzip
import struct

import pytest
import torch

import neurosush.data as data
from neurosush.data import load_mnist, read_idx

NAMES = {
    "train-images-idx3-ubyte": 3,
    "train-labels-idx1-ubyte": 1,
    "t10k-images-idx3-ubyte": 3,
    "t10k-labels-idx1-ubyte": 1,
}


def idx(values, shape, *, dtype=0x08, magic=0):
    """The bytes of an IDX file holding ``values`` (a flat list of bytes)."""
    header = struct.pack(">HBB", magic, dtype, len(shape)) + struct.pack(f">{len(shape)}I", *shape)
    return header + bytes(values)


def write_mnist(root, *, compress=False, skip=()):
    """Tiny files: 3 train and 2 test images of 2 x 2 pixels."""
    files = {
        "train-images-idx3-ubyte": idx(range(12), (3, 2, 2)),
        "train-labels-idx1-ubyte": idx([5, 0, 9], (3,)),
        "t10k-images-idx3-ubyte": idx(range(100, 108), (2, 2, 2)),
        "t10k-labels-idx1-ubyte": idx([7, 1], (2,)),
    }
    for name, raw in files.items():
        if name in skip:
            continue
        if compress:
            (root / f"{name}.gz").write_bytes(gzip.compress(raw))
        else:
            (root / name).write_bytes(raw)


@pytest.mark.parametrize("compress", [False, True])
def test_loads_plain_and_gzipped_files(tmp_path, compress):
    write_mnist(tmp_path, compress=compress)
    train_x, train_y, test_x, test_y = load_mnist(tmp_path)
    assert train_x.dtype == torch.uint8
    assert train_x.shape == (3, 2, 2)
    assert train_x.flatten().tolist() == list(range(12))
    assert train_y.dtype == torch.int64
    assert train_y.tolist() == [5, 0, 9]
    assert test_x.shape == (2, 2, 2)
    assert test_x[1, 1].tolist() == [106, 107]
    assert test_y.tolist() == [7, 1]


def test_plain_and_gzipped_files_can_be_mixed(tmp_path):
    write_mnist(tmp_path, compress=True, skip=("t10k-labels-idx1-ubyte",))
    (tmp_path / "t10k-labels-idx1-ubyte").write_bytes(idx([7, 1], (2,)))
    assert load_mnist(tmp_path)[3].tolist() == [7, 1]


def test_a_missing_file_explains_how_to_get_it(tmp_path):
    write_mnist(tmp_path, skip=("t10k-images-idx3-ubyte",))
    with pytest.raises(FileNotFoundError, match=r"t10k-images-idx3-ubyte.*download=True"):
        load_mnist(tmp_path)


def test_download_fetches_only_missing_files_atomically(tmp_path, monkeypatch):
    write_mnist(tmp_path, skip=("train-labels-idx1-ubyte",))
    urls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return gzip.compress(idx([5, 0, 9], (3,)))

    def urlopen(url, timeout):
        urls.append(url)
        return Response()

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    assert load_mnist(tmp_path, download=True)[1].tolist() == [5, 0, 9]
    assert urls == [data.MNIST_URL + "train-labels-idx1-ubyte.gz"]
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    def urlopen(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    with pytest.raises(FileNotFoundError, match="offline"):
        load_mnist(tmp_path, download=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (idx([1, 2], (2,), magic=1), "bad magic"),
        (idx([1, 2], (2,), dtype=0x0D), "unsigned byte"),
        (idx([1, 2, 3], (2,)), "declares 2 values, file holds 3"),
        (b"\x00\x00", "bad magic"),
        (b"\x00\x00\x08\x03\x00\x00", "truncated"),
    ],
)
def test_invalid_idx_files(tmp_path, raw, match):
    path = tmp_path / "bad"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=match):
        read_idx(path)


def test_the_dimensions_must_fit_the_kind_of_file(tmp_path):
    write_mnist(tmp_path)
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(idx(range(6), (3, 2)))
    with pytest.raises(ValueError, match="1 dimension"):
        load_mnist(tmp_path)


def test_images_and_labels_must_agree(tmp_path):
    write_mnist(tmp_path)
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(idx([1, 2], (2,)))
    with pytest.raises(ValueError, match="3 images but 2 labels"):
        load_mnist(tmp_path)
