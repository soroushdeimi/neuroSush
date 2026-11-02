"""Tests for binary frame encoding and decoding."""

from pathlib import Path

import pytest
import torch
from studio.stream.frames import (
    decode_frame,
    encode_scalar,
    encode_spikes,
    encode_vsample,
    encode_weights,
)


def get_golden_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "contracts" / "golden"


def test_golden_spikes() -> None:
    golden_path = get_golden_dir() / "spikes.bin"
    golden_bytes = golden_path.read_bytes()
    expected_hex = "0100000001000000640000000200000001020800"
    assert golden_bytes.hex() == expected_hex

    # Encode test
    # SPIKES (n=10, seq 1, step0 100, 2 steps; step100 fires {0,9}, step101 fires {3})
    payload = bytes([0x01, 0x02, 0x08, 0x00])
    encoded = encode_spikes(stream=0, seq=1, step0=100, nsteps=2, payload=payload)
    assert encoded == golden_bytes

    # Decode test
    decoded = decode_frame(golden_bytes)
    assert decoded["type"] == 1
    assert decoded["flags"] == 0
    assert decoded["stream"] == 0
    assert decoded["seq"] == 1
    assert decoded["step0"] == 100
    assert decoded["nsteps"] == 2
    assert decoded["payload"] == payload


def test_golden_vsample() -> None:
    golden_path = get_golden_dir() / "vsample.bin"
    golden_bytes = golden_path.read_bytes()
    expected_hex = (
        "02000100020000006400000002000000020000000000000005000000000082c2000081c2000071c20000f041"
    )
    assert golden_bytes.hex() == expected_hex

    # Encode test
    indices = [0, 5]
    values = [[-65.0, -64.5], [-60.25, 30.0]]
    encoded = encode_vsample(stream=1, seq=2, step0=100, indices=indices, values=values)
    assert encoded == golden_bytes

    # Decode test
    decoded = decode_frame(golden_bytes)
    assert decoded["type"] == 2
    assert decoded["stream"] == 1
    assert decoded["seq"] == 2
    assert decoded["step0"] == 100
    assert decoded["nsteps"] == 2
    assert decoded["indices"] == indices
    assert decoded["values"] == values


def test_golden_scalar() -> None:
    golden_path = get_golden_dir() / "scalar.bin"
    golden_bytes = golden_path.read_bytes()
    expected_hex = "03000000030000000000000003000000000000000000003f00004441"
    assert golden_bytes.hex() == expected_hex

    # Encode test
    values = [0.0, 0.5, 12.25]
    encoded = encode_scalar(stream=0, seq=3, step0=0, values=values)
    assert encoded == golden_bytes

    # Decode test
    decoded = decode_frame(golden_bytes)
    assert decoded["type"] == 3
    assert decoded["stream"] == 0
    assert decoded["seq"] == 3
    assert decoded["step0"] == 0
    assert decoded["nsteps"] == 3
    assert decoded["values"] == values


def test_golden_weights() -> None:
    golden_path = get_golden_dir() / "weights.bin"
    golden_bytes = golden_path.read_bytes()
    expected_hex = (
        "0401000004000000f40100000100000002000300"
        "04000400000000000000803e0000003f0000403f0000803f000080bf"
    )
    assert golden_bytes.hex() == expected_hex

    # Encode test
    weights = [0.0, 0.25, 0.5, 0.75, 1.0, -1.0]
    encoded = encode_weights(
        stream=0,
        seq=4,
        step0=500,
        rows=2,
        cols=3,
        row_stride=4,
        col_stride=4,
        weights=weights,
        flags=1,
    )
    assert encoded == golden_bytes

    # Decode test
    decoded = decode_frame(golden_bytes)
    assert decoded["type"] == 4
    assert decoded["flags"] == 1
    assert decoded["stream"] == 0
    assert decoded["seq"] == 4
    assert decoded["step0"] == 500
    assert decoded["nsteps"] == 1
    assert decoded["rows"] == 2
    assert decoded["cols"] == 3
    assert decoded["row_stride"] == 4
    assert decoded["col_stride"] == 4
    assert decoded["weights"] == weights


def test_torch_tensor_inputs() -> None:
    # Test encoding with torch tensors
    indices_t = torch.tensor([0, 5])
    values_t = torch.tensor([[-65.0, -64.5], [-60.25, 30.0]])
    encoded_v = encode_vsample(1, 2, 100, indices_t, values_t)
    assert decode_frame(encoded_v)["values"] == values_t.tolist()

    scalar_t = torch.tensor([0.0, 0.5, 12.25])
    encoded_s = encode_scalar(0, 3, 0, scalar_t)
    assert decode_frame(encoded_s)["values"] == scalar_t.tolist()

    weights_t = torch.tensor([[0.0, 0.25, 0.5], [0.75, 1.0, -1.0]])
    encoded_w = encode_weights(0, 4, 500, 2, 3, 4, 4, weights_t, flags=1)
    assert decode_frame(encoded_w)["weights"] == weights_t.flatten().tolist()


def test_error_cases() -> None:
    # Frame too short for header
    with pytest.raises(ValueError, match="Frame data too short"):
        decode_frame(b"short")

    # Unknown frame type
    bad_header = encode_scalar(0, 1, 0, [1.0])
    # Replace type byte (index 0) with 99
    bad_frame = bytes([99]) + bad_header[1:]
    with pytest.raises(ValueError, match="Unknown frame type"):
        decode_frame(bad_frame)

    # Truncated VSAMPLE payload
    vsample_bytes = encode_vsample(1, 1, 0, [0], [[1.0]])
    with pytest.raises(ValueError, match="payload"):
        decode_frame(vsample_bytes[: 16 + 2])  # cut off payload

    # Truncated SCALAR payload
    scalar_bytes = encode_scalar(0, 1, 0, [1.0, 2.0])
    with pytest.raises(ValueError, match="SCALAR payload truncated"):
        decode_frame(scalar_bytes[: 16 + 2])  # cut off half of float32

    # Truncated WEIGHTS payload
    weights_bytes = encode_weights(0, 1, 0, 2, 2, 1, 1, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="WEIGHTS payload truncated"):
        decode_frame(weights_bytes[: 16 + 10])  # cut off part of weights
