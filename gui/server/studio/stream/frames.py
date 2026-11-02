"""Binary frame encoding and decoding for WebSocket trace streaming."""

import struct
from typing import Any

import torch


def encode_header(
    type_val: int,
    flags: int,
    stream: int,
    seq: int,
    step0: int,
    nsteps: int,
) -> bytes:
    """Pack 16-byte frame header."""
    return struct.pack("<BBHIIHH", type_val, flags, stream, seq, step0, nsteps, 0)


def encode_spikes(
    stream: int,
    seq: int,
    step0: int,
    nsteps: int,
    payload: bytes,
    flags: int = 0,
) -> bytes:
    """Encode a SPIKES binary frame."""
    header = encode_header(1, flags, stream, seq, step0, nsteps)
    return header + payload


def encode_vsample(
    stream: int,
    seq: int,
    step0: int,
    indices: list[int] | torch.Tensor,
    values: list[list[float]] | torch.Tensor,
    flags: int = 0,
) -> bytes:
    """Encode a VSAMPLE binary frame."""
    indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
    values_list = values.tolist() if isinstance(values, torch.Tensor) else list(values)

    nsteps = len(values_list)
    count = len(indices_list)

    header = encode_header(2, flags, stream, seq, step0, nsteps)
    payload_header = struct.pack("<HH", count, 0)
    indices_bytes = struct.pack(f"<{count}I", *indices_list)

    flat_values = [v for step in values_list for v in step]
    values_bytes = struct.pack(f"<{len(flat_values)}f", *flat_values)

    return header + payload_header + indices_bytes + values_bytes


def encode_scalar(
    stream: int,
    seq: int,
    step0: int,
    values: list[float] | torch.Tensor,
    flags: int = 0,
) -> bytes:
    """Encode a SCALAR binary frame."""
    values_list = values.tolist() if isinstance(values, torch.Tensor) else list(values)

    nsteps = len(values_list)
    header = encode_header(3, flags, stream, seq, step0, nsteps)
    values_bytes = struct.pack(f"<{nsteps}f", *values_list)
    return header + values_bytes


def encode_weights(
    stream: int,
    seq: int,
    step0: int,
    rows: int,
    cols: int,
    row_stride: int,
    col_stride: int,
    weights: list[float] | torch.Tensor,
    flags: int = 0,
) -> bytes:
    """Encode a WEIGHTS binary frame."""
    if isinstance(weights, torch.Tensor):
        weights_list = weights.flatten().tolist()
    elif isinstance(weights, list) and len(weights) > 0 and isinstance(weights[0], list):
        weights_list = [w for row in weights for w in row]
    else:
        weights_list = list(weights)

    header = encode_header(4, flags, stream, seq, step0, 1)
    payload_header = struct.pack("<HHHH", rows, cols, row_stride, col_stride)
    weights_bytes = struct.pack(f"<{len(weights_list)}f", *weights_list)
    return header + payload_header + weights_bytes


def decode_frame(data: bytes) -> dict[str, Any]:
    """Decode a binary frame into its components and payload data."""
    if len(data) < 16:
        raise ValueError("Frame data too short for header")

    type_val, flags, stream, seq, step0, nsteps, _ = struct.unpack("<BBHIIHH", data[:16])
    payload = data[16:]

    result: dict[str, Any] = {
        "type": type_val,
        "flags": flags,
        "stream": stream,
        "seq": seq,
        "step0": step0,
        "nsteps": nsteps,
    }

    if type_val == 1:
        result["payload"] = payload
    elif type_val == 2:
        if len(payload) < 4:
            raise ValueError("VSAMPLE payload too short")
        count, _ = struct.unpack_from("<HH", payload, 0)
        indices_format = f"<{count}I"
        indices_size = struct.calcsize(indices_format)
        if len(payload) < 4 + indices_size:
            raise ValueError("VSAMPLE indices payload truncated")
        indices = list(struct.unpack_from(indices_format, payload, 4))

        values_offset = 4 + indices_size
        total_floats = nsteps * count
        values_format = f"<{total_floats}f"
        if len(payload) < values_offset + struct.calcsize(values_format):
            raise ValueError("VSAMPLE values payload truncated")
        flat_values = list(struct.unpack_from(values_format, payload, values_offset))

        values = [flat_values[i * count : (i + 1) * count] for i in range(nsteps)]
        result["indices"] = indices
        result["values"] = values
    elif type_val == 3:
        total_floats = nsteps
        format_str = f"<{total_floats}f"
        if len(payload) < struct.calcsize(format_str):
            raise ValueError("SCALAR payload truncated")
        values = list(struct.unpack_from(format_str, payload, 0))
        result["values"] = values
    elif type_val == 4:
        if len(payload) < 8:
            raise ValueError("WEIGHTS payload too short")
        rows, cols, row_stride, col_stride = struct.unpack_from("<HHHH", payload, 0)
        total_weights = rows * cols
        format_str = f"<{total_weights}f"
        if len(payload) < 8 + struct.calcsize(format_str):
            raise ValueError("WEIGHTS payload truncated")
        weights = list(struct.unpack_from(format_str, payload, 8))
        result["rows"] = rows
        result["cols"] = cols
        result["row_stride"] = row_stride
        result["col_stride"] = col_stride
        result["weights"] = weights
    else:
        raise ValueError(f"Unknown frame type: {type_val}")

    return result
