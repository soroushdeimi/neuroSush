"""Bounded shared mappings for one producer and independent display readers."""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import torch

from studio.stream.frames import encode_spikes

_HEADER = 16


class Ring:
    """Publish fixed-size slots without waiting for a reader or socket.

    The file is shared through mmap in both processes. Each reader owns its cursor;
    overwritten slots are display losses and never affect recorded traces.
    """

    def __init__(
        self, path: Path, *, create: bool = False, slots: int = 128, slot_size: int = 524288
    ) -> None:
        self.slots, self.slot_size = slots, slot_size
        self.file = path.open("w+b" if create else "r+b")
        if create:
            self.file.truncate(_HEADER + slots * slot_size)
        self.memory = mmap.mmap(self.file.fileno(), 0)
        if create:
            # Touch pages before stepping so publication needs no page allocation.
            self.memory[:] = bytes(len(self.memory))
        self.cursor = 0

    @property
    def head(self) -> int:
        """Return the last published slot number."""
        return int(struct.unpack_from("<Q", self.memory)[0])

    def write(self, frame: bytes) -> bool:
        """Publish into a generation-stamped slot without acquiring reader locks."""
        if len(frame) > self.slot_size - 16:
            raise ValueError("Frame exceeds ring slot size")
        head = self.head
        offset = _HEADER + (head % self.slots) * self.slot_size
        struct.pack_into("<Q", self.memory, offset, 0)
        struct.pack_into("<I", self.memory, offset + 8, len(frame))
        self.memory[offset + 16 : offset + 16 + len(frame)] = frame
        struct.pack_into("<Q", self.memory, offset, head + 1)
        struct.pack_into("<Q", self.memory, 0, head + 1)
        return True

    def read(self) -> tuple[list[bytes], int]:
        """Copy published slots, rejecting any overwritten during the copy."""
        head = self.head
        lost = max(0, head - self.cursor - self.slots)
        self.cursor = max(self.cursor, head - self.slots)
        frames = []
        while self.cursor < head:
            offset = _HEADER + (self.cursor % self.slots) * self.slot_size
            expected = self.cursor + 1
            generation = struct.unpack_from("<Q", self.memory, offset)[0]
            if generation == expected:
                size = struct.unpack_from("<I", self.memory, offset + 8)[0]
                frame = bytes(self.memory[offset + 16 : offset + 16 + size])
                if struct.unpack_from("<Q", self.memory, offset)[0] == expected:
                    frames.append(frame)
                else:
                    lost += 1
            else:
                lost += 1
            self.cursor += 1
        return frames, lost

    def close(self) -> None:
        """Release this process's mapping."""
        self.memory.close()
        self.file.close()


def materialize(frame: bytes) -> bytes:
    """Pack byte-per-neuron transport copies in the receiving process."""
    if len(frame) < 20 or frame[0] != 129:
        return frame
    _, flags, stream, seq, step0, count, _ = struct.unpack_from("<BBHIIHH", frame)
    size = struct.unpack_from("<I", frame, 16)[0]
    data = torch.frombuffer(bytearray(frame[20:]), dtype=torch.uint8).reshape(count, size)
    if size % 8:
        data = torch.nn.functional.pad(data, (0, -size % 8))
    packed = data[:, 0::8].clone()
    for bit in range(1, 8):
        packed.bitwise_or_(data[:, bit::8] << bit)
    return encode_spikes(stream, seq, step0, count, bytes(packed.flatten().tolist()), flags)


def remove_ring(root: Path) -> None:
    """Release the backing file after its producer has exited."""
    (root.parent / ".streams" / root.name / "stream.ring").unlink(missing_ok=True)
