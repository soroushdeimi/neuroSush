"""Learning curves of the spiking sequence memory and of the temporal memory it implements.

Both learn the sequences ABCDE and XBCDY (random SDRs; the two share their middle) over the
same repetitions. For every presented element the script records how many of its columns
burst (were not predicted). The results, with every parameter and seed, go to a JSON file.

Run: ``python experiments/sequence_learning.py --repetitions 12 --out sequence_learning.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.htm.sdr import random_sdr
from neurosush.htm.temporal_memory import TemporalMemory
from neurosush.neurons.axon import Axon
from neurosush.structure.sequence import Neuron, sequence_memory, sequence_timing


class Schedule(Behavior):
    """Fires the listed column neurons on every step: ``{step: [column, ...]}``."""

    order = Order.FIRE

    def __init__(self, schedule: dict[int, list[int]]) -> None:
        self.schedule = schedule

    def initialize(self, group: NeuronGroup) -> None:
        """Start silent."""
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group: NeuronGroup) -> None:
        """Fire this step's columns."""
        spikes = group.state(False, dtype=torch.bool)
        spikes[self.schedule.get(group.net.iteration, [])] = True
        group.spikes = spikes


def make_sequences(columns: int, active: int, length: int, seed: int) -> list[torch.Tensor]:
    """Two sequences that share all but their first and last elements."""
    first, second = (
        random_sdr(columns, active, batch=(length,), generator=torch.Generator().manual_seed(s))
        for s in (seed, seed + 1)
    )
    second[1 : length - 1] = first[1 : length - 1]
    return [first, second]


def spiking_bursts(order: list[torch.Tensor], args: argparse.Namespace) -> list[int]:
    """Bursting columns of every element shown to the spiking layer."""
    schedule: dict[int, list[int]] = {}
    starts: list[tuple[int, torch.Tensor]] = []
    t = 1
    for sequence in order:
        for x in sequence:
            for step in range(t, t + args.window):
                schedule.setdefault(step, []).extend(x.nonzero().flatten().tolist())
            starts.append((t, x))
            t += args.period
        t += 2 * args.period
    net = Network(dtype=torch.float64, seed=args.seed)
    columns = NeuronGroup(net, args.columns, [Schedule(schedule), Axon()])
    layer, _ = sequence_memory(
        net, columns, cells_per_column=args.cells, period=args.period, window=args.window
    )
    fired = torch.zeros(len(starts), layer.size, dtype=torch.bool)
    element = -1
    for _ in range(t):
        net.step()
        while element + 1 < len(starts) and starts[element + 1][0] <= net.iteration:
            element += 1
        if element >= 0 and net.iteration < starts[element][0] + args.period:
            fired[element] |= layer.spikes
    return [
        int(fired[i].view(args.columns, args.cells)[x.nonzero().flatten()].all(-1).sum())
        for i, (_, x) in enumerate(starts)
    ]


def memory_bursts(order: list[torch.Tensor], args: argparse.Namespace) -> list[int]:
    """Bursting columns of every element shown to the temporal memory."""
    tm = TemporalMemory(
        args.columns,
        args.cells,
        activation_threshold=6,
        min_threshold=4,
        max_new_synapses=8,
        max_synapses_per_segment=10,
        initial_permanence=0.21,
        predicted_decrement=0.05,
        seed=args.seed,
    )
    bursts = []
    for sequence in order:
        tm.reset()
        for x in sequence:
            tm.compute(x)
            active = tm.active_cells.view(args.columns, args.cells)[x.nonzero().flatten()]
            bursts.append(int(active.all(-1).sum()))
    return bursts


def main() -> None:
    """Run both models and write the learning curves."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--active", type=int, default=8)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--period", type=int, default=60)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="sequence_learning.json")
    args = parser.parse_args()
    sequences = make_sequences(args.columns, args.active, args.length, args.seed)
    order = [s for _ in range(args.repetitions) for s in sequences]
    start = time.perf_counter()
    spiking = spiking_bursts(order, args)
    reference = memory_bursts(order, args)
    per = 2 * args.length
    result = {
        "parameters": vars(args),
        "timing": vars(sequence_timing(args.period, args.window, Neuron())),
        "spiking": [spiking[i : i + per] for i in range(0, len(spiking), per)],
        "temporal_memory": [reference[i : i + per] for i in range(0, len(reference), per)],
        "seconds": time.perf_counter() - start,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for rep, (a, b) in enumerate(zip(result["spiking"], result["temporal_memory"], strict=True)):
        print(f"{rep:3d}  spiking {a}  temporal memory {b}")


if __name__ == "__main__":
    main()
