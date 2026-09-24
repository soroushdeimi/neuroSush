"""Learn high-order sequences with a spatial pooler, temporal memory and a classifier.

The sequences ``ABCDE`` and ``XBCDY`` share their middle ``BCD``: after ``D`` only the
first symbol tells whether ``E`` or ``Y`` comes next. Symbols are encoded as SDRs, the
spatial pooler turns them into column codes, the temporal memory learns the transitions
in context, and a softmax classifier reads the next symbol from the active cells. At first
every column bursts (anomaly 1); once the sequences are learned the anomaly drops to 0
after the first symbol and both endings are predicted correctly.

Run: ``python examples/sequence_prediction.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from neurosush.htm.classifier import SDRClassifier
from neurosush.htm.encoders import CategoryEncoder
from neurosush.htm.spatial_pooler import SpatialPooler
from neurosush.htm.temporal_memory import TemporalMemory

SEQUENCES = ("ABCDE", "XBCDY")
SYMBOLS = sorted(set("".join(SEQUENCES)))


@dataclass
class Result:
    """Per-epoch anomaly and the final predictions of every sequence's last symbol."""

    anomaly: list[float]
    endings: dict[str, str]

    @property
    def correct(self) -> bool:
        """Both endings predicted from their context."""
        return all(self.endings[s[:-1]] == s[-1] for s in SEQUENCES)


def run(seed: int = 0, epochs: int = 15) -> Result:
    """Train for ``epochs`` passes over the sequences and report what was learned."""
    encoder = CategoryEncoder(400, 20, len(SYMBOLS), seed=seed)
    pooler = SpatialPooler(400, 512, potential_radius=400, density=0.04, seed=seed)
    codes = encoder.encode(torch.arange(len(SYMBOLS)))
    for _ in range(10):  # settle the column code of every symbol, then keep it fixed
        pooler.compute(codes)
    columns = pooler.compute(codes, learn=False)
    memory = TemporalMemory(
        512,
        8,
        activation_threshold=12,
        min_threshold=8,
        initial_permanence=0.51,
        max_new_synapses=20,
        predicted_decrement=0.05,
        seed=seed,
    )
    classifier = SDRClassifier(memory.n_cells, len(SYMBOLS), lr=0.5)
    anomaly = []
    for _ in range(epochs):
        scores = []
        for sequence in SEQUENCES:
            memory.reset()
            for current, following in pairwise(sequence):
                cells = memory.compute(columns[SYMBOLS.index(current)])
                if current != sequence[0]:
                    scores.append(memory.anomaly)
                classifier.learn(cells, SYMBOLS.index(following))
            memory.compute(columns[SYMBOLS.index(sequence[-1])])
            scores.append(memory.anomaly)
        anomaly.append(sum(scores) / len(scores))
    endings = {}
    for sequence in SEQUENCES:
        memory.reset()
        for symbol in sequence[:-1]:
            cells = memory.compute(columns[SYMBOLS.index(symbol)], learn=False)
        endings[sequence[:-1]] = SYMBOLS[int(classifier.predict(cells))]
    return Result(anomaly, endings)


def main() -> None:
    """Print the anomaly curve and the predicted endings."""
    result = run()
    for epoch, score in enumerate(result.anomaly, 1):
        print(f"epoch {epoch:2d}  anomaly {score:.2f}")
    for context, symbol in result.endings.items():
        print(f"{context} -> {symbol}")


if __name__ == "__main__":
    main()
