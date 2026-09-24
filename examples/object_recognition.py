"""Columns that vote recognize objects in fewer touches (Lewis et al. 2019, Fig. 5).

A library holds random objects: one of ``features`` feature ids at every location of a
``5 x 5`` grid. Every column touches the object at its own location, keeps the
``(object, location)`` hypotheses consistent with what it sensed, and moves one step at
random; after every touch the columns vote on which objects remain. The script prints the
mean number of touches until exactly one object is left, for 1 to 8 columns.

Run: ``python examples/object_recognition.py``.
"""

from __future__ import annotations

import torch

from neurosush.htm.objects import ColumnEnsemble, ObjectLibrary


def touches_to_recognize(
    library: ObjectLibrary, obj: int, columns: int, generator: torch.Generator, limit: int = 50
) -> int:
    """Touches until ``columns`` voting columns single out ``obj``."""
    height, width = library.shape
    ensemble = ColumnEnsemble(library, columns)
    start = torch.randperm(height * width, generator=generator)[:columns].tolist()
    locations = [(i // width, i % width) for i in start]
    for touch in range(1, limit + 1):
        ensemble.sense([int(library.features[obj, y, x]) for y, x in locations])
        if ensemble.recognized() is not None:
            return touch
        moves = torch.randint(-1, 2, (columns, 2), generator=generator).tolist()
        ensemble.move([(dy, dx) for dy, dx in moves])
        locations = [
            ((y + dy) % height, (x + dx) % width)
            for (y, x), (dy, dx) in zip(locations, moves, strict=True)
        ]
    raise RuntimeError(f"object {obj} not recognized in {limit} touches")


def run(
    objects: int = 100, features: int = 30, columns: tuple[int, ...] = (1, 2, 4, 8), seed: int = 0
) -> dict[int, float]:
    """Mean touches to recognize every object, for each number of columns."""
    generator = torch.Generator().manual_seed(seed)
    library = ObjectLibrary.random(objects, (5, 5), features, generator=generator)
    return {
        count: sum(touches_to_recognize(library, obj, count, generator) for obj in range(objects))
        / objects
        for count in columns
    }


def main() -> None:
    """Print the mean number of touches per number of columns."""
    for columns, touches in run().items():
        print(f"{columns} column(s): {touches:.2f} touches")


if __name__ == "__main__":
    main()
