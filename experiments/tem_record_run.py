"""Export recorded TEM runs matching the shared viewer contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurosush.tem import record_run, write_run


def main() -> None:
    """Parse arguments and export recorded run JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=3)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--label", type=str, default="learned structure")
    parser.add_argument("--note", type=str, default="one seed; no claim of grid cells")
    args = parser.parse_args()

    run = record_run(
        args.checkpoint,
        worlds=args.worlds,
        size=args.size,
        steps=args.steps,
        units=args.units,
        seed=args.seed,
        train_seed=args.train_seed,
        label=args.label,
        note=args.note,
        probe_path=args.probe,
    )
    write_run(args.out, run)
    print(f"Recorded run exported to {args.out} (size: {args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
