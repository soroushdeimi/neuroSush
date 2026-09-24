"""Compare benchmark results with a baseline; exit 1 on a regression.

A case regresses when its normalized rate falls below ``(1 - tolerance)`` times the
baseline's. Cases missing on either side are reported, not failed.

Run: ``python benchmarks/compare.py benchmarks/baseline.json benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def ratios(baseline: dict, current: dict) -> dict[str, float | None]:
    """Current over baseline normalized rate per case (None when either side lacks it)."""
    base, now = baseline.get("cases", {}), current.get("cases", {})
    return {
        name: now[name]["normalized"] / base[name]["normalized"]
        if name in base and name in now
        else None
        for name in sorted(base.keys() | now.keys())
    }


def regressions(changes: dict[str, float | None], tolerance: float) -> list[str]:
    """Cases whose ratio is below ``1 - tolerance``."""
    return [name for name, ratio in changes.items() if ratio is not None and ratio < 1 - tolerance]


def table(changes: dict[str, float | None], tolerance: float) -> str:
    """A Markdown table of the changes."""
    rows = ["| case | change | status |", "|---|---|---|"]
    for name, ratio in changes.items():
        if ratio is None:
            rows.append(f"| {name} | n/a | missing on one side |")
        else:
            status = "regression" if ratio < 1 - tolerance else "ok"
            rows.append(f"| {name} | {ratio - 1:+.1%} | {status} |")
    return "\n".join(rows)


def main() -> int:
    """Print the comparison; return 1 if any case regressed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.3)
    args = parser.parse_args()
    current = json.loads(args.current.read_text(encoding="utf-8"))
    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; nothing to compare.")
        return 0
    changes = ratios(json.loads(args.baseline.read_text(encoding="utf-8")), current)
    print(table(changes, args.tolerance))
    return 1 if regressions(changes, args.tolerance) else 0


if __name__ == "__main__":
    sys.exit(main())
