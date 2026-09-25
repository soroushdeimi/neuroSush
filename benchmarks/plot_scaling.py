"""Draw the figures of docs/BENCHMARKS.md from the JSON that ``scaling.py`` writes.

Needs matplotlib, which neuroSush itself does not depend on (``pip install matplotlib``).

Run: ``python benchmarks/plot_scaling.py docs/figures/scaling-rtx3090.json docs/figures``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"graph": "#1b7f5b", "eager": "#8fb8a8", "cpu6": "#c9651a", "cpu1": "#e8b27e"}


def series(rows: list[dict], device: str, mode: str, threads: int | None) -> list[dict]:
    """The rows of one configuration, ordered by batch (unbatched first)."""
    chosen = [
        r
        for r in rows
        if r["device"] == device
        and r["mode"] == mode
        and r["threads"] == threads
        and "error" not in r
    ]
    return sorted(chosen, key=lambda r: r["batch"] or 0)


def style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Title, labels and a light grid."""
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def single_network(rows: list[dict], out: Path, machine: str) -> None:
    """Steps per second of one unbatched network, against real time at dt = 1 ms."""
    bars = [
        ("CPU, 1 thread", series(rows, "cpu", "eager", 1)[0], COLORS["cpu1"]),
        ("CPU, 6 threads", series(rows, "cpu", "eager", 6)[0], COLORS["cpu6"]),
        ("GPU, eager", series(rows, "cuda", "eager", None)[0], COLORS["eager"]),
        ("GPU, CUDA graph", series(rows, "cuda", "graph", None)[0], COLORS["graph"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [name for name, _, _ in bars]
    values = [row["steps_per_s"] for _, row, _ in bars]
    ax.barh(names, values, color=[color for _, _, color in bars])
    for y, value in enumerate(values):
        ax.text(value, y, f"  {value:,.0f}", va="center", fontsize=10)
    ax.axvline(1000, color="#555555", linestyle="--", linewidth=1, label="real time at dt = 1 ms")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, max(values) * 1.18)
    ax.invert_yaxis()
    style(ax, "One network: steps per second", "")
    ax.set_xlabel(f"steps per second (higher is better) · {machine}")
    fig.tight_layout()
    fig.savefig(out / "single-network.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def graph_speedup(rows: list[dict], out: Path, machine: str) -> None:
    """GPU steps per second, eager and replayed as a CUDA graph, across batch sizes."""
    eager, graph = series(rows, "cuda", "eager", None), series(rows, "cuda", "graph", None)
    labels = ["1" if r["batch"] is None else f"{r['batch']:,}" for r in eager]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        [r["steps_per_s"] for r in eager],
        width,
        label="eager (net.run)",
        color=COLORS["eager"],
    )
    ax.bar(
        [i + width / 2 for i in x],
        [r["steps_per_s"] for r in graph],
        width,
        label="CUDA graph (GraphStepper)",
        color=COLORS["graph"],
    )
    for i, (e, g) in enumerate(zip(eager, graph, strict=True)):
        ax.text(
            i + width / 2,
            g["steps_per_s"],
            f"{g['steps_per_s'] / e['steps_per_s']:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(list(x), labels)
    ax.set_xlabel(f"batch size (samples per step) · {machine}")
    style(ax, "GPU: CUDA graphs remove the per-step overhead", "steps per second")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gpu-graph-speedup.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def throughput(rows: list[dict], out: Path, machine: str) -> None:
    """Sample-steps per second against batch size, for every configuration (log-log)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, (device, mode, threads), color in [
        ("GPU, CUDA graph", ("cuda", "graph", None), COLORS["graph"]),
        ("GPU, eager", ("cuda", "eager", None), COLORS["eager"]),
        ("CPU, 6 threads", ("cpu", "eager", 6), COLORS["cpu6"]),
        ("CPU, 1 thread", ("cpu", "eager", 1), COLORS["cpu1"]),
    ]:
        rs = series(rows, device, mode, threads)
        ax.plot(
            [r["batch"] or 1 for r in rs],
            [r["sample_steps_per_s"] for r in rs],
            marker="o",
            color=color,
            label=label,
            linewidth=2,
        )
    ax.set_xscale("log", base=2)
    batches = [r["batch"] or 1 for r in series(rows, "cuda", "graph", None)]
    ax.set_xticks(batches, [f"{b:,}" for b in batches])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_xlabel(f"batch size (samples per step) · {machine}")
    style(ax, "Throughput grows with the batch", "sample-steps per second")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "throughput.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    """Read the results and write the three figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = data["results"]
    machine = f"{data['machine']['gpu']} machine, PyTorch {data['machine']['torch'].split('+')[0]}"
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "neurosush"})
    args.out.mkdir(parents=True, exist_ok=True)
    single_network(rows, args.out, machine)
    graph_speedup(rows, args.out, machine)
    throughput(rows, args.out, machine)


if __name__ == "__main__":
    main()
