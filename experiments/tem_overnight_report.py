"""Aggregate overnight TEM experiment results and evaluate pass/fail criteria."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, cast

# t-distribution critical values for 95% CI (two-tailed alpha=0.05)
# by degrees of freedom (df = n - 1)
T_CRITICAL: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def get_t_value(df: int) -> float:
    """Return t critical value for 95% CI given degrees of freedom."""
    if df <= 0:
        return 0.0
    if df in T_CRITICAL:
        return T_CRITICAL[df]
    if df > 30:
        return 1.96
    keys = sorted(T_CRITICAL.keys())
    for k in reversed(keys):
        if df >= k:
            return T_CRITICAL[k]
    return 1.96


def load_results(directory: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Load and filter overnight run JSON files across variants."""
    variants = ["v0", "v1", "v2", "v3v0", "v3v1"]
    aggregated: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    skipped: list[str] = []

    if not directory.exists():
        return aggregated, skipped

    for path in sorted(directory.glob("*.json")):
        name = path.name
        if "-s" not in name:
            continue
        variant_part = name.split("-s")[0]
        if variant_part not in variants:
            continue

        try:
            data = json.loads(path.read_text())
        except Exception as e:
            skipped.append(f"{path}: invalid json ({e})")
            continue

        status = data.get("status")
        if status != "complete":
            skipped.append(f"{path}: {status}")
            continue

        aggregated[variant_part].append(data)

    return aggregated, skipped


def analyze_variant(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across completed seeds for a variant and evaluate PASS rules."""
    if not runs:
        return {"n": 0}

    seed_metrics = []
    for run in runs:
        seed = run.get("seed")
        evals = run.get("evaluations", [])
        if not evals:
            continue

        accs = [float(e.get("accuracy", 0.0) or 0.0) for e in evals]
        freqs = [float(e.get("frequency", 0.0) or 0.0) for e in evals]
        sens = [float(e.get("sensory_only", 0.0) or 0.0) for e in evals]
        conds = [float(e.get("conditional", 0.0) or 0.0) for e in evals]
        node_conds = [float(e.get("node_conditional", 0.0) or 0.0) for e in evals]
        ces = [float(e.get("generative_ce", 0.0) or 0.0) for e in evals]
        r2s = [float(e.get("position_r2", 0.0) or 0.0) for e in evals]
        nulls = [float(e.get("shift_null_p95", 0.0) or 0.0) for e in evals]

        best_bases = [
            max(f, s, c, nc) for f, s, c, nc in zip(freqs, sens, conds, node_conds, strict=True)
        ]

        seed_metrics.append(
            {
                "seed": seed,
                "accuracy": fmean(accs),
                "frequency": fmean(freqs),
                "best_baseline": fmean(best_bases),
                "generative_ce": fmean(ces),
                "position_r2": fmean(r2s),
                "shift_null_p95": fmean(nulls),
                "max_grad": float(run.get("max_grad", 0.0) or 0.0),
                "p95_grad": float(run.get("p95_grad", 0.0) or 0.0),
            }
        )

    n = len(seed_metrics)
    if n == 0:
        return {"n": 0}

    acc_vals: list[float] = [float(cast(float, m["accuracy"])) for m in seed_metrics]
    freq_vals: list[float] = [float(cast(float, m["frequency"])) for m in seed_metrics]
    base_vals: list[float] = [float(cast(float, m["best_baseline"])) for m in seed_metrics]
    ce_vals: list[float] = [float(cast(float, m["generative_ce"])) for m in seed_metrics]
    r2_vals: list[float] = [float(cast(float, m["position_r2"])) for m in seed_metrics]
    null_vals: list[float] = [float(cast(float, m["shift_null_p95"])) for m in seed_metrics]
    grad_vals: list[float] = [float(cast(float, m["max_grad"])) for m in seed_metrics]
    p95_grad_vals: list[float] = [float(cast(float, m["p95_grad"])) for m in seed_metrics]

    acc_mean = fmean(acc_vals)
    acc_sd = stdev(acc_vals) if n > 1 else 0.0
    if n < 2:
        acc_ci = "n=1, no CI"
        acc_ci_lower = acc_mean
    else:
        se = acc_sd / math.sqrt(n)
        t_val = get_t_value(n - 1)
        me = t_val * se
        acc_ci = f"[{acc_mean - me:.4f}, {acc_mean + me:.4f}]"
        acc_ci_lower = acc_mean - me

    base_mean = fmean(base_vals)
    freq_mean = fmean(freq_vals)
    gap_mean = acc_mean - base_mean
    ce_mean = fmean(ce_vals)
    r2_mean = fmean(r2_vals)
    null_mean = fmean(null_vals)
    grad_mean = fmean(grad_vals)
    p95_grad_mean = fmean(p95_grad_vals)

    # PASS rule check:
    # 1. mean accuracy >= best_baseline + 0.05
    # 2. every seed's accuracy > its frequency
    # 3. CI lower bound of accuracy > mean frequency (or if n=1, accuracy > frequency)
    # 4. R2 claim only if mean R2 > mean null p95

    cond1 = acc_mean >= base_mean + 0.05
    cond2 = all(
        float(cast(float, m["accuracy"])) > float(cast(float, m["frequency"])) for m in seed_metrics
    )
    cond3 = acc_ci_lower > freq_mean

    r2_claimed = r2_mean > null_mean

    passed = (n >= 3) and cond1 and cond2 and cond3
    if n < 3:
        verdict = f"INCONCLUSIVE (n={n} seeds, need 3)"
    else:
        verdict = "PASS" if passed else "FAIL"
        if passed and r2_claimed:
            verdict += " (R2 claimed)"

    return {
        "n": n,
        "seeds": [m["seed"] for m in seed_metrics],
        "accuracy_mean": acc_mean,
        "accuracy_sd": acc_sd,
        "accuracy_ci": acc_ci,
        "accuracy_ci_lower": acc_ci_lower,
        "best_baseline_mean": base_mean,
        "frequency_mean": freq_mean,
        "gap_mean": gap_mean,
        "ce_mean": ce_mean,
        "position_r2_mean": r2_mean,
        "shift_null_p95_mean": null_mean,
        "max_grad_mean": grad_mean,
        "p95_grad_mean": p95_grad_mean,
        "r2_claimed": r2_claimed,
        "verdict": verdict,
    }


def generate_report(directory: Path) -> str:
    """Generate markdown report for overnight experiments."""
    aggregated, skipped = load_results(directory)
    variants = ["v0", "v1", "v2", "v3v0", "v3v1"]

    lines = [
        "# TEM Overnight Experiment Report",
        "",
        "Aggregated metrics across overnight experiment runs.",
        "",
    ]

    if skipped:
        lines.append("## Skipped / Incomplete Files")
        for s in skipped:
            lines.append(f"- `{s}`")
        lines.append("")

    lines.extend(
        [
            "## Summary Table",
            "",
            "| Variant | Seeds ($n$) | Accuracy (mean ± SD) | CI (95%) | "
            "Best Baseline | Gap | CE | Position R2 | Shift-Null p95 | "
            "Max Grad | p95 grad | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )

    for var in variants:
        runs = aggregated[var]
        res = analyze_variant(runs)
        var_label = f"{var}*" if var.startswith("v3") else var
        if res["n"] == 0:
            lines.append(f"| {var_label} | 0 | - | - | - | - | - | - | - | - | - | NO DATA |")
            continue

        acc_str = f"{res['accuracy_mean']:.4f} ± {res['accuracy_sd']:.4f}"
        ci_str = res["accuracy_ci"]
        base_str = f"{res['best_baseline_mean']:.4f}"
        gap_str = f"{res['gap_mean']:.4f}"
        ce_str = f"{res['ce_mean']:.4f}"
        r2_str = f"{res['position_r2_mean']:.4f}"
        null_str = f"{res['shift_null_p95_mean']:.4f}"
        grad_str = f"{res['max_grad_mean']:.1f}"
        p95_grad_str = f"{res['p95_grad_mean']:.1f}"
        verdict = res["verdict"]

        lines.append(
            f"| {var_label} | {res['n']} | {acc_str} | {ci_str} | "
            f"{base_str} | {gap_str} | {ce_str} | {r2_str} | "
            f"{null_str} | {grad_str} | {p95_grad_str} | {verdict} |"
        )

    lines.extend(
        [
            "",
            "### Notes",
            "- Variants marked with `*` (`v3v0*`, `v3v1*`) have structure imposed (rotation).",
            "- 95% CI computed via t-interval over seeds ($n$ small; `'n=1, no CI'` when $n=1$).",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point for generating overnight report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path("experiments/results/overnight"), help="Results directory"
    )
    parser.add_argument("--out", type=Path, help="Write markdown report to file")
    args = parser.parse_args()

    report = generate_report(args.dir)
    print(report, end="")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
