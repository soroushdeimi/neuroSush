"""Digitize published TEM figure crops for themed charts."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent
FIG_DIR = SCRIPTS / "site_assets" / "story-tem" / "paper-figures"
FACTS_PATH = SCRIPTS / "site_assets" / "story-tem" / "facts.json"
OUT_PATH = FIG_DIR / "native-data.json"

# jet lookup table for inverting heatmap colors to 0..1
_JET_LUT = (matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255).astype(int)


def _invert_jet(rgb):
    """Map an RGB pixel to its nearest jet colormap value in 0..1."""
    d = ((_JET_LUT - np.array(rgb, dtype=int)) ** 2).sum(axis=1)
    return int(np.argmin(d)) / 255


def _load_facts():
    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    return {f["id"]: f["value"] for f in facts}


def _open(name):
    path = FIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing source figure: {path}")
    return Image.open(path).convert("RGB")


def _is_blue(p):
    r, g, b = p
    return b > 120 and b - r > 50


def _is_red(p):
    r, g, b = p
    return r > 150 and g < 120 and b < 120


def _is_dark(p):
    r, g, b = p
    return r < 120 and g < 120 and b < 120


def _is_errgray(p):
    r, g, b = p
    return abs(r - g) < 25 and abs(g - b) < 25 and 60 < r < 160


def _bar_runs(px, x0, x1, ytop, ybot, color):
    """Contiguous column runs of a bar color, merged across antialias gaps."""
    test = _is_blue if color == "blue" else _is_red
    cols = [sum(1 for y in range(ytop, ybot) if test(px[x, y])) for x in range(x0, x1)]
    runs, start = [], None
    for i, n in enumerate(cols):
        if n > 0 and start is None:
            start = i
        if n == 0 and start is not None:
            runs.append((x0 + start, x0 + i))
            start = None
    if start is not None:
        runs.append((x0 + start, x1))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _bar_top(px, a, b, ytop, ybot, color):
    test = _is_blue if color == "blue" else _is_red
    tops = [y for y in range(ytop, ybot) for x in range(a + 2, b - 2) if test(px[x, y])]
    return min(tops)


def _err_cap(px, cx, top, ytop):
    """Top of the contiguous dark errorbar just above a bar top, or None."""
    darks = [y for y in range(ytop, top) if _is_dark(px[cx, y])]
    if not darks:
        return None
    run = [top - 1]
    for y in sorted(darks, reverse=True):
        if run and y == run[-1] - 1:
            run.append(y)
        elif y < top - 1:
            break
    return min(run)


def _fig11_line():
    """Grid emergence: fraction of grid cells vs training iterations."""
    px = _open("grid-emergence.png").load()
    y0, y100 = 348, 49  # 0% and 100%
    xs = [151, 229, 307, 384, 461, 539, 616]
    iters = [0, 4000, 8000, 12000, 16000, 20000, 24000]
    facts = _load_facts()
    points = []
    for x, it in zip(xs, iters):
        line = [y for y in range(49, 349) if _is_line(px[x, y])]
        yb = sum(line) / len(line)
        val = (y0 - yb) / (y0 - y100) * 100
        p = {"x": it, "y": round(val, 1)}
        # errorbar extents from dark gray, excluding axis rows
        err = [y for y in range(49, 349)
               if not (y <= 51 or y >= 346)
               and any(_is_errgray(px[x + dx, y]) for dx in (-2, -1, 0, 1, 2))]
        if err:
            p["high"] = round((y0 - min(err)) / (y0 - y100) * 100, 1)
            p["low"] = round((y0 - max(err)) / (y0 - y100) * 100, 1)
        points.append(p)
    # endpoint mean/sd are exact from the paper table
    last = points[-1]
    last["y"] = facts["grid_mean"]
    last["sd"] = facts["grid_sd"]
    return {
        "kind": "line",
        "title": "Grid-cell fraction during training",
        "source": "Kawahara & Fujisawa (2025 preprint), Fig. 3C",
        "provenance": "digitized",
        "approx": True,
        "xLabel": "training iterations",
        "yLabel": "percent of MEC II units",
        "series": [{"name": "grid fraction", "color": "blue", "points": points}],
        "notes": "Endpoint (24,000) mean/SD are exact from Table 2; other points are pixel-digitized.",
    }


def _is_line(p):
    r, g, b = p
    return b > 90 and b - r > 40 and r < 120


def _fig12_histograms():
    """CA1 spatial-information histograms, three conditions."""
    px = _open("ca1-theta.png").load()
    rects = [(75, 457, 268, 80), (558, 940, 270, 80), (1045, 1427, 268, 200)]
    groups = ["Before training", "After training", "Before training without theta"]
    bins = ["0-0.74", "0.74-1.49", "1.49-2.23", "2.23-2.98", "2.98-3.72",
            "3.72-4.47", "4.47-5.21", "5.21-5.96", "5.96-6.70", "6.70-7.45"]
    series = []
    for (x0, x1, yb, ymax), name in zip(rects, groups):
        runs = _bar_runs(px, x0, x1, 58, yb, "blue")
        w = (x1 - x0) / 10
        vals = [0.0] * 10
        for a, b in runs:
            idx = min(9, max(0, int(((a + b) / 2 - x0) // w)))
            top = _bar_top(px, a, b, 58, yb, "blue")
            vals[idx] = round((yb - top) / (yb - 57) * ymax, 1)
        points = [{"x": bins[i], "y": vals[i]} for i in range(10)]
        series.append({"name": name, "color": "blue", "points": points})
    return {
        "kind": "histograms",
        "title": "CA1 spatial-information distributions",
        "source": "Kawahara & Fujisawa (2025 preprint), Fig. 4C",
        "provenance": "digitized",
        "approx": True,
        "xLabel": "spatial information (bits)",
        "yLabel": "count",
        "series": series,
        "notes": "Bar heights digitized from pixels; error bars omitted (not robustly detectable).",
    }


def _sample_map(px, x, y, w, h, n=24):
    """Sample a w x h pixel block into an n x n grid of jet-inverted 0..1 values."""
    values = []
    for i in range(n):
        row = []
        py = y + (h * i) // n
        for j in range(n):
            pxl = x + (w * j) // n
            row.append(round(_invert_jet(px[pxl, py]), 3))
        values.append(row)
    return values


def _fig13_heatmaps():
    """Remapping: example MEC II and CA1 firing maps, two environments."""
    px = _open("remapping.png").load()
    mec = [(184, 96, 59, 59), (313, 96, 59, 59), (441, 96, 59, 59)]
    mec_b = [(184, 201, 59, 59), (313, 201, 59, 59), (441, 201, 59, 59)]
    ca1 = [(1277, 185, 60, 60), (1408, 185, 60, 60), (1541, 185, 60, 60)]
    ca1_b = [(1277, 318, 60, 60), (1408, 318, 60, 60), (1541, 318, 60, 60)]
    groups = [
        {"name": "MEC II", "maps": [
            {"label": f"cell {i+1} Env A", "values": _sample_map(px, *c)} for i, c in enumerate(mec)],
            "maps_b": [{"label": f"cell {i+1} Env B", "values": _sample_map(px, *c)} for i, c in enumerate(mec_b)]},
        {"name": "CA1", "maps": [
            {"label": f"cell {i+1} Env A", "values": _sample_map(px, *c)} for i, c in enumerate(ca1)],
            "maps_b": [{"label": f"cell {i+1} Env B", "values": _sample_map(px, *c)} for i, c in enumerate(ca1_b)]},
    ]
    # merge Env A and Env B maps per group
    out = []
    for g in groups:
        maps = g["maps"] + g["maps_b"]
        out.append({"name": g["name"], "maps": maps})
    return {
        "kind": "heatmaps",
        "title": "Field remapping across environments",
        "source": "Kawahara & Fujisawa (2025 preprint), Figs. 5A and 6A",
        "provenance": "digitized",
        "approx": True,
        "xLabel": "position",
        "yLabel": "position",
        "groups": out,
        "notes": "Digitized normalized firing maps (jet-inverted), not raw spikes or absolute Hz.",
    }


def _fig16_heatmaps():
    """Predictive MEC III: example firing maps, current vs next position."""
    px = _open("predictive-mec-iii.png").load()
    cols = [226, 349, 471]
    rows = [("MEC II current", 155, 60), ("MEC III current", 336, 64), ("MEC III next", 438, 62)]
    groups = []
    for i, c in enumerate(cols):
        maps = []
        for label, y, h in rows:
            maps.append({"label": label, "values": _sample_map(px, c, y, 60, h)})
        groups.append({"name": f"cell {i+1}", "maps": maps})
    return {
        "kind": "heatmaps",
        "title": "MEC III firing maps aligned to current and next position",
        "source": "Kawahara & Fujisawa (2025 preprint), Fig. 8A",
        "provenance": "digitized",
        "approx": True,
        "xLabel": "position",
        "yLabel": "position",
        "groups": groups,
        "notes": "Digitized normalized firing maps (jet-inverted), not raw spikes or absolute Hz.",
    }


def _fig14_bars():
    """Ablations: exact grid-cell fractions from Table 2."""
    facts = _load_facts()
    conds = [
        ("Baseline", "grid_mean", "grid_sd"),
        ("No feedback", "no_feedback", "no_feedback_sd"),
        ("No theta", "no_theta", "no_theta_sd"),
        ("No STDP", "no_stdp", "no_stdp_sd"),
        ("No sparsity", "no_sparsity", "no_sparsity_sd"),
        ("MSE only", "mse", "mse_sd"),
        ("No gain", "no_gain", "no_gain_sd"),
    ]
    points = [{"x": name, "y": facts[m], "sd": facts[s]} for name, m, s in conds]
    return {
        "kind": "bars",
        "title": "Grid-cell classification under ablations",
        "source": "Kawahara & Fujisawa (2025 preprint), Table 2",
        "provenance": "reported",
        "approx": False,
        "xLabel": "condition",
        "yLabel": "percent of MEC II units",
        "series": [{"name": "grid fraction", "color": "accent", "points": points}],
        "notes": "Exact means and SDs from Table 2 (four runs per condition).",
    }


def _fig15_bars():
    """Sensory ambiguity: grid fraction vs number of sensory neurons."""
    px = _open("sensory-ambiguity.png").load()
    y0, y50 = 502, 63  # 0% and 50%
    centers = [260, 424, 589, 753, 918]
    neurons = [10, 20, 40, 60, 80]
    points = []
    for cx, n in zip(centers, neurons):
        blues = [y for y in range(65, 500) for x in range(cx - 25, cx + 25) if _is_blue(px[x, y])]
        if not blues:
            # no colored bar detected; record the detected baseline, not an invented zero
            points.append({"x": n, "y": 0.0, "note": "no bar detected at this x"})
            continue
        top = min(blues)
        val = (y0 - top) / (y0 - y50) * 50
        p = {"x": n, "y": round(val, 1)}
        cap = _err_cap(px, cx, top, 65)
        if cap is not None:
            p["high"] = round((y0 - cap) / (y0 - y50) * 50, 1)
        points.append(p)
    return {
        "kind": "bars",
        "title": "Grid-cell fraction vs sensory ambiguity",
        "source": "Kawahara & Fujisawa (2025 preprint), Fig. 7",
        "provenance": "digitized",
        "approx": True,
        "xLabel": "sensory neurons",
        "yLabel": "percent of MEC II units",
        "series": [{"name": "grid fraction", "color": "blue", "points": points}],
        "notes": "Bar heights and error caps digitized; the 80-neuron bar is absent in the source.",
    }


def _fig17_bars():
    """Phase precession vs locking, MEC II and MEC III."""
    px = _open("phase-precession-locking.png").load()
    y0, y100 = 218, 37  # 0% and 100%
    facts = _load_facts()
    # (center, color, module, behavior)
    bars = [
        (216, "blue", "MEC II", "precession"),
        (254, "red", "MEC II", "locking"),
        (350, "blue", "MEC III", "precession"),
        (389, "red", "MEC III", "locking"),
    ]
    by_key = {}
    for cx, color, module, behavior in bars:
        test = _is_blue if color == "blue" else _is_red
        tops = [y for y in range(40, 218) for x in range(cx - 15, cx + 15) if test(px[x, y])]
        top = min(tops)
        val = (y0 - top) / (y0 - y100) * 100
        p = {"x": behavior, "y": round(val, 1)}
        cap = _err_cap(px, cx, top, 40)
        if cap is not None:
            p["high"] = round((y0 - cap) / (y0 - y100) * 100, 1)
        by_key[(module, behavior)] = p
    # exact reported means for the two dominant behaviors
    by_key[("MEC II", "precession")]["y"] = facts["precession_mean"]
    by_key[("MEC III", "locking")]["y"] = facts["locking_mean"]
    series = [
        {"name": "MEC II", "color": "blue",
         "points": [by_key[("MEC II", "precession")], by_key[("MEC II", "locking")]]},
        {"name": "MEC III", "color": "accent",
         "points": [by_key[("MEC III", "precession")], by_key[("MEC III", "locking")]]},
    ]
    return {
        "kind": "bars",
        "title": "Phase precession vs theta locking",
        "source": "Kawahara & Fujisawa (2025 preprint), Fig. 13C",
        "provenance": "digitized",
        "approx": True,
        "xLabel": "behavior",
        "yLabel": "percent of grid cells",
        "series": series,
        "notes": "MEC II precession (80.2) and MEC III locking (86.7) are exact; others digitized.",
    }


_BUILDERS = {
    "11": _fig11_line,
    "12": _fig12_histograms,
    "13": _fig13_heatmaps,
    "14": _fig14_bars,
    "15": _fig15_bars,
    "16": _fig16_heatmaps,
    "17": _fig17_bars,
}


def build():
    return {key: fn() for key, fn in _BUILDERS.items()}


def main(argv):
    data = build()
    text = json.dumps(data, indent=2, sort_keys=True)
    if "--check" in argv:
        if not OUT_PATH.is_file():
            print("native-data.json missing; run without --check to generate")
            return 1
        existing = OUT_PATH.read_text(encoding="utf-8")
        if existing != text + "\n":
            print("native-data.json differs from regenerated output")
            return 1
        print("native-data.json is reproducible")
        return 0
    OUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
