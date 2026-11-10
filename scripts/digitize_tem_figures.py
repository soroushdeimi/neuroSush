"""Digitize published TEM figure crops for themed charts."""

import json
import struct
import sys
import zlib
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
FIG_DIR = SCRIPTS / "site_assets" / "story-tem" / "paper-figures"
FACTS_PATH = SCRIPTS / "site_assets" / "story-tem" / "facts.json"
OUT_PATH = FIG_DIR / "native-data.json"

# jet lookup table for inverting heatmap colors to 0..1.
# Frozen from matplotlib.colormaps["jet"] at 256 stops (RGB, 0..255) so the
# digitizer stays stdlib-only while reproducing the exact original values.
_JET_LUT = (
    (0, 0, 127),
    (0, 0, 132),
    (0, 0, 136),
    (0, 0, 141),
    (0, 0, 145),
    (0, 0, 150),
    (0, 0, 154),
    (0, 0, 159),
    (0, 0, 163),
    (0, 0, 168),
    (0, 0, 172),
    (0, 0, 177),
    (0, 0, 182),
    (0, 0, 186),
    (0, 0, 191),
    (0, 0, 195),
    (0, 0, 200),
    (0, 0, 204),
    (0, 0, 209),
    (0, 0, 213),
    (0, 0, 218),
    (0, 0, 222),
    (0, 0, 227),
    (0, 0, 232),
    (0, 0, 236),
    (0, 0, 241),
    (0, 0, 245),
    (0, 0, 250),
    (0, 0, 254),
    (0, 0, 255),
    (0, 0, 255),
    (0, 0, 255),
    (0, 0, 255),
    (0, 4, 255),
    (0, 8, 255),
    (0, 12, 255),
    (0, 16, 255),
    (0, 20, 255),
    (0, 24, 255),
    (0, 28, 255),
    (0, 32, 255),
    (0, 36, 255),
    (0, 40, 255),
    (0, 44, 255),
    (0, 48, 255),
    (0, 52, 255),
    (0, 56, 255),
    (0, 60, 255),
    (0, 64, 255),
    (0, 68, 255),
    (0, 72, 255),
    (0, 76, 255),
    (0, 80, 255),
    (0, 84, 255),
    (0, 88, 255),
    (0, 92, 255),
    (0, 96, 255),
    (0, 100, 255),
    (0, 104, 255),
    (0, 108, 255),
    (0, 112, 255),
    (0, 116, 255),
    (0, 120, 255),
    (0, 124, 255),
    (0, 128, 255),
    (0, 132, 255),
    (0, 136, 255),
    (0, 140, 255),
    (0, 144, 255),
    (0, 148, 255),
    (0, 152, 255),
    (0, 156, 255),
    (0, 160, 255),
    (0, 164, 255),
    (0, 168, 255),
    (0, 172, 255),
    (0, 176, 255),
    (0, 180, 255),
    (0, 184, 255),
    (0, 188, 255),
    (0, 192, 255),
    (0, 196, 255),
    (0, 200, 255),
    (0, 204, 255),
    (0, 208, 255),
    (0, 212, 255),
    (0, 216, 255),
    (0, 220, 254),
    (0, 224, 250),
    (0, 228, 247),
    (2, 232, 244),
    (5, 236, 241),
    (8, 240, 237),
    (12, 244, 234),
    (15, 248, 231),
    (18, 252, 228),
    (21, 255, 225),
    (24, 255, 221),
    (28, 255, 218),
    (31, 255, 215),
    (34, 255, 212),
    (37, 255, 208),
    (41, 255, 205),
    (44, 255, 202),
    (47, 255, 199),
    (50, 255, 195),
    (54, 255, 192),
    (57, 255, 189),
    (60, 255, 186),
    (63, 255, 183),
    (66, 255, 179),
    (70, 255, 176),
    (73, 255, 173),
    (76, 255, 170),
    (79, 255, 166),
    (83, 255, 163),
    (86, 255, 160),
    (89, 255, 157),
    (92, 255, 154),
    (95, 255, 150),
    (99, 255, 147),
    (102, 255, 144),
    (105, 255, 141),
    (108, 255, 137),
    (112, 255, 134),
    (115, 255, 131),
    (118, 255, 128),
    (121, 255, 125),
    (124, 255, 121),
    (128, 255, 118),
    (131, 255, 115),
    (134, 255, 112),
    (137, 255, 108),
    (141, 255, 105),
    (144, 255, 102),
    (147, 255, 99),
    (150, 255, 95),
    (154, 255, 92),
    (157, 255, 89),
    (160, 255, 86),
    (163, 255, 83),
    (166, 255, 79),
    (170, 255, 76),
    (173, 255, 73),
    (176, 255, 70),
    (179, 255, 66),
    (183, 255, 63),
    (186, 255, 60),
    (189, 255, 57),
    (192, 255, 54),
    (195, 255, 50),
    (199, 255, 47),
    (202, 255, 44),
    (205, 255, 41),
    (208, 255, 37),
    (212, 255, 34),
    (215, 255, 31),
    (218, 255, 28),
    (221, 255, 24),
    (224, 255, 21),
    (228, 255, 18),
    (231, 255, 15),
    (234, 255, 12),
    (237, 255, 8),
    (241, 252, 5),
    (244, 248, 2),
    (247, 244, 0),
    (250, 240, 0),
    (254, 237, 0),
    (255, 233, 0),
    (255, 229, 0),
    (255, 226, 0),
    (255, 222, 0),
    (255, 218, 0),
    (255, 215, 0),
    (255, 211, 0),
    (255, 207, 0),
    (255, 203, 0),
    (255, 200, 0),
    (255, 196, 0),
    (255, 192, 0),
    (255, 189, 0),
    (255, 185, 0),
    (255, 181, 0),
    (255, 177, 0),
    (255, 174, 0),
    (255, 170, 0),
    (255, 166, 0),
    (255, 163, 0),
    (255, 159, 0),
    (255, 155, 0),
    (255, 152, 0),
    (255, 148, 0),
    (255, 144, 0),
    (255, 140, 0),
    (255, 137, 0),
    (255, 133, 0),
    (255, 129, 0),
    (255, 126, 0),
    (255, 122, 0),
    (255, 118, 0),
    (255, 115, 0),
    (255, 111, 0),
    (255, 107, 0),
    (255, 103, 0),
    (255, 100, 0),
    (255, 96, 0),
    (255, 92, 0),
    (255, 89, 0),
    (255, 85, 0),
    (255, 81, 0),
    (255, 77, 0),
    (255, 74, 0),
    (255, 70, 0),
    (255, 66, 0),
    (255, 63, 0),
    (255, 59, 0),
    (255, 55, 0),
    (255, 52, 0),
    (255, 48, 0),
    (255, 44, 0),
    (255, 40, 0),
    (255, 37, 0),
    (255, 33, 0),
    (255, 29, 0),
    (255, 26, 0),
    (255, 22, 0),
    (254, 18, 0),
    (250, 15, 0),
    (245, 11, 0),
    (241, 7, 0),
    (236, 3, 0),
    (232, 0, 0),
    (227, 0, 0),
    (222, 0, 0),
    (218, 0, 0),
    (213, 0, 0),
    (209, 0, 0),
    (204, 0, 0),
    (200, 0, 0),
    (195, 0, 0),
    (191, 0, 0),
    (186, 0, 0),
    (182, 0, 0),
    (177, 0, 0),
    (172, 0, 0),
    (168, 0, 0),
    (163, 0, 0),
    (159, 0, 0),
    (154, 0, 0),
    (150, 0, 0),
    (145, 0, 0),
    (141, 0, 0),
    (136, 0, 0),
    (132, 0, 0),
    (127, 0, 0),
)


def _invert_jet(rgb):
    """Map an RGB pixel to its nearest jet colormap value in 0..1."""
    best, idx = None, 0
    for i, (lr, lg, lb) in enumerate(_JET_LUT):
        dr, dg, db = lr - rgb[0], lg - rgb[1], lb - rgb[2]
        d = dr * dr + dg * dg + db * db
        if best is None or d < best:
            best, idx = d, i
    return idx / 255


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(path):
    """Decode an 8-bit RGB/RGBA PNG (filters 0..4, no interlace) to raw bytes."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    width = height = bitdepth = colortype = interlace = None
    idat = b""
    off = 8
    while off < len(data):
        length = struct.unpack(">I", data[off : off + 4])[0]
        ctype = data[off + 4 : off + 8]
        body = data[off + 8 : off + 8 + length]
        if ctype == b"IHDR":
            width, height, bitdepth, colortype, _c, _f, interlace = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        off += 12 + length
    if (colortype, bitdepth, interlace) not in ((2, 8, 0), (6, 8, 0)):
        raise ValueError(
            f"unsupported PNG format: color type {colortype}, bit depth {bitdepth}, "
            f"interlace {interlace} (need 8-bit RGB/RGBA, non-interlaced)"
        )
    bpp = 3 if colortype == 2 else 4
    stride = width * bpp
    raw = zlib.decompress(idat)
    if len(raw) != height * (stride + 1):
        raise ValueError("bad PNG scanline length")
    out = bytearray(height * stride)
    prev = bytearray(stride)
    for y in range(height):
        filt = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        if filt == 0:
            pass
        elif filt == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif filt == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 255
        else:
            raise ValueError(f"unsupported PNG filter {filt}")
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return width, height, bpp, bytes(out)


class _Png:
    """Minimal RGB image exposing the .load() pixel-index and .size interface."""

    def __init__(self, path):
        self._w, self._h, self._bpp, self._data = _decode_png(path)
        self.size = (self._w, self._h)

    def load(self):
        return self

    def __getitem__(self, xy):
        x, y = xy
        i = (y * self._w + x) * self._bpp
        return (self._data[i], self._data[i + 1], self._data[i + 2])


def _load_facts():
    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    return {f["id"]: f["value"] for f in facts}


def _open(name):
    path = FIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing source figure: {path}")
    return _Png(path)


def _is_blue(p):
    r, _, b = p
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
    for x, it in zip(xs, iters, strict=True):
        line = [y for y in range(49, 349) if _is_line(px[x, y])]
        yb = sum(line) / len(line)
        val = (y0 - yb) / (y0 - y100) * 100
        p = {"x": it, "y": round(val, 1)}
        # errorbar extents from dark gray, excluding axis rows
        err = [
            y
            for y in range(49, 349)
            if not (y <= 51 or y >= 346)
            and any(_is_errgray(px[x + dx, y]) for dx in (-2, -1, 0, 1, 2))
        ]
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
        "notes": (
            "Endpoint (24,000) mean/SD are exact from Table 2; other points are pixel-digitized."
        ),
    }


def _is_line(p):
    r, _, b = p
    return b > 90 and b - r > 40 and r < 120


def _fig12_histograms():
    """CA1 spatial-information histograms, three conditions."""
    px = _open("ca1-theta.png").load()
    rects = [(75, 457, 268, 80), (558, 940, 270, 80), (1045, 1427, 268, 200)]
    groups = ["Before training", "After training", "Before training without theta"]
    bins = [
        "0-0.74",
        "0.74-1.49",
        "1.49-2.23",
        "2.23-2.98",
        "2.98-3.72",
        "3.72-4.47",
        "4.47-5.21",
        "5.21-5.96",
        "5.96-6.70",
        "6.70-7.45",
    ]
    series = []
    for (x0, x1, yb, ymax), name in zip(rects, groups, strict=True):
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
        {
            "name": "MEC II",
            "maps": [
                {"label": f"cell {i + 1} Env A", "values": _sample_map(px, *c)}
                for i, c in enumerate(mec)
            ],
            "maps_b": [
                {"label": f"cell {i + 1} Env B", "values": _sample_map(px, *c)}
                for i, c in enumerate(mec_b)
            ],
        },
        {
            "name": "CA1",
            "maps": [
                {"label": f"cell {i + 1} Env A", "values": _sample_map(px, *c)}
                for i, c in enumerate(ca1)
            ],
            "maps_b": [
                {"label": f"cell {i + 1} Env B", "values": _sample_map(px, *c)}
                for i, c in enumerate(ca1_b)
            ],
        },
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
        groups.append({"name": f"cell {i + 1}", "maps": maps})
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
    for cx, n in zip(centers, neurons, strict=True):
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
        {
            "name": "MEC II",
            "color": "blue",
            "points": [by_key[("MEC II", "precession")], by_key[("MEC II", "locking")]],
        },
        {
            "name": "MEC III",
            "color": "accent",
            "points": [by_key[("MEC III", "precession")], by_key[("MEC III", "locking")]],
        },
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
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
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
