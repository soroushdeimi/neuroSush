"""Ramping activity from relaxation after a stimulus, with no prediction (an illustrative model).

Huang, Shamsnia, Chen, Wu, Stamm, Medico and Najafi (2026, Science Advances 12(34),
doi:10.1126/sciadv.aed6417) imaged mouse visual and parietal cortex during passive viewing of
200 ms stimuli. Stimulus-activated neurons ramp down after a stimulus and stimulus-inhibited
neurons ramp up before the next one; the ramps exist in naive animals, change at once when
the interval switches between blocks, and deviant stimuli evoke nearly the same response in
predictable (fixed 1.5 s) and irregular (0.5-2.5 s) contexts. The authors conclude that the
ramps are intrinsic timing, the relaxation of stimulus-evoked activity with heterogeneous
kinetics, which gives a population code for elapsed time, and not temporal prediction.

This is NOT a reproduction of the paper's data (the paper has no network model) but a small
mechanistic illustration of that interpretation. By construction the network has no
plasticity, no recurrent excitation, no memory of intervals and no predictive mechanism:

- ``stim`` (Poisson, 0 -> 120 Hz during each stimulus) drives excitatory population ``a``
  (100 cells) and an inhibitory SST-like population ``i`` (30 LIF cells); ``bg`` (constant
  10 Hz Poisson) gives ``a`` and the second excitatory population ``b`` (100 cells) a baseline.
- ``a`` cells are adaptive exponential LIF neurons (:class:`~neurosush.neurons.models.AdaptiveELIF`)
  with conductance synapses whose excitatory decay ``tau_exc`` is log-uniform over 0.1-2 s,
  and adaptation time constants ``tau_w`` also log-uniform over 0.1-2 s: after a stimulus they
  relax from the evoked rate with a different speed each (stimulus-activated cells).
- ``i`` inhibits ``b`` through conductances with a log-uniform inhibitory decay ``tau_inh``
  of 0.1-2 s: ``b`` is silenced during the stimulus and relaxes back with a different speed
  per cell (stimulus-inhibited cells, ramping up).

Analyses use spike counts in 50 ms bins convolved with a causal exponential calcium kernel
(300 ms decay, ``--calcium-tau 0`` turns it off), as for imaging data. Sessions: random ISIs
0.5-2.5 s; alternating 12-trial blocks of fixed 1.5 s and jittered 0.5-2.5 s ISIs with 10%
deviants (a stimulus 0.75 s after the previous one, in both block types; the paper has about
4%, more are needed here for few simulated minutes); alternating 12-trial blocks of fixed 1 s
and 2 s ISIs. ISIs run from stimulus offset to the next onset. Cells are classified on the
even trials of the random session (activated/inhibited: mean response over onset to 100 ms
after offset differs from the pre-stimulus mean by more than 3 standard errors); time
constants are fitted to odd trials (``c + A exp(-t / tau)`` over the 1.5 s after offset, kept
when R^2 > 0.6 and the sign of ``A`` matches the class); elapsed time (six 250 ms classes
within 1.5 s) is decoded from single 50 ms bins by a softmax regression on 10 principal
components, trained on even and tested on odd trials, with 40 cells per condition (``both``
is 20 + 20), 5 random cell subsets.

Everything below is measured from the model, not from the paper. Measured with the defaults
(8 sessions of 150 s, 727 stimuli, seed 0, about 1.5 min on CPU):

- Classification: 100/100 ``a`` cells activated, 91/100 ``b`` cells inhibited; no cell of
  the other kind.
- Ramps: the fitted time constants of the activated cells have a median of 0.77 s (IQR
  0.41-1.30 s) for the 49 cells below the 3 s grid limit; 51 cells hit the limit because a
  1.5 s window cannot constrain slower decays. The inhibited cells ramp up: 66 of 91 pass the
  fit criteria, 53 at the grid limit, the rest with median 0.77 s (IQR 0.62-1.44 s). The
  fitted tau is rank-correlated with the true kinetics (Spearman 0.86 for ``tau_exc`` of
  ``a``, 0.42 for ``tau_inh`` of ``b``).
- Elapsed-time decoding (chance 17%): activated-only 74.6 +- 0.9%, inhibited-only
  65.7 +- 1.6%, both 73.7 +- 1.5% (mean absolute error 64, 98 and 67 ms).
- Deviants: evoked response of the activated cells 5.24 Hz in the fixed context against
  5.01 Hz in the jittered one (n = 12 each; per-cell correlation 0.99), of the inhibited
  cells -0.63 against -0.03 Hz (correlation 0.80); both are smaller than the response to a
  regular stimulus at 1.5 s (8.76 and -1.96 Hz), because the cells have not yet relaxed
  after 0.75 s: the elapsed time, not the context, sets the response.
- Block switches: the first trial after a switch differs from the steady-state trace by
  0.14-0.28 of the trace range (noise level 0.03-0.10), but the second trial is already at
  0.05-0.14 (noise 0.03-0.10), so the change is complete within one interval. The first
  trial is not a prediction error: the slowest cells still carry the state left by the
  previous interval (checked in a separate run: trial 2 is at the noise level).

Paper, qualitatively: ramps of both signs, heterogeneous time constants, a population code
for time that is best with both groups together, deviant responses nearly identical across
contexts, immediate changes at block switches. The model agrees on most of these points, with
caveats: decoding with both groups is no better than with the activated cells alone (73.7
against 74.6%), the first trial after a switch is not yet fully at steady state, the
stimulus-specific, and no cell is of a mixed or ambiguous kind. The kinetics are put in by
hand (log-uniform time constants), so the spread of tau shown here is an assumption, not a
result. The deviant here differs from the paper's only in its timing.

Run: ``python examples/intrinsic_timing_ramps.py`` (about 1.5 min on CPU).
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise

import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import ConductanceIntegration
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, AdaptiveELIF, Fire
from neurosush.recording import SpikeCounter
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather

BIN = 50  # ms per analysis bin; every stimulus time is a multiple of it
STIM_BINS = 4  # 200 ms stimulus
STIM_HZ = 120.0  # rate of each stimulus input neuron during a stimulus
BG_HZ = 10.0  # rate of the background input neurons
TAU_LO, TAU_HI = 100.0, 2000.0  # ms, range of the heterogeneous kinetics


def _randint(lo: int, hi: int, gen: torch.Generator) -> int:
    """One integer uniform in ``[lo, hi]`` (inclusive)."""
    return int(torch.randint(lo, hi + 1, (1,), generator=gen))


def _rank(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(x)).double()


def log_uniform(n: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    """``n`` samples, uniform in ``log`` between ``lo`` and ``hi``."""
    u = torch.rand(n, generator=gen)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _lognormal(
    shape: tuple[int, ...], mean: float, gen: torch.Generator, sigma=0.5
) -> torch.Tensor:
    z = torch.randn(shape, generator=gen)
    return mean * torch.exp(sigma * z - sigma**2 / 2)


def build(
    n_a: int = 100,
    n_b: int = 100,
    n_i: int = 30,
    n_in: int = 100,
    *,
    batch: int = 1,
    seed: int = 0,
):
    """The network and the dict of its groups (no plasticity, no recurrent excitation).

    ``stim`` (Poisson, rate stepped up during stimuli) drives the adaptive excitatory
    population ``a`` and the inhibitory (SST-like) population ``i``; ``i`` inhibits the
    excitatory population ``b``; ``bg`` (constant Poisson) gives ``a`` and ``b`` a baseline.
    """
    gen = torch.Generator().manual_seed(seed + 1000)
    net = Network(seed=seed, batch_size=batch)
    stim = NeuronGroup(net, n_in, [PoissonInput(0.0), Axon()], name="stim")
    bg = NeuronGroup(net, n_in, [PoissonInput(BG_HZ / 1000.0), Axon()], name="bg")

    tau_a = log_uniform(n_a, TAU_LO, TAU_HI, gen)  # slow excitatory decay of the A cells
    tau_w = log_uniform(n_a, TAU_LO, TAU_HI, gen)  # adaptation time constant of the A cells
    tau_b = log_uniform(n_b, TAU_LO, TAU_HI, gen)  # slow inhibitory decay of the B cells

    def cells(n, model, name, inhibitory=False, *, tau_exc=5.0, tau_inh=50.0):
        conductance = ConductanceIntegration(
            e_exc=0.0, e_inh=-80.0, tau_exc=tau_exc, tau_inh=tau_inh
        )
        return NeuronGroup(
            net,
            n,
            [
                conductance,
                model,
                Fire(),
                SpikeCounter(),
                Axon(),
            ],
            inhibitory=inhibitory,
            name=name,
        )

    lif = {"tau": 20.0, "threshold": -50.0, "v_reset": -60.0, "v_rest": -70.0}
    a = cells(
        n_a,
        AdaptiveELIF(
            alpha=0.0,
            beta=BETA * (500.0 / tau_w) ** 0.5,
            tau_w=tau_w,
            delta=2.0,
            theta_rh=-52.0,
            **lif,
        ),
        "a",
        tau_exc=tau_a,
    )
    b = cells(
        n_b,
        AdaptiveELIF(alpha=0.0, beta=1.0, tau_w=500.0, delta=2.0, theta_rh=-52.0, **lif),
        "b",
        tau_inh=tau_b,
    )
    i = cells(n_i, LIF(**lif), "i", inhibitory=True)

    def connect(src, dst, w, name):
        SynapseGroup(net, src, dst, [WeightInit(weights=w), DenseInput(), SpikeGather()], name=name)

    # a spike's conductance integrates to weight * tau, so the A weights are scaled by
    # 1 / tau (same mean drive, slower kinetics); the stimulus weights by tau ** -0.5
    connect(bg, a, _lognormal((n_in, n_a), W_BG_A, gen) * (TAU_REF / tau_a), "bg_a")
    connect(bg, b, _lognormal((n_in, n_b), W_BG_B, gen), "bg_b")
    connect(
        stim,
        a,
        _lognormal((n_in, n_a), W_STIM, gen, 1.0) * (TAU_REF / tau_a) ** 0.5,
        "stim_a",
    )
    connect(stim, i, _lognormal((n_in, n_i), W_STIM_I, gen), "stim_i")
    # a spike's conductance integrates to weight * tau_inh, so scale the weights by
    # 1 / tau_inh: slow neurons get the same peak inhibition but a longer tail
    connect(i, b, _lognormal((n_i, n_b), W_INH, gen) * (200.0 / tau_b), "i_b")
    net.initialize()

    return (
        net,
        {"stim": stim, "a": a, "b": b, "i": i},
        {"tau_a": tau_a, "tau_b": tau_b, "tau_w": tau_w},
    )


TAU_REF = 5.0
TAU_MAX = 3.0  # s, upper limit of the time-constant fit grid
W_BG_A = 0.06
W_BG_B = 0.07
W_STIM = 0.0007
W_STIM_I = 0.01
W_INH = 0.006
BETA = 0.1


# ---------------------------------------------------------------- stimulus protocols


def make_schedule(
    protocol: str, n_bins: int, rng: torch.Generator, *, block: int = 12, p_deviant: float = 0.1
) -> tuple[torch.Tensor, list[dict]]:
    """Stimulus on/off per bin and the list of trials (onset, previous offset, context ...).

    ``random``: ISIs uniform 0.5-2.5 s. ``fixed_jitter``: alternating blocks of fixed 1.5 s
    and jittered 0.5-2.5 s ISIs, a fraction ``p_deviant`` of the trials being deviants (a
    stimulus 0.75 s after the previous one, in both block types). ``short_long``:
    alternating blocks of fixed 1 s and 2 s ISIs. ISIs are measured from stimulus offset to
    the next onset and are multiples of 50 ms.
    """
    on = torch.zeros(n_bins, dtype=torch.bool)
    trials: list[dict] = []
    onset, prev_off, k = 10, None, 0
    while onset + STIM_BINS < n_bins:
        j, b = k % block, k // block
        info = {"k": k, "deviant": False, "context": protocol, "index_in_block": j, "block": b}
        if protocol == "random":
            isi = _randint(10, 50, rng)
        elif protocol == "fixed_jitter":
            fixed = b % 2 == 0
            isi = 30 if fixed else _randint(10, 50, rng)
            info["context"] = "fixed" if fixed else "jitter"
            if k > 0 and float(torch.rand(1, generator=rng)) < p_deviant:
                isi, info["deviant"] = 15, True
        else:
            isi = 20 if b % 2 == 0 else 40
            info["context"] = "short" if b % 2 == 0 else "long"
        if k > 0:
            onset = prev_off + isi
            if onset + STIM_BINS >= n_bins:
                break
        info.update(onset=onset, offset=onset + STIM_BINS, isi=isi if k > 0 else None)
        on[onset : onset + STIM_BINS] = True
        trials.append(info)
        prev_off, k = onset + STIM_BINS, k + 1
    return on, trials


def simulate(
    protocols: list[str], seconds: float, *, seed: int = 0, calcium_tau: float = 300.0, **kwargs
) -> dict:
    """Run one network, one protocol per batch row, and return rates and trials per row."""
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))  # tiny tensors: more threads only add overhead
    net, groups, taus = build(batch=len(protocols), seed=seed, **kwargs)
    rng = torch.Generator().manual_seed(seed)
    n_bins = int(seconds * 1000 / BIN)
    schedules = [make_schedule(p, n_bins, rng) for p in protocols]
    on = torch.stack([s[0] for s in schedules]).to(net.dtype)
    stim = groups["stim"]
    counts = {n: torch.zeros(n_bins, len(protocols), groups[n].size) for n in "abi"}
    for t in range(n_bins):
        stim.rates.copy_((on[:, t] * STIM_HZ / 1000.0)[:, None].expand_as(stim.rates))
        before = {n: groups[n].spike_count.clone() for n in counts}
        net.run(BIN)
        for n in counts:
            counts[n][t] = groups[n].spike_count - before[n]
    torch.set_num_threads(threads)
    out = {"protocols": protocols, "trials": [s[1] for s in schedules], "taus": taus}
    for n in "ab":
        rate = counts[n].permute(1, 0, 2) / (BIN / 1000.0)  # (row, bin, cell) in Hz
        out[n] = calcium(rate, calcium_tau) if calcium_tau > 0 else rate
    out["rate_i"] = counts["i"].permute(1, 0, 2).mean(-1) / (BIN / 1000.0)
    return out


def calcium(rate: torch.Tensor, tau: float) -> torch.Tensor:
    """Causal exponential smoothing of the binned rates (a GCaMP-like decay, no rise)."""
    k = math.exp(-BIN / tau)
    out = torch.empty_like(rate)
    acc = torch.zeros_like(rate[:, 0])
    for t in range(rate.shape[1]):
        acc = k * acc + (1 - k) * rate[:, t]
        out[:, t] = acc
    return out


# ---------------------------------------------------------------- analyses


def _rows(out: dict, protocol: str) -> list[int]:
    return [r for r, p in enumerate(out["protocols"]) if p == protocol]


def _activity(out: dict) -> torch.Tensor:
    """Calcium-like activity of all cells, shape (row, bin, n_a + n_b)."""
    return torch.cat([out["a"], out["b"]], dim=2)


def _post_traces(out: dict, x: torch.Tensor, rows: list[int], n_bins: int, parity: int | None):
    """Activity in the ``n_bins`` bins after each offset whose next interval is long enough.

    Returns an array (trial, n_bins, cell).
    """
    traces = []
    for r in rows:
        trials = out["trials"][r]
        for cur, nxt in pairwise(trials):
            if nxt["isi"] >= n_bins and (parity is None or cur["k"] % 2 == parity):
                traces.append(x[r, cur["offset"] : cur["offset"] + n_bins])
    return torch.stack(traces)


def classify_cells(out: dict, x: torch.Tensor, t_crit: float = 3.0) -> torch.Tensor:
    """+1 stimulus-activated, -1 stimulus-inhibited, 0 neither, per cell.

    For each cell and each trial of the random-ISI sessions (even trials only, with a
    preceding interval of at least 1.5 s) the response is the mean activity from stimulus
    onset to 100 ms after offset minus the mean over the 200 ms before onset; a cell is
    activated (inhibited) when the mean difference over trials is above (below) ``t_crit``
    standard errors.
    """
    diffs = []
    for r in _rows(out, "random"):
        for tr in out["trials"][r]:
            if tr["k"] % 2 == 0 and tr["isi"] is not None and tr["isi"] >= 30:
                o, f = tr["onset"], tr["offset"]
                diffs.append(x[r, o : f + 2].mean(0) - x[r, o - STIM_BINS : o].mean(0))
    d = torch.stack(diffs)
    t = d.mean(0) / (d.std(0) / math.sqrt(len(d)) + 1e-9)
    return torch.where(t > t_crit, 1, torch.where(t < -t_crit, -1, 0))


def fit_exponential(trace: torch.Tensor, tau_grid: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Per column of ``trace`` (time, cell) fit ``c + A exp(-t / tau)``, t in seconds.

    For each tau on the grid ``c`` and ``A`` follow by linear least squares; returns tau
    (s), A and the explained variance. A ramp-down has ``A > 0``, a ramp-up ``A < 0``.
    """
    trace = trace.double()
    t = (torch.arange(trace.shape[0], dtype=torch.float64) + 0.5) * BIN / 1000.0
    best_sse = torch.full((trace.shape[1],), float("inf"), dtype=torch.float64)
    best = torch.zeros(3, trace.shape[1], dtype=torch.float64)
    for tau in tau_grid.double():
        design = torch.stack([torch.ones_like(t), torch.exp(-t / tau)], dim=1)
        coef = torch.linalg.lstsq(design, trace, driver="gelsd").solution
        sse = ((design @ coef - trace) ** 2).sum(0)
        better = sse < best_sse
        best_sse = torch.where(better, sse, best_sse)
        best[0, better], best[1, better] = tau, coef[1, better]
    var = ((trace - trace.mean(0)) ** 2).sum(0) + 1e-12
    return best[0], best[1], 1 - best_sse / var


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation."""
    return float(torch.corrcoef(torch.stack([_rank(a), _rank(b)]))[0, 1])


def time_constants(out: dict, x: torch.Tensor, kind: torch.Tensor) -> dict:
    """Exponential fits of the post-stimulus traces (odd trials, 1.5 s) per cell class."""
    traces = _post_traces(out, x, _rows(out, "random"), 30, parity=1).mean(0)  # (30, cell)
    grid = torch.logspace(math.log10(0.05), math.log10(TAU_MAX), 40, dtype=torch.float64)
    tau, amp, r2 = fit_exponential(traces, grid)
    truth = torch.cat([out["taus"]["tau_a"], out["taus"]["tau_b"]]).double() / 1000
    res = {}
    for name, sign in (("activated", 1), ("inhibited", -1)):
        ok = (kind == sign) & (torch.sign(amp) == sign) & (r2 > 0.6)
        res[name] = {
            "n": int((kind == sign).sum()),
            "fitted": int(ok.sum()),
            "tau": tau[ok],
            "rho": spearman(tau[ok], truth[ok]) if ok.sum() > 3 else float("nan"),
        }
    return res


def decode_time(
    out: dict,
    x: torch.Tensor,
    kind: torch.Tensor,
    *,
    seed: int = 0,
    repeats: int = 5,
    pcs: int = 10,
) -> dict:
    """Decode elapsed time since stimulus offset (six 250 ms classes) from single bins.

    Samples are the activity vectors of single 50 ms bins in the 1.5 s after a stimulus
    (random-ISI sessions); trials of even index train and trials of odd index test a
    softmax (logistic) regression on the leading principal components of standardised
    activity. Activated-only, inhibited-only and both populations use the same number of
    cells (``both`` takes half of each), drawn at random ``repeats`` times.
    """
    torch.manual_seed(seed)  # the randomised PCA
    rng = torch.Generator().manual_seed(seed)
    rows = _rows(out, "random")
    train = _post_traces(out, x, rows, 30, parity=0)
    test = _post_traces(out, x, rows, 30, parity=1)
    act, inh = torch.nonzero(kind == 1)[:, 0], torch.nonzero(kind == -1)[:, 0]
    n = min(len(act), len(inh), 40)
    labels = torch.arange(6).repeat_interleave(5)

    def choice(pool: torch.Tensor, k: int) -> torch.Tensor:
        return pool[torch.randperm(len(pool), generator=rng)[:k]]

    results: dict[str, list[tuple[float, float]]] = {"activated": [], "inhibited": [], "both": []}
    for _ in range(repeats):
        picks = {
            "activated": choice(act, n),
            "inhibited": choice(inh, n),
            "both": torch.cat([choice(act, n // 2), choice(inh, n - n // 2)]),
        }
        for name, cells in picks.items():
            results[name].append(_softmax_fit(train[..., cells], test[..., cells], labels, pcs))
    return {
        "n_cells": n,
        **{k: _mean_std(v) for k, v in results.items()},
    }


def _mean_std(v: list[tuple[float, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and (population) standard deviation over repeats, per metric."""
    t = torch.tensor(v, dtype=torch.float64)
    return t.mean(0), t.std(0, unbiased=False)


def _softmax_fit(train: torch.Tensor, test: torch.Tensor, labels: torch.Tensor, pcs: int):
    """Accuracy and mean absolute time error (ms) of a PCA + softmax regression."""
    xtr = train.reshape(-1, train.shape[-1]).float()
    xte = test.reshape(-1, test.shape[-1]).float()
    ytr = labels.repeat(len(train))
    yte = labels.repeat(len(test))
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    _, _, v = torch.pca_lowrank(xtr, q=pcs, center=False, niter=4)
    ztr, zte = xtr @ v, xte @ v
    sd = ztr.std(0)
    ztr, zte = ztr / sd, zte / sd
    model = torch.nn.Linear(pcs, 6)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(200):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(ztr), ytr).backward()
        opt.step()
    pred = model(zte).argmax(1)
    return float((pred == yte).float().mean()), float((pred - yte).abs().float().mean() * 250)


def deviant_responses(out: dict, x: torch.Tensor, kind: torch.Tensor) -> dict:
    """Evoked response of deviants in fixed and jittered blocks and of regular stimuli.

    The response is the mean from onset to 100 ms after offset minus the mean over the 200 ms
    before onset; regular stimuli are those at the expected 1.5 s in fixed blocks.
    """
    groups: dict[str, list[torch.Tensor]] = {"dev_fixed": [], "dev_jitter": [], "regular_fixed": []}
    for r in _rows(out, "fixed_jitter"):
        for tr in out["trials"][r]:
            if tr["isi"] is None:
                continue
            o, f = tr["onset"], tr["offset"]
            d = x[r, o : f + 2].mean(0) - x[r, o - STIM_BINS : o].mean(0)
            if tr["deviant"]:
                groups["dev_fixed" if tr["context"] == "fixed" else "dev_jitter"].append(d)
            elif tr["context"] == "fixed":
                groups["regular_fixed"].append(d)
    res = {k: torch.stack(v) for k, v in groups.items()}
    summary = {}
    for name, sign in (("activated", 1), ("inhibited", -1)):
        sel = kind == sign
        m = {k: v[:, sel].mean(0) for k, v in res.items()}  # per-cell means
        summary[name] = {
            "n_dev": (len(res["dev_fixed"]), len(res["dev_jitter"])),
            "fixed": float(m["dev_fixed"].mean()),
            "jitter": float(m["dev_jitter"].mean()),
            "regular": float(m["regular_fixed"].mean()),
            "corr": float(torch.corrcoef(torch.stack([m["dev_fixed"], m["dev_jitter"]]))[0, 1]),
        }
    return summary


def block_switch(out: dict, x: torch.Tensor, kind: torch.Tensor) -> dict:
    """Interval traces of the first two trials of a block versus the steady state.

    The traces cover 1 s after the previous offset, for 1 s and 2 s blocks, per cell class.

    Reports the root-mean-square difference between the mean trace of trial ``j`` of the
    block and the mean steady-state trace (trials 3+), relative to the trace's range, and
    the same quantity for randomly drawn sets of as many steady-state trials (the noise
    floor). Entries are ``(rms, noise floor, n)`` for ``j = 0, 1``; ``nan`` without data.
    """
    nan = (float("nan"), float("nan"), 0)
    early: dict[tuple[str, int], list[torch.Tensor]] = {}
    steady: dict[str, list[torch.Tensor]] = {"short": [], "long": []}
    for r in _rows(out, "short_long"):
        trials = out["trials"][r]
        for prev, tr in pairwise(trials):
            if tr["block"] == 0:
                continue  # no earlier block to switch from
            trace = x[r, prev["offset"] : prev["offset"] + 20]
            j = tr["index_in_block"]
            if j < 2:
                early.setdefault((tr["context"], j), []).append(trace)
            elif j >= 3:
                steady[tr["context"]].append(trace)
    res: dict = {}
    for name, sign in (("activated", 1), ("inhibited", -1)):
        sel = kind == sign
        for ctx in ("short", "long"):
            res[(name, ctx)] = [nan, nan]
            if len(steady[ctx]) < 4:
                continue
            st = torch.stack(steady[ctx])[:, :, sel]
            s_mean = st.mean((0, 2))
            span = float(s_mean.max() - s_mean.min()) + 1e-9
            for j in (0, 1):
                trials_j = early.get((ctx, j), [])
                if not trials_j or len(trials_j) >= len(st):
                    continue
                f = torch.stack(trials_j)[:, :, sel].mean((0, 2))
                floor = []
                for seed in range(100):
                    pick = torch.randperm(len(st), generator=torch.Generator().manual_seed(seed))
                    a_ = st[pick[: len(trials_j)]].mean((0, 2))
                    b_ = st[pick[len(trials_j) :]].mean((0, 2))
                    floor.append(float(((a_ - b_) ** 2).mean().sqrt()) / span)
                rms = float(((f - s_mean) ** 2).mean().sqrt()) / span
                res[(name, ctx)][j] = (rms, float(torch.tensor(floor).quantile(0.5)), len(trials_j))
    return res


# ---------------------------------------------------------------- driver


def run(
    seconds: float = 150.0, repeats: tuple[int, int, int] = (2, 3, 3), *, seed: int = 0, **kwargs
) -> dict:
    """Simulate the three protocols (``repeats`` independent sessions each) and analyse them."""
    protocols = (
        ["random"] * repeats[0] + ["fixed_jitter"] * repeats[1] + ["short_long"] * repeats[2]
    )
    out = simulate(protocols, seconds, seed=seed, **kwargs)
    x = _activity(out)
    kind = classify_cells(out, x)
    return {
        "out": out,
        "kind": kind,
        "n_a": out["a"].shape[2],
        "tau": time_constants(out, x, kind),
        "decode": decode_time(out, x, kind, seed=seed),
        "deviant": deviant_responses(out, x, kind),
        "switch": block_switch(out, x, kind),
    }


def report(res: dict) -> None:
    """Print the results summary."""
    out, kind, n_a = res["out"], res["kind"], res["n_a"]
    print(f"{len(out['protocols'])} sessions, {sum(len(t) for t in out['trials'])} stimuli")
    print(
        f"cells: {int((kind[:n_a] == 1).sum())}/{n_a} excitatory-A activated, "
        f"{int((kind[n_a:] == -1).sum())}/{len(kind) - n_a} excitatory-B inhibited "
        f"(A inhibited {int((kind[:n_a] == -1).sum())}, B activated {int((kind[n_a:] == 1).sum())})"
    )
    print("exponential time constants after stimulus offset (fits with R^2 > 0.6, sign ok):")
    for name, r in res["tau"].items():
        t = r["tau"]
        free = t[t < TAU_MAX * 0.99]
        qs = torch.tensor([0.25, 0.5, 0.75], dtype=free.dtype)
        q = torch.quantile(free, qs).tolist() if len(free) else [float("nan")] * 3
        print(
            f"  {name:9s}: {r['fitted']}/{r['n']} fitted, {len(t) - len(free)} at the "
            f"{TAU_MAX:g} s grid limit; the others: median {q[1]:.2f} s "
            f"(IQR {q[0]:.2f}-{q[2]:.2f}); Spearman vs true kinetics {r['rho']:.2f}"
        )
    d = res["decode"]
    print(f"decoding elapsed time (6 classes of 250 ms, chance 17%), {d['n_cells']} cells:")
    for name in ("activated", "inhibited", "both"):
        (acc, err), (sacc, _) = d[name]
        print(f"  {name:9s}: accuracy {100 * acc:.1f} +- {100 * sacc:.1f} %, error {err:.0f} ms")
    print("deviants (0.75 s) vs regular, evoked response in Hz (fixed / jitter / regular 1.5 s):")
    for name, v in res["deviant"].items():
        print(
            f"  {name:9s}: {v['fixed']:.2f} / {v['jitter']:.2f} / {v['regular']:.2f}, "
            f"n = {v['n_dev']}, per-cell corr fixed vs jitter {v['corr']:.2f}"
        )
    print("block switch, rms difference of the interval trace from steady state / trace range:")
    for (name, ctx), pair in res["switch"].items():
        cells = ", ".join(
            f"trial {j + 1}: {v[0]:.2f} (noise {v[1]:.2f}, n = {v[2]})" for j, v in enumerate(pair)
        )
        print(f"  {name:9s} {ctx:5s}: {cells}")


def main() -> None:
    """Run the three protocols and print the summary."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=150.0, help="simulated seconds per session")
    p.add_argument("--calcium-tau", type=float, default=300.0, help="ms; 0 analyses spike rates")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    report(run(args.seconds, seed=args.seed, calcium_tau=args.calcium_tau))


if __name__ == "__main__":
    main()
