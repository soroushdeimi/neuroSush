"""Kheradpisheh et al. (2018): an STDP-trained spiking convolutional network on MNIST.

Partial reproduction of "STDP-based spiking deep convolutional neural networks for object
recognition" (Neural Networks 99:56-67, arXiv:1611.01421), MNIST architecture:

* DoG filtering (ON and OFF centre, sigma 1 and 2, 7x7) and intensity-to-latency coding:
  one spike per active input, 30 time steps (``dog_kernel``, ``split_polarity``,
  ``intensity_to_latency``).
* conv1: 30 maps, 5x5, IF neurons (no leak), threshold 15, then 2x2 / stride 2 max pooling
  (earliest spike in the window).
* conv2: 100 maps, 5x5, threshold 10, then global max pooling.
* Every neuron fires at most once per image. Lateral inhibition: at each position only the
  earliest spike among the maps survives. Winner-take-all learning: the earliest neuron of
  each map (ties broken by potential) is the winner and the only one that learns, and it
  inhibits the other neurons around it from learning on this image.
* Simplified STDP, ``dw = a+ w (1 - w)`` if the presynaptic spike is not later than the
  postsynaptic one, else ``-a- w (1 - w)`` (also for inputs that never spike), a+ = 0.004,
  a- = 0.003 (the script defaults to 10x these, 0.04 and 0.03, because it sees only a few
  thousand images; the paper's rates gave about 55 % in a short run), weights initialised
  from N(0.8, 0.05). Layers are trained one after the other, without labels, until the
  convergence measure ``C = mean(w (1 - w)) < 0.01`` or the training images run out.
* Features: thresholds of the last layer set to infinity, the final potentials are max-pooled
  over positions (100 features per image).

Values come from the paper (MNIST section and Section 2): DoG sigma 1 and 2, 30 and 100 maps,
5x5 windows, thresholds 15 and 10, 2x2 stride-2 first pooling, global second pooling,
a+ = 0.004, a- = 0.003, N(0.8, 0.05) initial weights, 30 time steps, C < 0.01. Not verified
from the text: the DoG kernel size and padding (7x7, padding 3), the input threshold, the
winner-inhibition radius and the number of training images per layer (the paper trains on the
whole set, possibly several passes); these are choices of this script.

Measured (RTX 3060 laptop, about 1 minute, 3000 STDP images per layer, 10000 classifier
images, 10000 test images): 89.8 % with global pooling (100 features, ``--grid 1``, seed 0)
and 97.2 % with ``--grid 2`` (max over a 2x2 grid, 400 features, seed 1); the paper reports
98.4 % after training on the whole set. Single runs, no confidence intervals.

Deviations: the classifier is multinomial logistic regression trained with torch (the paper
used a linear SVM, C = 2.4); the default run trains on a few thousand images per layer and the
classifier on a subset, not 60000; the IF dynamics are computed in closed form (the potential
at time t is the convolution of the cumulative input spikes, ``conv2d_current``) instead of
stepping a ``Network``, because weights are constant during an image and neurons fire once.

Run: ``python examples/conv_stdp_mnist.py --data path/to/MNIST/raw --device cuda``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neurosush.data import load_mnist
from neurosush.encoding import intensity_to_latency
from neurosush.filters import dog_kernel
from neurosush.readout import accuracy
from neurosush.synapses.currents import conv2d_current
from neurosush.transforms import FilterBank, split_polarity

STEPS = 30
A_PAPER = (0.004, 0.003)  # the paper's a+, a-
LEARNING_SCALE = 10.0  # this script's default: 10x the paper's rates (few images)
CONVERGED = 0.01


def encode(images: torch.Tensor, *, threshold: float = 0.1, steps: int = STEPS) -> torch.Tensor:
    """DoG filter and latency-code images ``(N, 28, 28)`` into spike times ``(N, 2, 28, 28)``.

    ON and OFF channels share one per-image normalisation; a pixel that never spikes gets
    time ``steps``.
    """
    kernel = dog_kernel(7, 1.0, 2.0, zero_mean=True, dtype=torch.float32, device=images.device)
    bank = FilterBank(kernel[None, None], padding=3)
    out = []
    for img in images.float() / 255.0:
        dog = bank(img[None, None])[0]  # (1, 28, 28)
        pol = split_polarity(dog, dim=0)  # (2, 28, 28): ON, OFF
        pol = pol / pol.max().clamp_min(1e-6)
        spikes = intensity_to_latency(pol, steps, threshold=threshold)
        out.append(
            torch.where(spikes.any(0), spikes.float().argmax(0), torch.full_like(pol, steps))
        )
    return torch.stack(out).to(torch.uint8)


def potentials(times: torch.Tensor, weights: torch.Tensor, steps: int = STEPS) -> torch.Tensor:
    """IF potentials ``(B, T, F, h, w)`` of a conv layer for input spike times ``(B, C, H, W)``."""
    batch, channels, height, width = times.shape
    t = torch.arange(steps, device=times.device).view(1, steps, 1, 1, 1)
    cum = (times[:, None] <= t).flatten(2)  # (B, T, C*H*W) cumulative spikes
    out = conv2d_current(cum, weights, src_shape=(channels, height, width), stride=1, padding=0)
    return out.view(batch, steps, weights.shape[0], height - 4, width - 4)


def first_spikes(pot: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Spike time (``T`` when silent) and potential at that time for each neuron."""
    steps = pot.shape[1]
    crossed = pot >= threshold
    time_ = torch.where(
        crossed.any(1), crossed.float().argmax(1), torch.full_like(pot[:, 0], steps)
    )
    at = pot.gather(1, time_.long().clamp(max=steps - 1)[:, None])[:, 0]
    return time_, at


def lateral_inhibition(time_: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    """Keep only the earliest spike (ties: highest potential) over maps at each position."""
    steps_score = time_ * 1e6 - at
    winner = steps_score.argmin(1, keepdim=True)
    keep = torch.zeros_like(time_, dtype=torch.bool).scatter_(1, winner, True)
    return torch.where(keep, time_, torch.full_like(time_, STEPS))


def max_pool_times(time_: torch.Tensor, size: int = 2, stride: int = 2) -> torch.Tensor:
    """First-spike pooling: a pooled neuron spikes with the earliest spike of its window."""
    return -F.max_pool2d(-time_, size, stride)


@torch.no_grad()
def layer1(times: torch.Tensor, w1: torch.Tensor, thr: float, batch: int = 64) -> torch.Tensor:
    """Spike times ``(N, 30, 12, 12)`` after conv1, inhibition and pooling."""
    outs = []
    for i in range(0, len(times), batch):
        t, at = first_spikes(potentials(times[i : i + batch].float(), w1), thr)
        outs.append(max_pool_times(lateral_inhibition(t, at)).to(torch.uint8))
    return torch.cat(outs)


@torch.no_grad()
def features(
    times: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    thr1: float,
    batch: int = 64,
    grid: int = 1,
) -> torch.Tensor:
    """Max-pooled final potentials of conv2 (threshold infinite), ``(N, maps * grid**2)``.

    ``grid=1`` is the paper's global pooling; ``grid=g`` pools over a ``g x g`` grid of regions.
    """
    out = []
    for i in range(0, len(times), batch):
        pooled = layer1(times[i : i + batch], w1, thr1, batch)
        pot = potentials(pooled.float(), w2)[:, -1]  # final potential, (B, F, 8, 8)
        out.append(F.adaptive_max_pool2d(pot, grid).flatten(1))
    return torch.cat(out)


@torch.no_grad()
def train_layer(
    times: torch.Tensor,
    weights: torch.Tensor,
    threshold: float,
    *,
    a_plus: float,
    a_minus: float,
    radius: int,
    log: str,
) -> int:
    """Unsupervised WTA-STDP on one layer; returns the number of images used."""
    k = weights.shape[-1]
    used = 0
    for n in range(len(times)):
        x = times[n : n + 1].float()
        t, at = first_spikes(potentials(x, weights), threshold)
        score = (t * 1e6 - at)[0]  # (F, h, w); silent neurons score >= STEPS * 1e6 - at
        silent = t[0] >= STEPS
        score = score.masked_fill(silent, float("inf"))
        h, w = score.shape[-2:]
        flat = score.flatten().cpu()
        score = score.cpu()
        t0 = t[0].cpu()
        while True:
            idx = int(flat.argmin())
            if not torch.isfinite(flat[idx]):
                break
            f, rem = divmod(idx, h * w)
            y, xx = divmod(rem, w)
            patch = x[0, :, y : y + k, xx : xx + k]
            pre_spiked = patch <= t0[f, y, xx].to(x.device)
            sgn = pre_spiked.float() * (a_plus + a_minus) - a_minus
            wf = weights[f]
            weights[f] = (wf + sgn * wf * (1 - wf)).clamp_(0, 1)
            sc = flat.view(-1, h, w)
            sc[f] = float("inf")
            sc[:, max(0, y - radius) : y + radius + 1, max(0, xx - radius) : xx + radius + 1] = (
                float("inf")
            )
        used = n + 1
        if n % 200 == 199:
            c = (weights * (1 - weights)).mean().item()
            print(f"  {log}: image {n + 1}, C = {c:.4f}", flush=True)
            if c < CONVERGED:
                break
    return used


def fit_logreg(
    x: torch.Tensor, y: torch.Tensor, *, epochs: int = 300, wd: float = 1e-4
) -> torch.nn.Module:
    """Multinomial logistic regression (full-batch Adam) on standardised features."""
    model = torch.nn.Linear(x.shape[1], 10).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return model


def run(args: argparse.Namespace) -> dict[str, float]:
    """Train both layers, fit the classifier and return the metrics."""
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    tr_x, tr_y, te_x, te_y = load_mnist(args.data, download=True)
    n_clf = args.classifier_samples
    n_stdp = args.train_samples
    perm = torch.randperm(len(tr_x), generator=gen)
    stdp_idx, clf_idx = perm[:n_stdp], perm[n_stdp : n_stdp + n_clf]
    if len(clf_idx) < n_clf:
        clf_idx = perm[:n_clf]
    t0 = time.time()

    def prep(images: torch.Tensor) -> torch.Tensor:
        return encode(images.to(dev), threshold=args.input_threshold)

    def init(maps: int, ch: int) -> torch.Tensor:
        return (0.8 + 0.05 * torch.randn(maps, ch, 5, 5, generator=gen)).clamp(0, 1).to(dev)

    w1, w2 = init(args.maps1, 2), init(args.maps2, args.maps1)
    kw = {"a_plus": args.a_plus, "a_minus": args.a_minus, "radius": args.radius}
    inp = prep(tr_x[stdp_idx])
    print(f"encoded {len(inp)} images in {time.time() - t0:.1f}s", flush=True)
    u1 = train_layer(inp, w1, args.thr1, log="conv1", **kw)
    print(f"conv1 trained on {u1} images ({time.time() - t0:.0f}s)", flush=True)
    l1 = layer1(inp, w1, args.thr1)
    u2 = train_layer(l1, w2, args.thr2, log="conv2", **kw)
    print(f"conv2 trained on {u2} images ({time.time() - t0:.0f}s)", flush=True)

    xt = features(prep(tr_x[clf_idx]), w1, w2, args.thr1, grid=args.grid)
    xe = features(prep(te_x[: args.test_samples]), w1, w2, args.thr1, grid=args.grid)
    mu, sd = xt.mean(0), xt.std(0).clamp_min(1e-6)
    model = fit_logreg((xt - mu) / sd, tr_y[clf_idx].to(dev))
    with torch.no_grad():
        pred_te = model((xe - mu) / sd).argmax(1).cpu()
        pred_tr = model((xt - mu) / sd).argmax(1).cpu()
    res = {
        "train_accuracy": accuracy(pred_tr, tr_y[clf_idx]),
        "test_accuracy": accuracy(pred_te, te_y[: args.test_samples]),
        "images_conv1": u1,
        "images_conv2": u2,
        "seconds": time.time() - t0,
    }
    print(json.dumps(res))
    return res


def main() -> None:
    """Parse the command line and run."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/MNIST/raw"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-samples", type=int, default=3000, help="STDP images per layer")
    p.add_argument("--classifier-samples", type=int, default=10000)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--maps1", type=int, default=30)
    p.add_argument("--maps2", type=int, default=100)
    p.add_argument("--thr1", type=float, default=15.0)
    p.add_argument("--thr2", type=float, default=10.0)
    p.add_argument("--a-plus", type=float, default=A_PAPER[0] * LEARNING_SCALE)
    p.add_argument("--a-minus", type=float, default=A_PAPER[1] * LEARNING_SCALE)
    p.add_argument("--radius", type=int, default=2, help="learning inhibition radius")
    p.add_argument("--input-threshold", type=float, default=0.1)
    p.add_argument("--grid", type=int, default=1, help="1 = global pooling (paper)")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
