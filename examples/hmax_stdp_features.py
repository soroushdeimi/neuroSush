"""Spiking HMAX activity: STDP feature learning on C1 spikes (reference solution).

A single convolutional layer of F = 4 feature neurons learns its 3x3 kernels over the
4 C1 orientation channels by simplified STDP (Kheradpisheh et al. 2018, Sec. 2,
Eq. 3-4) on top of the Serre et al. 2007 front end (TPAMI 29(3), Sec. 2): S1 with the
first scale pair of the Gabor bank, C1 pooling over the single band (0, 1, 8, 4), and
intensity-to-latency encoding of the normalised C1 maps. Trained on horizontal and
vertical bars, the features specialise on the two orthogonal orientations.

Measured with seed 0 (30 bars per class, 3 epochs, THRESH = 2.0): the convergence
index (Eq. 4) falls from 0.2492 to 0.0842 and the dominant orientation channels are
[2, 2, 0, 0] (channel 0 = horizontal axis, channel 2 = vertical axis of the Serre
bank). A sweep of THRESH over 1.0..4.0 keeps the same split at every value.

Run: ``python examples/hmax_stdp_features.py`` (about 10 s on CPU).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from neurosush.encoding import intensity_to_latency
from neurosush.hmax.c1 import c1_pool
from neurosush.hmax.gabor import serre2007_gabor_bank
from neurosush.hmax.probe import bar_image
from neurosush.hmax.s1 import s1_response
from neurosush.hmax.wta import select_conv_winners, simplified_stdp, stdp_convergence

N_FEATURES = 4
A_PLUS = 0.05
A_MINUS = 0.04
THRESH = 2.0
BANK = serre2007_gabor_bank()


def make_images(n_per_class: int, generator: torch.Generator) -> torch.Tensor:
    """Horizontal and vertical bars with noise, one image per row.

    Class 0 is the horizontal bar, class 1 the vertical bar. The centre of each bar
    is drawn so the 15x3 bar fits inside the 32x32 image, and 0.02 * uniform noise
    is added.

    Args:
        n_per_class: Number of images per class.
        generator: Torch generator for the bar centres and the noise.

    Returns:
        A tensor of shape (2 * n_per_class, 1, 32, 32).

    Raises:
        ValueError: If n_per_class < 1.
    """
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be at least 1, got {n_per_class}")

    images = []
    for cls in range(2):
        orientation = "h" if cls == 0 else "v"
        for _ in range(n_per_class):
            # Draw the centre so the 15x3 bar fits inside the 32x32 image.
            if orientation == "h":
                row = int(torch.randint(1, 31, (1,), generator=generator))
                col = int(torch.randint(7, 25, (1,), generator=generator))
            else:
                row = int(torch.randint(7, 25, (1,), generator=generator))
                col = int(torch.randint(1, 31, (1,), generator=generator))
            img = bar_image(32, 15, 3, orientation, (row, col))
            img = img + 0.02 * torch.rand(32, 32, generator=generator)
            images.append(img)
    return torch.stack(images).unsqueeze(1)


def encode(
    image: torch.Tensor, steps: int = 8, threshold: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one image as C1 spikes and first-spike times.

    S1 with the first scale pair of the Serre bank, C1 pooling with the single band
    (0, 1, 8, 4), normalisation to [0, 1], then intensity-to-latency encoding.

    Args:
        image: Float tensor of shape (1, 1, 32, 32) or (32, 32).
        steps: Number of time steps of the spike train.
        threshold: Activity threshold for the encoding.

    Returns:
        A tuple (t_in, spikes): t_in has shape (4, 7, 7) and holds the index of the
        first True step per input neuron (+inf where it never spikes); spikes is the
        bool tensor of shape (steps, 4, 7, 7).

    Raises:
        ValueError: If steps < 1.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    x = image if image.dim() == 4 else image.unsqueeze(0)
    s1 = s1_response(x, BANK[:2])
    c1 = c1_pool(s1, [(0, 1, 8, 4)])[0]
    c1 = c1 / (c1.max() + 1e-8)
    spikes = intensity_to_latency(c1, steps, threshold=threshold).squeeze(1)
    mask = spikes.any(dim=0)
    t_in = torch.where(mask, spikes.int().argmax(dim=0).float(), torch.full((), math.inf))
    return t_in, spikes


def layer_forward(
    spikes: torch.Tensor, weights: torch.Tensor, thresh: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the feature layer over the spike train.

    The membrane potential at step t is the convolution of the cumulative spikes up
    to and including t with the (F, 4, 3, 3) weights.

    Args:
        spikes: Bool spike train of shape (steps, 4, 7, 7).
        weights: Feature weights of shape (F, 4, 3, 3).
        thresh: Potential threshold for a feature neuron to fire.

    Returns:
        A tuple (first_time, pot_final): first_time has shape (F, 5, 5) and holds the
        first t with pot_t >= thresh (+inf where never); pot_final is the potential
        at the last step, shape (F, 5, 5).
    """
    first_time = torch.full((N_FEATURES, 5, 5), math.inf)
    pot_final = None
    for t in range(spikes.shape[0]):
        cum = spikes[: t + 1].sum(dim=0).float().unsqueeze(0)
        pot = F.conv2d(cum, weights).squeeze(0)
        pot_final = pot
        hit = (pot >= thresh) & torch.isinf(first_time)
        first_time = torch.where(hit, torch.full((), float(t)), first_time)
    return first_time, pot_final


def train_step(
    weights: torch.Tensor,
    t_in: torch.Tensor,
    spikes: torch.Tensor,
    thresh: float = THRESH,
    a_plus: float = A_PLUS,
    a_minus: float = A_MINUS,
) -> torch.Tensor:
    """One STDP update on one image.

    The winners are selected from the feature layer's first-spike times and final
    potentials; every winner's kernel is updated with simplified STDP against the
    first-spike times of its 3x3 input window.

    Args:
        weights: Feature weights of shape (F, 4, 3, 3).
        t_in: First-spike times of the input neurons, shape (4, 7, 7).
        spikes: Bool spike train of shape (steps, 4, 7, 7).
        thresh: Potential threshold for a feature neuron to fire.
        a_plus: LTP learning rate.
        a_minus: LTD learning rate magnitude.

    Returns:
        A new weight tensor of shape (F, 4, 3, 3).
    """
    first_time, pot_final = layer_forward(spikes, weights, thresh)
    winners = select_conv_winners(first_time, pot_final, k=N_FEATURES, radius=2)
    weights = weights.clone()
    for f, y, x in winners.tolist():
        pre = t_in[:, y : y + 3, x : x + 3]
        weights[f] = simplified_stdp(weights[f], pre, float(first_time[f, y, x]), a_plus, a_minus)
    return weights


def train(
    images: torch.Tensor,
    generator: torch.Generator,
    epochs: int = 3,
    thresh: float = THRESH,
) -> tuple[torch.Tensor, list[float]]:
    """Learn the feature weights on the images in a fixed order.

    Args:
        images: Tensor of shape (N, 1, 32, 32).
        generator: Torch generator for the initial weights.
        epochs: Number of passes over the images.
        thresh: Potential threshold for a feature neuron to fire.

    Returns:
        A tuple (weights, history): the final weights of shape (F, 4, 3, 3) and the
        convergence index (Eq. 4) before training and after every epoch.
    """
    weights = 0.45 + 0.1 * torch.rand(N_FEATURES, 4, 3, 3, generator=generator)
    history = [float(stdp_convergence(weights))]
    for _ in range(epochs):
        for img in images:
            t_in, spikes = encode(img)
            weights = train_step(weights, t_in, spikes, thresh)
        history.append(float(stdp_convergence(weights)))
    return weights, history


def dominant_orientations(weights: torch.Tensor) -> list[int]:
    """Dominant input channel per feature: argmax over the 4 channels of W[f].sum((1, 2))."""
    return [int(weights[f].sum((1, 2)).argmax()) for f in range(N_FEATURES)]


def main() -> None:
    """Train on 30 bars per class and print the convergence and dominant orientations."""
    generator = torch.Generator().manual_seed(0)
    images = make_images(30, generator)
    weights, history = train(images, torch.Generator().manual_seed(0))
    print(f"convergence: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"dominant orientations: {dominant_orientations(weights)}")


if __name__ == "__main__":
    main()
