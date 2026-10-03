"""Supervised learning with predictive coding on MNIST (Whittington and Bogacz 2017).

Whittington and Bogacz (2017, Neural Computation 29:1229) showed that a hierarchical
predictive coding network, where every layer predicts the layer below and weights change
with local Hebbian-like rules from prediction errors, approximates backpropagation when it
is trained with the input clamped at the top and the target at the bottom.

Here a :class:`~neurosush.predictive_coding.PredictiveCodingNetwork` with layers
``[10, hidden, 784]`` has the one-hot label at layer 0 and the image (scaled to [0, 1]) at
the top. Training: clamp both, relax the hidden layer for ``--infer-steps`` steps, then one
local weight update per mini-batch. Testing: the top-down prediction ``predict(image)[0]``;
the class is its argmax.

Deviations from the paper: the paper's experiments use small settings and analytic tests;
this is a plain tanh MLP of one hidden layer trained with mini-batches by a fixed number of
inference steps, with the hidden layer's variance lowered to ``--hidden-variance`` so that
the weight updates approach backprop's (the paper's limit). Weights are initialized
randomly, there are no biases, and no hyper-parameter search was done.

Run: ``python examples/predictive_coding_mnist.py --data path/to/MNIST/raw`` (a folder of
IDX files; downloaded when missing). Defaults: 10 epochs on 10000 training samples,
128 hidden units, mini-batches of 20.

Measured on CPU with the defaults (seed 0, about 30 s): test accuracy on the 10000 test
images 0.838 after the first epoch, 0.907 after five and 0.915 after ten. For reference a
fully trained MLP reaches about 98 %; this small run is not tuned for that.
"""

from __future__ import annotations

import argparse
import time

import torch

from neurosush.data import load_mnist
from neurosush.predictive_coding import PredictiveCodingNetwork


def train_epoch(
    net: PredictiveCodingNetwork,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch: int = 20,
    infer_steps: int = 50,
    infer_rate: float = 0.1,
    rate: float = 0.1,
    generator: torch.Generator | None = None,
) -> None:
    """One pass over ``images`` (``(n, 784)`` floats in [0, 1]) with one-hot ``labels``."""
    order = torch.randperm(len(images), generator=generator)
    for start in range(0, len(images), batch):
        idx = order[start : start + batch]
        x = images[idx]
        target = torch.nn.functional.one_hot(labels[idx], net.sizes[0]).to(net.dtype)
        mu = net.infer(target, top=x, steps=infer_steps, rate=infer_rate)
        net.learn(mu, rate)


def accuracy(net: PredictiveCodingNetwork, images: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of ``images`` whose predicted label (argmax of layer 0) is right."""
    with torch.no_grad():
        predicted = torch.cat([net.predict(chunk)[0].argmax(-1) for chunk in images.split(1000)])
    return float((predicted == labels).float().mean())


def run(
    images: torch.Tensor,
    labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    hidden: int = 128,
    epochs: int = 1,
    hidden_variance: float = 0.1,
    seed: int = 0,
    **kwargs: float,
) -> list[float]:
    """Train and return the test accuracy after every epoch."""
    net = PredictiveCodingNetwork(
        [10, hidden, 784], activation="tanh", variances=[1.0, hidden_variance, 1.0], seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(epochs):
        train_epoch(net, images, labels, generator=generator, **kwargs)
        scores.append(accuracy(net, test_images, test_labels))
    return scores


def main() -> None:
    """Load MNIST, train and print the test accuracy."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/MNIST/raw", help="folder of MNIST IDX files")
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--hidden-variance", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--infer-steps", type=int, default=50)
    p.add_argument("--infer-rate", type=float, default=0.1)
    p.add_argument("--rate", type=float, default=0.1, help="weight learning rate")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    xtr, ytr, xte, yte = load_mnist(args.data, download=True)
    xtr = xtr.reshape(len(xtr), -1).float()[: args.train_samples] / 255.0
    xte = xte.reshape(len(xte), -1).float()[: args.test_samples] / 255.0
    ytr, yte = ytr[: args.train_samples], yte[: args.test_samples]
    t0 = time.perf_counter()
    scores = run(
        xtr,
        ytr,
        xte,
        yte,
        hidden=args.hidden,
        epochs=args.epochs,
        hidden_variance=args.hidden_variance,
        seed=args.seed,
        batch=args.batch,
        infer_steps=args.infer_steps,
        infer_rate=args.infer_rate,
        rate=args.rate,
    )
    for epoch, score in enumerate(scores, 1):
        print(f"epoch {epoch}: test accuracy {score:.4f}")
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
