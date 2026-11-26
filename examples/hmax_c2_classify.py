"""Analog HMAX classification of horizontal vs vertical bars (Serre et al. 2007, Sec. 2).

The reference solution for the WP8 task: build a bar dataset, extract C2 features with
the ``neurosush.hmax`` building blocks, learn C1 prototypes from the training set, and
fit a small softmax classifier on the C2 features.

Run: ``python examples/hmax_c2_classify.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurosush.hmax.c1 import c1_bands_from_spec, c1_pool
from neurosush.hmax.c2 import c2_global_max
from neurosush.hmax.gabor import SERRE2007_BANDS, serre2007_gabor_bank
from neurosush.hmax.probe import bar_image
from neurosush.hmax.s1 import s1_response
from neurosush.hmax.s2 import s2_rbf, sample_c1_patches

BAR_LENGTHS = (15, 19, 23, 27, 31)
BAR_WIDTH = 3
NOISE_SCALE = 0.05
BATCH_SIZE = 16


def _c1_of_images(images: torch.Tensor, bank: list[torch.Tensor]) -> list[torch.Tensor]:
    """C1 maps (bands 0-3) of a batch of images.

    Args:
        images: Float tensor of shape (B, 1, H, W).
        bank: The Serre 2007 Gabor bank.

    Returns:
        One C1 tensor ``(B, T, H_b, W_b)`` per band, for the first four bands.
    """
    s1 = s1_response(images, bank, padding="same")
    return c1_pool(s1, c1_bands_from_spec(list(SERRE2007_BANDS[:4])))


def make_bar_dataset(
    n_per_class: int, size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """A dataset of horizontal (class 0) and vertical (class 1) bars.

    Each bar has width 3, a length drawn uniformly from ``BAR_LENGTHS``, and a centre
    drawn uniformly so the bar fits inside the image. Uniform noise of scale
    ``NOISE_SCALE`` is added. All class-0 images come first, then all class-1 images.

    Args:
        n_per_class: Number of images per class; must be at least 1.
        size: Side length of each square image; must be at least 40.
        generator: Torch generator for all randomness.

    Returns:
        ``(images, labels)`` where images has shape ``(2 * n_per_class, 1, size, size)``
        in float32 and labels has shape ``(2 * n_per_class,)`` in long.

    Raises:
        ValueError: If n_per_class < 1 or size < 40.
    """
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be at least 1, got {n_per_class}")
    if size < 40:
        raise ValueError(f"size must be at least 40, got {size}")

    images: list[torch.Tensor] = []
    for orientation in ("h", "v"):
        for _ in range(n_per_class):
            length = int(torch.randint(len(BAR_LENGTHS), (1,), generator=generator).item())
            length = BAR_LENGTHS[length]
            if orientation == "h":
                row = int(torch.randint(size, (1,), generator=generator).item())
                col = int(
                    torch.randint(length, size - length + 1, (1,), generator=generator).item()
                )
            else:
                row = int(
                    torch.randint(length, size - length + 1, (1,), generator=generator).item()
                )
                col = int(torch.randint(size, (1,), generator=generator).item())
            img = bar_image(size, length, BAR_WIDTH, orientation, (row, col))
            img = img + NOISE_SCALE * torch.rand(size, size, generator=generator)
            images.append(img)

    return torch.stack(images).unsqueeze(1), torch.cat(
        [torch.zeros(n_per_class, dtype=torch.long), torch.ones(n_per_class, dtype=torch.long)]
    )


def hmax_c2(
    images: torch.Tensor,
    prototypes_by_size: dict[int, torch.Tensor],
    bands: tuple[int, ...] = (0, 1, 2, 3),
    beta: float = 0.05,
) -> torch.Tensor:
    """C2 features of every image: global max of the S2 responses, per prototype.

    For each image the pipeline runs S1 (Serre 2007 bank, same padding), C1 (first
    ``bands`` bands), then, for every prototype size, the S2 response of all bands whose
    C1 map is at least that size wide, pooled with a global max. The per-size results are
    concatenated over sizes. Images are processed in batches of ``BATCH_SIZE``.

    Args:
        images: Float tensor of shape (N, 1, H, W).
        prototypes_by_size: Mapping from prototype side length ``n`` to a tensor
            ``(P, T, n, n)`` of stored C1 prototypes.
        bands: C1 band indices to use; must be a non-empty subset of 0-3.
        beta: Inverse squared width of the radial basis.

    Returns:
        A tensor of shape ``(N, P_total)`` where ``P_total`` is the sum of the prototype
        counts over all sizes.

    Raises:
        ValueError: If images is not 4-D with one channel, bands is empty or out of
            range, or a prototype tensor is not 4-D.
    """
    if images.dim() != 4 or images.shape[1] != 1:
        raise ValueError(f"images must be 4-D (N, 1, H, W), got shape {tuple(images.shape)}")
    if not bands:
        raise ValueError("bands must not be empty")
    if min(bands) < 0 or max(bands) > 3:
        raise ValueError(f"bands must be a subset of 0-3, got {bands}")
    for n, protos in prototypes_by_size.items():
        if protos.dim() != 4:
            raise ValueError(
                f"prototypes for size {n} must be 4-D, got shape {tuple(protos.shape)}"
            )

    bank = serre2007_gabor_bank()
    sizes = sorted(prototypes_by_size)
    n_images = images.shape[0]
    out = torch.empty(n_images, sum(protos.shape[0] for protos in prototypes_by_size.values()))

    for start in range(0, n_images, BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE]
        c1 = _c1_of_images(batch, bank)
        cols = 0
        for n in sizes:
            protos = prototypes_by_size[n]
            s2 = [s2_rbf(c1[b], protos, beta=beta) for b in bands if c1[b].shape[-1] >= n]
            out[start : start + batch.shape[0], cols : cols + protos.shape[0]] = c2_global_max(s2)
            cols += protos.shape[0]
    return out


def learn_prototypes(
    train_images: torch.Tensor,
    sizes: tuple[int, ...] = (4, 8),
    per_size: int = 20,
    generator: torch.Generator | None = None,
) -> dict[int, torch.Tensor]:
    """Sample C1 prototypes from the training images, per prototype size.

    The C1 maps (bands 0-3) of every training image are computed once. For each size,
    ``per_size`` patches are drawn: a training image index is chosen uniformly from the
    generator, then ``sample_c1_patches`` draws one patch of that size from the image's
    C1 bands.

    Args:
        train_images: Float tensor of shape (N, 1, H, W).
        sizes: Prototype side lengths; must be non-empty.
        per_size: Number of patches per size; must be at least 1.
        generator: Torch generator for all randomness.

    Returns:
        A mapping from each size to a tensor ``(per_size, T, n, n)`` of patches.

    Raises:
        ValueError: If sizes is empty, per_size < 1, or train_images is empty.
    """
    if not sizes:
        raise ValueError("sizes must not be empty")
    if per_size < 1:
        raise ValueError(f"per_size must be at least 1, got {per_size}")
    if train_images.shape[0] < 1:
        raise ValueError(f"train_images must not be empty, got {train_images.shape[0]}")
    if generator is None:
        generator = torch.Generator()

    bank = serre2007_gabor_bank()
    c1 = _c1_of_images(train_images, bank)
    n = train_images.shape[0]
    out: dict[int, torch.Tensor] = {}
    for size in sizes:
        patches: list[torch.Tensor] = []
        for _ in range(per_size):
            idx = int(torch.randint(n, (1,), generator=generator).item())
            patches.append(sample_c1_patches([c1[b][idx] for b in range(4)], 1, size, generator))
        out[size] = torch.cat(patches)
    return out


@dataclass(frozen=True)
class LinearModel:
    """A fitted softmax regression: weights, bias and the training standardisation."""

    w: torch.Tensor
    b: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def fit_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.5,
    generator: torch.Generator | None = None,
) -> LinearModel:
    """Fit a softmax regression to ``y`` on the standardised ``x`` by full-batch GD.

    ``x`` is standardised with the training mean and standard deviation (per feature).
    The weights start at zero, so the fit is deterministic for a given ``steps`` and
    ``lr``; ``generator`` is accepted for signature compatibility and unused.

    Args:
        x: Features of shape (N, D).
        y: Labels of shape (N,) in 0..K-1.
        steps: Number of full-batch gradient steps.
        lr: Learning rate.
        generator: Unused; kept for a uniform signature.

    Returns:
        A ``LinearModel`` with the weights, bias and the training mean/std.

    Raises:
        ValueError: If x is not 2-D, y is not 1-D, or the row counts differ.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
    if y.dim() != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must share the row count, got {tuple(x.shape)} vs {tuple(y.shape)}"
        )

    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-8)
    xz = (x - mean) / std
    k = int(y.max().item()) + 1
    w = torch.zeros(k, x.shape[1])
    b = torch.zeros(k)
    for _ in range(steps):
        logits = xz @ w.T + b
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        grad_w = xz.T @ (logp.exp() - torch.nn.functional.one_hot(y, k).float()) / x.shape[0]
        grad_b = (logp.exp() - torch.nn.functional.one_hot(y, k).float()).mean(dim=0)
        w -= lr * grad_w.T
        b -= lr * grad_b
    return LinearModel(w, b, mean, std)


def accuracy(model: LinearModel, x: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of ``y`` predicted correctly by ``model`` on the standardised ``x``.

    Args:
        model: A fitted ``LinearModel``.
        x: Features of shape (N, D).
        y: Labels of shape (N,).

    Returns:
        The accuracy in [0, 1].
    """
    xz = (x - model.mean) / model.std
    return float((xz @ model.w.T + model.b).argmax(dim=1).eq(y).float().mean().item())


def main() -> None:
    """Train and evaluate the HMAX C2 classifier on the bar dataset."""
    for seed in (0, 1):
        g = torch.Generator().manual_seed(seed)
        train = make_bar_dataset(30, 64, g)
        test = make_bar_dataset(30, 64, torch.Generator().manual_seed(seed + 1000))
        protos = learn_prototypes(train[0], sizes=(4, 8), per_size=20, generator=g)
        x_train = hmax_c2(train[0], protos)
        x_test = hmax_c2(test[0], protos)
        model = fit_linear(x_train, train[1])
        print(
            f"seed {seed}: train {accuracy(model, x_train, train[1]):.3f} "
            f"test {accuracy(model, x_test, test[1]):.3f}"
        )


if __name__ == "__main__":
    main()
