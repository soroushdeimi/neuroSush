"""Read out a label from an unsupervised spiking layer (Diehl and Cook 2015).

Learning uses no labels. Afterwards each neuron is assigned the class it responds to most
on labelled samples (:func:`assign_labels`), and a new sample gets the class whose assigned
neurons respond most (:func:`classify`). Counts have shape ``(samples, neurons)``, for
example the ``spike_count`` that :class:`~neurosush.recording.SpikeCounter` leaves on a
group after each sample.
"""

from __future__ import annotations

import torch
from torch import Tensor


def assign_labels(counts: Tensor, labels: Tensor, classes: int = 10) -> Tensor:
    """The class each neuron responds to most, on average over the samples of that class.

    Args:
        counts: Responses ``(samples, neurons)``.
        labels: Class of every sample ``(samples,)``, in ``[0, classes)``.
        classes: Number of classes.

    Returns:
        Long tensor ``(neurons,)``: the class with the highest mean response (the lowest
        class on a tie), or ``-1`` for a neuron with no response on any sample, which
        :func:`classify` ignores. A class without samples has mean response 0.

    Raises:
        ValueError: If the shapes do not match or a label is outside ``[0, classes)``.
    """
    if counts.dim() != 2:
        raise ValueError(f"counts must have shape (samples, neurons), got {tuple(counts.shape)}")
    if labels.shape != counts.shape[:1]:
        raise ValueError(f"labels must have shape ({counts.shape[0]},), got {tuple(labels.shape)}")
    _check_labels(labels, classes)
    counts = counts.to(torch.get_default_dtype()) if not counts.is_floating_point() else counts
    mean = torch.zeros(classes, counts.shape[1], dtype=counts.dtype, device=counts.device)
    for c in range(classes):
        mask = labels == c
        if mask.any():
            mean[c] = counts[mask].mean(0)
    assignment = mean.argmax(0)
    return assignment.masked_fill(counts.sum(0) == 0, -1)


def classify(counts: Tensor, assignment: Tensor, classes: int = 10) -> Tensor:
    """The class whose assigned neurons have the highest mean response.

    Args:
        counts: Responses ``(samples, neurons)``.
        assignment: Class of every neuron ``(neurons,)`` from :func:`assign_labels`;
            ``-1`` neurons are ignored.
        classes: Number of classes.

    Returns:
        Long tensor ``(samples,)``: the winning class (the lowest on a tie). A class with no
        assigned neuron scores ``-inf``, so it wins only when no class has a neuron, which
        gives class 0.

    Raises:
        ValueError: If the shapes do not match.
    """
    if counts.dim() != 2:
        raise ValueError(f"counts must have shape (samples, neurons), got {tuple(counts.shape)}")
    if assignment.shape != counts.shape[1:]:
        raise ValueError(
            f"assignment must have shape ({counts.shape[1]},), got {tuple(assignment.shape)}"
        )
    counts = counts.to(torch.get_default_dtype()) if not counts.is_floating_point() else counts
    votes = torch.full(
        (counts.shape[0], classes), -float("inf"), dtype=counts.dtype, device=counts.device
    )
    for c in range(classes):
        mask = assignment == c
        if mask.any():
            votes[:, c] = counts[:, mask].mean(1)
    return votes.argmax(1)


def accuracy(pred: Tensor, labels: Tensor) -> float:
    """The fraction of predictions equal to the labels.

    Raises:
        ValueError: If the shapes differ or there are no samples.
    """
    if pred.shape != labels.shape:
        raise ValueError(
            f"pred and labels must match, got {tuple(pred.shape)}, {tuple(labels.shape)}"
        )
    if pred.numel() == 0:
        raise ValueError("accuracy needs at least one sample")
    return float((pred == labels.to(pred.device)).float().mean())


def _check_labels(labels: Tensor, classes: int) -> None:
    if classes < 1:
        raise ValueError(f"classes must be positive, got {classes}")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= classes):
        raise ValueError(f"labels must be in [0, {classes}), got {labels.tolist()}")
