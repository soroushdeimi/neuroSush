"""Softmax regression from SDRs to class labels."""

from __future__ import annotations

import torch


class SDRClassifier:
    """Maps an SDR to class probabilities ``softmax(W x)`` and learns online.

    One learning step is gradient descent on the cross-entropy of a batch:
    ``W += lr * mean_i (onehot(y_i) - p_i) x_i^T``.

    Args:
        n: SDR size.
        classes: Number of classes.
        lr: Learning rate.
        device: Device of the weights.
    """

    def __init__(
        self, n: int, classes: int, *, lr: float = 0.1, device: torch.device | str = "cpu"
    ) -> None:
        if n < 1 or classes < 2:
            raise ValueError(f"need n >= 1 and classes >= 2, got n={n}, classes={classes}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = lr
        self.weights = torch.zeros(classes, n, device=device)

    def probabilities(self, sdr: torch.Tensor) -> torch.Tensor:
        """Class probabilities of shape ``(*batch, classes)``."""
        return torch.softmax(sdr.to(self.weights.dtype) @ self.weights.T, dim=-1)

    def predict(self, sdr: torch.Tensor) -> torch.Tensor:
        """Most likely class of every SDR."""
        return self.probabilities(sdr).argmax(-1)

    def learn(self, sdr: torch.Tensor, label: torch.Tensor | int) -> None:
        """One gradient step on the cross-entropy of a batch of SDRs and labels."""
        x = sdr.to(self.weights.dtype).reshape(-1, self.weights.shape[1])
        y = torch.as_tensor(label, device=x.device).reshape(-1)
        target = torch.nn.functional.one_hot(y, self.weights.shape[0]).to(x.dtype)
        error = target - self.probabilities(x)
        self.weights += self.lr * error.T @ x / len(x)
