import pytest
import torch

from neurosush.htm.classifier import SDRClassifier
from neurosush.htm.sdr import random_sdr, subsample


def test_update_is_cross_entropy_gradient_descent():
    g = torch.Generator().manual_seed(0)
    clf = SDRClassifier(40, 3, lr=0.5)
    clf.weights = torch.randn(3, 40, generator=g)
    x = random_sdr(40, 6, batch=(5,), generator=g)
    y = torch.tensor([0, 2, 1, 1, 0])
    w = clf.weights.clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(x.float() @ w.T, y)
    loss.backward()
    clf.learn(x, y)
    torch.testing.assert_close(clf.weights, (w - 0.5 * w.grad).detach())


def test_probabilities_sum_to_one_and_start_uniform():
    clf = SDRClassifier(10, 4)
    p = clf.probabilities(random_sdr(10, 3, batch=(2,)))
    torch.testing.assert_close(p, torch.full((2, 4), 0.25))


def test_learns_noisy_prototypes():
    g = torch.Generator().manual_seed(1)
    prototypes = random_sdr(500, 30, batch=(8,), generator=g)
    clf = SDRClassifier(500, 8, lr=0.5)
    for _ in range(40):
        labels = torch.randint(8, (32,), generator=g)
        clf.learn(subsample(prototypes[labels], 18, generator=g), labels)
    labels = torch.randint(8, (200,), generator=g)
    predictions = clf.predict(subsample(prototypes[labels], 18, generator=g))
    assert (predictions == labels).float().mean().item() == 1.0


def test_invalid_arguments():
    with pytest.raises(ValueError, match="classes"):
        SDRClassifier(10, 1)
    with pytest.raises(ValueError, match="lr"):
        SDRClassifier(10, 2, lr=0.0)
