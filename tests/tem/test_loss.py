from math import log

import pytest
import torch

from neurosush.tem import TEMConfig
from neurosush.tem.loss import (
    activity_loss,
    sensory_loss,
    sparsity_loss,
    squared_loss,
    total_loss,
)
from neurosush.tem.model import TEMOutput


def test_individual_hand_values():
    logits = torch.tensor([[0.0, log(3)], [0.0, 0.0]])
    target = torch.tensor([1, 0])
    assert sensory_loss(logits, target).item() == pytest.approx((log(4 / 3) + log(2)) / 2)
    a = torch.tensor([[1.0, -3.0], [2.0, 0.0]])
    b = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
    assert squared_loss(a, b).item() == 9 / 4
    assert activity_loss(a).item() == 14 / 4
    assert sparsity_loss(a).item() == 6 / 4


def test_total_terms_and_ramp_hand_values():
    logits = torch.zeros(1, 2)
    g_inf, g_prior = torch.tensor([[1.0, -3.0]]), torch.zeros(1, 2)
    p_inf, p_gen = torch.tensor([[2.0, -2.0]]), torch.zeros(1, 2)
    output = TEMOutput(logits, logits, logits, g_prior, g_inf, p_gen, p_inf)
    config = TEMConfig(sensory=2, compressed=3, modules=(2,), latent_ramp=10)
    loss, terms = total_loss(output, torch.tensor([0]), config, iteration=5)
    expected = {
        "L_x_gen": log(2),
        "L_x_g": log(2),
        "L_x_p": log(2),
        "L_p": 4,
        "L_g": 5,
        "reg_g": 5,
        "reg_p": 2,
    }
    assert {k: v.item() for k, v in terms.items()} == pytest.approx(expected)
    assert loss.item() == pytest.approx(3 * log(2) + 0.5 * 9 + 0.01 * 5 + 0.02 * 2)


def test_mask_and_empty_mask_remain_differentiable():
    logits = torch.zeros(2, 2, requires_grad=True)
    mask = torch.tensor([True, False])
    target = torch.tensor([0, 1])
    assert sensory_loss(logits, target, mask).item() == pytest.approx(log(2))
    empty = sensory_loss(logits, target, ~torch.ones(2, dtype=torch.bool))
    assert empty.item() == 0
    empty.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    with pytest.raises(ValueError, match="mask"):
        sensory_loss(logits, target, torch.ones(2))
