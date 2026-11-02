from dataclasses import replace

import pytest
import torch

from neurosush.tem import TEM, TEMConfig, train
from neurosush.tem.loss import loss_weights, total_loss
from neurosush.tem.model import TEMOutput


def test_calibrated_ratios_and_memory_mask() -> None:
    logits = torch.zeros(2, 20)
    g = torch.ones(2, 32)
    p = torch.ones(2, 320)
    output = TEMOutput(logits, logits, logits, g * 0, g, p * 0, p, p * 3)
    c = TEMConfig()
    _, old = total_loss(output, torch.zeros(2, dtype=torch.long), c)
    c = replace(c, calibrated_reductions=True, inference_memory_loss=True)
    _, new = total_loss(
        output, torch.zeros(2, dtype=torch.long), c, mask=torch.tensor([True, False])
    )
    assert new["L_g"] == old["L_g"] * 16
    assert new["L_p"] == old["L_p"] * 160
    assert new["reg_g"] == old["reg_g"] * 32
    assert new["reg_p"] == old["reg_p"] * 320
    assert new["L_p_x"] == 640
    _, empty = total_loss(
        output, torch.zeros(2, dtype=torch.long), c, mask=torch.zeros(2, dtype=torch.bool)
    )
    assert all(value == 0 for value in empty.values())
    assert loss_weights(c, 1000)["L_p_x"] == 0.25
    assert loss_weights(c, 4000)["reg_p"] == 0
    assert loss_weights(c, 2000)["reg_p"] == pytest.approx(0.01)


def test_bounded_activation_hand_value_and_derivative() -> None:
    c = TEMConfig(iterations=1, bounded_attractor_activation=True)
    model = TEM(c, generator=torch.Generator().manual_seed(0))
    cue = torch.tensor([[-5.0, 0.5]], requires_grad=True)
    result = model._retrieve(cue, torch.zeros(1, 2, 2))
    assert result.tolist()[0] == pytest.approx([-0.00008, 0.4])
    torch.autograd.backward(result.sum())
    assert cue.grad is not None
    assert cue.grad[0, 0] == 0
    assert result.min() >= -0.01


def test_empty_memory_and_corrupt_retrieval_precision() -> None:
    c = TEMConfig(reliable_fusion=True)
    model = TEM(c, generator=torch.Generator().manual_seed(0))
    state = model.reset(2)
    for _ in range(5):
        output, state = model(torch.tensor([0, 1]), torch.tensor([4, 4]), state, rate=0)
        torch.testing.assert_close(output.g_inf, output.g_prior, rtol=0, atol=1e-6)
    mean = torch.ones(2, 12, requires_grad=True)
    error = torch.tensor([0.0, 10.0], requires_grad=True)
    sigma = model._memory_sigma(mean, error, 0)
    assert bool((sigma[1].reciprocal().square() < sigma[0].reciprocal().square()).all())
    torch.autograd.backward(sigma.sum())
    assert mean.grad is None
    assert error.grad is None
    assert bool((sigma > c.sigma_floor).all())


def test_flags_off_match_default_exactly() -> None:
    base = TEMConfig()
    explicit = replace(
        base,
        inference_memory_loss=False,
        calibrated_reductions=False,
        reliable_fusion=False,
        bounded_attractor_activation=False,
    )
    a = TEM(base, generator=torch.Generator().manual_seed(17))
    b = TEM(explicit, generator=torch.Generator().manual_seed(17))
    sa, sb = a.reset(1), b.reset(1)
    for label in (0, 2, 1):
        oa, sa = a(torch.tensor([label]), torch.tensor([4]), sa)
        ob, sb = b(torch.tensor([label]), torch.tensor([4]), sb)
        assert torch.equal(oa.g_inf, ob.g_inf)
        assert torch.equal(oa.logits_gen, ob.logits_gen)


def test_term_gradients_and_update_callback() -> None:
    c = TEMConfig(sensory=3, compressed=3, modules=(2,), batch=1, bptt=5, walk_len=10)
    updates: list[int] = []
    result = train(
        c,
        steps=10,
        seed=1,
        log_grads=True,
        grad_every=1,
        on_update=lambda history: updates.append(len(history)),
    )
    assert updates == [1, 2]
    assert "grad_term/L_x_gen" in result.history[-1]
    assert result.history[-1]["grad_term/L_x_gen"] > 0


def test_memory_loss_gradients_and_missing_output() -> None:
    p = torch.tensor([[1.0, 3.0], [10.0, 20.0]], requires_grad=True)
    memory = torch.tensor([[0.0, 1.0], [0.0, 0.0]], requires_grad=True)
    logits = torch.zeros(2, 2)
    g = torch.zeros(2, 2)
    output = TEMOutput(logits, logits, logits, g, g, p, p, memory)
    c = TEMConfig(inference_memory_loss=True)
    _, terms = total_loss(output, torch.tensor([0, 0]), c, mask=torch.tensor([True, False]))
    assert terms["L_p_x"] == 2.5
    torch.autograd.backward(terms["L_p_x"])
    torch.testing.assert_close(p.grad, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(memory.grad, torch.tensor([[-1.0, -2.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="requires p_memory"):
        total_loss(replace(output, p_memory=None), torch.tensor([0, 0]), c)


def test_new_horizon_validation() -> None:
    c = TEMConfig()
    with pytest.raises(ValueError, match="correction_ramp"):
        replace(c, correction_ramp=0)
    with pytest.raises(ValueError, match="reg_p_horizon"):
        replace(c, reg_p_horizon=0)
    with pytest.raises(ValueError, match="reg_g_horizon"):
        replace(c, reg_g_horizon=0)
    with pytest.raises(ValueError, match="sigma_floor"):
        replace(c, sigma_floor=0)
