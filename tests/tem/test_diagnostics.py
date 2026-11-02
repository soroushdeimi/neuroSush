import pytest
import torch

from neurosush.tem import TEMConfig, train
from neurosush.tem.diagnostics import gradient_norms, saturation
from neurosush.tem.model import Dense, TEMOutput


def test_saturation_hand_values():
    g = torch.tensor([[-1.0, 1.0, 0.5, -0.5]])
    p = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    logits = torch.zeros(1, 2)
    output = TEMOutput(logits, logits, logits, g, g * 0.5, p, p)
    values = saturation(output)
    assert values["g_prior_cap"].item() == 0.5
    assert values["g_inf_cap"].item() == 0
    assert values["p_gen_cap"].item() == 0.25
    assert values["g_inf_rms"].item() == pytest.approx((0.625 * 0.25) ** 0.5)
    assert values["p_gen_rms"].item() == 0.5


def test_gradient_norms_hand_values_without_clipping_or_mutation():
    layer = Dense(2, 2, torch.Generator().manual_seed(0))
    layer.weight.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    assert gradient_norms(layer) == {"grad/weight": 5.0, "grad/bias": 0.0}
    assert torch.equal(layer.weight.grad, torch.tensor([[3.0, 4.0], [0.0, 0.0]]))


def test_diagnostics_do_not_change_training():
    config = TEMConfig(
        sensory=4, compressed=4, modules=(3,), hidden=5, batch=2, bptt=5, walk_len=10, widths=(2,)
    )
    ordinary = train(config, steps=10, seed=0)
    diagnosed = train(config, steps=10, seed=0, diagnostics=True)
    for a, b in zip(ordinary.history, diagnosed.history, strict=True):
        assert a == {key: b[key] for key in a}
        squared_norm = sum(value**2 for key, value in b.items() if key.startswith("grad/"))
        assert squared_norm**0.5 == pytest.approx(b["grad_norm"], rel=1e-5)
    assert all(
        torch.equal(value, diagnosed.model.state_dict()[name])
        for name, value in ordinary.model.state_dict().items()
    )
