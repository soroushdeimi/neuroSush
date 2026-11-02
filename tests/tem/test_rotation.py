"""Exact geometry and interface checks for torus path integration."""

import math

import pytest
import torch

from neurosush.tem.rotation import Rotation


def rotation() -> Rotation:
    result = Rotation(4, torch.Generator().manual_seed(8))
    with torch.no_grad():
        result.angles.copy_(torch.tensor([[math.pi / 2, 0.2], [0.3, -0.4]]))
    return result


def test_hand_rotation_and_stay() -> None:
    module = rotation()
    g = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        module(g, torch.tensor([0])), torch.tensor([[0.0, 1.0, 0.0, 0.0]]), atol=1e-6, rtol=0
    )
    assert torch.equal(module(g, torch.tensor([4])), g)


def test_norm_inverse_loop_and_commutation() -> None:
    module = rotation()
    g = torch.tensor([[3.0, 4.0, -2.0, 7.0]]).repeat(5, 1)
    actions = torch.arange(5)
    moved = module(g, actions)
    torch.testing.assert_close(moved.reshape(5, 2, 2).norm(dim=-1), g.reshape(5, 2, 2).norm(dim=-1))
    torch.testing.assert_close(module(moved, torch.tensor([1, 0, 3, 2, 4])), g, atol=1e-6, rtol=0)
    loop = g
    for action in (0, 2, 1, 3):
        loop = module(loop, torch.full((5,), action))
    torch.testing.assert_close(loop, g, atol=1e-6, rtol=0)
    right, up = torch.zeros(5, dtype=torch.long), torch.full((5,), 2)
    torch.testing.assert_close(module(module(g, right), up), module(module(g, up), right))


def test_closed_form_and_gradients() -> None:
    module = rotation().double()
    g = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    moved = g
    for _ in range(17):
        moved = module(moved, torch.tensor([0]))
    angle = 17 * module.angles[0]
    x, y = g.reshape(1, 2, 2).unbind(-1)
    expected = torch.stack(
        (x * angle.cos() - y * angle.sin(), x * angle.sin() + y * angle.cos()), -1
    )
    torch.testing.assert_close(moved, expected.flatten(1))
    torch.autograd.backward(moved.sum())
    assert g.grad is not None
    assert bool(torch.isfinite(g.grad).all())
    assert module.angles.grad is not None
    assert bool(torch.isfinite(module.angles.grad).all())
    assert bool((module.angles.grad[0] != 0).all())


@pytest.mark.parametrize("size", [0, 3, -2, True])
def test_invalid_size(size: int) -> None:
    with pytest.raises(ValueError, match=f"size.*{size}"):
        Rotation(size, torch.Generator())


@pytest.mark.parametrize(
    ("g", "action", "match"),
    [
        (torch.zeros(1, 3), torch.tensor([0]), "g.*3"),
        (torch.zeros(1, 4), torch.tensor([[0]]), "action shape"),
        (torch.zeros(1, 4), torch.tensor([5]), "action.*5"),
        (torch.zeros(1, 4), torch.tensor([-1]), "action.*-1"),
        (torch.zeros(1, 4), torch.tensor([0.0]), "action dtype.*float32"),
    ],
)
def test_invalid_inputs(g: torch.Tensor, action: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rotation()(g, action)


def test_model_hook_and_config() -> None:
    from neurosush.tem import TEM, TEMConfig

    config = TEMConfig(modules=(4, 2), transition="rotation")
    model = TEM(config, generator=torch.Generator().manual_seed(7))
    state = model.reset(2)
    for i, g in enumerate(state.g):
        torch.testing.assert_close(g.norm(dim=-1), torch.ones(2))
        assert isinstance(model.transitions[i], Rotation)
        moved = model.transition(g * 4, torch.tensor([0, 2]), i)
        torch.testing.assert_close(moved.norm(dim=-1), torch.full((2,), 4.0))
        torch.testing.assert_close(
            model.transition(moved, torch.tensor([1, 3]), i), g * 4, atol=1e-6, rtol=0
        )
    with pytest.raises(ValueError, match=r"modules.*3"):
        TEMConfig(modules=(3,), transition="rotation")
    with pytest.raises(ValueError, match=r"transition.*unknown"):
        TEMConfig(transition="unknown")  # type: ignore[arg-type]
