"""Public validation remains checked before trusted module execution."""

import pytest
import torch

from neurosush.tem import TEM, TEMConfig
from neurosush.tem.model import precision_mean


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize(
    ("observation", "action", "message"),
    [
        ([-1], [0], "observation must be in [0, 20), got tensor([-1])"),
        ([20], [0], "observation must be in [0, 20), got tensor([20])"),
        ([0], [-1], "action must be in [0, 4], got tensor([-1])"),
        ([0], [5], "action must be in [0, 4], got tensor([5])"),
    ],
)
def test_public_range_errors(rotation, observation, action, message):
    model = TEM(
        TEMConfig(transition="rotation" if rotation else "matrix"),
        generator=torch.Generator().manual_seed(2),
    )
    with pytest.raises(ValueError, match="must be") as error:
        model(torch.tensor(observation), torch.tensor(action), model.reset(1))
    assert str(error.value) == message


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("index", [2, 3])
def test_public_sigma_errors(value, index):
    inputs = [torch.ones(1) for _ in range(4)]
    inputs[index] = torch.tensor([value])
    with pytest.raises(ValueError, match="must be") as error:
        precision_mean(*inputs)
    name = "sigma_prior" if index == 2 else "sigma_memory"
    assert str(error.value) == f"{name} must be finite and positive, got {[value]}"


@pytest.mark.parametrize("rotation", [False, True])
def test_public_shape_and_dtype_errors(rotation):
    model = TEM(
        TEMConfig(transition="rotation" if rotation else "matrix"),
        generator=torch.Generator().manual_seed(2),
    )
    state = model.reset(1)
    with pytest.raises(ValueError, match="observation must have shape") as error:
        model(torch.tensor([[0]]), torch.tensor([0]), state)
    assert str(error.value) == "observation must have shape (1,), got torch.Size([1, 1])"
    with pytest.raises(ValueError, match="action shape must match") as error:
        model(torch.tensor([0]), torch.tensor([[0]]), state)
    assert str(error.value) == "action shape must match torch.Size([1]), got torch.Size([1, 1])"
    if rotation:
        with pytest.raises(ValueError, match="action dtype must be") as error:
            model(torch.tensor([0]), torch.tensor([0.0]), state)
        assert str(error.value) == "action dtype must be torch.int64, got torch.float32"
    else:
        with pytest.raises(RuntimeError, match="one_hot is only applicable") as error:
            model(torch.tensor([0]), torch.tensor([0.0]), state)
        assert str(error.value) == "one_hot is only applicable to index tensor of type LongTensor."


def test_oracle_does_not_use_action_values():
    model = TEM(TEMConfig(), generator=torch.Generator().manual_seed(2))
    state = model.reset(1)
    output, _ = model(torch.tensor([0]), torch.tensor([99]), state, prior_override=state.g)
    assert torch.equal(output.g_prior, torch.cat(state.g, -1))
