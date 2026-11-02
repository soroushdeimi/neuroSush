import pytest
import torch

from neurosush.core.network import Network
from neurosush.structure.literals import ALLOWED_FUNCTIONS, decode_literals
from neurosush.viz.trace import pack_array


@pytest.mark.parametrize("name", ["os.system", "torch.abs", "missing"])
def test_functions_are_closed_and_report_pointer(name):
    with pytest.raises(ValueError, match="/params/a~1b/0"):
        decode_literals({"a/b": [{"$fn": name}]}, Network(), path="/params")


def test_packed_tensor_and_nested_plain_data():
    tensor = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    result = decode_literals({"x": [{"$tensor": pack_array(tensor)}], "a": "os.system"}, Network())
    assert torch.equal(result["x"][0], tensor)
    assert result["a"] == "os.system"


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("uniform", {"low": -2, "high": 3}),
        ("normal", {"mean": 2, "std": 0.5}),
        ("constant", {"value": 7}),
    ],
)
def test_initializers_use_network_generator(kind, options):
    literal = {"$init": kind, "shape": [3, 4], **options}
    net = Network(seed=7)
    actual = decode_literals(literal, net)
    manual = Network(seed=7)
    if kind == "uniform":
        expected = torch.rand((3, 4), generator=manual.generator) * 5 - 2
    elif kind == "normal":
        expected = torch.randn((3, 4), generator=manual.generator) * 0.5 + 2
    else:
        expected = torch.full((3, 4), 7.0)
    assert torch.equal(actual, expected)
    assert torch.equal(net.generator.get_state(), manual.generator.get_state())


def test_assets_are_resolved_and_tensor_shape_checked():
    net = Network()
    frames = torch.ones(2, 3)
    assert decode_literals({"$asset": "frames"}, net, {"frames": frames}) is frames
    tensor = decode_literals(
        {"$tensor": {"asset": "frames", "shape": [2, 3]}}, net, lambda name: frames
    )
    assert torch.equal(tensor, frames)
    assert tensor.data_ptr() != frames.data_ptr()
    for assets in (None, {}):
        with pytest.raises(ValueError, match="/frames"):
            decode_literals({"$asset": "missing"}, net, assets, "/frames")
    with pytest.raises(ValueError, match="shape mismatch"):
        decode_literals({"$tensor": {"asset": "frames", "shape": [6]}}, net, {"frames": frames})


@pytest.mark.parametrize(
    "literal",
    [
        {"$init": "eval", "shape": [1]},
        {"$init": "normal", "shape": [1], "std": -1},
        {"$init": "uniform", "shape": [1], "high": -1},
        {"$init": "constant", "shape": [1], "extra": 1},
        {"$tensor": {}},
        {"$asset": "x", "$fn": "x"},
    ],
)
def test_malformed_literals_fail_with_pointer(literal):
    with pytest.raises(ValueError, match="/params"):
        decode_literals(literal, Network(), path="/params")


def test_allowlisted_library_initializer():
    assert set(ALLOWED_FUNCTIONS) == {"sparse_random", "fixed_in_degree"}
    fn = decode_literals({"$fn": "fixed_in_degree", "kwargs": {"in_degree": 2}}, Network())
    src, dst = fn(4, 3, generator=torch.Generator().manual_seed(7))
    assert src.shape == dst.shape == (6,)
    assert torch.equal(torch.bincount(dst), torch.tensor([2, 2, 2]))
