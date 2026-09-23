import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.synapses.currents import (
    AvgPool2dInput,
    Conv2dInput,
    DenseInput,
    LateralInput,
    Local2dInput,
    OneToOneInput,
    SparseInput,
)
from neurosush.synapses.init import WeightInit


def build(src_shape, dst_shape, init, input_behavior, inhibitory=False, same_group=False):
    net = Network()
    src = NeuronGroup(net, src_shape, inhibitory=inhibitory)
    dst = src if same_group else NeuronGroup(net, dst_shape)
    behaviors = [input_behavior] if init is None else [init, input_behavior]
    syn = SynapseGroup(net, src, dst, behaviors=behaviors)
    net.initialize()
    return syn


class TestDenseInput:
    def test_initial_state(self):
        syn = build(2, 3, WeightInit(mode="ones"), DenseInput())
        assert syn.I.tolist() == [0.0, 0.0, 0.0]
        assert syn.pre_spike.tolist() == [False, False]
        assert syn.connectivity == "dense"
        assert syn.input is syn.behaviors[1]

    def test_forward_uses_pre_spikes_and_coefficient(self):
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        syn = build(2, 2, WeightInit(weights=w), DenseInput(coef=0.5))
        syn.pre_spike = torch.tensor([True, True])
        syn.behaviors[1].forward(syn)
        assert syn.I.tolist() == [2.0, 3.0]

    def test_inhibitory_source_gives_negative_current(self):
        syn = build(1, 1, WeightInit(mode="ones"), DenseInput(), inhibitory=True)
        syn.pre_spike = torch.tensor([True])
        syn.behaviors[1].forward(syn)
        assert syn.I.tolist() == [-1.0]

    def test_needs_weights(self):
        with pytest.raises(RuntimeError, match="WeightInit"):
            build(2, 2, None, DenseInput())

    def test_weight_shape_is_checked(self):
        with pytest.raises(ValueError, match="shape"):
            build(2, 3, WeightInit(mode="ones", shape=(3, 2)), DenseInput())


class TestOneToOneInput:
    def test_forward(self):
        syn = build(
            3, 3, WeightInit(weights=torch.tensor([1.0, 2.0, 3.0]), shape=(3,)), OneToOneInput()
        )
        syn.pre_spike = torch.tensor([True, False, True])
        syn.behaviors[1].forward(syn)
        assert syn.I.tolist() == [1.0, 0.0, 3.0]
        assert syn.connectivity == "one_to_one"

    def test_sizes_must_match(self):
        with pytest.raises(ValueError, match="size"):
            build(3, 2, WeightInit(mode="ones", shape=(3,)), OneToOneInput())


class TestSparseInput:
    def test_forward(self):
        syn = build(4, 4, WeightInit(mode="ones", density=0.5, sparse=True), SparseInput())
        syn.pre_spike = torch.ones(4, dtype=torch.bool)
        syn.behaviors[1].forward(syn)
        assert syn.I.sum().item() == 8.0
        assert syn.connectivity == "sparse"

    def test_needs_sparse_weights(self):
        with pytest.raises(RuntimeError, match="sparse"):
            build(2, 2, WeightInit(mode="ones"), SparseInput())


class TestConv2dInput:
    def test_forward(self):
        syn = build(
            (1, 3, 3), (2, 2, 2), WeightInit(mode="ones", shape=(2, 1, 2, 2)), Conv2dInput()
        )
        syn.pre_spike = torch.ones(9, dtype=torch.bool)
        syn.behaviors[1].forward(syn)
        assert syn.I.tolist() == [4.0] * 8
        assert syn.connectivity == "conv2d"

    @pytest.mark.parametrize(
        ("dst_shape", "weight_shape", "match"),
        [
            ((2, 2, 2), (2, 2, 2, 2), "in_channels"),
            ((3, 2, 2), (2, 1, 2, 2), "out_channels"),
            ((2, 3, 2), (2, 1, 2, 2), "height"),
            ((2, 2, 3), (2, 1, 2, 2), "width"),
        ],
    )
    def test_shape_validation(self, dst_shape, weight_shape, match):
        with pytest.raises(ValueError, match=match):
            build((1, 3, 3), dst_shape, WeightInit(mode="ones", shape=weight_shape), Conv2dInput())

    def test_stride_and_padding(self):
        syn = build(
            (1, 4, 4),
            (1, 3, 3),
            WeightInit(mode="ones", shape=(1, 1, 2, 2)),
            Conv2dInput(stride=2, padding=1),
        )
        syn.pre_spike = torch.ones(16, dtype=torch.bool)
        syn.behaviors[1].forward(syn)
        assert syn.I.view(3, 3)[0, 0].item() == 1.0
        assert syn.I.view(3, 3)[1, 1].item() == 4.0


class TestLocal2dInput:
    def test_forward_and_validation(self):
        syn = build(
            (1, 2, 2),
            (1, 1, 2),
            WeightInit(mode="ones", shape=(1, 2, 2)),
            Local2dInput(kernel_size=(2, 1)),
        )
        syn.pre_spike = torch.ones(4, dtype=torch.bool)
        syn.behaviors[1].forward(syn)
        assert syn.I.tolist() == [2.0, 2.0]
        assert syn.connectivity == "local2d"

    def test_weight_shape_must_match_geometry(self):
        with pytest.raises(ValueError, match="shape"):
            build(
                (1, 2, 2),
                (1, 1, 2),
                WeightInit(mode="ones", shape=(1, 3, 2)),
                Local2dInput(kernel_size=(2, 1)),
            )


class TestLateralInput:
    def test_forward(self):
        w = torch.ones(1, 1, 1, 3, 3)
        syn = build(
            (1, 3, 3),
            None,
            WeightInit(weights=w, shape=(1, 1, 1, 3, 3)),
            LateralInput(),
            same_group=True,
        )
        syn.pre_spike = torch.zeros(9, dtype=torch.bool)
        syn.pre_spike[4] = True
        syn.behaviors[1].forward(syn)
        assert syn.I.tolist() == [1.0] * 9

    def test_requires_same_group(self):
        with pytest.raises(ValueError, match="same group"):
            build(
                (1, 3, 3), (1, 3, 3), WeightInit(mode="ones", shape=(1, 1, 1, 3, 3)), LateralInput()
            )

    def test_requires_odd_kernel(self):
        with pytest.raises(ValueError, match="odd"):
            build(
                (1, 3, 3),
                None,
                WeightInit(mode="ones", shape=(1, 1, 1, 2, 3)),
                LateralInput(),
                same_group=True,
            )


class TestAvgPool2dInput:
    def test_forward_without_weights(self):
        syn = build((2, 4, 4), (2, 2, 2), None, AvgPool2dInput())
        syn.pre_spike = torch.ones(32, dtype=torch.bool)
        syn.behaviors[0].forward(syn)
        assert syn.I.tolist() == [1.0] * 8

    def test_depth_must_match(self):
        with pytest.raises(ValueError, match="depth"):
            build((2, 4, 4), (3, 2, 2), None, AvgPool2dInput())
