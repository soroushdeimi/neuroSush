import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.synapses.init import DelayInit, WeightInit, sparse_random


def synapse(init, n_src=3, n_dst=4, seed=0, **kwargs):
    net = Network(seed=seed)
    syn = SynapseGroup(
        net, NeuronGroup(net, n_src), NeuronGroup(net, n_dst), behaviors=[init], **kwargs
    )
    net.initialize()
    return syn


class TestSparseRandom:
    def test_exact_count_unique_and_sorted(self):
        src, dst = sparse_random(10, 20, 0.25, generator=torch.Generator().manual_seed(0))
        assert src.numel() == dst.numel() == 50
        pairs = (src * 20 + dst).tolist()
        assert pairs == sorted(set(pairs))
        assert int(src.max()) < 10
        assert int(dst.max()) < 20

    def test_reproducible(self):
        a = sparse_random(5, 5, 0.5, generator=torch.Generator().manual_seed(1))
        b = sparse_random(5, 5, 0.5, generator=torch.Generator().manual_seed(1))
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

    @pytest.mark.parametrize("density", [0.0, 1.5])
    def test_invalid_density(self, density):
        with pytest.raises(ValueError, match="density"):
            sparse_random(3, 3, density)


class TestWeightInit:
    def test_uniform_dense_shape(self):
        syn = synapse(WeightInit(mode="uniform"))
        assert syn.weights.shape == (3, 4)
        assert syn.weights.dtype == torch.float32
        assert bool(((syn.weights >= 0) & (syn.weights < 1)).all())

    def test_seeded(self):
        a = synapse(WeightInit(mode="normal"), seed=5).weights
        b = synapse(WeightInit(mode="normal"), seed=5).weights
        assert torch.equal(a, b)

    def test_constant_scale_and_offset(self):
        syn = synapse(WeightInit(mode=0.5, scale=2.0, offset=-0.25))
        assert syn.weights.tolist() == [[0.75] * 4] * 3

    @pytest.mark.parametrize(("mode", "value"), [("zeros", 0.0), ("ones", 1.0)])
    def test_named_constant_modes(self, mode, value):
        assert synapse(WeightInit(mode=mode)).weights.eq(value).all()

    def test_explicit_weights_are_copied(self):
        w = torch.arange(12, dtype=torch.float64).view(3, 4)
        syn = synapse(WeightInit(weights=w))
        assert syn.weights.dtype == torch.float32
        w[0, 0] = 100.0
        assert syn.weights[0, 0].item() == 0.0

    def test_custom_shape(self):
        syn = synapse(WeightInit(mode="ones", shape=(2, 1, 3, 3)))
        assert syn.weights.shape == (2, 1, 3, 3)

    def test_fn_is_applied_before_scale(self):
        syn = synapse(WeightInit(mode="ones", fn=lambda w: w * 3, scale=2.0))
        assert syn.weights.eq(6.0).all()

    def test_dense_density_masks_weights(self):
        syn = synapse(WeightInit(mode="ones", density=0.5), n_src=100, n_dst=100)
        assert syn.weights.mean().item() == pytest.approx(0.5, abs=0.05)

    def test_sparse_storage(self):
        syn = synapse(WeightInit(mode="ones", density=0.5, sparse=True), n_src=4, n_dst=4)
        assert syn.weights.shape == (8,)
        assert syn.src_idx.shape == syn.dst_idx.shape == (8,)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "mode"),
            ({"mode": "ones", "weights": torch.ones(3, 4)}, "mode"),
            ({"mode": "cauchy"}, "mode"),
            ({"weights": torch.ones(3, 4), "scale": 2.0}, "weights"),
            ({"mode": "ones", "density": 0.0}, "density"),
            ({"mode": "ones", "sparse": True, "shape": (2, 2, 2)}, "sparse"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WeightInit(**kwargs)

    def test_explicit_weights_must_match_the_shape(self):
        with pytest.raises(ValueError, match="shape"):
            synapse(WeightInit(weights=torch.ones(2, 2)))


class TestDelayInit:
    def test_constant_source_delay(self):
        syn = synapse(DelayInit(delays=2))
        assert syn.src_delay.tolist() == [2, 2, 2]
        assert syn.dst_delay.tolist() == [0, 0, 0, 0]

    def test_tensor_destination_delay(self):
        syn = synapse(DelayInit(delays=torch.tensor([0, 1, 2, 3]), side="dst"))
        assert syn.dst_delay.tolist() == [0, 1, 2, 3]
        assert syn.dst_delay.dtype == torch.long

    def test_random_delays_below_max(self):
        syn = synapse(DelayInit(max_delay=3), n_src=200)
        assert set(syn.src_delay.tolist()) == {0, 1, 2}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "delays"),
            ({"delays": 1, "max_delay": 2}, "delays"),
            ({"delays": -1}, "non-negative"),
            ({"max_delay": 0}, "max_delay"),
            ({"delays": 1, "side": "both"}, "side"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DelayInit(**kwargs)

    def test_tensor_of_wrong_size(self):
        with pytest.raises(ValueError, match="shape"):
            synapse(DelayInit(delays=torch.tensor([1, 2])))
