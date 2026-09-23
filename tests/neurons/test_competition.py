import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.competition import KWTA, InherentNoise, kwta_losers
from neurosush.neurons.models import LIF

LIF_ARGS = {"tau": 10.0, "threshold": 0.0, "v_reset": -1.0, "v_rest": -0.5}


class TestKwtaLosers:
    def test_keeps_the_k_highest_above_threshold(self):
        v = torch.tensor([3.0, 1.0, 5.0, -1.0, 2.0])
        losers = kwta_losers(v, 0.0, k=2)
        assert losers.tolist() == [False, True, False, False, True]

    def test_no_losers_when_few_candidates(self):
        v = torch.tensor([3.0, -1.0, -2.0])
        assert not kwta_losers(v, 0.0, k=2).any()

    def test_ties_keep_the_lower_index(self):
        v = torch.tensor([1.0, 1.0, 1.0])
        assert kwta_losers(v, 0.0, k=1).tolist() == [False, True, True]

    def test_per_neuron_threshold(self):
        v = torch.tensor([3.0, 2.0, 1.0])
        losers = kwta_losers(v, torch.tensor([5.0, 0.0, 0.0]), k=1)
        assert losers.tolist() == [False, False, True]

    def test_competition_along_depth(self):
        # shape (depth=2, height=1, width=2): each column competes across depth
        v = torch.tensor([3.0, 1.0, 2.0, 4.0])
        losers = kwta_losers(v, 0.0, k=1, shape=(2, 1, 2), dim=0)
        assert losers.tolist() == [False, True, True, False]

    def test_competition_along_width(self):
        v = torch.tensor([3.0, 1.0, 2.0, 4.0])
        losers = kwta_losers(v, 0.0, k=1, shape=(2, 1, 2), dim=2)
        assert losers.tolist() == [False, True, True, False]

    def test_input_is_not_modified(self):
        v = torch.tensor([3.0, 2.0])
        kwta_losers(v, 0.0, k=1)
        assert v.tolist() == [3.0, 2.0]


class TestKWTA:
    def test_resets_losers_before_firing(self):
        net = Network()
        ng = NeuronGroup(net, 4, behaviors=[LIF(**LIF_ARGS), KWTA(k=1)])
        net.initialize()
        ng.v = torch.tensor([2.0, 3.0, -0.2, 1.0])
        ng.behaviors[1].forward(ng)
        assert ng.v.tolist() == pytest.approx([-1.0, 3.0, -0.2, -1.0])

    def test_uses_group_shape_for_dim(self):
        net = Network()
        ng = NeuronGroup(net, (2, 1, 2), behaviors=[LIF(**LIF_ARGS), KWTA(k=1, dim=0)])
        net.initialize()
        ng.v = torch.tensor([3.0, 1.0, 2.0, 4.0])
        ng.behaviors[1].forward(ng)
        assert ng.v.tolist() == [3.0, -1.0, -1.0, 4.0]

    def test_needs_a_neuron_model(self):
        net = Network()
        NeuronGroup(net, 2, behaviors=[KWTA(k=1)])
        with pytest.raises(RuntimeError, match="model"):
            net.initialize()

    @pytest.mark.parametrize(("kwargs", "match"), [({"k": 0}, "k"), ({"k": 1, "dim": 3}, "dim")])
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            KWTA(**kwargs)


class TestInherentNoise:
    def test_uniform_noise_is_seeded_and_scaled(self):
        def run(seed):
            net = Network(seed=seed)
            ng = NeuronGroup(
                net, 1000, behaviors=[LIF(**LIF_ARGS), InherentNoise(scale=2.0, offset=1.0)]
            )
            net.initialize()
            before = ng.v.clone()
            ng.behaviors[1].forward(ng)
            return ng.v - before

        a, b = run(3), run(3)
        assert torch.equal(a, b)
        assert bool(((a >= 1.0) & (a < 3.0)).all())
        assert a.mean().item() == pytest.approx(2.0, abs=0.1)

    def test_normal_noise(self):
        net = Network(seed=0)
        ng = NeuronGroup(
            net, 5000, behaviors=[LIF(**LIF_ARGS), InherentNoise(scale=0.5, distribution="normal")]
        )
        net.initialize()
        before = ng.v.clone()
        ng.behaviors[1].forward(ng)
        delta = ng.v - before
        assert delta.mean().item() == pytest.approx(0.0, abs=0.05)
        assert delta.std().item() == pytest.approx(0.5, abs=0.05)

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="distribution"):
            InherentNoise(distribution="cauchy")

    def test_runs_between_dynamics_and_fire(self):
        from neurosush.core.order import Order

        assert Order.NEURON_DYNAMICS < InherentNoise.order < KWTA.order < Order.FIRE
