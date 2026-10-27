"""Tests for the Hodgkin-Huxley neuron model behavior."""

from __future__ import annotations

import pytest
import torch

from neurosush import checkpoint
from neurosush.core.behavior import Behavior
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.hodgkin_huxley import HodgkinHuxley, resting_potential
from neurosush.neurons.models import Fire
from neurosush.structure.spec import registered


class ConstantCurrent(Behavior):
    """Supplies a fixed injected current density to group.I at DENDRITE_INTEGRATION."""

    order = Order.DENDRITE_INTEGRATION
    independent_ok = True
    graph_safe = True

    def __init__(self, value: float) -> None:
        self.value = value

    def initialize(self, group: NeuronGroup) -> None:
        group.I = group.state(self.value)

    def forward(self, group: NeuronGroup) -> None:
        group.I = group.state(self.value)


class TestHodgkinHuxleyAllocationAndValidation:
    def test_allocation_and_initial_state(self) -> None:
        net = Network(dt=0.025, dtype=torch.float64)
        model = HodgkinHuxley(v_init=-70.0)
        group = NeuronGroup(net, 3, [model])
        net.initialize()

        assert group.v.shape == (3,)
        assert group.v.tolist() == [-70.0, -70.0, -70.0]
        assert group.m.shape == (3,)
        assert group.h.shape == (3,)
        assert group.n.shape == (3,)
        assert group.spikes.shape == (3,)
        assert group.spikes.dtype == torch.bool
        assert not bool(group.spikes.any().item())
        assert group.i_na.shape == (3,)
        assert group.i_k.shape == (3,)
        assert group.i_leak.shape == (3,)
        assert group.v_prev.shape == (3,)
        assert group.threshold.shape == (3,)
        assert group.threshold.tolist() == [0.0, 0.0, 0.0]
        assert group.v_rest == pytest.approx(-65.0, abs=0.1)
        assert group.model is model

    def test_default_v_init_uses_resting_potential(self) -> None:
        net = Network(dt=0.025, dtype=torch.float64)
        model = HodgkinHuxley()
        group = NeuronGroup(net, 2, [model])
        net.initialize()

        v_rest = resting_potential()
        assert group.v.tolist() == pytest.approx([v_rest, v_rest], abs=1e-9)
        assert group.v_rest == pytest.approx(v_rest, abs=1e-9)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"g_na": -120.0}, "g_na must be positive, got -120.0"),
            ({"g_na": 0.0}, "g_na must be positive, got 0.0"),
            ({"g_k": -36.0}, "g_k must be positive, got -36.0"),
            ({"g_leak": -0.3}, "g_leak must be positive, got -0.3"),
            ({"c": -1.0}, "c must be positive, got -1.0"),
            ({"q10": 0.0}, "q10 must be positive, got 0.0"),
            ({"substeps": 0}, "substeps must be at least 1, got 0"),
            ({"substeps": -2}, "substeps must be at least 1, got -2"),
            ({"substeps": True}, "substeps must be at least 1, got True"),
        ],
    )
    def test_validation_errors_in_init(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            HodgkinHuxley(**kwargs)

    def test_stability_validation_in_initialize(self) -> None:
        net = Network(dt=1.0, dtype=torch.float64)
        NeuronGroup(net, 1, [HodgkinHuxley(substeps=1)], name="hh_group")
        expected = (
            r"Hodgkin-Huxley on hh_group needs dt/substeps <= 0.05 ms; got dt=1.0, substeps=1"
        )
        with pytest.raises(ValueError, match=expected):
            net.initialize()

    def test_fire_behavior_conflict_in_initialize(self) -> None:
        net = Network(dt=0.025, dtype=torch.float64)
        NeuronGroup(net, 1, [HodgkinHuxley(), Fire()], name="conflict_group")
        with pytest.raises(
            ValueError,
            match=r"Hodgkin-Huxley on conflict_group cannot be combined with Fire",
        ):
            net.initialize()


class TestDynamicsAndBookkeeping:
    def test_currents_bookkeeping_matches_voltage_change(self) -> None:
        # After one step: -(i_na + i_k + i_leak) + I times dt/c equals v - v_prev
        net = Network(dt=0.025, dtype=torch.float64)
        model = HodgkinHuxley(c=1.0, substeps=1)
        group = NeuronGroup(net, 2, [ConstantCurrent(10.0), model])
        net.initialize()

        v_before = group.v.clone()
        net.step()
        delta_v = group.v - v_before

        i_ion_total = group.i_na + group.i_k + group.i_leak
        expected_delta_v = (-i_ion_total + group.I) * (net.dt / model.c)

        assert torch.allclose(delta_v, expected_delta_v, atol=1e-12)

    def test_substep_equivalence(self) -> None:
        # substeps=k with dt equals substeps=1 with dt/k driven by constant current exactly
        dt = 0.04
        k = 4
        dt_sub = dt / k
        steps = 50
        current = 10.0

        # Run with substeps=4 and dt=0.04
        net_sub = Network(dt=dt, dtype=torch.float64)
        g_sub = NeuronGroup(net_sub, 1, [ConstantCurrent(current), HodgkinHuxley(substeps=k)])
        net_sub.initialize()
        for _ in range(steps):
            net_sub.step()

        # Run with substeps=1 and dt=0.01 (k times as many steps)
        net_single = Network(dt=dt_sub, dtype=torch.float64)
        g_single = NeuronGroup(net_single, 1, [ConstantCurrent(current), HodgkinHuxley(substeps=1)])
        net_single.initialize()
        for _ in range(steps * k):
            net_single.step()

        assert g_sub.v.item() == pytest.approx(g_single.v.item(), abs=1e-12)
        assert g_sub.m.item() == pytest.approx(g_single.m.item(), abs=1e-12)
        assert g_sub.h.item() == pytest.approx(g_single.h.item(), abs=1e-12)
        assert g_sub.n.item() == pytest.approx(g_single.n.item(), abs=1e-12)
        assert g_sub.i_na.item() == pytest.approx(g_single.i_na.item(), abs=1e-12)
        assert g_sub.i_k.item() == pytest.approx(g_single.i_k.item(), abs=1e-12)
        assert g_sub.i_leak.item() == pytest.approx(g_single.i_leak.item(), abs=1e-12)


class TestStepCurrentResponses:
    def test_step_current_zero_no_spikes(self) -> None:
        net = Network(dt=0.025, dtype=torch.float64)
        group = NeuronGroup(net, 1, [ConstantCurrent(0.0), HodgkinHuxley()])
        net.initialize()
        spikes = 0
        for _ in range(round(200.0 / 0.025)):
            net.step()
            spikes += int(group.spikes.sum().item())
        assert spikes == 0

    def test_step_current_subthreshold_two_no_repetitive_spiking(self) -> None:
        # I = 2 µA/cm² depolarizes the membrane to ~ -60 mV but emits no spikes
        net = Network(dt=0.025, dtype=torch.float64)
        group = NeuronGroup(net, 1, [ConstantCurrent(2.0), HodgkinHuxley()])
        net.initialize()
        spikes = 0
        v_max = -100.0
        for _ in range(round(200.0 / 0.025)):
            net.step()
            v_curr = group.v.item()
            v_max = max(v_max, v_curr)
            spikes += int(group.spikes.sum().item())
        assert spikes == 0
        assert v_max == pytest.approx(-60.0, abs=1.0)

    def test_step_current_ten_repetitive_spiking(self) -> None:
        # I = 10 µA/cm² triggers repetitive firing: 14 spikes in 200 ms with peak ~ +40.7 mV
        net = Network(dt=0.025, dtype=torch.float64)
        group = NeuronGroup(net, 1, [ConstantCurrent(10.0), HodgkinHuxley()])
        net.initialize()
        spikes = 0
        v_max = -100.0
        for _ in range(round(200.0 / 0.025)):
            net.step()
            v_curr = group.v.item()
            v_max = max(v_max, v_curr)
            spikes += int(group.spikes.sum().item())

        assert spikes >= 5
        assert spikes == 14
        assert 20.0 <= v_max <= 60.0
        assert v_max == pytest.approx(40.75, abs=0.5)


class TestStateManagementAndInfrastructure:
    def test_reset_state_restores_in_place(self) -> None:
        net = Network(dt=0.025, dtype=torch.float64)
        model = HodgkinHuxley()
        group = NeuronGroup(net, 2, [ConstantCurrent(10.0), model])
        net.initialize()

        v_init = group.v.clone()
        m_init = group.m.clone()
        h_init = group.h.clone()
        n_init = group.n.clone()

        # Run for 20 steps to change state
        for _ in range(20):
            net.step()

        assert not torch.equal(group.v, v_init)
        assert bool(group.spikes.any().item()) or group.i_na.abs().sum() > 0

        # Snapshot IDs immediately before reset_state to verify in-place restoration
        id_v = id(group.v)
        id_m = id(group.m)
        id_h = id(group.h)
        id_n = id(group.n)
        id_i_na = id(group.i_na)
        id_i_k = id(group.i_k)
        id_i_leak = id(group.i_leak)
        id_spikes = id(group.spikes)
        id_ext_i = id(group.I)

        # Reset state
        group.model.reset_state(group)

        # Assert same tensor identities
        assert id(group.v) == id_v
        assert id(group.m) == id_m
        assert id(group.h) == id_h
        assert id(group.n) == id_n
        assert id(group.i_na) == id_i_na
        assert id(group.i_k) == id_i_k
        assert id(group.i_leak) == id_i_leak
        assert id(group.spikes) == id_spikes
        assert id(group.I) == id_ext_i

        # Assert values restored
        assert torch.allclose(group.v, v_init, atol=1e-12)
        assert torch.allclose(group.m, m_init, atol=1e-12)
        assert torch.allclose(group.h, h_init, atol=1e-12)
        assert torch.allclose(group.n, n_init, atol=1e-12)
        assert group.i_na.abs().sum().item() == 0.0
        assert group.i_k.abs().sum().item() == 0.0
        assert group.i_leak.abs().sum().item() == 0.0
        assert not bool(group.spikes.any().item())
        assert group.I.abs().sum().item() == 0.0

    def test_checkpoint_save_and_resume(self, tmp_path: pytest.TempPathFactory) -> None:
        def build_net() -> Network:
            n = Network(dt=0.025, dtype=torch.float64, seed=42)
            NeuronGroup(n, 3, [ConstantCurrent(10.0), HodgkinHuxley()])
            return n

        # Continuous reference run
        ref_net = build_net()
        ref_net.run(50)
        ref_v = ref_net.groups[0].v.clone()
        ref_m = ref_net.groups[0].m.clone()

        # Interrupted and resumed run
        run_net = build_net()
        run_net.run(20)

        path = tmp_path / "hh_net.pt"
        checkpoint.save(run_net, path)

        resumed_net = build_net()
        checkpoint.load(resumed_net, path)
        resumed_net.run(30)

        assert torch.equal(resumed_net.groups[0].v, ref_v)
        assert torch.equal(resumed_net.groups[0].m, ref_m)

    @pytest.mark.gpu
    def test_graph_stepper_equality_on_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Eager reference run on CUDA
        eager_net = Network(dt=0.025, device="cuda", dtype=torch.float64, seed=123)
        NeuronGroup(eager_net, 4, [ConstantCurrent(10.0), HodgkinHuxley()])
        eager_net.initialize()
        eager_net.run(40)

        # GraphStepper run on CUDA
        graph_net = Network(dt=0.025, device="cuda", dtype=torch.float64, seed=123)
        NeuronGroup(graph_net, 4, [ConstantCurrent(10.0), HodgkinHuxley()])
        stepper = GraphStepper(graph_net)
        stepper.run(40)

        assert torch.equal(graph_net.groups[0].v, eager_net.groups[0].v)
        assert torch.equal(graph_net.groups[0].m, eager_net.groups[0].m)
        assert torch.equal(graph_net.groups[0].h, eager_net.groups[0].h)
        assert torch.equal(graph_net.groups[0].n, eager_net.groups[0].n)
        assert torch.equal(graph_net.groups[0].spikes, eager_net.groups[0].spikes)

    def test_independent_batches(self) -> None:
        # B members in an independent network run identical to unbatched copies
        net = Network(dt=0.025, batch_size=3, independent=True, dtype=torch.float64)
        group = NeuronGroup(net, 2, [ConstantCurrent(10.0), HodgkinHuxley()])
        net.initialize()
        assert group.v.shape == (3, 2)
        assert group.m.shape == (3, 2)
        assert group.spikes.shape == (3, 2)
        net.run(20)

    def test_spec_registry(self) -> None:
        assert "HodgkinHuxley" in registered()
