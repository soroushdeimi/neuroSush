import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Compartment, Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order


class Recorder(Behavior):
    """Appends (label, phase, iteration) to a shared log."""

    def __init__(self, label, log, order=Order.NEURON_DYNAMICS):
        self.label = label
        self.log = log
        self.order = order

    def initialize(self, host):
        self.log.append((self.label, "init", host))

    def forward(self, host):
        net = host if isinstance(host, Network) else host.net
        self.log.append((self.label, "step", net.iteration))


class TestNetwork:
    def test_defaults(self):
        net = Network()
        assert net.dt == 1.0
        assert net.dtype == torch.float32
        assert net.device == torch.device("cpu")
        assert net.iteration == 0
        assert net.initialized is False
        assert net.groups == []
        assert net.synapses == []

    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_dt_must_be_positive(self, dt):
        with pytest.raises(ValueError, match="dt"):
            Network(dt=dt)

    def test_dtype_must_be_floating(self):
        with pytest.raises(TypeError, match="dtype"):
            Network(dtype=torch.int64)

    def test_seed_makes_random_draws_reproducible(self):
        a = NeuronGroup(Network(seed=7), 5).rand()
        b = NeuronGroup(Network(seed=7), 5).rand()
        assert torch.equal(a, b)

    def test_run_counts_iterations(self):
        net = Network()
        net.run(3)
        assert net.iteration == 3
        assert net.initialized is True

    def test_run_rejects_negative_steps(self):
        with pytest.raises(ValueError, match="steps"):
            Network().run(-1)

    def test_initialize_twice_is_an_error(self):
        net = Network()
        net.initialize()
        with pytest.raises(RuntimeError, match="initialized"):
            net.initialize()

    def test_iteration_is_one_during_first_step(self):
        log = []
        net = Network(behaviors=[Recorder("n", log, order=Order.PAYOFF)])
        net.step()
        assert log[-1] == ("n", "step", 1)


class TestSchedule:
    def test_behaviors_run_by_order_then_registration(self):
        log = []
        net = Network(behaviors=[Recorder("net-late", log, order=Order.PLASTICITY)])
        NeuronGroup(
            net,
            2,
            behaviors=[
                Recorder("fire", log, order=Order.FIRE),
                Recorder("dyn", log, order=Order.NEURON_DYNAMICS),
            ],
        )
        NeuronGroup(net, 2, behaviors=[Recorder("dyn2", log, order=Order.NEURON_DYNAMICS)])
        net.step()
        inits = [label for label, phase, _ in log if phase == "init"]
        steps = [label for label, phase, _ in log if phase == "step"]
        assert inits == ["dyn", "dyn2", "fire", "net-late"]
        assert steps == inits

    def test_initialize_receives_the_host(self):
        log = []
        net = Network()
        group = NeuronGroup(net, 3, behaviors=[Recorder("g", log)])
        net.initialize()
        assert log == [("g", "init", group)]

    def test_disabled_behavior_is_skipped(self):
        log = []
        rec = Recorder("g", log)
        net = Network()
        NeuronGroup(net, 1, behaviors=[rec])
        net.initialize()
        rec.enabled = False
        net.run(2)
        assert [entry for entry in log if entry[1] == "step"] == []

    def test_schedule_lists_hosts_and_behaviors(self):
        rec = Recorder("g", [])
        net = Network()
        group = NeuronGroup(net, 1, behaviors=[rec])
        net.initialize()
        assert net.schedule == [(group, rec)]

    def test_behavior_without_order_is_rejected(self):
        class NoOrder(Behavior):
            pass

        with pytest.raises(TypeError, match="order"):
            NeuronGroup(Network(), 1, behaviors=[NoOrder()])

    def test_non_behavior_is_rejected(self):
        with pytest.raises(TypeError, match="Behavior"):
            NeuronGroup(Network(), 1, behaviors=[object()])

    def test_behavior_instance_cannot_be_shared(self):
        rec = Recorder("g", [])
        net = Network()
        NeuronGroup(net, 1, behaviors=[rec])
        with pytest.raises(ValueError, match="already attached"):
            NeuronGroup(net, 1, behaviors=[rec])

    def test_no_new_objects_after_initialize(self):
        net = Network()
        net.initialize()
        with pytest.raises(RuntimeError, match="initialized"):
            NeuronGroup(net, 1)


class TestNeuronGroup:
    def test_int_size_is_a_flat_shape(self):
        group = NeuronGroup(Network(), 4)
        assert group.shape == (1, 1, 4)
        assert group.size == 4

    def test_three_dimensional_shape(self):
        group = NeuronGroup(Network(), (2, 3, 4))
        assert (group.depth, group.height, group.width) == (2, 3, 4)
        assert group.size == 24

    @pytest.mark.parametrize("shape", [0, -3, (2, 3), (1, 0, 2), (1.5, 1, 1), "4"])
    def test_invalid_shape(self, shape):
        with pytest.raises(ValueError, match="shape"):
            NeuronGroup(Network(), shape)

    def test_default_names_are_unique(self):
        net = Network()
        assert [NeuronGroup(net, 1).name for _ in range(2)] == ["ng0", "ng1"]

    def test_duplicate_name_is_rejected(self):
        net = Network()
        NeuronGroup(net, 1, name="exc")
        with pytest.raises(ValueError, match="exc"):
            NeuronGroup(net, 1, name="exc")

    def test_tags_and_inhibitory_flag(self):
        group = NeuronGroup(Network(), 1, tags=("L4", "exc"), inhibitory=True)
        assert group.tags == frozenset({"L4", "exc"})
        assert group.inhibitory is True

    def test_vector_uses_network_dtype_and_fill(self):
        group = NeuronGroup(Network(dtype=torch.float64), 3)
        v = group.vector(-65.0)
        assert v.dtype == torch.float64
        assert v.tolist() == [-65.0, -65.0, -65.0]
        assert group.vector(dtype=torch.bool).tolist() == [False, False, False]

    def test_random_vectors_have_group_size(self):
        group = NeuronGroup(Network(seed=0), 6)
        assert group.rand().shape == (6,)
        assert group.randn().shape == (6,)
        assert bool(((group.rand() >= 0) & (group.rand() < 1)).all())

    def test_group_is_registered(self):
        net = Network()
        group = NeuronGroup(net, 1)
        assert net.groups == [group]
        assert group.net is net


class TestSynapseGroup:
    def test_registers_as_afferent_and_efferent(self):
        net = Network()
        a, b = NeuronGroup(net, 2), NeuronGroup(net, 3)
        syn = SynapseGroup(net, a, b, compartment="distal")
        assert syn.compartment is Compartment.DISTAL
        assert b.afferent[Compartment.DISTAL] == [syn]
        assert a.efferent[Compartment.DISTAL] == [syn]
        assert b.afferent[Compartment.PROXIMAL] == []
        assert net.synapses == [syn]

    def test_default_delays_are_zero(self):
        net = Network()
        syn = SynapseGroup(net, NeuronGroup(net, 2), NeuronGroup(net, 3))
        assert syn.src_delay.tolist() == [0, 0]
        assert syn.dst_delay.tolist() == [0, 0, 0]
        assert syn.src_delay.dtype == torch.long
        assert syn.weights is None

    def test_default_name_and_compartment(self):
        net = Network()
        syn = SynapseGroup(net, NeuronGroup(net, 1), NeuronGroup(net, 1))
        assert syn.name == "sg0"
        assert syn.compartment is Compartment.PROXIMAL

    def test_groups_must_belong_to_the_network(self):
        net, other = Network(), Network()
        with pytest.raises(ValueError, match="network"):
            SynapseGroup(net, NeuronGroup(net, 1), NeuronGroup(other, 1))

    def test_unknown_compartment(self):
        net = Network()
        with pytest.raises(ValueError, match="compartment"):
            SynapseGroup(net, NeuronGroup(net, 1), NeuronGroup(net, 1), compartment="basal")
